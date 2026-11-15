import os, cv2, torch, numpy as np, shutil, glob, csv, gc, sys, random, pandas as pd
from ultralytics import YOLO

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM3_PATH = os.path.join(BASE_DIR, "sam3")
if SAM3_PATH not in sys.path: sys.path.append(SAM3_PATH)

GLOBAL_SAM_MODEL = None
GLOBAL_SAM_PREDICTOR = None

class TAAMEngine:
    def __init__(self, workspace, config, log_app):
        self.workspace = os.path.abspath(workspace)
        self.config = config
        self.log_app = log_app
        self.predictor = None
        self.stop_flag = False 
        self.colors = [(57, 255, 20), (50, 50, 255), (255, 100, 50), (0, 255, 255)]

    def log(self, msg):
        if self.log_app: self.log_app.emit(msg)
        else: print(msg)

    def load_sam3(self):
        global GLOBAL_SAM_MODEL, GLOBAL_SAM_PREDICTOR
        try:
            if GLOBAL_SAM_MODEL is not None:
                self.predictor = GLOBAL_SAM_PREDICTOR
                self.log("AI: SAM3 already loaded in memory.")
                return True

            from sam3.model_builder import build_sam3_video_model
            self.log("AI: Loading SAM3 Architecture...")
            with torch.inference_mode():
                GLOBAL_SAM_MODEL = build_sam3_video_model()
                GLOBAL_SAM_PREDICTOR = GLOBAL_SAM_MODEL.tracker
                GLOBAL_SAM_PREDICTOR.backbone = GLOBAL_SAM_MODEL.detector.backbone
            self.predictor = GLOBAL_SAM_PREDICTOR
            self.log("AI: SAM3 Loaded Successfully.")
            return True
        except Exception as e:
            self.log(f"AI: ❌ Failed to load SAM3: {e}")
            self._free_sam3_memory()
            return False

    def _free_sam3_memory(self):
        """Safely delete SAM3 from GPU memory."""
        global GLOBAL_SAM_MODEL, GLOBAL_SAM_PREDICTOR
        try:
            if GLOBAL_SAM_MODEL is not None:
                self.log("AI: Releasing SAM3 from GPU memory...")
                del GLOBAL_SAM_MODEL
                GLOBAL_SAM_MODEL = None
            if GLOBAL_SAM_PREDICTOR is not None:
                del GLOBAL_SAM_PREDICTOR
                GLOBAL_SAM_PREDICTOR = None
            self.predictor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.log("AI: SAM3 memory cleared successfully.")
        except Exception as e:
            self.log(f"AI: Error during SAM3 memory cleanup: {e}")

    def run_full_pipeline(self, video_paths, annotations_map, model_name, progress_callback):
        try:
            video_info = self.run_sam3_only(video_paths, annotations_map, model_name, progress_callback)
            
            # Deactivated forceful purging of the compiled global model in active runs to prevent SIGSEGV
            # self._free_sam3_memory()
            
            if video_info and not self.stop_flag:
                return self.run_yolo_only(video_info, model_name, progress_callback)
            return False
        except Exception as e:
            self.log(f"PIPELINE ERROR: {e}")
            return False

    def run_sam3_only(self, video_paths, annotations_map, model_name, progress_callback):
        try:
            if not self.load_sam3():
                return None

            exp_root = os.path.join(self.workspace, "Experiments", model_name)
            ds_dir = os.path.abspath(os.path.join(self.workspace, "Datasets", model_name))
            pool_dir = os.path.join(ds_dir, "sampling_pool")
            os.makedirs(exp_root, exist_ok=True); os.makedirs(ds_dir, exist_ok=True); os.makedirs(pool_dir, exist_ok=True)

            video_info = {}
            for v_idx, v_path in enumerate(video_paths):
                if self.stop_flag: break

                v_filename_full = os.path.basename(v_path)
                v_name = os.path.splitext(v_filename_full)[0]
                self.log(f"PROJECT: Tracking {v_name} ({v_idx+1}/{len(video_paths)})")
                v_exp_dir = os.path.join(exp_root, v_name); os.makedirs(v_exp_dir, exist_ok=True)

                csv_p = self._process_video_sam3(
                    v_path,
                    annotations_map.get(v_path, {}),
                    v_exp_dir,
                    pool_dir,
                    progress_callback
                )

                if csv_p:
                    video_info[v_path] = csv_p

            return video_info
        except RuntimeError as re:
            self.log(f"SAM3 MEMORY ERROR: {re}")
            return None
        except Exception as e:
            self.log(f"SAM3 ERROR: {e}")
            return None

    def run_yolo_only(self, video_info, model_name, progress_callback):
        try:
            ds_dir = os.path.abspath(os.path.join(self.workspace, "Datasets", model_name))
            
            self.log("SYSTEM: Preparing YOLO dataset folders...")

            # Clean and regenerate everything under ds_dir EXCEPT sampling_pool to ensure fully fresh splits
            if os.path.exists(ds_dir):
                for item in os.listdir(ds_dir):
                    item_path = os.path.join(ds_dir, item)
                    if item == "sampling_pool":
                        continue
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
            else:
                os.makedirs(ds_dir, exist_ok=True)

            if not video_info:
                self.log(f"🔍 YOLO-Only Mode: Discovering existing SAM3 outputs for {model_name}...")
                video_info = self._discover_existing_data(model_name)

            clean_video_info = {}
            for v_path, csv_p in (video_info or {}).items():
                if os.path.exists(csv_p):
                    clean_video_info[v_path] = csv_p
                else:
                    self.log(f"SYSTEM: Skipping deleted SAM3 output for video: {os.path.basename(str(v_path))}")

            if not clean_video_info:
                self.log("❌ Error: No valid SAM3 tracking folders found.")
                return False

            yaml_path = self._generate_yolo_dataset(clean_video_info, ds_dir, progress_callback)

            if yaml_path and not self.stop_flag:
                # Crucial step: delete all stale YOLO dataset cache files to force binary rebuilding
                for cache_file in glob.glob(os.path.join(ds_dir, "**", "*.cache"), recursive=True):
                    try:
                        os.remove(cache_file)
                    except Exception:
                        pass

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Set task and default weights dynamically based on Semantic Segmentation vs others
                is_semantic = self.config.get('task_type') == "Semantic Segmentation"
                if is_semantic:
                    task = "semantic"
                    default_weights = "yolo26n-sem.pt"
                else:
                    task = "segment" if self.config['task_type'] == "Segmentation" else "detect"
                    default_weights = self.config.get('yolo_ver') or "yolov8n.pt"
                
                cust_w = self.config.get('custom_weights')
                model_base = cust_w if (cust_w and os.path.exists(cust_w)) else default_weights

                self.log(f"🚀 Starting YOLO Training [Task: {task.upper()}]...")

                model = YOLO(model_base)
                model.train(
                    data=yaml_path,
                    epochs=self.config['epochs'],
                    imgsz=self.config['imgsz'],
                    batch=self.config['yolo_batch'],
                    project=os.path.join(self.workspace, "Models"),
                    name=model_name,
                    workers=0,
                    device=0,
                    task=task,
                    exist_ok=True
                )

                self.log("✅ YOLO Training Completed Successfully.")
                return True

            self.log("❌ ERROR: Dataset preparation failed.")
            return False

        except Exception as e:
            self.log(f"YOLO ERROR: {e}")
            return False

    def _extract_single_frame_on_the_fly(self, video_path, frame_idx, pool_dir, target_filename):
        """
        Extracts a single frame from the video file on the fly and saves it to the pool directory.
        Used to dynamically heal the dataset if sampling_pool went missing.
        """
        actual_video_path = video_path
        if not os.path.exists(actual_video_path):
            # Look inside workspace if absolute path was moved
            search_pattern = os.path.join(self.workspace, "**", os.path.basename(video_path))
            found_paths = glob.glob(search_pattern, recursive=True)
            if found_paths:
                actual_video_path = found_paths[0]
            else:
                return None
                
        try:
            cap = cv2.VideoCapture(actual_video_path)
            if not cap.isOpened():
                return None
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                save_path = os.path.join(pool_dir, target_filename)
                cv2.imwrite(save_path, frame)
                return save_path
        except Exception:
            pass
        return None

    def _generate_yolo_dataset(self, video_info_map, dataset_root, cb):
        all_p = []
        
        # Determine the current expected pool directory based on model name
        model_name = os.path.basename(dataset_root)
        current_pool_dir = os.path.abspath(os.path.join(self.workspace, "Datasets", model_name, "sampling_pool"))
        os.makedirs(current_pool_dir, exist_ok=True)

        total_videos = len(video_info_map)
        for v_idx, (v_path, csv_p) in enumerate(video_info_map.items()):
            if not os.path.exists(csv_p):
                continue

            df = pd.read_csv(csv_p, sep='\t')
            df.columns = df.columns.str.strip()
            
            # Clean up the paths in the dataframe
            df["Image_Path"] = df["Image_Path"].astype(str).str.strip()
            
            # Defensive Path Resolution: Reconstruct missing images on-the-fly from original video
            resolved_paths = []
            total_rows = len(df)
            for idx, r_idx in df.iterrows():
                if self.stop_flag:
                    return None
                
                img_p = r_idx["Image_Path"]
                f_idx = int(r_idx["Global_Frame_ID"])
                
                # Dynamic visual progress logging during the scanning & healing phase (allocated to 0%-40%)
                if idx % 10 == 0 or idx == total_rows - 1:
                    overall_progress = int((v_idx / total_videos) * 40 + (idx / total_rows) * (40 / total_videos))
                    cb(overall_progress, f"Scanning Pool: {v_idx+1}/{total_videos} | Frame {f_idx:,} ({idx+1}/{total_rows})")
                
                if os.path.exists(img_p):
                    resolved_paths.append(img_p)
                else:
                    reconstructed = os.path.join(current_pool_dir, os.path.basename(img_p))
                    if os.path.exists(reconstructed):
                        resolved_paths.append(reconstructed)
                    else:
                        # Re-extract the frame directly from the source video file on the fly
                        cb(overall_progress, f"Healing Pool: Extracting frame {f_idx:,} from video...")
                        extracted_p = self._extract_single_frame_on_the_fly(v_path, f_idx, current_pool_dir, os.path.basename(img_p))
                        if extracted_p and os.path.exists(extracted_p):
                            resolved_paths.append(extracted_p)
                        else:
                            resolved_paths.append("")  # Exclude frame
            
            df["Image_Path"] = resolved_paths
            df = df[df["Image_Path"] != ""]
            
            if df.empty:
                continue

            for fid in df["Global_Frame_ID"].unique():
                all_p.append((v_path, fid, csv_p))

        if not all_p:
            self.log("ERROR: No valid images found in pool for dataset generation.")
            return None

        random.shuffle(all_p)
        max_f = self.config.get('max_frames', 500)
        sampled = all_p[:max_f]
        
        tr_ratio = self.config.get('tr', 70) / 100.0
        va_ratio = self.config.get('va', 20) / 100.0
        tr_lim = int(len(sampled) * tr_ratio)
        va_lim = tr_lim + int(len(sampled) * va_ratio)

        is_semantic = self.config.get('task_type') == "Semantic Segmentation"
        label_or_mask_folder = "masks" if is_semantic else "labels"

        for s in ['train', 'val', 'test']:
            os.makedirs(os.path.join(dataset_root, "images", s), exist_ok=True)
            os.makedirs(os.path.join(dataset_root, label_or_mask_folder, s), exist_ok=True)

        class_names = self.config.get('class_names', ['object'])
        bbox_mode = self.config.get('bbox_mode', 'Dynamic (From Mask)')
        fixed_size = self.config.get('fixed_bbox_size', 60)
        include_background = self.config.get('include_background', True)

        for i, (v_path, f_idx, csv_p) in enumerate(sampled):
            if self.stop_flag:
                return None

            if i < tr_lim: split = "train"
            elif i < va_lim: split = "val"
            else: split = "test"

            df = pd.read_csv(csv_p, sep='\t')
            df.columns = df.columns.str.strip()
            df["Image_Path"] = df["Image_Path"].astype(str).str.strip()
            
            rows = df[df["Global_Frame_ID"] == f_idx]
            if rows.empty:
                continue

            img_src = rows.iloc[0]["Image_Path"]
            
            # Apply dynamic path reconstruction if the copied location has shifted
            if not os.path.exists(img_src):
                img_src = os.path.join(current_pool_dir, os.path.basename(img_src))
                
            tmp_img = cv2.imread(img_src)
            if tmp_img is None: continue
            fh, fw = tmp_img.shape[:2]

            vid_name = os.path.splitext(os.path.basename(v_path))[0]
            base_n = f"{vid_name}_frame_{int(f_idx):06d}_{random.randint(100,999)}.jpg"
            shutil.copy(img_src, os.path.join(dataset_root, "images", split, base_n))

            # Render single-channel PNG masks if semantic, else generate standard coordinates labels txt
            if is_semantic:
                # If background is excluded, initialize the canvas with value '255' (YOLO ignore value)
                mask_img = np.zeros((fh, fw), dtype=np.uint8) if include_background else np.full((fh, fw), 255, dtype=np.uint8)
                
                for _, r in rows.iterrows():
                    poly = str(r.get("Polygon_Coords", ""))
                    cls_id = int(r.get("Class_ID", 0))
                    
                    # Shift class IDs up by 1 if background is included
                    semantic_val = (cls_id + 1) if include_background else cls_id
                    
                    coords = [list(map(float, p.split(','))) for p in poly.split(';') if ',' in p]
                    if coords:
                        pts = np.array(coords, dtype=np.int32)
                        cv2.fillPoly(mask_img, [pts], int(semantic_val))
                
                mask_name = base_n.replace(".jpg", ".png")
                cv2.imwrite(os.path.join(dataset_root, "masks", split, mask_name), mask_img)
            else:
                with open(os.path.join(dataset_root, "labels", split, base_n.replace(".jpg", ".txt")), "w") as f_lbl:
                    for _, r in rows.iterrows():
                        poly = str(r.get("Polygon_Coords", ""))
                        cls_id = int(r.get("Class_ID", 0))

                        if self.config['task_type'] == "Segmentation":
                            f_lbl.write(
                                f"{cls_id} " +
                                " ".join([f"{float(p.split(',')[0])/fw:.6f} {float(p.split(',')[1])/fh:.6f}"
                                        for p in poly.split(';') if ',' in p]) + "\n"
                            )
                        elif bbox_mode == "Fixed (Centered)":
                            cx, cy = float(r["Centroid_X"]), float(r["Centroid_Y"])
                            n_cx = cx / fw
                            n_cy = cy / fh
                            n_w = fixed_size / fw
                            n_h = fixed_size / fh
                            f_lbl.write(f"{cls_id} {n_cx:.6f} {n_cy:.6f} {n_w:.6f} {n_h:.6f}\n")
                        else:
                            coords = [list(map(float, p.split(','))) for p in poly.split(';') if ',' in p]
                            if coords:
                                xs, ys = [c[0] for c in coords], [c[1] for c in coords]
                                min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
                                bw, bh = max_x - min_x, max_y - min_y
                                f_lbl.write(
                                    f"{cls_id} {(min_x+bw/2)/fw:.6f} {(min_y+bh/2)/fh:.6f} {bw/fw:.6f} {bh/fh:.6f}\n"
                                )

            # Dynamic visual progress reporting during the splitting & label compilation phase (allocated to 40%-90%)
            cb(40 + int((i/len(sampled))*50), f"YOLO: Preparing {split} split ({i+1}/{len(sampled)})")

        cb(95, "YOLO: Finalizing dataset configuration...")
        yaml_p = os.path.join(dataset_root, "data.yaml")
        with open(yaml_p, "w") as f:
            f.write(f"path: {os.path.abspath(dataset_root)}\n")
            f.write(f"train: images/train\nval: images/val\ntest: images/test\n")
            
            if is_semantic:
                f.write(f"masks:\n  train: masks/train\n  val: masks/val\n  test: masks/test\n")
                f.write(f"names:\n")
                if include_background:
                    f.write(f"  0: background\n")
                    for idx, name in enumerate(class_names):
                        f.write(f"  {idx + 1}: {name}\n")
                else:
                    for idx, name in enumerate(class_names):
                        f.write(f"  {idx}: {name}\n")
            else:
                f.write(f"names:\n")
                for idx, name in enumerate(class_names):
                    f.write(f"  {idx}: {name}\n")

        return yaml_p

    def _discover_existing_data(self, model_name):
        """
        Scans strictly inside the Experiments folder matching the active model name.
        Resolves matching video source locations dynamically to keep path mapping consistent.
        """
        discovered = {}
        search_path = os.path.join(self.workspace, "Experiments", model_name)
        csv_files = glob.glob(os.path.join(search_path, "**", "*_data.csv"), recursive=True)
        
        for cp in csv_files:
            v_name = os.path.basename(cp).replace("_data.csv", "")
            
            # Find matching video file path in workspace to establish correct keys mapping
            video_matches = glob.glob(os.path.join(self.workspace, "**", f"{v_name}.*"), recursive=True)
            video_matches = [m for m in video_matches if os.path.splitext(m)[1].lower() in ['.mp4', '.avi', '.mov', '.mkv']]
            
            v_path = video_matches[0] if video_matches else f"{v_name}.mp4"
            discovered[v_path] = cp
            
        return discovered

    def _process_video_sam3(self, v_path, ann, out_dir, pool_dir, cb):
        temp, proc = os.path.join(out_dir, "temp"), os.path.join(out_dir, "proc")
        os.makedirs(temp, exist_ok=True); os.makedirs(proc, exist_ok=True)
        name = os.path.splitext(os.path.basename(v_path))[0]
        csv_p = os.path.join(out_dir, f"{name}_data.csv")
        vid_p = os.path.join(out_dir, f"{name}_tracked.mp4")
        
        with open(csv_p, 'w', newline='') as f: 
            csv.writer(f, delimiter='\t').writerow(["Global_Frame_ID", "Object_ID", "Class_ID", "Centroid_X", "Centroid_Y", "Size_Pixels", "Size_um2", "Polygon_Coords", "Image_Path"])
        cap = cv2.VideoCapture(v_path); fps, w, h = cap.get(5), int(cap.get(3)), int(cap.get(4))
        chunk_paths = self._split_video(cap, fps, w, h, temp); cap.release()

        prompts = {}
        if 0 in ann:
            for i, (rect, cid) in enumerate(ann[0]):
                prompts[i+1] = {'pt': (rect.center().x() / w, rect.center().y() / h), 'cls': cid}

        total_c = len(chunk_paths)
        for i, cp in enumerate(chunk_paths):
            if self.stop_flag or not prompts: break
            cb(10 + int((i/total_c)*40), f"SAM3 Chunk {i+1}/{total_c}")
            save_p = os.path.join(proc, f"p_{os.path.basename(cp)}")
            
            prompts = self._track_chunk(cp, i, prompts, fps, csv_p, pool_dir, w, h, save_p)

        self._stitch(proc, vid_p, fps, w, h)
        shutil.rmtree(temp, ignore_errors=True); shutil.rmtree(proc, ignore_errors=True)
        return csv_p

    def _track_chunk(self, cp, c_idx, prompts, fps, csv_path, pool_dir, w_orig, h_orig, save_p):
        obj_batch_size = int(self.config.get('sam_batch', 1))
        if obj_batch_size < 1: obj_batch_size = 1 
        
        prompt_items = list(prompts.items())
        all_frames_data = {} 
        last_masks_for_stitch = {}
        
        cap_temp = cv2.VideoCapture(cp)
        frame_count = int(cap_temp.get(7))
        cap_temp.release()

        for b_start in range(0, len(prompt_items), obj_batch_size):
            batch = prompt_items[b_start : b_start + obj_batch_size]
            self.log(f"AI: Tracking Object Batch {b_start//obj_batch_size + 1} (Size: {len(batch)})")

            state = self.predictor.init_state(video_path=cp, offload_video_to_cpu=True)
            
            try:
                with torch.inference_mode():
                    for oid, data in batch:
                        pt_norm = data['pt']
                        self.predictor.add_new_points(
                            state, 0, oid,
                            torch.tensor([[pt_norm[0], pt_norm[1]]], dtype=torch.float32, device="cuda"),
                            torch.tensor([1], dtype=torch.int32, device="cuda")
                        )

                    gen = self.predictor.propagate_in_video(
                        state, 
                        start_frame_idx=0, 
                        max_frame_num_to_track=frame_count, 
                        reverse=False, 
                        propagate_preflight=True
                    )

                    for f_idx, oids, _, masks, _ in gen:
                        if f_idx not in all_frames_data:
                            all_frames_data[f_idx] = {}
                        
                        for j, oid in enumerate(oids):
                            mask_np = (masks[j] > 0.0).cpu().numpy().squeeze()
                            centroid, area = self._get_centroid_area(mask_np)
                            if centroid:
                                poly_str = self._get_polygon_str(mask_np)
                                all_frames_data[f_idx][oid] = {
                                    'pt': centroid, 
                                    'cls': prompts[oid]['cls'], 
                                    'area': area, 
                                    'poly': poly_str
                                }
                                if f_idx == frame_count - 1:
                                    last_masks_for_stitch[oid] = mask_np

                self.predictor.clear_all_points_in_video(state)
            
            except Exception as e:
                self.log(f"AI BATCH ERROR: {e}")
            
            finally:
                del state
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        video_base = os.path.splitext(os.path.basename(csv_path))[0].replace("_data", "")
        cap_read = cv2.VideoCapture(cp)
        with open(csv_path, 'a', newline='') as f_csv:
            csv_w = csv.writer(f_csv, delimiter='\t')
            for f_idx in range(frame_count):
                ret, frame = cap_read.read()
                if not ret: break

                if f_idx in all_frames_data and all_frames_data[f_idx]:
                    global_frame_id = (c_idx * frame_count) + f_idx
                    img_name = f"{video_base}_frame_{int(global_frame_id):06d}.jpg"
                    img_p = os.path.abspath(os.path.join(pool_dir, img_name))
                    cv2.imwrite(img_p, frame)

                    for oid, info in all_frames_data[f_idx].items():
                        csv_w.writerow([
                            global_frame_id, oid, info['cls'],
                            info['pt'][0], info['pt'][1], info['area'], 0, info['poly'], img_p
                        ])
        cap_read.release()

        self._render(cp, save_p, fps, all_frames_data)

        next_prompts = {}
        for oid in prompts.keys():
            if oid in last_masks_for_stitch:
                m = last_masks_for_stitch[oid]
                c, _ = self._get_centroid_area(m)
                if c:
                    next_prompts[oid] = {
                        'pt': (c[0]/w_orig, c[1]/h_orig), 
                        'cls': prompts[oid]['cls']
                    }
        return next_prompts

    def _render(self, inp, out, fps, data):
        cap = cv2.VideoCapture(inp)
        w, h = int(cap.get(3)), int(cap.get(4))
        
        # Robust safety check to prevent 0x0 VideoWriter failures on drive sync latency
        if w <= 0 or h <= 0:
            ret, first_frame = cap.read()
            if ret and first_frame is not None:
                h, w = first_frame.shape[:2]
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                w, h = 1920, 1080  # Fallback dimensions
                
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        curr_f = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if curr_f in data:
                for oid, info in data[curr_f].items():
                    color = self.colors[info['cls'] % len(self.colors)]
                    cv2.circle(frame, info['pt'], 5, color, -1)
                    cv2.putText(frame, f"ID:{oid}", (info['pt'][0]+8, info['pt'][1]-8), 0, 0.6, (255,255,255), 2)
            writer.write(frame); curr_f += 1
        cap.release(); writer.release()

    def _get_centroid_area(self, mask):
        a = np.count_nonzero(mask); rows, cols = np.where(mask)
        return ((int(np.mean(cols)), int(np.mean(rows))), a) if len(rows) > 0 else (None, 0)

    def _get_polygon_str(self, mask):
        m8 = (mask.astype(np.uint8)) * 255; cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return ""
        c = max(cnts, key=cv2.contourArea)
        return ";".join([f"{p[0][0]},{p[0][1]}" for p in cv2.approxPolyDP(c, 0.005*cv2.arcLength(c, True), True)])

    def _split_video(self, cap, fps, w, h, out_dir):
        per = int(fps * self.config['chunk_duration']); paths, idx = [], 0
        
        # Robust safety dimensions check on split
        if w <= 0 or h <= 0:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w <= 0 or h <= 0:
                w, h = 1920, 1080
                
        while True:
            ret, frame = cap.read()
            if not ret: break
            out_f = os.path.abspath(os.path.join(out_dir, f"c_{idx:03d}.mp4"))
            writer = cv2.VideoWriter(out_f, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            cnt = 0
            while cnt < per:
                if frame is None: break
                writer.write(frame); cnt += 1
                ret, frame = cap.read()
                if not ret: break
            writer.release(); paths.append(out_f); idx += 1
        return paths

    def _stitch(self, src_dir, out_path, fps, w, h):
        files = sorted(glob.glob(os.path.join(src_dir, "*.mp4")))
        
        # Robust dimensions check on stitch files
        if (w <= 0 or h <= 0) and files:
            cap_test = cv2.VideoCapture(files[0])
            w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w <= 0 or h <= 0:
                ret, first_frame = cap_test.read()
                if ret and first_frame is not None:
                    h, w = first_frame.shape[:2]
            cap_test.release()
            
        if w <= 0 or h <= 0:
            w, h = 1920, 1080
            
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in files:
            cap = cv2.VideoCapture(f)
            while True:
                ret, frame = cap.read()
                if not ret: break
                writer.write(frame)
            cap.release()
        writer.release()