import os, time, random, traceback
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
from PyQt6.QtCore import QThread, pyqtSignal

try:
    import cv2
    import torch
    from ultralytics import YOLO
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass

class Stopwatch:
    def __init__(self): self.start_time = 0
    def start(self): self.start_time = time.time()
    def get_elapsed_time(self, as_float=False):
        el = time.time() - self.start_time
        return el if as_float else time.strftime("%H:%M:%S", time.gmtime(el))
    def get_etr(self, current, total):
        if current <= 0 or total <= 0: return "--:--:--"
        etr = ((time.time() - self.start_time) / current) * (total - current)
        return time.strftime("%H:%M:%S", time.gmtime(etr))


# ==============================================================================
# AMADEUS VISUAL-KINEMATIC TRACKER (Real-Time DeepSORT Alternative)
# ==============================================================================
class AmadeusTrack:
    def __init__(self, track_id, det):
        self.track_id = track_id
        self.cx = det['cx']
        self.cy = det['cy']
        self.box = det['box']
        self.vel = (0.0, 0.0)
        self.features = deque(maxlen=100)
        if 'feature' in det:
            self.features.append(det['feature'])
            
        self.time_since_update = 0
        self.hits = 1
        self.frozen = False
        
        # Store latest full det info (skeleton etc.)
        self.latest_det = det

    def predict(self):
        # Predict forward using momentum decayed by friction
        if self.time_since_update > 0:
            self.vel = (self.vel[0] * 0.9, self.vel[1] * 0.9)
            
        self.cx += self.vel[0]
        self.cy += self.vel[1]
        
        # Shift bounding box
        bw = self.box[2] - self.box[0]
        bh = self.box[3] - self.box[1]
        self.box = [self.cx - bw/2, self.cy - bh/2, self.cx + bw/2, self.cy + bh/2]
        
        self.time_since_update += 1

    def update(self, det):
        self.time_since_update = 0
        self.hits += 1
        
        # Update Velocity
        self.vel = (det['cx'] - self.cx, det['cy'] - self.cy)
        
        self.cx = det['cx']
        self.cy = det['cy']
        self.box = det['box']
        
        if not self.frozen and 'feature' in det:
            self.features.append(det['feature'])
            
        self.latest_det = det

class AmadeusOnlineTracker:
    def __init__(self, config):
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        self.w_dist = config.get('weight_dist', 0.3)
        self.w_vis = config.get('weight_vis', 0.4)
        self.w_dir = config.get('weight_dir', 0.3)
        self.max_n = config.get('max_n', 999)
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        # 1. Predict Track States
        for track in self.tracks:
            track.predict()
            
        # Freeze visual templates for tracks that are occluding/colliding (< 120px)
        for i, t1 in enumerate(self.tracks):
            t1.frozen = False
            for j, t2 in enumerate(self.tracks):
                if i != j:
                    dist = np.hypot(t1.cx - t2.cx, t1.cy - t2.cy)
                    if dist < 120.0:
                        t1.frozen = True
                        break

        matched, unmatched_dets, unmatched_trks = [], list(range(len(detections))), list(range(len(self.tracks)))

        # 2. Stage-1: Joint Cost Matching (Distance + Visual + Direction)
        if len(self.tracks) > 0 and len(detections) > 0:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    # Spatial
                    dist = np.hypot(track.cx - det['cx'], track.cy - det['cy'])
                    if dist > 250.0:
                        cost_matrix[t_idx, d_idx] = 1e6
                        continue
                    dist_cost = min(1.0, dist / 250.0)
                    
                    # Visual Feature (Bhattacharyya)
                    vis_cost = 0.0
                    if 'feature' in det and track.features:
                        median_feat = np.median(list(track.features), axis=0).astype(np.float32)
                        det_feat = det['feature'].astype(np.float32)
                        vis_cost = cv2.compareHist(median_feat, det_feat, cv2.HISTCMP_BHATTACHARYYA)
                        
                    # Directional
                    dir_cost = 0.0
                    mag_v = np.hypot(track.vel[0], track.vel[1])
                    disp_x = det['cx'] - track.cx
                    disp_y = det['cy'] - track.cy
                    mag_d = np.hypot(disp_x, disp_y)
                    if mag_v > 1e-3 and mag_d > 1e-3:
                        cos_sim = (track.vel[0] * disp_x + track.vel[1] * disp_y) / (mag_v * mag_d)
                        dir_cost = 1.0 - max(0.0, cos_sim)
                        
                    cost_matrix[t_idx, d_idx] = (self.w_dist * dist_cost) + (self.w_vis * vis_cost) + (self.w_dir * dir_cost)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self.tracks)))
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 2.0:
                    matched.append((r, c))
                    unmatched_dets.remove(c)
                    unmatched_trks.remove(r)

        # 3. Update Matched Tracks
        for t_idx, d_idx in matched:
            self.tracks[t_idx].update(detections[d_idx])

        # 4. Initialize New Tracks
        for d_idx in unmatched_dets:
            if len([t for t in self.tracks if t.time_since_update == 0]) < self.max_n:
                self.tracks.append(AmadeusTrack(self.next_id, detections[d_idx]))
                self.next_id += 1

        # 5. Remove Lost Tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 6. Extract Valid Results (Hits >= min_hits or new but active)
        valid_results = []
        for t in self.tracks:
            if t.time_since_update == 0 and (t.hits >= self.min_hits or self.min_hits == 1):
                out_det = t.latest_det.copy()
                out_det['track_id'] = t.track_id
                valid_results.append(out_det)
                
        return valid_results


class ArenaWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)

    def __init__(self, config, workspace):
        super().__init__()
        self.config = config
        self.workspace = workspace
        self.is_running = True
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        
        self.b_thick = self.config.get("bbox_thick", 3)
        self.l_scale = self.config.get("lbl_scale", 0.6)
        self.l_thick = self.config.get("lbl_thick", 2)
        self.dot_sz = self.config.get("dot_size", 5)

        self.class_palette = [
            (57, 255, 20), (255, 0, 127), (0, 255, 255), (255, 102, 0),
            (204, 0, 255), (0, 102, 255), (255, 255, 255)
        ]

        skeleton_rois_list = self.config.get('skeleton_rois', [])
        if not skeleton_rois_list or "all" in [str(x).lower() for x in skeleton_rois_list]:
            self.skeleton_roi_set = None
        else:
            self.skeleton_roi_set = set()
            for x in skeleton_rois_list:
                parts = str(x).split('_')
                if len(parts) > 1 and parts[1].isdigit():
                    self.skeleton_roi_set.add(int(parts[1]))
                    
        self.all_arenas = []

    def stop(self):
        self.is_running = False

    def _get_grid_cells(self, roi):
        cells = []
        rx, ry, rw, rh = roi['x'], roi['y'], roi['w'], roi['h']
        r, c = roi['grid']
        cw, ch = rw/c, rh/r
        for row in range(r):
            for col in range(c):
                cells.append({'type':'rect','x':rx+col*cw,'y':ry+row*ch,'w':cw,'h':ch})
        return cells

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if (boxAArea + boxBArea - interArea) == 0: return 0.0
        return interArea / float(boxAArea + boxBArea - interArea)

    def _extract_fish_keypoints(self, mask_res, fallback_cx, fallback_cy, crop_offset_x, crop_offset_y):
        y_indices, x_indices = np.nonzero(mask_res)
        if len(x_indices) < 15:
            fallback = (fallback_cx + crop_offset_x, fallback_cy + crop_offset_y)
            return fallback, fallback, fallback, fallback, fallback
            
        h_mask, w_mask = mask_res.shape[:2]
        kernel_sz = max(3, int(min(h_mask, w_mask) * 0.15))
        if kernel_sz % 2 == 0:
            kernel_sz += 1
            
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_sz, kernel_sz))
        core_mask = cv2.morphologyEx(mask_res, cv2.MORPH_OPEN, morph_kernel)
        
        if np.count_nonzero(core_mask) < 15:
            core_mask = mask_res
            
        M = cv2.moments(core_mask)
        if M['m00'] > 0:
            lcx = M['m10'] / M['m00']
            lcy = M['m01'] / M['m00']
        else:
            lcx, lcy = fallback_cx, fallback_cy
            
        center_pt = (lcx + crop_offset_x, lcy + crop_offset_y)
        
        contours, _ = cv2.findContours(mask_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            fallback = (fallback_cx + crop_offset_x, fallback_cy + crop_offset_y)
            return fallback, fallback, fallback, fallback, fallback
            
        cnt = max(contours, key=cv2.contourArea)
        cnt_pts = cnt.reshape(-1, 2).astype(np.float32)
        
        dists_to_center = (cnt_pts[:, 0] - lcx)**2 + (cnt_pts[:, 1] - lcy)**2
        tail_idx = np.argmax(dists_to_center)
        ltail = (float(cnt_pts[tail_idx][0]), float(cnt_pts[tail_idx][1]))
        
        mu20 = M['mu20']
        mu02 = M['mu02']
        mu11 = M['mu11']
        angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        projected = (cnt_pts[:, 0] - lcx) * cos_a + (cnt_pts[:, 1] - lcy) * sin_a
        tail_proj = (ltail[0] - lcx) * cos_a + (ltail[1] - lcy) * sin_a
        
        if tail_proj > 0:
            head_candidates_mask = projected < 0
        else:
            head_candidates_mask = projected > 0
            
        opposite_indices = np.nonzero(head_candidates_mask)[0]
        if len(opposite_indices) == 0:
            opposite_indices = np.arange(len(cnt_pts))
            
        opposite_pts = cnt_pts[opposite_indices]
        dists_to_tail = (opposite_pts[:, 0] - ltail[0])**2 + (opposite_pts[:, 1] - ltail[1])**2
        head_idx = opposite_indices[np.argmax(dists_to_tail)]
        lhead = (float(cnt_pts[head_idx][0]), float(cnt_pts[head_idx][1]))
        
        head_pt = (lhead[0] + crop_offset_x, lhead[1] + crop_offset_y)
        tail_pt = (ltail[0] + crop_offset_x, ltail[1] + crop_offset_y)
        end1_global = (lhead[0] + crop_offset_x, lhead[1] + crop_offset_y)
        end2_global = (ltail[0] + crop_offset_x, ltail[1] + crop_offset_y)
        
        return head_pt, center_pt, tail_pt, end1_global, end2_global

    def _extract_visual_fingerprint(self, bgr_frame, mask_res_crop, b, cx1, cy1, cx2, cy2):
        """Extracts 66-D Feature Vector: 64-bin HSV Histogram + Aspect Ratio + Normalized Area"""
        # Crop actual BGR image
        bgr_crop = bgr_frame[cy1:cy2, cx1:cx2]
        
        # Ensure sizes match (defensive programming)
        if bgr_crop.shape[:2] != mask_res_crop.shape[:2]:
            mask_res_crop = cv2.resize(mask_res_crop, (bgr_crop.shape[1], bgr_crop.shape[0]))
            
        hsv_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        
        # Calculate 2D Histogram for H and S channels (8 bins each = 64 bins total)
        hist = cv2.calcHist([hsv_crop], [0, 1], mask_res_crop, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_flat = hist.flatten()
        
        # Append Aspect Ratio and Area
        w = b[2] - b[0]
        h = b[3] - b[1]
        aspect_ratio = w / h if h > 0 else 1.0
        norm_area = np.count_nonzero(mask_res_crop) / (bgr_frame.shape[0] * bgr_frame.shape[1])
        
        feature_vector = np.concatenate([hist_flat, [aspect_ratio, norm_area]])
        return feature_vector

    def run(self):
        try:
            start_batch_time = time.time()
            bg_frame = None 
            
            self.all_arenas = []
            for s in self.config['rois']:
                if s['type'] == 'grid': self.all_arenas.extend(self._get_grid_cells(s))
                else: self.all_arenas.append(s)
            self.log_signal.emit(f"📐 GEOMETRY: {len(self.all_arenas)} arenas mapped.")

            model_path = os.path.join(self.workspace, "Models", self.config['model_name'], "weights", "best.pt")
            model = YOLO(model_path)
            class_names = model.names
            
            device_target, use_half = 'cpu', False
            if torch.cuda.is_available():
                self.log_signal.emit(f"🔍 Hardware: {torch.cuda.get_device_name(0)}")
                dummy_img = np.zeros((160, 160, 3), dtype=np.uint8)
                try:
                    model.predict(dummy_img, device=0, half=True, verbose=False)
                    device_target, use_half = 0, True
                    self.log_signal.emit("✅ GPU FP16 Mode Active.")
                except Exception:
                    try:
                        model.predict(dummy_img, device=0, half=False, verbose=False)
                        device_target, use_half = 0, False
                        self.log_signal.emit("✅ GPU FP32 Mode Active.")
                    except:
                        self.log_signal.emit("❌ GPU Failed. CPU Fallback.")

            for v_idx, v_path in enumerate(self.config['videos']):
                if not self.is_running: break
                
                base_n = os.path.splitext(os.path.basename(v_path))[0]
                out_dir = os.path.join(self.workspace, "Advanced_Results", base_n)
                os.makedirs(out_dir, exist_ok=True)
                
                try:
                    v_cap = cv2.VideoCapture(v_path)
                    width, height = int(v_cap.get(3)), int(v_cap.get(4))
                    fps = v_cap.get(cv2.CAP_PROP_FPS) or 30.0
                    total_frames = int(v_cap.get(7)) 
                    
                    ret_bg, bg_frame = v_cap.read()
                    if not ret_bg or bg_frame is None:
                        bg_frame = None
                        
                    v_cap.release()
                    v_cap = cv2.VideoCapture(v_path)
                    
                    inf_writer = None
                    try:
                        if self.config['save_inference_vid']:
                            inf_writer = cv2.VideoWriter(os.path.join(out_dir, f"{base_n}_inference.mp4"),
                                                         cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                            
                        raw_frame_data = defaultdict(list)
                        is_seg = self.config['task_type'] == "Segmentation"
                        is_semantic_seg = self.config['task_type'] == "Semantic Segmentation"
                        
                        sample_rate = self.config.get('frame_sample_rate', 1)
                        if sample_rate < 1:
                            sample_rate = 1
                            
                        stopwatch = Stopwatch(); stopwatch.start()
                        frame_count_fps, fps_time, current_fps = 0, 0, 0.0
                        
                        results_gen = model.predict(
                            source=v_path,
                            conf=self.config['conf'],
                            stream=True,
                            device=device_target,
                            half=use_half,
                            vid_stride=sample_rate,
                            verbose=False
                        )
                        
                        # Initialize AMADEUS Trackers (One per tank)
                        use_amadeus = self.config.get('method') == "AMADEUS Visual-Kinematic"
                        if use_amadeus:
                            self.log_signal.emit("🔥 Initializing Real-Time AMADEUS Visual-Kinematic Tracker...")
                            tank_trackers = {i+1: AmadeusOnlineTracker(self.config) for i in range(len(self.all_arenas))}
                        
                        frame_idx = 0
                        for res in results_gen:
                            if not self.is_running: break
                            
                            f_real_idx = frame_idx * sample_rate
                            if f_real_idx >= total_frames:
                                f_real_idx = total_frames - 1
                                
                            frame = res.orig_img.copy() if inf_writer else None
                            bgr_orig = res.orig_img  # Keep for color extraction
                            
                            frame_detections = defaultdict(list)
                            
                            # 1. Processing Standard Detection & Instance Segmentation YOLO Models
                            if not is_semantic_seg and res.boxes is not None and len(res.boxes) > 0:
                                boxes_np = res.boxes.cpu()
                                has_native_masks = is_seg and res.masks is not None and res.masks.xy is not None

                                for j in range(len(boxes_np)):
                                    b = boxes_np.xyxy[j].numpy()
                                    cid = int(boxes_np.cls[j])
                                    conf = float(boxes_np.conf[j])
                                    color = self.class_palette[cid % len(self.class_palette)]
                                    
                                    cx, cy, poly_str, poly_pts = (b[0]+b[2])/2.0, (b[1]+b[3])/2.0, "", None
                                    head_pt = (cx, cy)
                                    tail_pt = (cx, cy)
                                    end1 = (cx, cy)
                                    end2 = (cx, cy)
                                    feature_vector = None

                                    if has_native_masks and len(res.masks.xy) > j:
                                        native_poly = res.masks.xy[j]
                                        if len(native_poly) > 0:
                                            poly_pts = native_poly.astype(np.int32)
                                            poly_str = ";".join([f"{int(p[0])},{int(p[1])}" for p in poly_pts])
                                            
                                            pad = 5
                                            cx1 = max(0, int(b[0]) - pad)
                                            cy1 = max(0, int(b[1]) - pad)
                                            cx2 = min(width, int(b[2]) + pad)
                                            cy2 = min(height, int(b[3]) + pad)
                                            
                                            mask_res_crop = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.uint8)
                                            local_poly_pts = poly_pts - [cx1, cy1]
                                            cv2.drawContours(mask_res_crop, [local_poly_pts], -1, 255, -1)
                                            
                                            if use_amadeus:
                                                feature_vector = self._extract_visual_fingerprint(bgr_orig, mask_res_crop, b, cx1, cy1, cx2, cy2)
                                            
                                            if self.config.get('extract_skeleton', False):
                                                head_pt, center_pt, tail_pt, end1, end2 = self._extract_fish_keypoints(
                                                    mask_res_crop, (b[0]+b[2])/2.0 - cx1, (b[1]+b[3])/2.0 - cy1, cx1, cy1
                                                )
                                                cx, cy = center_pt
                                            else:
                                                M = cv2.moments(mask_res_crop)
                                                if M["m00"] != 0:
                                                    cx, cy = M["m10"]/M["m00"] + cx1, M["m01"]/M["m00"] + cy1
                                                else:
                                                    cx, cy = (b[0]+b[2])/2.0, (b[1]+b[3])/2.0

                                    tank_num = None
                                    for a_idx, arena in enumerate(self.all_arenas):
                                        ax, ay, aw, ah = arena['x'], arena['y'], arena['w'], arena['h']
                                        if arena['type'] == 'circle':
                                            acx, acy = ax + aw / 2.0, ay + ah / 2.0
                                            # FIXED THE MAJOR BUG: Ellipse Equation Corrected
                                            dx = (cx - acx) / (aw / 2.0)
                                            dy = (cy - acy) / (ah / 2.0)
                                            if (dx**2 + dy**2) <= 1.0: 
                                                tank_num = a_idx + 1
                                                break
                                        else:
                                            if (ax <= cx <= ax + aw and ay <= cy <= ay + ah): 
                                                tank_num = a_idx + 1
                                                break
                                    
                                    if tank_num is not None:
                                        run_skeleton_on_this_tank = False
                                        if self.config.get('extract_skeleton', False):
                                            if self.skeleton_roi_set is None or tank_num in self.skeleton_roi_set:
                                                run_skeleton_on_this_tank = True

                                        if not run_skeleton_on_this_tank:
                                            head_pt = (cx, cy)
                                            tail_pt = (cx, cy)
                                            end1 = (cx, cy)
                                            end2 = (cx, cy)

                                        det_dict = {
                                            'frame_idx': f_real_idx, 'tank_number': tank_num, 'class_name': class_names[cid],
                                            'conf': conf, 'x1': b[0], 'y1': b[1], 'x2': b[2], 'y2': b[3], 'box': [b[0], b[1], b[2], b[3]],
                                            'cx': cx, 'cy': cy, 
                                            'polygon': poly_str, 'poly_pts': poly_pts, 'cid': cid
                                        }
                                        if feature_vector is not None:
                                            det_dict['feature'] = feature_vector
                                            
                                        if run_skeleton_on_this_tank:
                                            det_dict.update({
                                                'head_x': head_pt[0], 'head_y': head_pt[1],
                                                'tail_x': tail_pt[0], 'tail_y': tail_pt[1],
                                                'init_head_x': head_pt[0], 'init_head_y': head_pt[1],
                                                'init_tail_x': tail_pt[0], 'init_tail_y': tail_pt[1],
                                                'end1_x': end1[0], 'end1_y': end1[1],
                                                'end2_x': end2[0], 'end2_y': end2[1]
                                            })
                                            
                                        frame_detections[tank_num].append(det_dict)

                            # If using Online AMADEUS Tracker, run it here natively
                            if use_amadeus:
                                for tank_num in range(1, len(self.all_arenas) + 1):
                                    dets = frame_detections.get(tank_num, [])
                                    valid_results = tank_trackers[tank_num].update(dets)
                                    raw_frame_data[f_real_idx].extend(valid_results)
                            else:
                                for tank_num, dets in frame_detections.items():
                                    raw_frame_data[f_real_idx].extend(dets)

                            frame_idx += 1
                            frame_count_fps += 1
                            curr_time = stopwatch.get_elapsed_time(as_float=True)
                            if curr_time > fps_time + 1.0:
                                current_fps = frame_count_fps / (curr_time - fps_time)
                                frame_count_fps, fps_time = 0, curr_time

                            if total_frames > 0 and (f_real_idx % 500 == 0 or f_real_idx >= total_frames - sample_rate):
                                pct = min(100, int((f_real_idx + 1) * 100 / total_frames))
                                self.progress_signal.emit(pct, f"Inference: {f_real_idx}/{total_frames} | {current_fps:.1f} FPS | ETR: {stopwatch.get_etr(f_real_idx, total_frames)}")
                                
                    finally:
                        v_cap.release()
                    
                    if not self.is_running: break

                    self.log_signal.emit(f"\n--- [PHASE 2] Tracking & Optimization ---")
                    
                    # Only do offline tracking if AMADEUS is OFF
                    use_amadeus = self.config.get('method') == "AMADEUS Visual-Kinematic"
                    if not use_amadeus:
                        tracked_detections = defaultdict(list)
                        num_tanks = len(self.all_arenas)
                        
                        try:
                            if self.config['method'] == "Custom Force-N":
                                self.log_signal.emit("Applying Standard Force-N tracking...")
                                active_tracks = {t: {} for t in range(1, num_tanks + 1)}
                                for frame_idx_t in range(total_frames):
                                    dets_this_frame = raw_frame_data.get(frame_idx_t, [])
                                    for tank_num in range(1, num_tanks + 1):
                                        tank_dets = [d for d in dets_this_frame if d.get('tank_number') == tank_num]
                                        tank_dets.sort(key=lambda x: x.get('conf', 0.0), reverse=True)
                                        tank_dets = tank_dets[:self.config['max_n']]
                                        tank_active = active_tracks[tank_num]
                                        
                                        if len(tank_active) < self.config['max_n']:
                                            unassigned_dets = []
                                            for det in tank_dets:
                                                if len(tank_active) < self.config['max_n']:
                                                    new_id = len(tank_active) + 1
                                                    det['track_id'] = new_id
                                                    tank_active[new_id] = {'pos': (det['cx'], det['cy'])}
                                                    tracked_detections[frame_idx_t].append(det)
                                                else: 
                                                    unassigned_dets.append(det)
                                            tank_dets = unassigned_dets 
                                        
                                        if not tank_dets: continue 
                                            
                                        track_ids = list(tank_active.keys())
                                        cost_matrix = np.full((len(track_ids), len(tank_dets)), 1e6) 
                                        for i, tid in enumerate(track_ids):
                                            for j, det in enumerate(tank_dets):
                                                active_pos = tank_active[tid].get('pos', (det['cx'], det['cy']))
                                                cost_matrix[i, j] = np.hypot(active_pos[0] - det['cx'], active_pos[1] - det['cy'])

                                        try:
                                            row_ind, col_ind = linear_sum_assignment(cost_matrix)
                                        except Exception as e:
                                            self.log_signal.emit(f"⚠️ Tracking assignment error bypassed: {str(e)}")
                                            break
                                        
                                        for r, c in zip(row_ind, col_ind):
                                            if cost_matrix[r, c] < 300.0:
                                                tid = track_ids[r]; matched_det = tank_dets[c].copy()
                                                matched_det['track_id'] = tid
                                                tank_active[tid] = {'pos': (matched_det['cx'], matched_det['cy'])}
                                                tracked_detections[frame_idx_t].append(matched_det)
                                                
                            elif self.config['method'] == "Confidence Filter":
                                self.log_signal.emit("Applying Confidence Filter...")
                                for frame_idx_t, dets in raw_frame_data.items():
                                    tank_groups = defaultdict(list)
                                    for det in dets: tank_groups[det.get('tank_number')].append(det)
                                    for tank_num, tank_dets in tank_groups.items():
                                        tank_dets.sort(key=lambda x: x.get('conf', 0.0), reverse=True)
                                        valid_dets = tank_dets[:self.config['max_n']]
                                        for i, d in enumerate(valid_dets):
                                            d['track_id'] = i + 1
                                            tracked_detections[frame_idx_t].append(d)
                        except Exception as track_err:
                            self.log_signal.emit(f"⚠️ Warning: Tracking pipeline exception encountered: {str(track_err)}. Automatically falling back to raw IDs.")
                            tracked_detections = raw_frame_data
                    else:
                        # Data is already tracked and fully structured natively by AMADEUS
                        tracked_detections = raw_frame_data

                    # KINEMATIC DIRECTION ESTIMATION (Forward/Backward Movement Tracking)
                    if self.config.get('calc_direction', True):
                        self.log_signal.emit("⏱️ Calculating Kinematic Movement Directions (Forward/Backward)...")
                        dir_gap_sec = float(self.config.get('dir_gap_sec', 0.5))
                        frame_gap = max(1, int(fps * dir_gap_sec))
                        
                        num_tanks = len(self.all_arenas)
                        for tank_num in range(1, num_tanks + 1):
                            tank_frames = defaultdict(list)
                            for frame_idx_t in range(total_frames):
                                for d in tracked_detections.get(frame_idx_t, []):
                                    if d.get('tank_number') == tank_num:
                                        tank_frames[d.get('track_id')].append((frame_idx_t, d))
                                        
                            for tid, track_timeline in tank_frames.items():
                                track_timeline.sort(key=lambda x: x[0])
                                
                                t_indices = [x[0] for x in track_timeline]
                                t_data = [x[1] for x in track_timeline]
                                
                                for i in range(len(t_indices)):
                                    current_f = t_indices[i]
                                    d = t_data[i]
                                    d['movement_direction'] = "Stationary"
                                    
                                    target_past_f = current_f - frame_gap
                                    if target_past_f < 0:
                                        continue
                                        
                                    past_i = None
                                    for idx_past in range(i-1, -1, -1):
                                        if t_indices[idx_past] <= target_past_f:
                                            past_i = idx_past
                                            break
                                            
                                    if past_i is not None:
                                        past_d = t_data[past_i]
                                        
                                        vx = d['cx'] - past_d['cx']
                                        vy = d['cy'] - past_d['cy']
                                        mag_v = np.hypot(vx, vy)
                                        
                                        if mag_v > 2.0:
                                            if 'head_x' in d and 'tail_x' in d:
                                                # Center to Head Vector
                                                hx = d['head_x'] - d['cx']
                                                hy = d['head_y'] - d['cy']
                                                
                                                dot_product = (vx * hx) + (vy * hy)
                                                if dot_product > 0:
                                                    d['movement_direction'] = "Forward"
                                                else:
                                                    d['movement_direction'] = "Backward"
                                            else:
                                                d['movement_direction'] = "Moving"

                    # ==============================================================
                    # PHASE 3: RENDER VIDEO & ANALYTICS EXPORT
                    # ==============================================================
                    if self.config['save_video']:
                        self.log_signal.emit(f"\n--- [PHASE 3] Rendering Tracked Video ---")
                        v_cap = cv2.VideoCapture(v_path)
                        trk_writer = cv2.VideoWriter(os.path.join(out_dir, f"{base_n}_tracked.mp4"),
                                                     cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                        try:
                            f_idx = 0 
                            for f_idx in range(total_frames):
                                if not self.is_running: break
                                ret, frame = v_cap.read()
                                if not ret or frame is None or frame.size == 0: 
                                    break
                                
                                for d in tracked_detections.get(f_idx, []):
                                    x1, y1, x2, y2 = map(int, [d.get('x1', 0), d.get('y1', 0), d.get('x2', 0), d.get('y2', 0)])
                                    cx = int(d.get('cx', (x1 + x2) / 2.0))
                                    cy = int(d.get('cy', (y1 + y2) / 2.0))
                                    t_id = d.get('track_id', 0)
                                    color = self.class_palette[int(t_id) % len(self.class_palette)]
                                    poly_pts = d.get('poly_pts', None)

                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), self.b_thick)
                                    
                                    has_mask_visual = True
                                    if has_mask_visual and poly_pts is not None:
                                        ov = frame.copy()
                                        cv2.fillPoly(ov, [poly_pts], color)
                                        cv2.addWeighted(ov, 0.4, frame, 0.6, 0, frame)
                                        cv2.drawContours(frame, [poly_pts], -1, color, max(1, self.b_thick - 1))
                                    
                                    cv2.circle(frame, (cx, cy), self.dot_sz, (0, 0, 255), -1)      # Default Red center
                                    
                                    if 'head_x' in d and 'tail_x' in d:
                                        hx, hy = int(d['head_x']), int(d['head_y'])
                                        tx, ty = int(d['tail_x']), int(d['tail_y'])
                                        cv2.circle(frame, (hx, hy), self.dot_sz, (180, 105, 255), -1)   # Pink Head
                                        cv2.circle(frame, (cx, cy), self.dot_sz, (0, 0, 255), -1)       # Red Center
                                        cv2.circle(frame, (tx, ty), self.dot_sz, (255, 0, 0), -1)       # Blue Tail
                                    
                                    movement_tag = d.get('movement_direction', "")
                                    if movement_tag == "Forward":
                                        movement_str = "[FWD]"
                                    elif movement_tag == "Backward":
                                        movement_str = "[BWD]"
                                    else:
                                        movement_str = ""
                                        
                                    label = f"A{int(d.get('tank_number', 0))}|ID{t_id} {movement_str} {d.get('class_name', '')}"
                                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, self.l_scale, self.l_thick)
                                    cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+10, y1), (255, 255, 255), -1)
                                    cv2.putText(frame, label, (x1+5, y1-7), cv2.FONT_HERSHEY_DUPLEX, self.l_scale, (0, 0, 0), self.l_thick, cv2.LINE_AA)
                                
                                if frame.shape[1] != width or frame.shape[0] != height:
                                    frame = cv2.resize(frame, (width, height))
                                trk_writer.write(frame)

                                if total_frames > 0 and (f_idx % 500 == 0 or f_idx == total_frames - 1):
                                    pct = min(100, int((f_idx + 1) * 100 / total_frames))
                                    self.progress_signal.emit(pct, f"Rendering: {f_idx}/{total_frames}")
                        finally:
                            v_cap.release()
                            if trk_writer is not None: 
                                trk_writer.release()

                    if self.is_running:
                        self.log_signal.emit("Exporting Analytics Data...")
                        flat_data = [d for f, dets in tracked_detections.items() for d in dets]
                        self._generate_analytics(base_n, flat_data, bg_frame, out_dir, width, height)
                
                except Exception as file_err:
                    self.log_signal.emit(f"❌ Error encountered on file {base_n}: {str(file_err)}")
                    print(f"=== INDIVIDUAL RUNTIME FILE ERROR: {base_n} ===")
                    traceback.print_exc()
                    continue

            time_str = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_batch_time))
            self.finished_signal.emit(f"✅ Batch Completed in {time_str}")
        except Exception as e: 
            print("=== WORKER CRASH TRACEBACK ===")
            traceback.print_exc()
            self.finished_signal.emit(f"❌ Error: {str(e)}")

    def _generate_analytics(self, base_n, flat_data, bg, out, width, height):
        if not flat_data: return
        df = pd.DataFrame(flat_data)
        
        required_cols = [
            'frame_idx', 'tank_number', 'track_id', 'class_name', 
            'movement_direction',
            'cx', 'cy', 'head_x', 'head_y', 'tail_x', 'tail_y', 
            'x1', 'y1', 'x2', 'y2', 'conf', 'polygon'
        ]
        for col in required_cols:
            if col not in df.columns:
                if col in ['frame_idx', 'tank_number', 'track_id']:
                    df[col] = 1
                elif col in ['cx', 'cy', 'head_x', 'head_y', 'tail_x', 'tail_y', 'x1', 'y1', 'x2', 'y2', 'conf']:
                    df[col] = 0.0
                else:
                    df[col] = ""

        export_df = df[[
            'frame_idx', 'tank_number', 'track_id', 'class_name', 
            'movement_direction',
            'cx', 'cy', 'head_x', 'head_y', 'tail_x', 'tail_y', 
            'x1', 'y1', 'x2', 'y2', 'conf', 'polygon'
        ]]
        export_df.columns = [
            "Frame", "Arena", "ID", "Class", "Direction",
            "X", "Y", "Head_X", "Head_Y", "Tail_X", "Tail_Y", 
            "x1", "y1", "x2", "y2", "Conf", "Polygon"
        ]

        # 1. Standard CSV Export
        if self.config.get('save_csv'):
            try:
                export_df.to_csv(os.path.join(out, f"{base_n}_tracked_data.csv"), index=False)
            except Exception as e:
                self.log_signal.emit(f"⚠️ Failed to save CSV: {str(e)}")

        # 2. Centroid CSV Export (including keypoint columns)
        if self.config.get('save_centroid_csv'):
            try:
                cent_df = export_df[['Frame', 'Arena', 'ID', 'Class', 'Direction', 'X', 'Y', 'Head_X', 'Head_Y', 'Tail_X', 'Tail_Y']]
                cent_df.to_csv(os.path.join(out, f"{base_n}_centroids.csv"), index=False)
            except Exception as e:
                self.log_signal.emit(f"⚠️ Failed to save Centroids CSV: {str(e)}")

        # 3. Excel (By Tank) Export
        if self.config.get('save_excel_tank'):
            try:
                MAX_EXCEL_ROWS = 1048576
                MAX_DATA_ROWS = MAX_EXCEL_ROWS - 1
                base_path = os.path.join(out, f"{base_n}_by_tank.xlsx")
                total_rows = len(export_df)

                if total_rows <= MAX_DATA_ROWS:
                    with pd.ExcelWriter(base_path, engine='openpyxl') as writer:
                        export_df.to_excel(writer, sheet_name="Master_Data", index=False)
                        for a_id in sorted(export_df['Arena'].unique()):
                            arena_df = export_df[export_df['Arena'] == a_id]
                            if not arena_df.empty:
                                arena_df.to_excel(writer, sheet_name=f"Arena_{int(a_id)}", index=False)
                else:
                    num_files = (total_rows // MAX_DATA_ROWS) + 1
                    for i in range(num_files):
                        start_row = i * MAX_DATA_ROWS
                        end_row = min((i + 1) * MAX_DATA_ROWS, total_rows)
                        chunk_df = export_df.iloc[start_row:end_row]
                        excel_file = os.path.join(out, f"{base_n}_by_tank_Part_{i+1}.xlsx")

                        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                            chunk_df.to_excel(writer, sheet_name="Master_Data", index=False)
                            for a_id in sorted(chunk_df['Arena'].unique()):
                                arena_df = chunk_df[chunk_df['Arena'] == a_id]
                                if not arena_df.empty:
                                    arena_df.to_excel(writer, sheet_name=f"Arena_{int(a_id)}", index=False)
            except Exception as e:
                self.log_signal.emit(f"⚠️ Failed Excel By Tank: {str(e)}")

        # 4. Excel (By Track) Export
        if self.config.get('save_excel_track'):
            try:
                MAX_EXCEL_ROWS = 1048576
                MAX_DATA_ROWS = MAX_EXCEL_ROWS - 1
                base_path = os.path.join(out, f"{base_n}_by_track.xlsx")
                total_rows = len(export_df)

                if total_rows <= MAX_DATA_ROWS:
                    with pd.ExcelWriter(base_path, engine='openpyxl') as writer:
                        export_df.to_excel(writer, sheet_name="Master_Data", index=False)
                        for t_id in sorted(export_df['ID'].unique()):
                            track_df = export_df[export_df['ID'] == t_id]
                            if not track_df.empty:
                                track_df.to_excel(writer, sheet_name=f"Track_{int(t_id)}", index=False)
                else:
                    num_files = (total_rows // MAX_DATA_ROWS) + 1
                    for i in range(num_files):
                        start_row = i * MAX_DATA_ROWS
                        end_row = min((i + 1) * MAX_DATA_ROWS, total_rows)
                        chunk_df = export_df.iloc[start_row:end_row]
                        excel_file = os.path.join(out, f"{base_n}_by_track_Part_{i+1}.xlsx")

                        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                            chunk_df.to_excel(writer, sheet_name="Master_Data", index=False)
                            for t_id in sorted(chunk_df['ID'].unique()):
                                track_df = chunk_df[chunk_df['ID'] == t_id]
                                if not track_df.empty:
                                    track_df.to_excel(writer, sheet_name=f"Track_{int(t_id)}", index=False)
            except Exception as e:
                self.log_signal.emit(f"⚠️ Failed Excel By Track: {str(e)}")

        # 5. Dynamic 3D Locomotion Reconstruction & Visualization Engine
        if self.config.get('reconstruct_3d') and not export_df.empty:
            try:
                self.log_signal.emit("⏳ Processing 3D Trajectory Reconstruction (Top + Side)...")
                
                # Dynamically extract integer IDs for Top and Side arenas from configuration
                top_arena_idx = int(str(self.config.get('top_view_arena', 'Arena_1')).split('_')[1])
                side_arena_idx = int(str(self.config.get('side_view_arena', 'Arena_2')).split('_')[1])
                px_to_cm = float(self.config.get('px_to_cm', 10.0))
                
                # Fetch designed arena boundary geometry parameters
                top_arena_geometry = self.all_arenas[top_arena_idx - 1]
                side_arena_geometry = self.all_arenas[side_arena_idx - 1]
                
                tx_start = top_arena_geometry['x']
                ty_start = top_arena_geometry['y']
                tw = top_arena_geometry['w']
                
                sx_start = side_arena_geometry['x']
                sy_start = side_arena_geometry['y']
                sw = side_arena_geometry['w']
                sy_end = sy_start + side_arena_geometry['h']
                
                rows_3d_data = []
                
                # Match Top and Side coordinates frame-by-frame strictly by ID (Track ID)
                for t in sorted(export_df['Frame'].unique()):
                    top_frame_df = export_df[(export_df['Frame'] == t) & (export_df['Arena'] == top_arena_idx)]
                    side_frame_df = export_df[(export_df['Frame'] == t) & (export_df['Arena'] == side_arena_idx)]
                    
                    if top_frame_df.empty or side_frame_df.empty:
                        continue
                        
                    # Match strictly by matching track IDs
                    for _, top_row in top_frame_df.iterrows():
                        tid = top_row['ID']
                        # Find the corresponding Side View detection with the same Track ID
                        side_row_match = side_frame_df[side_frame_df['ID'] == tid]
                        
                        if not side_row_match.empty:
                            side_row = side_row_match.iloc[0]
                            
                            # Convert pixels to centimeters using Top (X, Y) and Side (X, Z) coordinates
                            # Invert side Y to represent Z height from floor
                            x_cm = (float(top_row['X']) - tx_start) / px_to_cm
                            y_cm = (float(top_row['Y']) - ty_start) / px_to_cm
                            z_cm = (sy_end - float(side_row['Y'])) / px_to_cm
                            
                            rows_3d_data.append({
                                "Frame": int(t),
                                "ID": int(tid),
                                "X_cm": x_cm,
                                "Y_cm": y_cm,
                                "Z_cm": z_cm
                            })
                            
                if rows_3d_data:
                    df_3d = pd.DataFrame(rows_3d_data)
                    
                    # 1. Export spreadsheet dataset
                    df_3d.to_csv(os.path.join(out, f"{base_n}_3d_coords.csv"), index=False)
                    self.log_signal.emit(f"✅ Exported 3D coordinates: {base_n}_3d_coords.csv")
                    
                    # 2. Render high-resolution interactive 3D Matplotlib visual
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # Fetch dynamic line thickness from config
                    linewidth_3d = float(self.config.get('line_thick_3d', 1.5))
                    
                    # Plot distinct trajectory lines for each animal ID
                    unique_tids = sorted(df_3d['ID'].unique())
                    for idx_t, t_id in enumerate(unique_tids):
                        subset = df_3d[df_3d['ID'] == t_id].sort_values("Frame")
                        color_render = self.class_palette[idx_t % len(self.class_palette)]
                        color_normalized = (color_render[2]/255.0, color_render[1]/255.0, color_render[0]/255.0)  # BGR to RGB
                        
                        ax.plot(
                            subset['Y_cm'].values,  # Plotted on the X-axis of the graph
                            subset['X_cm'].values,  # Plotted on the Y-axis of the graph
                            subset['Z_cm'].values,  # Plotted on the Z-axis of the graph (Z remains unchanged)
                            label=f"Track_{t_id}",
                            color=color_normalized,
                            linewidth=linewidth_3d
                        )
                        
                    ax.set_xlabel("Y Axis (cm) [Width]")   # Swapped label
                    ax.set_ylabel("X Axis (cm) [Length]")  # Swapped label
                    ax.set_zlabel("Z Axis (cm) [Height]")
                    ax.set_title("3D Locomotion Trajectories")
                    
                    # Setup physical tank limits:
                    ax.set_xlim(0, df_3d['Y_cm'].max() * 1.1)
                    ax.set_ylim(df_3d['X_cm'].max() * 1.1, 0)  # Inverted X limits
                    ax.set_zlim(0, df_3d['Z_cm'].max() * 1.1)
                    
                    # Set the viewing angle so that the vertical Z-axis starts exactly at the opposite corner (X=0.0, Y=Y_max)
                    ax.view_init(elev=20, azim=45)
                    
                    ax.grid(True)
                    ax.legend()
                    
                    plt.tight_layout()
                    plot_path_3d = os.path.join(out, f"{base_n}_3d_trajectories.png")
                    plt.savefig(plot_path_3d, dpi=300)
                    plt.close(fig)
                    self.log_signal.emit(f"✅ Exported 3D Trajectories image: {base_n}_3d_trajectories.png")
                else:
                    self.log_signal.emit("⚠️ 3D Warning: No overlapping temporal frames found between top and side views.")
            except Exception as e3d:
                self.log_signal.emit(f"⚠️ 3D Error encountered: {str(e3d)}")
                traceback.print_exc()

        # 6. Render Trajectories (Original 2D)
        if self.config.get('save_traj') and bg is not None:
            try:
                if len(bg.shape) == 2:
                    canvas = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
                elif bg.shape[2] == 1:
                    canvas = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
                else:
                    canvas = bg.copy()

                for a_id in sorted(export_df['Arena'].unique()):
                    for o_id in sorted(export_df[export_df['Arena']==a_id]['ID'].unique()):
                        subset = export_df[(export_df['Arena']==a_id)&(export_df['ID']==o_id)].sort_values("Frame")
                        subset['diff'] = subset['Frame'].diff().fillna(1)
                        subset['group'] = (subset['diff'] > 1).cumsum()
                        
                        for _, group in subset.groupby('group'):
                            pts = group[['X','Y']].values.astype(np.int32)
                            random.seed(int(a_id*100 + o_id))
                            clr = (random.randint(60,255), random.randint(60,255), random.randint(60,255))
                            if len(pts)>1: cv2.polylines(canvas, [pts], False, clr, 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(out, f"{base_n}_trajectories.png"), canvas)
            except Exception as e:
                self.log_signal.emit(f"⚠️ Failed to save trajectories visual: {str(e)}")

        # 7. Render Heatmap (Original 2D)
        if self.config.get('save_heat') and bg is not None:
            try:
                if len(bg.shape) == 2:
                    bg_color = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
                elif bg.shape[2] == 1:
                    bg_color = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
                else:
                    bg_color = bg.copy()

                h, w = bg_color.shape[:2]; accum = np.zeros((h, w), dtype=np.float32)
                for _, r in export_df.iterrows(): cv2.circle(accum, (int(r['X']), int(r['Y'])), 10, 0.4, -1)
                accum = cv2.normalize(cv2.GaussianBlur(accum, (51, 51), 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                heatmap = cv2.applyColorMap(accum, cv2.COLORMAP_JET)
                
                if bg_color.shape[:2] == heatmap.shape[:2]:
                    cv2.imwrite(os.path.join(out, f"{base_n}_heatmap.png"), cv2.addWeighted(bg_color, 0.6, heatmap, 0.4, 0))
                else:
                    cv2.imwrite(os.path.join(out, f"{base_n}_heatmap.png"), heatmap)
            except Exception as e:
                self.log_signal.emit(f"⚠️ Failed to save heatmap visual: {str(e)}")  