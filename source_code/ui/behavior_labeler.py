import os
import glob
import csv
import cv2
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, 
    QLabel, QSlider, QLineEdit, QComboBox, QMessageBox, QGroupBox,
    QFormLayout, QSpinBox, QDoubleSpinBox, QInputDialog, QScrollArea, QWidget
)
from PyQt6.QtCore import Qt, QRectF, QPoint
from PyQt6.QtGui import QImage, QPixmap, QShortcut, QKeySequence

class BehaviorLabelerDialog(QDialog):
    """
    Advanced Behavioral Labeling Panel featuring mouse scroll wheel zoom,
    pan navigation, hotkey shortcuts, behavioral list editing, frame range propagation,
    and visual style overrides (class color mapping, box thickness, and text scale).
    """
    def __init__(self, workspace, experiment_name, class_names, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(self.workspace, "Experiments", self.experiment_name)
        self.behavior_classes = class_names if class_names else ["wandering", "fighting", "resting", "normal"]
        
        self.csv_files = []
        self.current_csv_path = None
        self.csv_headers = []
        self.tracking_data = {}  # frame_idx -> list of rows
        self.unique_object_ids = set()
        
        self.current_frame_idx = 0
        self.frame_list = []
        self.selected_object_id = -1
        
        # Zoom & Pan States
        self.zoom_factor = 1.0
        self.zoom_center_x = 0.0
        self.zoom_center_y = 0.0
        self.pan_active = False
        self.last_mouse_pos = QPoint()
        
        # Consistent class color mapping palette (BGR format for OpenCV drawing)
        self.colors = [
            (57, 255, 20),    # Neon Green
            (255, 100, 50),   # Bright Blue
            (50, 50, 255),    # Bright Red
            (0, 255, 255),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 165, 255),    # Orange
            (200, 200, 0),    # Cyan
            (130, 0, 130),    # Purple
            (128, 128, 255)   # Pink
        ]
        
        self.setWindowTitle("🧠 Advanced Behavioral Verification Panel")
        self.resize(1200, 780)
        
        self.setup_ui()
        self.setup_shortcuts()
        self.scan_for_experiments()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # --- LEFT PANEL: VIEWPORT & NAVIGATION ---
        left_layout = QVBoxLayout()
        
        top_ctrl = QHBoxLayout()
        top_ctrl.addWidget(QLabel("Select Track CSV File:"))
        self.combo_csv = QComboBox()
        self.combo_csv.setCursor(Qt.CursorShape.PointingHandCursor)
        self.combo_csv.currentIndexChanged.connect(self.load_selected_csv)
        top_ctrl.addWidget(self.combo_csv, 1)
        
        self.btn_reset_view = QPushButton("Reset Zoom")
        self.btn_reset_view.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset_view.clicked.connect(self.reset_zoom)
        top_ctrl.addWidget(self.btn_reset_view)
        left_layout.addLayout(top_ctrl)
        
        self.lbl_screen = QLabel()
        self.lbl_screen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_screen.setStyleSheet("background-color: #050505; border: 2px solid #333; border-radius: 4px;")
        self.lbl_screen.setMinimumSize(480, 320)
        self.lbl_screen.installEventFilter(self)
        self.lbl_screen.setMouseTracking(True)
        left_layout.addWidget(self.lbl_screen, 1)
        
        self.lbl_info = QLabel("Frame Index: 0 | Selected Object: None")
        self.lbl_info.setStyleSheet("font-family: 'Consolas'; font-size: 11px; color: #39FF14; background: #111; padding: 6px;")
        left_layout.addWidget(self.lbl_info)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider.valueChanged.connect(self.seek_frame)
        left_layout.addWidget(self.slider)
        
        play_ctrls = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Back (A)")
        self.btn_prev.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_prev.clicked.connect(lambda: self.step_frame(-1))
        
        self.btn_next = QPushButton("Forward (D) ▶")
        self.btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_next.clicked.connect(lambda: self.step_frame(1))
        
        play_ctrls.addWidget(self.btn_prev)
        play_ctrls.addWidget(self.btn_next)
        
        self.lbl_help_shortcuts = QLabel("💡 Hotkeys: [A/D] Prev/Next Frame | [Ctrl+S] Save | [1-9] Quick-Assign")
        self.lbl_help_shortcuts.setStyleSheet("font-size: 10px; color: #aaa; padding-left: 10px;")
        play_ctrls.addWidget(self.lbl_help_shortcuts, 1)
        left_layout.addLayout(play_ctrls)
        
        main_layout.addLayout(left_layout, 3)
        
        # --- RIGHT PANEL: CONFIGURATION & VIEWPORT PREFERENCES (SCROLLABLE) ---
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFixedWidth(320)
        right_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        right_content = QWidget()
        right_panel = QVBoxLayout(right_content)
        right_panel.setContentsMargins(0, 0, 0, 0)
        
        # Viewport Display Controls Group
        grp_display = QGroupBox("Viewport Display Options")
        disp_lay = QFormLayout(grp_display)
        
        self.spin_box_thickness = QSpinBox()
        self.spin_box_thickness.setRange(1, 10)
        self.spin_box_thickness.setValue(2)
        self.spin_box_thickness.setSuffix(" px")
        self.spin_box_thickness.valueChanged.connect(lambda: self.show_frame_by_index(self.current_frame_idx))
        
        self.spin_text_scale = QDoubleSpinBox()
        self.spin_text_scale.setRange(0.2, 3.0)
        self.spin_text_scale.setSingleStep(0.1)
        self.spin_text_scale.setValue(0.5)
        self.spin_text_scale.setSuffix(" scale")
        self.spin_text_scale.valueChanged.connect(lambda: self.show_frame_by_index(self.current_frame_idx))
        
        disp_lay.addRow("Line Thickness:", self.spin_box_thickness)
        disp_lay.addRow("Text Scale:", self.spin_text_scale)
        right_panel.addWidget(grp_display)
        
        # Behavioral Classes Editor Group
        grp_behaviors = QGroupBox("Behavior Classes Management")
        behav_lay = QVBoxLayout(grp_behaviors)
        
        self.list_behaviors = QListWidget()
        self.list_behaviors.setCursor(Qt.CursorShape.PointingHandCursor)
        self.list_behaviors.addItems(self.behavior_classes)
        self.list_behaviors.setMaximumHeight(120)
        behav_lay.addWidget(self.list_behaviors)
        
        add_lay = QHBoxLayout()
        self.edit_new_behavior = QLineEdit()
        self.edit_new_behavior.setPlaceholderText("New Class...")
        self.btn_add_behavior = QPushButton("Add")
        self.btn_add_behavior.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_behavior.clicked.connect(self.add_behavior_class)
        add_lay.addWidget(self.edit_new_behavior)
        add_lay.addWidget(self.btn_add_behavior)
        behav_lay.addLayout(add_lay)
        
        edit_del_lay = QHBoxLayout()
        self.btn_edit_class = QPushButton("✏️ Rename")
        self.btn_edit_class.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_edit_class.clicked.connect(self.edit_behavior_class)
        
        self.btn_del_class = QPushButton("🗑️ Delete")
        self.btn_del_class.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_del_class.setStyleSheet("background-color: #8c1c1c; color: white;")
        self.btn_del_class.clicked.connect(self.delete_behavior_class)
        
        edit_del_lay.addWidget(self.btn_edit_class)
        edit_del_lay.addWidget(self.btn_del_class)
        behav_lay.addLayout(edit_del_lay)
        
        right_panel.addWidget(grp_behaviors)
        
        # Interactive Range Assignment Group
        grp_assign = QGroupBox("Assign Behaviors")
        assign_lay = QVBoxLayout(grp_assign)
        
        self.btn_assign_current = QPushButton("Apply to Current Frame only")
        self.btn_assign_current.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_assign_current.clicked.connect(self.assign_behavior_current_frame)
        assign_lay.addWidget(self.btn_assign_current)
        
        # Frame Range controls
        range_box = QGroupBox("Apply over Frame Range")
        range_lay = QFormLayout(range_box)
        
        self.spin_from_frame = QSpinBox()
        self.spin_from_frame.setRange(0, 999999)
        self.spin_to_frame = QSpinBox()
        self.spin_to_frame.setRange(0, 999999)
        
        self.btn_set_start = QPushButton("Set Current Frame as Start")
        self.btn_set_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_set_start.clicked.connect(lambda: self.spin_from_frame.setValue(self.frame_list[self.current_frame_idx]))
        
        self.btn_set_end = QPushButton("Set Current Frame as End")
        self.btn_set_end.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_set_end.clicked.connect(lambda: self.spin_to_frame.setValue(self.frame_list[self.current_frame_idx]))
        
        range_lay.addRow("Start Frame:", self.spin_from_frame)
        range_lay.addRow(self.btn_set_start)
        range_lay.addRow("End Frame:", self.spin_to_frame)
        range_lay.addRow(self.btn_set_end)
        
        self.btn_apply_range = QPushButton("Apply to Selection in Range")
        self.btn_apply_range.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_apply_range.setStyleSheet("background-color: #0078d4; color: white; font-weight: bold;")
        self.btn_apply_range.clicked.connect(self.assign_behavior_range)
        range_lay.addRow(self.btn_apply_range)
        
        assign_lay.addWidget(range_box)
        
        self.btn_assign_track = QPushButton("Apply to Entire Object ID Sequence")
        self.btn_assign_track.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_assign_track.clicked.connect(self.assign_behavior_sequence)
        assign_lay.addWidget(self.btn_assign_track)
        
        right_panel.addWidget(grp_assign)
        
        self.btn_save_csv = QPushButton("💾 Save Changes (Ctrl+S)")
        self.btn_save_csv.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save_csv.setStyleSheet("background-color: #e8a00c; color: white; font-weight: bold; padding: 10px;")
        self.btn_save_csv.clicked.connect(self.save_data_to_csv)
        right_panel.addWidget(self.btn_save_csv)
        
        right_panel.addStretch()
        
        btn_close = QPushButton("Return to Main Panel")
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.setStyleSheet("background-color: #444; color: white; padding: 10px;")
        btn_close.clicked.connect(self.save_and_accept)
        right_panel.addWidget(btn_close)
        
        self.btn_proceed = QPushButton("Proceed to YOLO Training ✔️")
        self.btn_proceed.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_proceed.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 12px;")
        self.btn_proceed.clicked.connect(self.confirm_and_close)
        right_panel.addWidget(self.btn_proceed)
        
        right_scroll.setWidget(right_content)
        main_layout.addWidget(right_scroll, 1)

    def setup_shortcuts(self):
        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save.activated.connect(self.save_data_to_csv)
        
        self.shortcut_prev_a = QShortcut(QKeySequence("A"), self)
        self.shortcut_prev_a.activated.connect(lambda: self.step_frame(-1))
        self.shortcut_next_d = QShortcut(QKeySequence("D"), self)
        self.shortcut_next_d.activated.connect(lambda: self.step_frame(1))
        
        for i in range(1, 10):
            shortcut = QShortcut(QKeySequence(str(i)), self)
            shortcut.activated.connect(lambda idx=i-1: self.quick_assign_class_index(idx))

    def scan_for_experiments(self):
        if not os.path.exists(self.experiment_dir):
            return
        pattern = os.path.join(self.experiment_dir, "**", "*.csv")
        self.csv_files = glob.glob(pattern, recursive=True)
        self.combo_csv.clear()
        
        if self.csv_files:
            for filepath in self.csv_files:
                parts = filepath.split(os.sep)
                self.combo_csv.addItem(os.path.join(parts[-2], parts[-1]), filepath)
        else:
            self.combo_csv.addItem("No tracking data discovered yet.")

    def load_selected_csv(self, index):
        if index < 0 or index >= len(self.csv_files):
            return
        csv_path = self.combo_csv.itemData(index)
        self.current_csv_path = csv_path
        self.tracking_data = {}
        self.unique_object_ids = set()
        self.frame_list = []
        
        try:
            df = pd.read_csv(csv_path, sep=None, engine='python')
            df.columns = df.columns.str.strip()
            df.rename(columns=lambda x: x.lstrip('\ufeff').strip(), inplace=True)
            
            required = ["Global_Frame_ID", "Object_ID", "Class_ID", "Centroid_X", "Centroid_Y", "Polygon_Coords", "Image_Path"]
            for col in required:
                if col not in df.columns:
                    raise KeyError(f"Missing column: '{col}'. Found: {list(df.columns)}")

            self.csv_headers = list(df.columns)
            
            for _, row in df.iterrows():
                frame_id = int(row["Global_Frame_ID"])
                obj_id = int(row["Object_ID"])
                class_id = int(row["Class_ID"])
                centroid_x = float(row["Centroid_X"])
                centroid_y = float(row["Centroid_Y"])
                polygon_str = str(row["Polygon_Coords"])
                image_path = str(row["Image_Path"])
                
                self.unique_object_ids.add(obj_id)
                if frame_id not in self.tracking_data:
                    self.tracking_data[frame_id] = []
                    
                self.tracking_data[frame_id].append({
                    "Object_ID": obj_id,
                    "Class_ID": class_id,
                    "Centroid_X": centroid_x,
                    "Centroid_Y": centroid_y,
                    "Polygon_Coords": polygon_str,
                    "Image_Path": image_path,
                    "raw_row": list(row.values)
                })
            
            self.frame_list = sorted(list(self.tracking_data.keys()))
            self.slider.setRange(0, len(self.frame_list) - 1)
            
            if self.frame_list:
                self.spin_from_frame.setRange(self.frame_list[0], self.frame_list[-1])
                self.spin_from_frame.setValue(self.frame_list[0])
                self.spin_to_frame.setRange(self.frame_list[0], self.frame_list[-1])
                self.spin_to_frame.setValue(self.frame_list[-1])
                
            self.current_frame_idx = 0
            self.selected_object_id = -1
            self.slider.setValue(0)
            self.reset_zoom()
            self.show_frame_by_index(0)
        except Exception as e:
            QMessageBox.critical(self, "Read Error", f"CSV Parsing Error: {str(e)}")

    def seek_frame(self, val):
        self.current_frame_idx = val
        self.show_frame_by_index(val)

    def step_frame(self, delta):
        target = max(0, min(len(self.frame_list) - 1, self.current_frame_idx + delta))
        self.slider.setValue(target)

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.zoom_center_x = 0.0
        self.zoom_center_y = 0.0
        self.show_frame_by_index(self.current_frame_idx)

    def show_frame_by_index(self, index):
        if not self.frame_list or index < 0 or index >= len(self.frame_list):
            return
        frame_id = self.frame_list[index]
        detections = self.tracking_data.get(frame_id, [])
        if not detections:
            return
            
        img_path = detections[0]["Image_Path"]
        if not os.path.exists(img_path):
            self.lbl_screen.setText(f"Missing image asset:\n{img_path}")
            return
            
        frame = cv2.imread(img_path)
        if frame is None:
            return
            
        overlay_frame = frame.copy()
        
        # User defined style params
        base_thickness = self.spin_box_thickness.value()
        txt_scale = self.spin_text_scale.value()
        
        for det in detections:
            obj_id = det["Object_ID"]
            class_id = det["Class_ID"]
            pts = self.parse_polygon_coords(det["Polygon_Coords"])
            
            if pts:
                pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                is_selected = (obj_id == self.selected_object_id)
                
                # Fetch class-specific color map indices
                class_color = self.colors[class_id % len(self.colors)]
                thickness = base_thickness if not is_selected else base_thickness + 2
                
                # Draw dynamic polygon contours
                cv2.polylines(overlay_frame, [pts_np], True, class_color, thickness)
                
                # Draw selection bounding indicators if selected
                if is_selected:
                    xs, ys = [pt[0] for pt in pts], [pt[1] for pt in pts]
                    cv2.rectangle(overlay_frame, (int(min(xs)) - 3, int(min(ys)) - 3), 
                                  (int(max(xs)) + 3, int(max(ys)) + 3), (0, 0, 255), 1)
                
                label_name = self.behavior_classes[class_id] if 0 <= class_id < len(self.behavior_classes) else f"Class {class_id}"
                cx, cy = int(det["Centroid_X"]), int(det["Centroid_Y"])
                cv2.putText(overlay_frame, f"ID {obj_id}: {label_name}", (cx - 30, cy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, txt_scale, class_color, 1, cv2.LINE_AA)
                            
        h, w = overlay_frame.shape[:2]
        if self.zoom_center_x == 0.0 and self.zoom_center_y == 0.0:
            self.zoom_center_x = w / 2.0
            self.zoom_center_y = h / 2.0
            
        if self.zoom_factor > 1.0:
            new_w = int(w / self.zoom_factor)
            new_h = int(h / self.zoom_factor)
            
            x1 = int(self.zoom_center_x - new_w / 2.0)
            y1 = int(self.zoom_center_y - new_h / 2.0)
            
            x1 = max(0, min(w - new_w, x1))
            y1 = max(0, min(h - new_h, y1))
            x2 = x1 + new_w
            y2 = y1 + new_h
            
            overlay_frame = overlay_frame[y1:y2, x1:x2]
            
        rgb = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * rgb.shape[1]
        q_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(q_img)
        self.lbl_screen.setPixmap(pix.scaled(self.lbl_screen.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        
        sel_text = f"Track ID {self.selected_object_id}" if self.selected_object_id != -1 else "None Selected"
        self.lbl_info.setText(f"Frame Index: {frame_id} | Object Selected: {sel_text} | Zoom: {self.zoom_factor:.1f}x")

    def parse_polygon_coords(self, coord_str):
        pts = []
        if not coord_str:
            return pts
        for p in coord_str.split(';'):
            if ',' in p:
                parts = p.split(',')
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        return pts

    def eventFilter(self, obj, event):
        if obj == self.lbl_screen:
            if event.type() == event.Type.Wheel:
                delta = event.angleDelta().y()
                if delta > 0:
                    self.zoom_factor = min(10.0, self.zoom_factor + 0.3)
                else:
                    self.zoom_factor = max(1.0, self.zoom_factor - 0.3)
                
                if self.zoom_factor == 1.0:
                    self.zoom_center_x = 0.0
                    self.zoom_center_y = 0.0
                self.show_frame_by_index(self.current_frame_idx)
                return True
                
            elif event.type() == event.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.pan_active = True
                    self.last_mouse_pos = event.position().toPoint()
                    return True
                    
            elif event.type() == event.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.pan_active = False
                    pos = event.position().toPoint()
                    if (pos - self.last_mouse_pos).manhattanLength() < 5:
                        self.handle_screen_click(pos.x(), pos.y())
                    return True
                    
            elif event.type() == event.Type.MouseMove:
                if self.pan_active and self.zoom_factor > 1.0:
                    curr_pos = event.position().toPoint()
                    diff = curr_pos - self.last_mouse_pos
                    self.last_mouse_pos = curr_pos
                    
                    pix_size = self.lbl_screen.pixmap().size() if self.lbl_screen.pixmap() else self.lbl_screen.size()
                    frame_id = self.frame_list[self.current_frame_idx]
                    detections = self.tracking_data[frame_id]
                    img_path = detections[0]["Image_Path"]
                    test_img = cv2.imread(img_path)
                    if test_img is not None:
                        h, w = test_img.shape[:2]
                        scale_x = (w / self.zoom_factor) / pix_size.width()
                        scale_y = (h / self.zoom_factor) / pix_size.height()
                        
                        self.zoom_center_x -= diff.x() * scale_x
                        self.zoom_center_y -= diff.y() * scale_y
                        
                        self.zoom_center_x = max(0.0, min(float(w), self.zoom_center_x))
                        self.zoom_center_y = max(0.0, min(float(h), self.zoom_center_y))
                        
                        self.show_frame_by_index(self.current_frame_idx)
                    return True
                    
        return super().eventFilter(obj, event)

    def handle_screen_click(self, label_x, label_y):
        if not self.frame_list or not self.lbl_screen.pixmap():
            return
        frame_id = self.frame_list[self.current_frame_idx]
        detections = self.tracking_data.get(frame_id, [])
        if not detections:
            return
            
        pix_size = self.lbl_screen.pixmap().size()
        lbl_w = self.lbl_screen.width()
        lbl_h = self.lbl_screen.height()
        
        offset_x = (lbl_w - pix_size.width()) / 2.0
        offset_y = (lbl_h - pix_size.height()) / 2.0
        
        click_x_in_pix = label_x - offset_x
        click_y_in_pix = label_y - offset_y
        
        if click_x_in_pix < 0 or click_y_in_pix < 0 or click_x_in_pix > pix_size.width() or click_y_in_pix > pix_size.height():
            return
            
        img_path = detections[0]["Image_Path"]
        test_img = cv2.imread(img_path)
        if test_img is None:
            return
        frame_h, frame_w = test_img.shape[:2]
        
        if self.zoom_factor > 1.0:
            crop_w = frame_w / self.zoom_factor
            crop_h = frame_h / self.zoom_factor
            
            x1 = self.zoom_center_x - crop_w / 2.0
            y1 = self.zoom_center_y - crop_h / 2.0
            x1 = max(0, min(frame_w - crop_w, x1))
            y1 = max(0, min(frame_h - crop_h, y1))
            
            target_x = x1 + (click_x_in_pix * (crop_w / pix_size.width()))
            target_y = y1 + (click_y_in_pix * (crop_h / pix_size.height()))
        else:
            target_x = click_x_in_pix * (frame_w / pix_size.width())
            target_y = click_y_in_pix * (frame_h / pix_size.height())
        
        closest_obj_id = -1
        min_dist = float('inf')
        
        for det in detections:
            pts = self.parse_polygon_coords(det["Polygon_Coords"])
            if pts:
                xs, ys = [pt[0] for pt in pts], [pt[1] for pt in pts]
                rect = QRectF(min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
                if rect.contains(target_x, target_y):
                    cx, cy = det["Centroid_X"], det["Centroid_Y"]
                    dist = (cx - target_x) ** 2 + (cy - target_y) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        closest_obj_id = det["Object_ID"]
                        
        self.selected_object_id = closest_obj_id
        self.show_frame_by_index(self.current_frame_idx)

    def add_behavior_class(self):
        new_beh = self.edit_new_behavior.text().strip().lower()
        if new_beh and new_beh not in self.behavior_classes:
            self.behavior_classes.append(new_beh)
            self.list_behaviors.addItem(new_beh)
            self.edit_new_behavior.clear()

    def edit_behavior_class(self):
        selected_item = self.list_behaviors.currentItem()
        if not selected_item:
            return
        row = self.list_behaviors.row(selected_item)
        old_val = selected_item.text()
        
        new_val, ok = QInputDialog.getText(self, "Rename Class", f"Rename behavioral class '{old_val}' to:", text=old_val)
        if ok and new_val.strip():
            new_val_clean = new_val.strip().lower()
            self.behavior_classes[row] = new_val_clean
            selected_item.setText(new_val_clean)
            self.show_frame_by_index(self.current_frame_idx)

    def delete_behavior_class(self):
        selected_item = self.list_behaviors.currentItem()
        if not selected_item:
            return
        row = self.list_behaviors.row(selected_item)
        class_name = selected_item.text()
        
        reply = QMessageBox.question(
            self, "Delete Class", f"Remove behavioral class '{class_name}' from selection menu?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.behavior_classes.pop(row)
            self.list_behaviors.takeItem(row)
            self.show_frame_by_index(self.current_frame_idx)

    def assign_behavior_current_frame(self):
        if self.selected_object_id == -1:
            QMessageBox.warning(self, "Selection", "Please click to select an animal in the screen viewport.")
            return
        sel_row = self.list_behaviors.currentRow()
        if sel_row < 0:
            QMessageBox.warning(self, "Selection", "Please select a target behavioral class from the list.")
            return
        frame_id = self.frame_list[self.current_frame_idx]
        for det in self.tracking_data.get(frame_id, []):
            if det["Object_ID"] == self.selected_object_id:
                det["Class_ID"] = sel_row
                break
        self.show_frame_by_index(self.current_frame_idx)

    def assign_behavior_range(self):
        if self.selected_object_id == -1:
            QMessageBox.warning(self, "Selection", "Please click to select an animal in the screen viewport.")
            return
        sel_row = self.list_behaviors.currentRow()
        if sel_row < 0:
            QMessageBox.warning(self, "Selection", "Please select a target behavioral class from the list.")
            return
            
        start_f = self.spin_from_frame.value()
        end_f = self.spin_to_frame.value()
        
        if start_f > end_f:
            QMessageBox.warning(self, "Validation", "Start Frame cannot be greater than End Frame.")
            return
            
        count_updated = 0
        for frame_id in range(start_f, end_f + 1):
            if frame_id in self.tracking_data:
                for det in self.tracking_data[frame_id]:
                    if det["Object_ID"] == self.selected_object_id:
                        det["Class_ID"] = sel_row
                        count_updated += 1
                        
        self.show_frame_by_index(self.current_frame_idx)
        QMessageBox.information(self, "Completed", f"Updated label class for Object ID {self.selected_object_id} in {count_updated} matching frames.")

    def assign_behavior_sequence(self):
        if self.selected_object_id == -1:
            return
        sel_row = self.list_behaviors.currentRow()
        if sel_row < 0:
            return
        for frame_id, detections in self.tracking_data.items():
            for det in detections:
                if det["Object_ID"] == self.selected_object_id:
                    det["Class_ID"] = sel_row
        self.show_frame_by_index(self.current_frame_idx)

    def quick_assign_class_index(self, index):
        if self.selected_object_id == -1 or index < 0 or index >= len(self.behavior_classes):
            return
        frame_id = self.frame_list[self.current_frame_idx]
        for det in self.tracking_data.get(frame_id, []):
            if det["Object_ID"] == self.selected_object_id:
                det["Class_ID"] = index
                break
        self.show_frame_by_index(self.current_frame_idx)

    def save_data_to_csv(self):
            if not self.current_csv_path:
                return False
            try:
                # Force write tab-separated format consistently to align with the YOLO reader
                temp_rows = []
                h_map = {name.strip(): idx for idx, name in enumerate(self.csv_headers)}
                for frame_id in sorted(self.tracking_data.keys()):
                    for det in self.tracking_data[frame_id]:
                        row = det["raw_row"]
                        row[h_map["Class_ID"]] = str(det["Class_ID"])
                        temp_rows.append(row)
                
                with open(self.current_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(self.csv_headers)
                    writer.writerows(temp_rows)
                return True
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Unable to write to CSV tracking file: {str(e)}")
                return False

    def save_and_accept(self):
        """Forces an automatic disk-save before exiting the dialog."""
        if self.save_data_to_csv():
            self.accept()
        else:
            reply = QMessageBox.question(
                self, "Save Verification", 
                "Could not write tracking data to the CSV file. Exit anyway without saving?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.accept()

    def confirm_and_close(self):
        if self.save_data_to_csv():
            self.accept()

    def closeEvent(self, event):
        super().closeEvent(event)