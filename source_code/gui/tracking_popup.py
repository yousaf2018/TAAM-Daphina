import os
import traceback
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

class AdvancedTrackingPopup(QDialog):
    def __init__(self, workspace, video_paths, model_names, rois, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TAAM | EthoGrid Tracking Engine")
        self.resize(1150, 950)
        self.workspace, self.video_paths, self.model_names, self.rois = workspace, video_paths, model_names, rois
        self.setStyleSheet("QDialog { background-color: #2b2b2b; color: white; } QGroupBox { font-weight: bold; color: #00a2ed; }")
        
        # Dynamically generate the exact list of active arenas based on the designed ROIs
        self.arenas_list = []
        idx = 1
        for s in self.rois:
            if s['type'] == 'grid':
                r, c = s['grid']
                for row in range(r):
                    for col in range(c):
                        self.arenas_list.append(f"Arena_{idx}")
                        idx += 1
            else:
                self.arenas_list.append(f"Arena_{idx}")
                idx += 1

        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); form = QVBoxLayout(content)

        # 1. Inputs
        g1 = QGroupBox("1. Batch Selection"); l1 = QVBoxLayout()
        self.vid_list = QListWidget()
        for v in self.video_paths: self.vid_list.addItem(os.path.basename(v))
        self.vid_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection); self.vid_list.selectAll()
        self.model_combo = QComboBox(); self.model_combo.addItems(self.model_names)
        l1.addWidget(QLabel("Videos:")); l1.addWidget(self.vid_list); l1.addWidget(QLabel("Model:")); l1.addWidget(self.model_combo)
        g1.setLayout(l1); form.addWidget(g1)

        # 2. Parameters
        g2 = QGroupBox("2. Inference & Tracking Tuning"); l2 = QFormLayout()
        self.combo_task = QComboBox(); self.combo_task.addItems(["Detection", "Segmentation", "Semantic Segmentation"])
        self.combo_method = QComboBox(); self.combo_method.addItems(["AMADEUS Visual-Kinematic", "Custom Force-N", "Confidence Filter", "Norfair", "BoTSORT"])
        self.spin_conf = QDoubleSpinBox(); self.spin_conf.setRange(0.01, 1.0); self.spin_conf.setValue(0.25)
        self.spin_max_n = QSpinBox(); self.spin_max_n.setRange(1, 999); self.spin_max_n.setValue(1)
        self.spin_iou = QDoubleSpinBox(); self.spin_iou.setRange(0.01, 1.0); self.spin_iou.setValue(0.45)
        self.spin_time_gap = QDoubleSpinBox(); self.spin_time_gap.setRange(0.1, 999.0); self.spin_time_gap.setValue(2.0)
        self.spin_sample_rate = QSpinBox(); self.spin_sample_rate.setRange(1, 100); self.spin_sample_rate.setValue(1)
        self.chk_auto_stitch = QCheckBox("Enable Auto-Stitching (Hungarian Ghost Match)"); self.chk_auto_stitch.setChecked(True)
        
        # Skeleton Keypoints extraction checkbox (Dynamic to Task selection)
        self.chk_extract_skeleton = QCheckBox("Extract Fish Keypoints (Head, Center, Tail)")
        self.chk_extract_skeleton.setChecked(False)
        self.chk_extract_skeleton.setEnabled(False)
        self.chk_extract_skeleton.toggled.connect(self.on_skeleton_checkbox_toggled)

        # Dynamic Checkable List representation for mapped arenas
        self.roi_list_widget = QListWidget()
        self.roi_list_widget.setFixedHeight(120)
        self.roi_list_widget.setEnabled(False)
        for arena_name in self.arenas_list:
            item = QListWidgetItem(arena_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.roi_list_widget.addItem(item)

        self.chk_include_background = QCheckBox("Include Background Class in Semantic Model")
        self.chk_include_background.setChecked(True)
        self.chk_include_background.setEnabled(False)
        
        l2.addRow("Task Type:", self.combo_task); l2.addRow("Tracking Algorithm:", self.combo_method)
        l2.addRow("Confidence Threshold:", self.spin_conf); l2.addRow("Max Animals per Tank/Arena:", self.spin_max_n)
        l2.addRow("NMS IOU Threshold:", self.spin_iou); l2.addRow("Trajectory Time Gap (sec):", self.spin_time_gap)
        l2.addRow("Frame Sample Rate:", self.spin_sample_rate); l2.addRow("", self.chk_auto_stitch)
        l2.addRow("", self.chk_extract_skeleton)
        l2.addRow("Skeleton Target ROIs:", self.roi_list_widget)
        l2.addRow("", self.chk_include_background)
        g2.setLayout(l2); form.addWidget(g2)
        
        self.combo_task.currentTextChanged.connect(self.on_task_changed)

        # 3. AMADEUS Multi-Staged Association & Kinematics
        g_multi = QGroupBox("3. AMADEUS Multi-Staged Association & Kinematics"); l_multi = QFormLayout()
        
        self.spin_max_age = QSpinBox(); self.spin_max_age.setRange(1, 500); self.spin_max_age.setValue(30)
        self.spin_min_hits = QSpinBox(); self.spin_min_hits.setRange(1, 50); self.spin_min_hits.setValue(3)
        self.spin_weight_dist = QDoubleSpinBox(); self.spin_weight_dist.setRange(0.0, 1.0); self.spin_weight_dist.setValue(0.3); self.spin_weight_dist.setSingleStep(0.1)
        self.spin_weight_vis = QDoubleSpinBox(); self.spin_weight_vis.setRange(0.0, 1.0); self.spin_weight_vis.setValue(0.4); self.spin_weight_vis.setSingleStep(0.1)
        self.spin_weight_dir = QDoubleSpinBox(); self.spin_weight_dir.setRange(0.0, 1.0); self.spin_weight_dir.setValue(0.3); self.spin_weight_dir.setSingleStep(0.1)
        
        self.chk_calc_direction = QCheckBox("Estimate Kinematic Direction (Forward/Backward)")
        self.chk_calc_direction.setChecked(True)
        self.spin_dir_gap = QDoubleSpinBox(); self.spin_dir_gap.setRange(0.1, 10.0); self.spin_dir_gap.setValue(0.5); self.spin_dir_gap.setSuffix(" sec")

        l_multi.addRow("Max Age (Frames to keep lost track alive):", self.spin_max_age)
        l_multi.addRow("Min Hits (Frames to confirm detection):", self.spin_min_hits)
        l_multi.addRow("Spatial Distance Weight (AMADEUS):", self.spin_weight_dist)
        l_multi.addRow("Visual Template Weight (AMADEUS):", self.spin_weight_vis)
        l_multi.addRow("Directional Momentum Weight (AMADEUS):", self.spin_weight_dir)
        l_multi.addRow("", self.chk_calc_direction)
        l_multi.addRow("Direction Trajectory Gap:", self.spin_dir_gap)
        g_multi.setLayout(l_multi); form.addWidget(g_multi)

        # 4. 3D Locomotion Reconstruction parameters
        g_3d = QGroupBox("4. 3D Locomotion Reconstruction (Top + Side Panels)"); l_3d = QFormLayout()
        self.chk_reconstruct_3d = QCheckBox("Enable 3D Trajectory Reconstruction")
        self.chk_reconstruct_3d.setChecked(False)
        
        self.spin_px_to_cm = QDoubleSpinBox()
        self.spin_px_to_cm.setRange(0.1, 1000.0)
        self.spin_px_to_cm.setValue(10.0)
        self.spin_px_to_cm.setSuffix(" px/cm")
        
        self.combo_top_view = QComboBox()
        self.combo_top_view.addItems(self.arenas_list)
        
        self.combo_side_view = QComboBox()
        self.combo_side_view.addItems(self.arenas_list)
        
        l_3d.addRow("", self.chk_reconstruct_3d)
        l_3d.addRow("Calibration Factor:", self.spin_px_to_cm)
        l_3d.addRow("Top View Panel:", self.combo_top_view)
        l_3d.addRow("Side View Panel:", self.combo_side_view)
        g_3d.setLayout(l_3d); form.addWidget(g_3d)

        # 5. Visualization Tuning
        g_viz = QGroupBox("5. Visual Output Adjustments"); l_viz = QFormLayout()
        self.spin_bbox_thick = QSpinBox(); self.spin_bbox_thick.setRange(1, 15); self.spin_bbox_thick.setValue(3)
        self.spin_lbl_scale = QDoubleSpinBox(); self.spin_lbl_scale.setRange(0.1, 5.0); self.spin_lbl_scale.setSingleStep(0.1); self.spin_lbl_scale.setValue(0.6)
        self.spin_lbl_thick = QSpinBox(); self.spin_lbl_thick.setRange(1, 10); self.spin_lbl_thick.setValue(2)
        self.spin_dot_size = QSpinBox(); self.spin_dot_size.setRange(1, 30); self.spin_dot_size.setValue(5)
        
        l_viz.addRow("Bounding Box Thickness:", self.spin_bbox_thick)
        l_viz.addRow("Label Text Size (Scale):", self.spin_lbl_scale)
        l_viz.addRow("Label Text Thickness:", self.spin_lbl_thick)
        l_viz.addRow("Centroid Dot Size:", self.spin_dot_size)
        g_viz.setLayout(l_viz); form.addWidget(g_viz)

        # 6. Export
        g3 = QGroupBox("6. Scientific Analytics & Video Exports"); l3 = QGridLayout()
        self.chk_inf_vid = QCheckBox("Save Inference Video"); self.chk_inf_vid.setChecked(True)
        self.chk_track_vid = QCheckBox("Save Tracked Video"); self.chk_track_vid.setChecked(True)
        self.chk_csv = QCheckBox("Export Standard CSV"); self.chk_csv.setChecked(True)
        self.chk_centroid = QCheckBox("Export Centroid CSV"); self.chk_centroid.setChecked(True)
        self.chk_xlsx_track = QCheckBox("Export Excel (By Track)"); self.chk_xlsx_track.setChecked(True)
        self.chk_xlsx_tank = QCheckBox("Export Excel (By Tank)"); self.chk_xlsx_tank.setChecked(True)
        self.chk_traj = QCheckBox("Save Trajectory Image"); self.chk_traj.setChecked(True)
        self.chk_heat = QCheckBox("Save Heatmap Image"); self.chk_heat.setChecked(True)
        
        l3.addWidget(self.chk_inf_vid, 0, 0); l3.addWidget(self.chk_track_vid, 0, 1)
        l3.addWidget(self.chk_csv, 1, 0); l3.addWidget(self.chk_centroid, 1, 1)
        l3.addWidget(self.chk_xlsx_track, 2, 0); l3.addWidget(self.chk_xlsx_tank, 2, 1)
        l3.addWidget(self.chk_traj, 3, 0); l3.addWidget(self.chk_heat, 3, 1)
        g3.setLayout(l3); form.addWidget(g3)

        scroll.setWidget(content); layout.addWidget(scroll)

        self.btn_go = QPushButton("🚀 LAUNCH BATCH PIPELINE")
        self.btn_go.setFixedHeight(60); self.btn_go.setStyleSheet("background: #28a745; font-weight: bold; font-size: 14px;")
        self.btn_go.clicked.connect(self.on_launch_clicked) 
        layout.addWidget(self.btn_go)

    def on_task_changed(self, task_name):
        is_segmentation = (task_name in ["Segmentation", "Semantic Segmentation"])
        is_semantic = (task_name == "Semantic Segmentation")
        self.chk_extract_skeleton.setEnabled(is_segmentation)
        self.chk_include_background.setEnabled(is_semantic)
        if not is_segmentation: self.chk_extract_skeleton.setChecked(False)
        if not is_semantic: self.chk_include_background.setChecked(False)
        self.on_skeleton_checkbox_toggled()

    def on_skeleton_checkbox_toggled(self):
        is_active = self.chk_extract_skeleton.isEnabled() and self.chk_extract_skeleton.isChecked()
        self.roi_list_widget.setEnabled(is_active)
        self.chk_calc_direction.setEnabled(is_active)
        self.spin_dir_gap.setEnabled(is_active)

    def on_launch_clicked(self):
        self.save_settings()
        self.accept()

    def load_settings(self):
        settings = QSettings("TAAM", "AdvancedTracking")
        if settings.contains("task_type"):
            self.combo_task.setCurrentText(settings.value("task_type", "Detection"))
            self.combo_method.setCurrentText(settings.value("method", "AMADEUS Visual-Kinematic"))
            self.spin_conf.setValue(float(settings.value("conf", 0.25)))
            self.spin_max_n.setValue(int(settings.value("max_n", 1)))
            self.spin_iou.setValue(float(settings.value("iou", 0.45)))
            self.spin_time_gap.setValue(float(settings.value("time_gap", 2.0)))
            self.spin_sample_rate.setValue(int(settings.value("sample_rate", 1)))
            self.chk_auto_stitch.setChecked(settings.value("auto_stitch", True, type=bool))
            
            self.chk_extract_skeleton.setChecked(settings.value("extract_skeleton", False, type=bool))
            self.chk_include_background.setChecked(settings.value("include_background", True, type=bool))

            self.spin_max_age.setValue(int(settings.value("max_age", 30)))
            self.spin_min_hits.setValue(int(settings.value("min_hits", 3)))
            self.spin_weight_dist.setValue(float(settings.value("weight_dist", 0.3)))
            self.spin_weight_vis.setValue(float(settings.value("weight_vis", 0.4)))
            self.spin_weight_dir.setValue(float(settings.value("weight_dir", 0.3)))
            self.chk_calc_direction.setChecked(settings.value("calc_direction", True, type=bool))
            self.spin_dir_gap.setValue(float(settings.value("dir_gap_sec", 0.5)))

            self.chk_reconstruct_3d.setChecked(settings.value("reconstruct_3d", False, type=bool))
            self.spin_px_to_cm.setValue(float(settings.value("px_to_cm", 10.0)))
            self.combo_top_view.setCurrentText(settings.value("top_view_arena", "Arena_1"))
            self.combo_side_view.setCurrentText(settings.value("side_view_arena", "Arena_2"))

            saved_rois = settings.value("skeleton_rois", [])
            if saved_rois:
                for i in range(self.roi_list_widget.count()):
                    item = self.roi_list_widget.item(i)
                    if item.text() in saved_rois:
                        item.setCheckState(Qt.CheckState.Checked)
                    else:
                        item.setCheckState(Qt.CheckState.Unchecked)

            self.spin_bbox_thick.setValue(int(settings.value("bbox_thick", 3)))
            self.spin_lbl_scale.setValue(float(settings.value("lbl_scale", 0.6)))
            self.spin_lbl_thick.setValue(int(settings.value("lbl_thick", 2)))
            self.spin_dot_size.setValue(int(settings.value("dot_size", 5)))

            self.chk_inf_vid.setChecked(settings.value("inf_vid", True, type=bool))
            self.chk_track_vid.setChecked(settings.value("track_vid", True, type=bool))
            self.chk_csv.setChecked(settings.value("csv", True, type=bool))
            self.chk_centroid.setChecked(settings.value("centroid", True, type=bool))
            self.chk_xlsx_track.setChecked(settings.value("xlsx_track", True, type=bool))
            self.chk_xlsx_tank.setChecked(settings.value("xlsx_tank", True, type=bool))
            self.chk_traj.setChecked(settings.value("traj", True, type=bool))
            self.chk_heat.setChecked(settings.value("heat", True, type=bool))
        
        self.on_task_changed(self.combo_task.currentText())

    def save_settings(self):
        settings = QSettings("TAAM", "AdvancedTracking")
        settings.setValue("task_type", self.combo_task.currentText())
        settings.setValue("method", self.combo_method.currentText())
        settings.setValue("conf", self.spin_conf.value())
        settings.setValue("max_n", self.spin_max_n.value())
        settings.setValue("iou", self.spin_iou.value())
        settings.setValue("time_gap", self.spin_time_gap.value())
        settings.setValue("sample_rate", self.spin_sample_rate.value())
        settings.setValue("auto_stitch", self.chk_auto_stitch.isChecked())
        
        settings.setValue("extract_skeleton", self.chk_extract_skeleton.isChecked())
        settings.setValue("include_background", self.chk_include_background.isChecked())

        settings.setValue("max_age", self.spin_max_age.value())
        settings.setValue("min_hits", self.spin_min_hits.value())
        settings.setValue("weight_dist", self.spin_weight_dist.value())
        settings.setValue("weight_vis", self.spin_weight_vis.value())
        settings.setValue("weight_dir", self.spin_weight_dir.value())
        settings.setValue("calc_direction", self.chk_calc_direction.isChecked())
        settings.setValue("dir_gap_sec", self.spin_dir_gap.value())

        settings.setValue("reconstruct_3d", self.chk_reconstruct_3d.isChecked())
        settings.setValue("px_to_cm", self.spin_px_to_cm.value())
        settings.setValue("top_view_arena", self.combo_top_view.currentText())
        settings.setValue("side_view_arena", self.combo_side_view.currentText())

        checked_rois = [self.roi_list_widget.item(i).text() for i in range(self.roi_list_widget.count()) if self.roi_list_widget.item(i).checkState() == Qt.CheckState.Checked]
        settings.setValue("skeleton_rois", checked_rois)

        settings.setValue("bbox_thick", self.spin_bbox_thick.value())
        settings.setValue("lbl_scale", self.spin_lbl_scale.value())
        settings.setValue("lbl_thick", self.spin_lbl_thick.value())
        settings.setValue("dot_size", self.spin_dot_size.value())

        settings.setValue("inf_vid", self.chk_inf_vid.isChecked())
        settings.setValue("track_vid", self.chk_track_vid.isChecked())
        settings.setValue("csv", self.chk_csv.isChecked())
        settings.setValue("centroid", self.chk_centroid.isChecked())
        settings.setValue("xlsx_track", self.chk_xlsx_track.isChecked())
        settings.setValue("xlsx_tank", self.chk_xlsx_tank.isChecked())
        settings.setValue("traj", self.chk_traj.isChecked())
        settings.setValue("heat", self.chk_heat.isChecked())

    def get_config(self):
        v_idx = [i for i in range(self.vid_list.count()) if self.vid_list.item(i).isSelected()]
        checked_rois = [self.roi_list_widget.item(i).text() for i in range(self.roi_list_widget.count()) if self.roi_list_widget.item(i).checkState() == Qt.CheckState.Checked]
        
        return {
            "videos": [self.video_paths[i] for i in v_idx],
            "model_name": self.model_combo.currentText(),
            "task_type": self.combo_task.currentText(),
            "method": self.combo_method.currentText(),
            "conf": self.spin_conf.value(),
            "max_n": self.spin_max_n.value(),
            "iou_threshold": self.spin_iou.value(),
            "time_gap_seconds": self.spin_time_gap.value(),
            "frame_sample_rate": self.spin_sample_rate.value(),
            "auto_stitch": self.chk_auto_stitch.isChecked(),
            
            "extract_skeleton": self.chk_extract_skeleton.isChecked(),
            "skeleton_rois": checked_rois,
            "include_background": self.chk_include_background.isChecked(),

            "max_age": self.spin_max_age.value(),
            "min_hits": self.spin_min_hits.value(),
            "weight_dist": self.spin_weight_dist.value(),
            "weight_vis": self.spin_weight_vis.value(),
            "weight_dir": self.spin_weight_dir.value(),
            "calc_direction": self.chk_calc_direction.isChecked(),
            "dir_gap_sec": self.spin_dir_gap.value(),
            
            "reconstruct_3d": self.chk_reconstruct_3d.isChecked(),
            "px_to_cm": self.spin_px_to_cm.value(),
            "top_view_arena": self.combo_top_view.currentText(),
            "side_view_arena": self.combo_side_view.currentText(),
            
            "bbox_thick": self.spin_bbox_thick.value(),
            "lbl_scale": self.spin_lbl_scale.value(),
            "lbl_thick": self.spin_lbl_thick.value(),
            "dot_size": self.spin_dot_size.value(),

            "rois": self.rois,
            "save_inference_vid": self.chk_inf_vid.isChecked(),
            "save_video": self.chk_track_vid.isChecked(),
            "save_csv": self.chk_csv.isChecked(),
            "save_centroid_csv": self.chk_centroid.isChecked(),
            "save_excel_track": self.chk_xlsx_track.isChecked(),
            "save_excel_tank": self.chk_xlsx_tank.isChecked(),
            "save_traj": self.chk_traj.isChecked(),
            "save_heat": self.chk_heat.isChecked()
        }