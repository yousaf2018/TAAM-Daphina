import os, cv2, pandas as pd, numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QListWidget, QSplitter, QFrame)
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent
from PyQt6.QtCore import Qt

class BehaviorAnnotatorWidget(QWidget):
    # Signal to tell MainWindow to switch back to main view
    back_to_main = pyqtSignal() 
    # Signal to notify that new labels are ready for processing
    labels_ready = pyqtSignal(str) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracking_df = None
        self.video_cap = None
        self.current_frame_lt = 0
        self.all_frames_in_csv = []
        self.labels_session = []
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- LEFT PANEL (Controls) ---
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        
        self.btn_back = QPushButton("⬅ Back to Main")
        self.btn_back.clicked.connect(lambda: self.back_to_main.emit())

        self.frame_list = QListWidget()
        self.frame_list.itemClicked.connect(self.on_list_click)

        self.btn_export = QPushButton("💾 Export Labels")
        self.btn_export.clicked.connect(self.finalize_session)

        left_layout.addWidget(self.btn_back)
        left_layout.addWidget(QLabel("<b>Frames</b>"))
        left_layout.addWidget(self.frame_list)
        left_layout.addWidget(self.btn_export)

        # --- RIGHT PANEL (Viewer) ---
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)

        self.video_display = QLabel("Please load data from Main Window first")
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display.setStyleSheet("background: black;")

        # Quick Labels Bar
        label_bar = QHBoxLayout()
        self.btn_f = QPushButton("F: FIGHT"); self.btn_f.setStyleSheet("background:#e74c3c; color:white;")
        self.btn_c = QPushButton("C: CHASE"); self.btn_c.setStyleSheet("background:#f1c40f; color:black;")
        self.btn_i = QPushButton("I: INTERACT"); self.btn_i.setStyleSheet("background:#2ecc71; color:white;")
        
        for b in [self.btn_f, self.btn_c, self.btn_i]:
            label_bar.addWidget(b)
        
        self.btn_f.clicked.connect(lambda: self.apply_label("fighting"))
        self.btn_c.clicked.connect(lambda: self.apply_label("chasing"))
        self.btn_i.clicked.connect(lambda: self.apply_label("interaction"))

        right_layout.addWidget(self.video_display, 8)
        right_layout.addLayout(label_bar)

        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)
        layout.addWidget(self.splitter)

    def load_data(self, csv_path, video_path):
        """Called by MainWindow to inject data into this widget."""
        try:
            self.tracking_df = pd.read_csv(csv_path)
            self.video_cap = cv2.VideoCapture(video_path)
            self.all_frames_in_csv = sorted(self.tracking_df['frame_idx'].unique().tolist())
            self.frame_list.clear()
            self.frame_list.addItems([str(f)	for f in self.all_frames_in_csv])
            self.current_frame_lt = self.all_frames_in_csv[0]
            self.render_frame()
        except Exception as e:
            print(f"Error loading data into Annotator: {e}")

    def render_frame(self):
        if not self.video_cap: return
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_lt)
        ret, frame = self.video_cap.read()
        if ret:
            # Draw existing tracking info from CSV
            frame_data = self.tracking_df[self.tracking_df['frame_idx'] == self.current_frame_lt]
            for _, row in frame_data.iterrows():
                cv2_color = (0, 255, 0)
                cv2.putText(frame, f"ID:{row['track_id']}", (int(row['cx']-15), int(row['cy']-15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cv_color, 1)
            
            # Convert to QImage
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            img = QImage(rgb.data, w, h, w*c, QImage.Format.Format_RGB8	88).scaled(
                self.video_display.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.video_display.setPixmap(QPixmap.fromImage(img))

    def apply_label(self, label_name):
        # Check if frame already labeled in this session
        self.labels_session = [item for item in self.labels_session if item['frame'] != self.current_frame_lt]
        self.labels_session.append({'frame': self.current_frame_lt, 'label': label_name})
        self.render_frame()

    def on_list_click(self, item):
        self.current_frame_lt = int(item.text())
        self.render_frame()

    def finalize_session(self):
        if not self.labels_session: return
        # Save to a CSV in the same folder as the original tracking csv
        save_path = self.tracking_df.iloc[0]['frame_idx'] # Placeholder logic
        # In real usage, pass the directory from MainWindow
        self.labels_ready.emit("Annotation Complete")

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Right: self.next_frame()
        elif event.key() == Qt.Key.Key_Left: self.prev_frame()
        elif event.key() == Qt.Key.Key_F: self.apply_label("fighting")
        elif event.key() == Qt.Key.Key_C: self.apply_label("chasing")
        elif event.key() == Qt.Key.Key_I: self.apply_label("interaction")

    def next_frame(self):
        idx = self.all_frames_in_csv.index(self.current_frame_lt)
        if idx < len(self.all_frames_in_csv)-1:
            self.current_frame_lt = self.all_frames_in_csv[idx+1]
            self.render_frame()

    def prev_frame(self):
        idx = self.all_frames_in_csv.index(self.current_frame_lt)
        if idx > 0:
            self.current_frame_lt = self.all_frames_in_csv[idx-1]
            self.render_frame()