import cv2
import numpy as np

# ==============================================================================
# 1. AQUARIUM PHYSICAL CONFIGURATION (Edit these values to match your tank)
# ==============================================================================
# Enter the actual physical dimensions of your aquarium in centimeters (or inches)
TANK_PHYSICAL_WIDTH_CM = 120.0  # From left side to right side of the tank
TANK_PHYSICAL_HEIGHT_CM = 60.0  # From bottom glass to top rim (Side View height)
TANK_PHYSICAL_DEPTH_CM = 50.0   # From front glass to back glass (Top View depth)

# Define how many pixels should represent 1 cm in the output video.
# Increasing this value increases the output resolution but takes more processing time.
PIXELS_PER_CM = 8.0  

# ==============================================================================
# Global variables for mouse interactions
# ==============================================================================
selected_points = []
scale_factor_disp = 1.0

def select_points_callback(event, x, y, flags, param):
    """Handles user clicks for ROI selection and scales them back to original resolution."""
    global selected_points, scale_factor_disp
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) < 4:
            orig_x = int(x / scale_factor_disp)
            orig_y = int(y / scale_factor_disp)
            selected_points.append((orig_x, orig_y))
            
            # Draw visual feedback
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(param, f"P{len(selected_points)}", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("Calibration - Step 2: Select ROI Corners", param)

def draw_overlay_instructions(img, lines):
    """Draws a semi-transparent black bar with instruction text at the top of the frame."""
    h, w = img.shape[:2]
    box_h = len(lines) * 25 + 20
    # Create an overlay layer
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, box_h), (0, 0, 0), -1)
    # Apply transparency
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    # Draw text
    for i, line in enumerate(lines):
        cv2.putText(img, line, (15, 25 + i * 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def undistort_frame(frame, k1, k2, zoom):
    """Applies manual radial undistortion to straighten curved lines."""
    h, w = frame.shape[:2]
    f = w * zoom
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)
    D = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
    
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_32FC1)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

def reduce_glare_reflections(img):
    """Balances contrast using CLAHE in LAB space to mitigate highlights/glare."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def run_lens_tuning_gui(raw_frame, camera_name, disp_w, disp_h):
    """Opens a UI window with sliders to let the user manually flatten the lens curves."""
    win_name = f"Step 1: Straighten Lens - {camera_name}"
    cv2.namedWindow(win_name)
    
    # Trackbars (offset by 500 to allow negative/positive values)
    cv2.createTrackbar("K1 (Lens Curve)", win_name, 500, 1000, lambda x: None)
    cv2.createTrackbar("K2 (Edge Curve)", win_name, 500, 1000, lambda x: None)
    cv2.createTrackbar("Zoom / Crop Adjust", win_name, 100, 200, lambda x: None)
    
    while True:
        pos_k1 = cv2.getTrackbarPos("K1 (Lens Curve)", win_name)
        pos_k2 = cv2.getTrackbarPos("K2 (Edge Curve)", win_name)
        pos_zoom = cv2.getTrackbarPos("Zoom / Crop Adjust", win_name)
        
        # Normalize variables
        val_k1 = (pos_k1 - 500) / 1000.0
        val_k2 = (pos_k2 - 500) / 5000.0
        val_zoom = pos_zoom / 100.0
        if val_zoom == 0: val_zoom = 0.1
        
        # Apply current calibration setup
        corrected = undistort_frame(raw_frame, val_k1, val_k2, val_zoom)
        disp_img = cv2.resize(corrected, (disp_w, disp_h))
        
        # Draw green guidelines to help align curves
        cv2.line(disp_img, (0, disp_h // 2), (disp_w, disp_h // 2), (0, 255, 0), 1)
        cv2.line(disp_img, (disp_w // 2, 0), (disp_w // 2, disp_h), (0, 255, 0), 1)
        
        # Text Overlay Instructions
        instructions = [
            f"CAMERA: {camera_name}",
            "1. Move 'K1' to the LEFT to fix barrel/fisheye curve.",
            "2. Adjust 'Zoom' to remove black corner boundaries if they appear.",
            "3. Align the curved lines of the tank to match the straight green lines.",
            "Press 'ENTER' or 'SPACE' when lines look straight."
        ]
        draw_overlay_instructions(disp_img, instructions)
        
        cv2.imshow(win_name, disp_img)
        key = cv2.waitKey(30) & 0xFF
        if key in [13, 32]:  # Enter or Space
            cv2.destroyWindow(win_name)
            return val_k1, val_k2, val_zoom
        elif key == 27:  # Esc
            cv2.destroyAllWindows()
            exit()

def run_roi_selection_gui(corrected_frame, camera_name, disp_w, disp_h, scale_val, plane_description):
    """Opens a UI window to select the 4 coordinates to execute the perspective warp."""
    global selected_points, scale_factor_disp
    selected_points = []
    scale_factor_disp = scale_val
    
    disp_img = cv2.resize(corrected_frame, (disp_w, disp_h))
    clone = disp_img.copy()
    
    win_name = "Calibration - Step 2: Select ROI Corners"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, select_points_callback, clone)
    
    while True:
        # We re-draw instructions on a copy so we don't overwrite point selection drawings
        temp_draw = clone.copy()
        instructions = [
            f"CAMERA: {camera_name}",
            f"TARGET PLANE: Select the 4 corners of the {plane_description}",
            "Click points precisely in this order:",
            "  1. Top-Left  |  2. Top-Right  |  3. Bottom-Left  |  4. Bottom-Right",
            "Press 'ENTER' to confirm when exactly 4 markers are visible."
        ]
        draw_overlay_instructions(temp_draw, instructions)
        
        cv2.imshow(win_name, temp_draw)
        key = cv2.waitKey(30) & 0xFF
        if key == 13:  # Enter
            if len(selected_points) == 4:
                cv2.destroyWindow(win_name)
                return np.float32(selected_points)
            else:
                print(f"Error: 4 points required. You have selected {len(selected_points)}.")
        elif key == 27:  # Esc
            cv2.destroyAllWindows()
            exit()

# ==============================================================================
# Main Program Loop
# ==============================================================================
def main():
    # Load input video
    video_path = '/media/mahmood-yousaf/Expansion/3D/3D videos/20260813_Blood parrot vs Jaguar cichlid.mkv'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video path: {video_path}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not retrieve reference frame.")
        cap.release()
        return

    # Split side-by-side feed into left and right sub-frames
    tot_h, tot_w = first_frame.shape[:2]
    half_w = tot_w // 2
    
    side_view_raw = first_frame[:, :half_w]
    top_view_raw = first_frame[:, half_w:]

    # Calculate proportional GUI window size to prevent display overflow
    max_disp_w, max_disp_h = 1024, 576
    scale_factor = min(max_disp_w / half_w, max_disp_h / tot_h)
    disp_w = int(half_w * scale_factor)
    disp_h = int(tot_h * scale_factor)

    # 1. Calibrate Left Camera (Side View)
    k1_L, k2_L, zoom_L = run_lens_tuning_gui(side_view_raw, "LEFT (Side View)", disp_w, disp_h)
    straight_side = undistort_frame(side_view_raw, k1_L, k2_L, zoom_L)
    src_pts_L = run_roi_selection_gui(
        straight_side, "LEFT (Side View)", disp_w, disp_h, scale_factor, 
        "BACK glass wall (or front glass) of your aquarium."
    )

    # 2. Calibrate Right Camera (Top View)
    k1_R, k2_R, zoom_R = run_lens_tuning_gui(top_view_raw, "RIGHT (Top View)", disp_w, disp_h)
    straight_top = undistort_frame(top_view_raw, k1_R, k2_R, zoom_R)
    src_pts_R = run_roi_selection_gui(
        straight_top, "RIGHT (Top View)", disp_w, disp_h, scale_factor, 
        "BOTTOM glass floor (or top rim frame) of your aquarium."
    )

    # Calculate exact scaled output dimensions based on user physical parameters
    out_w_side = int(TANK_PHYSICAL_WIDTH_CM * PIXELS_PER_CM)
    out_h_side = int(TANK_PHYSICAL_HEIGHT_CM * PIXELS_PER_CM)

    out_w_top = int(TANK_PHYSICAL_WIDTH_CM * PIXELS_PER_CM)
    out_h_top = int(TANK_PHYSICAL_DEPTH_CM * PIXELS_PER_CM)

    # Define standard flat destination point grids
    dst_pts_side = np.float32([
        [0, 0],
        [out_w_side, 0],
        [0, out_h_side],
        [out_w_side, out_h_side]
    ])

    dst_pts_top = np.float32([
        [0, 0],
        [out_w_top, 0],
        [0, out_h_top],
        [out_w_top, out_h_top]
    ])

    # Compute matrices
    M_L = cv2.getPerspectiveTransform(src_pts_L, dst_pts_side)
    M_R = cv2.getPerspectiveTransform(src_pts_R, dst_pts_top)

    # To concatenate left and right back together side-by-side, they must share the same height.
    # We choose the maximum height and pad the shorter frame dynamically to avoid deformation.
    final_h = max(out_h_side, out_h_top)
    final_w = out_w_side + out_w_top

    # Setup output video writer
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('normalized_alignment_output.mp4', fourcc, fps, (final_w, final_h))

    # Reset video capture to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Configure resizable execution display window
    cv2.namedWindow('Processing Execution Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Processing Execution Frame', disp_w * 2, disp_h)

    print("\n--- Running Video Correction Pipeline ---")
    print("Writing output video to 'normalized_alignment_output.mp4'. Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Split video frame
        left_raw = frame[:, :half_w]
        right_raw = frame[:, half_w:]
        
        # Step A: Remove lens curves (Undistort)
        left_straight = undistort_frame(left_raw, k1_L, k2_L, zoom_L)
        right_straight = undistort_frame(right_raw, k1_R, k2_R, zoom_R)
        
        # Step B: Contrast balance / Glare mitigation
        left_straight = reduce_glare_reflections(left_straight)
        right_straight = reduce_glare_reflections(right_straight)
        
        # Step C: Warp perspectives to linear output planes
        left_warped = cv2.warpPerspective(left_straight, M_L, (out_w_side, out_h_side))
        right_warped = cv2.warpPerspective(right_straight, M_R, (out_w_top, out_h_top))
        
        # Step D: Height alignment padding and concatenation
        # Pad left frame if its output height is smaller than final height
        pad_left = np.zeros((final_h, out_w_side, 3), dtype=np.uint8)
        pad_left[:out_h_side, :out_w_side] = left_warped

        # Pad right frame if its output height is smaller than final height
        pad_right = np.zeros((final_h, out_w_top, 3), dtype=np.uint8)
        pad_right[:out_h_top, :out_w_top] = right_warped

        # Combine side-by-side
        combined_frame = np.hstack((pad_left, pad_right))
        
        # Write output frame
        out.write(combined_frame)
        
        # Display runtime preview
        cv2.imshow('Processing Execution Frame', combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\nProcessing complete. Saved as 'normalized_alignment_output.mp4'.")

if __name__ == '__main__':
    main()