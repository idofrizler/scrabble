import cv2
import numpy as np
import argparse
import sys
import time

class ScrabbleTracker:
    def __init__(self, video_path, manual_corners=None):
        self.cap = cv2.VideoCapture(video_path)
        self.points = []
        if manual_corners:
            self.points = manual_corners
            
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        # State variables
        self.prev_gray = None
        self.ref_gray = None
        self.ref_keypoints = None
        self.ref_descriptors = None
        
        # Tracking state
        self.current_corners = None # The smoothed current position
        self.last_valid_matrix = None 

        # New: 2D board representation
        self.board_size = (450, 450)  # Width, Height of the 2D board
        self.grid_size = 15
        self.cell_size = self.board_size[0] // self.grid_size, self.board_size[1] // self.grid_size

        # Tile detection state
        self.reference_cell_colors = None  # 15x15 array of reference LAB colors
        self.cell_change_duration_ms = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)  # ms of consecutive change
        self.cell_no_change_duration_ms = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)  # ms of no change (for unlocking)
        self.locked_cells = np.zeros((self.grid_size, self.grid_size), dtype=bool)  # locked tile flags
        self.last_frame_time = None  # for delta time calculation
        
        # Tile detection parameters
        self.COLOR_CHANGE_THRESHOLD = 25.0  # LAB color difference threshold
        self.LOCK_IN_TIME_MS = 8000.0  # 8 seconds to lock
        self.UNLOCK_TIME_MS = 5000.0  # 5 seconds to unlock

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                # Visual feedback
                cv2.circle(self.display_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('Mark 4 Corners', self.display_frame)

    def initialize(self):
        ret, frame = self.cap.read()
        if not ret: 
            print("Error: Could not read video.")
            return False
        
        self.ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.display_frame = frame.copy()

        # If corners weren't passed via CLI, ask user to click
        if len(self.points) != 4:
            cv2.imshow('Mark 4 Corners', self.display_frame)
            cv2.setMouseCallback('Mark 4 Corners', self.click_event)
            
            print("Click the 4 corners (TL, TR, BR, BL)...")
            while len(self.points) < 4:
                cv2.waitKey(1)
            
            cv2.destroyWindow('Mark 4 Corners')

        # Output for user to copy-paste later
        flat_coords = ",".join([f"{int(x)},{int(y)}" for x, y in self.points])
        print(f"\n--- CONFIGURATION FOR NEXT RUN ---")
        print(f"Corner Args: --corners \"{flat_coords}\"")
        print(f"----------------------------------\n")

        # Convert to numpy format
        self.ref_corners = np.array(self.points, dtype=np.float32).reshape(-1, 1, 2)
        self.current_corners = self.ref_corners.copy()

        # Generate mask to ignore background features in reference
        mask = np.zeros(self.ref_gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(self.points)], 255)
        
        # Compute reference features (The "Fingerprint" of the board)
        self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(self.ref_gray, mask)
        
        return True

    def get_cell_average_color(self, warped_board_lab, row, col):
        """Get the average LAB color of a specific cell."""
        x1 = col * self.cell_size[0]
        y1 = row * self.cell_size[1]
        x2 = x1 + self.cell_size[0]
        y2 = y1 + self.cell_size[1]
        
        # Use center region of cell (avoid grid lines)
        margin = 3
        cell_region = warped_board_lab[y1+margin:y2-margin, x1+margin:x2-margin]
        
        if cell_region.size == 0:
            return np.array([0, 128, 128], dtype=np.float32)  # Default gray in LAB
        
        return np.mean(cell_region, axis=(0, 1)).astype(np.float32)

    def capture_reference_colors(self, warped_board):
        """Capture reference colors for all cells from the initial board state."""
        warped_lab = cv2.cvtColor(warped_board, cv2.COLOR_BGR2LAB)
        self.reference_cell_colors = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.reference_cell_colors[row, col] = self.get_cell_average_color(warped_lab, row, col)
        
        print("Reference cell colors captured.")

    def detect_cell_changes(self, warped_board, delta_ms):
        """Detect color changes in each cell and update lock state."""
        warped_lab = cv2.cvtColor(warped_board, cv2.COLOR_BGR2LAB)
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                current_color = self.get_cell_average_color(warped_lab, row, col)
                ref_color = self.reference_cell_colors[row, col]
                
                # Calculate color difference (Euclidean distance in LAB space)
                color_diff = np.linalg.norm(current_color - ref_color)
                
                is_changed = color_diff > self.COLOR_CHANGE_THRESHOLD
                
                if self.locked_cells[row, col]:
                    # Cell is locked - check for unlock condition
                    if is_changed:
                        # Still different from reference, reset no-change counter
                        self.cell_no_change_duration_ms[row, col] = 0
                    else:
                        # Returned to reference color
                        self.cell_no_change_duration_ms[row, col] += delta_ms
                        if self.cell_no_change_duration_ms[row, col] >= self.UNLOCK_TIME_MS:
                            # Unlock the cell
                            self.locked_cells[row, col] = False
                            self.cell_no_change_duration_ms[row, col] = 0
                            self.cell_change_duration_ms[row, col] = 0
                else:
                    # Cell is not locked - check for lock condition
                    if is_changed:
                        self.cell_change_duration_ms[row, col] += delta_ms
                        self.cell_no_change_duration_ms[row, col] = 0
                        
                        if self.cell_change_duration_ms[row, col] >= self.LOCK_IN_TIME_MS:
                            # Lock the cell
                            self.locked_cells[row, col] = True
                            self.cell_change_duration_ms[row, col] = 0
                    else:
                        # No change detected, reset change counter
                        self.cell_change_duration_ms[row, col] = 0

    def draw_cell_overlays(self, warped_board):
        """Draw visual overlays for detecting and locked cells."""
        overlay = warped_board.copy()
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x1 = col * self.cell_size[0]
                y1 = row * self.cell_size[1]
                x2 = x1 + self.cell_size[0]
                y2 = y1 + self.cell_size[1]
                
                if self.locked_cells[row, col]:
                    # Draw semi-transparent green overlay for locked cells
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                elif self.cell_change_duration_ms[row, col] > 0:
                    # Draw yellow overlay with intensity based on progress
                    progress = min(self.cell_change_duration_ms[row, col] / self.LOCK_IN_TIME_MS, 1.0)
                    # Yellow color with varying intensity
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
                    
                    # Draw ms text
                    ms_text = f"{int(self.cell_change_duration_ms[row, col])}"
                    font_scale = 0.3
                    text_size = cv2.getTextSize(ms_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                    text_x = x1 + (self.cell_size[0] - text_size[0]) // 2
                    text_y = y1 + (self.cell_size[1] + text_size[1]) // 2
                    cv2.putText(overlay, ms_text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
        
        # Blend overlay with original (0.3 = 30% overlay, 70% original)
        alpha = 0.3
        result = cv2.addWeighted(overlay, alpha, warped_board, 1 - alpha, 0)
        
        # Draw checkmarks on locked cells (on top of blend)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.locked_cells[row, col]:
                    x1 = col * self.cell_size[0]
                    y1 = row * self.cell_size[1]
                    cx = x1 + self.cell_size[0] // 2
                    cy = y1 + self.cell_size[1] // 2
        
        return result

    def process_video(self):
        # Smoothing factor (Lower = smoother but more lag, Higher = more responsive)
        ALPHA = 0.4 
        # Max pixels the board corners are allowed to move in 1 frame before we call it a "glitch"
        MAX_SHIFT_THRESHOLD = 50.0 

        # New: Define the destination corners for the 2D board
        dst_corners = np.array([
            [0, 0],
            [self.board_size[0] - 1, 0],
            [self.board_size[0] - 1, self.board_size[1] - 1],
            [0, self.board_size[1] - 1]
        ], dtype=np.float32)

        # Initialize timing
        self.last_frame_time = time.time()
        reference_captured = False

        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            # Calculate delta time
            current_time = time.time()
            delta_ms = (current_time - self.last_frame_time) * 1000.0
            self.last_frame_time = current_time
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Feature Matching (ORB)
            # We use this every frame for simplicity. 
            # (Optimization: Use Optical Flow for speed, ORB for correction. 
            # But for stability with Scrabble boards, straight ORB is often less drift-prone)
            kp2, des2 = self.orb.detectAndCompute(frame_gray, None)
            
            found_board = False
            num_inliers = 0
            shift_metric = 0.0

            if des2 is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(self.ref_descriptors, des2)
                matches = sorted(matches, key=lambda x: x.distance)

                # Need enough matches to form a homography
                if len(matches) > 15:
                    src_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    # RANSAC to filter outliers
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if M is not None:
                        num_inliers = np.sum(mask) # Confidence metric
                        
                        # Project new raw corners
                        new_corners = cv2.perspectiveTransform(self.ref_corners, M)
                        
                        # Calculate shift from previous frame (L2 distance)
                        shift_metric = np.linalg.norm(new_corners - self.current_corners) / 4
                        
                        # FILTERING:
                        # Only update if the shift is realistic (prevents teleporting/glitches)
                        if shift_metric < MAX_SHIFT_THRESHOLD:
                            # Apply Smoothing (Exponential Moving Average)
                            self.current_corners = (1 - ALPHA) * self.current_corners + ALPHA * new_corners
                            found_board = True
                        else:
                            # If shift is too huge, we treat it as a glitch and keep old corners
                            pass 

            # --- VISUALIZATION ---
            # Draw the board
            color = (0, 255, 0) if found_board else (0, 0, 255) # Green if tracking, Red if lost/coasting
            cv2.polylines(frame, [np.int32(self.current_corners)], True, color, 3)

            # Draw Corners and Orientation
            tl = self.current_corners[0][0]
            cv2.putText(frame, "TL", (int(tl[0]), int(tl[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw Stats Overlay
            cv2.rectangle(frame, (10, 10), (250, 100), (0,0,0), -1)
            cv2.putText(frame, f"Inliers: {num_inliers}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Shift: {shift_metric:.2f} px", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if shift_metric > MAX_SHIFT_THRESHOLD:
                cv2.putText(frame, "GLITCH IGNORED", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow('Scrabble Tracker', frame)
            
            # New: Create and display the 2D board
            if self.current_corners is not None:
                # Get the perspective transform matrix
                transform_matrix = cv2.getPerspectiveTransform(self.current_corners, dst_corners)
                
                # Warp the original frame to get the top-down 2D board
                warped_board = cv2.warpPerspective(frame, transform_matrix, self.board_size)
                
                # Capture reference colors on first valid frame
                if not reference_captured:
                    self.capture_reference_colors(warped_board)
                    reference_captured = True
                
                # Detect cell changes and update lock states
                if self.reference_cell_colors is not None:
                    self.detect_cell_changes(warped_board, delta_ms)
                    
                    # Draw cell overlays (yellow for detecting, green for locked)
                    warped_board = self.draw_cell_overlays(warped_board)
                
                # Draw the 15x15 grid (on top of overlays)
                for i in range(1, self.grid_size):
                    # Vertical lines
                    cv2.line(warped_board, (i * self.cell_size[0], 0), (i * self.cell_size[0], self.board_size[1]), (255, 255, 255), 1)
                    # Horizontal lines
                    cv2.line(warped_board, (0, i * self.cell_size[1]), (self.board_size[0], i * self.cell_size[1]), (255, 255, 255), 1)
                
                # Display the 2D board
                cv2.imshow('2D Board', warped_board)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Track Scrabble Board')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--corners', type=str, help='Comma separated coordinates: x1,y1,x2,y2,x3,y3,x4,y4', default=None)
    
    args = parser.parse_args()
    
    manual_corners = None
    if args.corners:
        try:
            c = list(map(int, args.corners.split(',')))
            manual_corners = [(c[i], c[i+1]) for i in range(0, len(c), 2)]
        except:
            print("Error parsing corners. Please use format: x1,y1,x2,y2,x3,y3,x4,y4")
            sys.exit(1)

    tracker = ScrabbleTracker(args.video_path, manual_corners)
    if tracker.initialize():
        tracker.process_video()
