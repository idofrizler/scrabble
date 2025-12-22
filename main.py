import cv2
import numpy as np
import argparse
import sys

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

    def process_video(self):
        # Smoothing factor (Lower = smoother but more lag, Higher = more responsive)
        ALPHA = 0.4 
        # Max pixels the board corners are allowed to move in 1 frame before we call it a "glitch"
        MAX_SHIFT_THRESHOLD = 50.0 

        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
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