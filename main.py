import cv2
import numpy as np
import argparse
import sys
import time
import threading
import queue
import pytesseract


class OCRService:
    """Async OCR service that processes tile recognition with multiple worker threads."""
    
    NUM_WORKERS = 4  # Number of parallel OCR workers
    
    def __init__(self):
        self.queue = queue.Queue()
        self.pending = {}  # (row, col) -> request_id
        self.pending_lock = threading.Lock()  # Protect pending dict access
        self.request_counter = 0
        self.results = queue.Queue()  # Thread-safe results queue
        
        # Start worker pool
        self.workers = []
        for i in range(self.NUM_WORKERS):
            worker = threading.Thread(target=self._process_loop, daemon=True, name=f"OCR-Worker-{i}")
            worker.start()
            self.workers.append(worker)
    
    def submit(self, tile_face, row, col):
        """Submit a tile for OCR processing."""
        with self.pending_lock:
            self.request_counter += 1
            req_id = self.request_counter
            self.pending[(row, col)] = req_id
        self.queue.put((tile_face.copy(), row, col, req_id))
    
    def cancel(self, row, col):
        """Cancel pending OCR request for a cell (called when cell unlocks)."""
        with self.pending_lock:
            self.pending.pop((row, col), None)
    
    def is_pending(self, row, col):
        """Check if there's a pending OCR request for this cell."""
        with self.pending_lock:
            return (row, col) in self.pending
    
    def get_results(self):
        """Get all available OCR results (non-blocking)."""
        results = []
        while True:
            try:
                result = self.results.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results
    
    def _process_loop(self):
        """Background thread that processes OCR requests."""
        while True:
            try:
                tile_face, row, col, req_id = self.queue.get()
                
                # Check if request is still valid (not cancelled)
                with self.pending_lock:
                    if self.pending.get((row, col)) != req_id:
                        continue  # Stale request, skip
                
                # Perform OCR (outside lock - this is the slow part)
                letter, confidence, tile_clean = self._recognize_tile(tile_face)
                
                # Check again after OCR (cell might have unlocked during processing)
                with self.pending_lock:
                    if self.pending.get((row, col)) != req_id:
                        continue  # Request cancelled during OCR
                    
                    # Remove from pending
                    self.pending.pop((row, col), None)
                
                # Add to results (queue is already thread-safe)
                self.results.put((row, col, letter, confidence, tile_clean))
                    
            except Exception as e:
                print(f"OCR Worker Error: {e}")
    
    def get_smart_tile_letter(self, tile_face):
            """
            Advanced preprocessing to isolate the letter from tile borders/shadows.
            Strategy: Find the 'best' contour based on size + centrality, preserving holes.
            """
            # 1. Grayscale
            if len(tile_face.shape) == 3:
                gray = cv2.cvtColor(tile_face, cv2.COLOR_BGR2GRAY)
            else:
                gray = tile_face

            h, w = gray.shape

            # 2. Threshold (Inverted)
            # Letter becomes WHITE, Wood becomes BLACK
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # 3. Morphological Opening (Disconnect letter from border)
            # If the letter 'R' touches the black border slightly, this cuts that link.
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

            # 4. Find Contours WITH HIERARCHY to preserve holes
            # RETR_CCOMP: retrieves all contours and organizes them into a 2-level hierarchy
            # (external contours and their holes)
            cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            if not cnts or hierarchy is None: 
                return gray # Fail safe

            hierarchy = hierarchy[0]  # Get actual hierarchy array
            
            best_cnt_idx = None
            max_score = 0
            
            # Center of the image
            center_x, center_y = w // 2, h // 2

            for i, c in enumerate(cnts):
                # Only consider top-level contours (parent == -1)
                # hierarchy[i] = [next, prev, first_child, parent]
                if hierarchy[i][3] != -1:
                    continue  # This is a hole, not an outer contour
                
                # --- FILTER 1: BORDER TOUCHING ---
                # If the contour touches the edge of the image (within 2px), it's likely a border artifact.
                x, y, cw, ch = cv2.boundingRect(c)
                if x <= 2 or y <= 2 or (x + cw) >= w - 2 or (y + ch) >= h - 2:
                    continue

                # --- FILTER 2: SIZE ---
                area = cv2.contourArea(c)
                # Must be at least 5% of the tile area (to be visible) 
                # and less than 80% (to not be the whole box)
                if area < (h * w * 0.05) or area > (h * w * 0.8):
                    continue

                # --- FILTER 3: ASPECT RATIO ---
                # Letters are generally somewhat square-ish (0.2 to 3.0 ratio).
                # Long thin lines (shadows) are rejected.
                aspect_ratio = float(cw) / ch
                if aspect_ratio < 0.2 or aspect_ratio > 4.0:
                    continue

                # --- SCORING: CENTRALITY ---
                # Calculate distance from blob center to image center
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    dist_from_center = np.sqrt((cX - center_x)**2 + (cY - center_y)**2)
                    
                    # Score = Area / Distance (Big blobs near center win)
                    # We add 1 to distance to avoid division by zero
                    score = area / (dist_from_center + 1.0)
                    
                    if score > max_score:
                        max_score = score
                        best_cnt_idx = i

            # 5. Draw the Winner WITH HOLES PRESERVED
            if best_cnt_idx is not None:
                # Create a clean white canvas
                clean_img = np.full_like(gray, 255)
                
                # Draw the outer contour filled in BLACK
                cv2.drawContours(clean_img, cnts, best_cnt_idx, 0, -1)
                
                # Now find and draw holes (children of this contour) in WHITE
                # to "cut out" the holes
                child_idx = hierarchy[best_cnt_idx][2]  # first_child
                while child_idx != -1:
                    # Draw hole in WHITE to cut it out
                    cv2.drawContours(clean_img, cnts, child_idx, 255, -1)
                    child_idx = hierarchy[child_idx][0]  # next sibling
                
                return clean_img
            
            # If no valid letter found, return a blank white image (safest fallback)
            return np.full_like(gray, 255)

    def _recognize_tile(self, tile_face):
        """Detects the letter on a Scrabble tile. Returns (letter, confidence, processed_image)."""
        # Use the smart cleaning method that removes score number via contour filtering
        clean_img = self.get_smart_tile_letter(tile_face)

        # OCR Config
        # --psm 10: Treat the image as a single character
        # whitelist: Only allow A-Z (ignores numbers and symbols)
        custom_config = r'--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        try:
            # Use image_to_data to get confidence scores
            data = pytesseract.image_to_data(clean_img, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Find the best detection with confidence
            best_letter = None
            best_conf = 0
            
            for i, text in enumerate(data['text']):
                clean_text = text.strip().upper()
                conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
                
                if len(clean_text) == 1 and clean_text.isalpha() and conf > best_conf:
                    best_letter = clean_text
                    best_conf = conf
            
            if best_letter:
                # Convert clean_img to 3-channel for consistent caching
                clean_img_color = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2BGR)
                return (best_letter, best_conf, clean_img_color)
                
        except Exception as e:
            print(f"OCR Error: {e}")
        
        # Convert to color for consistent caching
        clean_img_color = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2BGR)
        return ("?", 0, clean_img_color)

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

        # 3D Pose State
        self.rvec = None
        self.tvec = None
        
        # Camera Matrix (Approximate, assuming generic webcam)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        focal_length = self.frame_width 
        center = (self.frame_width/2, self.frame_height/2)
        
        self.camera_matrix = np.array(
             [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype = "double"
        )
        self.dist_coeffs = np.zeros((4,1)) 

        # Tile Geometry
        # Scrabble tiles are ~19mm wide, ~4mm high. Ratio ~0.2
        # Negative because we extrude "up" (Z- direction relative to board plane)
        self.TILE_HEIGHT_RATIO = 0.25

        # Board state stores tuples: (letter, confidence) or None for empty cells
        self.board_state = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Cache of tile images sent to OCR: (row, col) -> tile_image
        self.tile_image_cache = {}
        
        # OCR retry tracking: (row, col) -> last_attempt_time
        self.last_ocr_attempt_time = {}
        self.OCR_RETRY_INTERVAL_MS = 1000.0  # Retry every 1 second
        self.OCR_MIN_CONFIDENCE = 50  # Retry if confidence below this
        
        # Async OCR service
        self.ocr_service = OCRService()

    def draw_digital_board(self):
        """Creates a clean digital visualization of the board state with confidence coloring."""
        # Create a blank beige image (resembling a board)
        board_img = np.zeros((450, 450, 3), dtype=np.uint8)
        board_img[:] = (220, 245, 245) # Beige background (BGR)
        
        step_x = 450 // self.grid_size
        step_y = 450 // self.grid_size

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Draw grid lines
                x1 = col * step_x
                y1 = row * step_y
                cv2.rectangle(board_img, (x1, y1), (x1 + step_x, y1 + step_y), (150, 150, 150), 1)
                
                # Draw the letter if it exists
                cell_data = self.board_state[row][col]
                if cell_data is not None:
                    letter, confidence = cell_data
                    
                    # Calculate tile background color based on confidence (0-100)
                    # High confidence (90+): Green-ish
                    # Medium confidence (60-90): Yellow-ish  
                    # Low confidence (<60): Red-ish
                    if confidence >= 90:
                        # Green tint: BGR (180, 255, 200) - light green
                        tile_color = (180, 255, 200)
                    elif confidence >= 60:
                        # Yellow/Orange tint: BGR (150, 220, 255) - light orange
                        # Interpolate between green and orange
                        t = (confidence - 60) / 30.0  # 0 to 1
                        tile_color = (
                            int(150 + t * 30),   # B: 150 -> 180
                            int(220 + t * 35),   # G: 220 -> 255
                            int(255 - t * 55)    # R: 255 -> 200
                        )
                    else:
                        # Red tint: BGR (150, 150, 255) - light red
                        # Interpolate between orange and red
                        t = confidence / 60.0  # 0 to 1
                        tile_color = (
                            int(150),            # B: 150
                            int(150 + t * 70),   # G: 150 -> 220
                            int(255)             # R: 255
                        )
                    
                    # Draw a tile background with confidence color
                    cv2.rectangle(board_img, (x1+2, y1+2), (x1 + step_x-2, y1 + step_y-2), tile_color, -1)
                    
                    # Center the text
                    text_size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    text_x = x1 + (step_x - text_size[0]) // 2
                    text_y = y1 + (step_y + text_size[1]) // 2
                    
                    cv2.putText(board_img, letter, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    # Draw small confidence value in corner
                    conf_text = f"{confidence}"
                    cv2.putText(board_img, conf_text, (x1 + 3, y1 + step_y - 3), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.25, (80, 80, 80), 1)
        
        return board_img

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                # Visual feedback
                cv2.circle(self.display_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('Mark 4 Corners', self.display_frame)

    def on_board_click(self, event, x, y, flags, params):
        """Handle clicks on the Digital Game State window to show cached tile images."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Calculate which cell was clicked
            step_x = 450 // self.grid_size
            step_y = 450 // self.grid_size
            col = x // step_x
            row = y // step_y
            
            # Bounds check
            if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                cell_data = self.board_state[row][col]
                if cell_data is not None and (row, col) in self.tile_image_cache:
                    letter, confidence = cell_data
                    tile_img = self.tile_image_cache[(row, col)]
                    
                    # Scale up the tile image for better visibility
                    display_size = (300, 300)
                    tile_display = cv2.resize(tile_img, display_size, interpolation=cv2.INTER_NEAREST)
                    
                    # Add text overlay with detection info
                    cv2.putText(tile_display, f"Cell: ({row},{col})", (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(tile_display, f"Letter: {letter}", (10, 55), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(tile_display, f"Confidence: {confidence}%", (10, 85), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow('Tile Debug View', tile_display)
                else:
                    print(f"No cached image for cell ({row},{col})")

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

    def recognize_tile(self, tile_face):
        """
        Detects the letter on a Scrabble tile.
        """
        # 1. Crop center to remove the small score number (bottom right)
        # We remove 15% from edges to isolate the big letter
        h, w = tile_face.shape[:2]
        margin_h = int(h * 0.15)
        margin_w = int(w * 0.15)
        
        # Focus on the center-ish part
        roi = tile_face[margin_h:h-margin_h, margin_w:w-margin_w]
        
        # 2. Pre-processing (Make it look like a scanned document)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Otsu's thresholding (Automatic black/white contrast)
        # THRESH_BINARY_INV because Tesseract prefers black text on white background
        # (Scrabble is usually black text on light wood, so usually standard Binary is fine, 
        # but sometimes inverting helps if lighting is dark. Try both.)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Optional: Denoise if video is grainy
        thresh = cv2.medianBlur(thresh, 3)

        # 3. OCR Config
        # --psm 10: Treat the image as a single character
        # whitelist: Only allow A-Z (ignores numbers and symbols)
        custom_config = r'--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        try:
            letter = pytesseract.image_to_string(thresh, config=custom_config)
            clean_letter = letter.strip().upper()
            
            # Filter: ensure it's exactly 1 letter
            if len(clean_letter) == 1 and clean_letter.isalpha():
                return clean_letter, thresh # Return thresh for debug viewing
                
        except Exception as e:
            print(f"OCR Error: {e}")
            
        return "?", thresh

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
            
            # Keep a clean copy of the frame for tile extraction (before any drawing)
            frame_clean = frame.copy()
            
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

                            self.update_3d_pose()
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
                    warped_board_display = self.draw_cell_overlays(warped_board)
                
                    # Draw Grid Lines
                    for i in range(1, self.grid_size):
                        cv2.line(warped_board_display, (i*self.cell_size[0], 0), (i*self.cell_size[0], self.board_size[1]), (255,255,255), 1)
                        cv2.line(warped_board_display, (0, i*self.cell_size[1]), (self.board_size[0], i*self.cell_size[1]), (255,255,255), 1)
                    
                    cv2.imshow('2D Board State', warped_board_display)

                    # --- ASYNC OCR PROCESSING ---
                    current_time_ms = current_time * 1000.0
                    
                    # 1. Collect completed OCR results (now includes confidence and processed image)
                    for (r, c, letter, confidence, tile_clean) in self.ocr_service.get_results():
                        # Only apply if cell is still locked (wasn't cancelled)
                        if self.locked_cells[r, c]:
                            existing = self.board_state[r][c]
                            
                            # Update if: no existing result OR new result has higher confidence
                            if existing is None:
                                self.board_state[r][c] = (letter, confidence)
                                self.tile_image_cache[(r, c)] = tile_clean
                                print(f"Recognized {letter} (conf={confidence}) at ({r},{c})")
                            elif confidence > existing[1]:
                                # New result is better - update
                                old_letter, old_conf = existing
                                self.board_state[r][c] = (letter, confidence)
                                self.tile_image_cache[(r, c)] = tile_clean
                                print(f"Updated ({r},{c}): {old_letter}({old_conf}) -> {letter}({confidence})")
                    
                    # 2. Process cells - submit new OCR requests or retry low-confidence ones
                    for r in range(self.grid_size):
                        for c in range(self.grid_size):
                            
                            if self.locked_cells[r, c]:
                                # Cell is LOCKED
                                existing = self.board_state[r][c]
                                needs_ocr = False
                                
                                if existing is None:
                                    # No result yet - need OCR
                                    needs_ocr = True
                                elif existing[1] < self.OCR_MIN_CONFIDENCE:
                                    # Low confidence - retry after interval
                                    last_attempt = self.last_ocr_attempt_time.get((r, c), 0)
                                    if current_time_ms - last_attempt >= self.OCR_RETRY_INTERVAL_MS:
                                        needs_ocr = True
                                
                                # Submit OCR if needed and not already pending
                                if needs_ocr and not self.ocr_service.is_pending(r, c):
                                    tile_face = self.extract_tile_face(frame_clean, r, c, output_size=(100, 100), margin_scale=0.1)
                                    if tile_face is not None:
                                        self.last_ocr_attempt_time[(r, c)] = current_time_ms
                                        self.ocr_service.submit(tile_face, r, c)
                            else:
                                # Cell is UNLOCKED - cancel any pending OCR and clear state
                                self.ocr_service.cancel(r, c)
                                self.board_state[r][c] = None
                                self.tile_image_cache.pop((r, c), None)
                                self.last_ocr_attempt_time.pop((r, c), None)

                    # Render the separate digital window
                    digital_board = self.draw_digital_board()
                    cv2.imshow('Digital Game State', digital_board)
                    cv2.setMouseCallback('Digital Game State', self.on_board_click)

                # Draw the 15x15 grid (on top of overlays)
                for i in range(1, self.grid_size):
                    # Vertical lines
                    cv2.line(warped_board, (i * self.cell_size[0], 0), (i * self.cell_size[0], self.board_size[1]), (255, 255, 255), 1)
                    # Horizontal lines
                    cv2.line(warped_board, (0, i * self.cell_size[1]), (self.board_size[0], i * self.cell_size[1]), (255, 255, 255), 1)
                
                # Check locked cells and show the most recently locked one
                rows, cols = np.where(self.locked_cells)
                if len(rows) > 0:
                    r, c = rows[-1], cols[-1] # Get last locked tile                        

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def update_3d_pose(self):
        """Calculates camera pose based on the 4 tracked board corners"""
        if self.current_corners is None: return

        # Real world coordinates of the board corners (Z=0)
        # We use grid units: (0,0), (15,0), (15,15), (0,15)
        obj_pts = np.array([
            [0, 0, 0],
            [self.grid_size, 0, 0],
            [self.grid_size, self.grid_size, 0],
            [0, self.grid_size, 0]
        ], dtype=np.float32)

        img_pts = self.current_corners.reshape(-1, 2).astype(np.float32)

        # Solve PnP
        ret, self.rvec, self.tvec = cv2.solvePnP(obj_pts, img_pts, self.camera_matrix, self.dist_coeffs)

    def extract_tile_face(self, frame, row, col, output_size=(100, 100), margin_scale=0.2):
        """
        Extracts the tile face accounting for height (3D) to avoid perspective distortion.
        """
        if self.rvec is None or self.tvec is None:
            return None

        # Z is "up" out of the board. 
        h_val = -1.0 * self.TILE_HEIGHT_RATIO 
        
        m = margin_scale / 2.0
        
        # 3D points of the tile TOP face
        top_face_3d = np.array([
            [col - m,     row - m,     h_val], # TL
            [col + 1 + m, row - m,     h_val], # TR
            [col + 1 + m, row + 1 + m, h_val], # BR
            [col - m,     row + 1 + m, h_val]  # BL
        ], dtype=np.float32)

        # Project 3D points -> 2D Image pixels
        image_points, _ = cv2.projectPoints(top_face_3d, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        image_points = image_points.reshape(-1, 2).astype(np.float32)

        # Warp extracted quad to flat square
        dst_pts = np.array([
            [0, 0],
            [output_size[0]-1, 0],
            [output_size[0]-1, output_size[1]-1],
            [0, output_size[1]-1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(image_points, dst_pts)
        tile_face = cv2.warpPerspective(frame, M, output_size)
        
        return tile_face

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
