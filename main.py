import cv2
import numpy as np
import argparse
import sys
import time
import threading
import queue
import re
import os
import uuid
import pytesseract
from rapidfuzz import fuzz, process


class OCRService:
    """Async OCR service that processes tile recognition with multiple worker threads."""
    
    NUM_WORKERS = 2  # Number of parallel OCR workers
    
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
    # Standard Scrabble letter point values
    LETTER_VALUES = {
        'A': 1, 'E': 1, 'I': 1, 'O': 1, 'U': 1, 'L': 1, 'N': 1, 'S': 1, 'T': 1, 'R': 1,
        'D': 2, 'G': 2,
        'B': 3, 'C': 3, 'M': 3, 'P': 3,
        'F': 4, 'H': 4, 'V': 4, 'W': 4, 'Y': 4,
        'K': 5,
        'J': 8, 'X': 8,
        'Q': 10, 'Z': 10
    }
    
    # Special cells on standard 15x15 Scrabble board
    # DL = Double Letter, TL = Triple Letter, DW = Double Word, TW = Triple Word
    SPECIAL_CELLS = {
        # Triple Word (red corners and edge midpoints)
        'TW': {(0, 0), (0, 7), (0, 14), (7, 0), (7, 14), (14, 0), (14, 7), (14, 14)},
        # Double Word (pink diagonals and center)
        'DW': {(1, 1), (2, 2), (3, 3), (4, 4), (7, 7),
               (1, 13), (2, 12), (3, 11), (4, 10),
               (10, 4), (11, 3), (12, 2), (13, 1),
               (10, 10), (11, 11), (12, 12), (13, 13)},
        # Triple Letter (dark blue)
        'TL': {(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13),
               (9, 1), (9, 5), (9, 9), (9, 13), (13, 5), (13, 9)},
        # Double Letter (light blue)
        'DL': {(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14),
               (6, 2), (6, 6), (6, 8), (6, 12), (7, 3), (7, 11),
               (8, 2), (8, 6), (8, 8), (8, 12), (11, 0), (11, 7), (11, 14),
               (12, 6), (12, 8), (14, 3), (14, 11)}
    }
    
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
        self.LOCK_IN_TIME_MS = 12000.0  # 12 seconds to lock
        self.UNLOCK_TIME_MS = 5000.0  # 5 seconds to unlock
        
        # Perspective-based tile obstruction offset (pixels to ignore at bottom of each cell)
        self.tile_obstruction_offset = 0  # Will be calculated based on board perspective

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
        
        # Turn management state
        self.confirmed_board_state = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.previous_pending_tiles = set()  # For tracking stability
        self.pending_stable_since = None  # Timestamp when pending tiles became stable
        self.turn_number = 0
        self.awaiting_confirmation = False
        self.detected_words = []  # List of (word_cells, new_tile_mask) tuples
        self.turn_error_message = None  # Error message to display
        self.turn_error_time = None  # When error was set (for timeout)
        self.PENDING_STABLE_TIME_MS = 5000.0  # 5 seconds of stability before validating
        
        # Confirmation UI dimensions
        self.CONFIRM_PANEL_HEIGHT = 90  # Increased to fit word info + buttons
        self.CONFIRM_BUTTON_WIDTH = 100
        self.CANCEL_BUTTON_WIDTH = 100
        
        # Cache of tile images sent to OCR: (row, col) -> processed tile_image
        self.tile_image_cache = {}
        # Cache of original tile captures (before processing): (row, col) -> original tile_image
        self.tile_original_cache = {}
        
        # OCR retry tracking: (row, col) -> last_attempt_time
        self.last_ocr_attempt_time = {}
        self.OCR_RETRY_INTERVAL_MS = 1000.0  # Retry every 1 second
        self.OCR_MIN_CONFIDENCE = 50  # Retry if confidence below this
        
        # Async OCR service
        self.ocr_service = OCRService()
        
        # Dictionary for word validation
        self.dictionary = set()
        self.dictionary_by_length = {}  # length -> list of words (for pattern matching)
        self.load_dictionary()
        
        # Word validation state (for confirmation UI)
        self.word_validations = []  # List of (word_string, is_valid, suggestions)
        self.selected_corrections = {}  # word_index -> selected_correction
        self.MAX_SUGGESTIONS = 5  # Max suggestions to show
        
        # Manual word input state
        self.manual_input_active = False
        self.manual_input_word_idx = None
        self.manual_input_text = ""
        
        # Manual letter overrides: (row, col) -> letter
        # These take precedence over OCR results for display and dataset saving
        # Use '*' for blank tiles
        self.manual_overrides = {}
        
        # Tile debug view double-click tracking
        self.last_tile_click_time = 0
        self.last_tile_click_cell = None
        self.DOUBLE_CLICK_THRESHOLD_MS = 500.0
        
        # Override input state (for tile debug view)
        self.override_input_active = False
        self.override_input_cell = None
        self.override_input_text = ""
        
        # Dataset saving state
        self.dataset_dir = None  # Set via set_dataset_dir()
        self.dataset_last_save_time = {}  # (row, col) -> timestamp_ms of last save
        self.DATASET_SAVE_INTERVAL_MS = 5000.0  # Save every 5 seconds per tile
        self.DATASET_MIN_CONFIDENCE = 50  # Only save tiles with >=50% confidence

    def set_dataset_dir(self, dataset_dir):
        """Enable dataset saving to the specified directory."""
        self.dataset_dir = dataset_dir
        
        # Create directory structure
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create A-Z folders
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            os.makedirs(os.path.join(dataset_dir, letter), exist_ok=True)
        
        # Create BLANK folder for unknown tiles
        os.makedirs(os.path.join(dataset_dir, "BLANK"), exist_ok=True)
        
        print(f"Dataset saving enabled -> {dataset_dir}/")
        print(f"  Save interval: {self.DATASET_SAVE_INTERVAL_MS/1000.0}s per tile")
        print(f"  Min confidence: {self.DATASET_MIN_CONFIDENCE}%")
    
    def save_tile_to_dataset(self, row, col, letter, confidence, tile_face, current_time_ms):
        """
        Save a tile image to the dataset folder if conditions are met.
        Uses manual override letter if present.
        Returns True if saved, False otherwise.
        """
        if self.dataset_dir is None:
            return False
        
        # Check for manual override - use override letter instead of OCR result
        override_letter = self.manual_overrides.get((row, col))
        if override_letter:
            # Override takes precedence - use it with 100% confidence
            save_letter = override_letter
            save_confidence = 100
        else:
            # Use OCR result
            save_letter = letter
            save_confidence = confidence
            
            # Check confidence threshold (only for OCR results)
            if save_confidence < self.DATASET_MIN_CONFIDENCE:
                return False
        
        # Check time since last save for this cell
        last_save = self.dataset_last_save_time.get((row, col), 0)
        if current_time_ms - last_save < self.DATASET_SAVE_INTERVAL_MS:
            return False
        
        # Determine folder: '*' -> BLANK, letter -> letter folder, '?' -> BLANK
        if save_letter == '*':
            folder = "BLANK"
        elif save_letter == '?' or not save_letter.isalpha():
            folder = "BLANK"
        else:
            folder = save_letter.upper()
        
        # Generate unique filename and save
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(self.dataset_dir, folder, filename)
        
        cv2.imwrite(filepath, tile_face)
        
        # Update last save time
        self.dataset_last_save_time[(row, col)] = current_time_ms
        
        override_indicator = " [override]" if override_letter else ""
        print(f"  Dataset: Saved {save_letter} ({save_confidence}%) -> {folder}/{filename}{override_indicator}")
        return True

    def load_dictionary(self):
        """Load Scrabble dictionary from words.txt file."""
        dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'words.txt')
        
        if not os.path.exists(dict_path):
            print(f"Warning: Dictionary file not found at {dict_path}")
            return
        
        try:
            with open(dict_path, 'r') as f:
                for line in f:
                    word = line.strip().upper()
                    if word and word.isalpha():
                        self.dictionary.add(word)
                        length = len(word)
                        if length not in self.dictionary_by_length:
                            self.dictionary_by_length[length] = []
                        self.dictionary_by_length[length].append(word)
            
            print(f"Loaded {len(self.dictionary)} words from dictionary")
        except Exception as e:
            print(f"Error loading dictionary: {e}")
    
    def validate_word(self, word):
        """
        Validate a word against the dictionary.
        Returns (is_valid, suggestions) where suggestions is a list of possible corrections.
        """
        word_upper = word.upper()
        
        # Check for exact match
        if word_upper in self.dictionary:
            return True, []
        
        # If word contains '?', find pattern matches
        if '?' in word_upper:
            suggestions = self.find_pattern_matches(word_upper)
            return False, suggestions[:self.MAX_SUGGESTIONS]
        
        # Otherwise, find similar words using fuzzy matching
        suggestions = self.find_similar_words(word_upper)
        return False, suggestions[:self.MAX_SUGGESTIONS]
    
    def find_pattern_matches(self, pattern):
        """
        Find words that match a pattern with '?' as wildcard.
        Example: 'C?T' matches 'CAT', 'COT', 'CUT', etc.
        """
        length = len(pattern)
        if length not in self.dictionary_by_length:
            return []
        
        # Convert pattern to regex
        regex_pattern = '^' + pattern.replace('?', '.') + '$'
        regex = re.compile(regex_pattern)
        
        matches = []
        for word in self.dictionary_by_length[length]:
            if regex.match(word):
                matches.append(word)
        
        return matches
    
    def find_similar_words(self, word, max_distance=2):
        """
        Find words similar to the given word using fuzzy matching.
        Returns list of (word, score) sorted by similarity.
        """
        length = len(word)
        candidates = []
        
        # Search words of similar length
        for l in range(max(2, length - 1), length + 2):
            if l in self.dictionary_by_length:
                candidates.extend(self.dictionary_by_length[l])
        
        if not candidates:
            return []
        
        # Use rapidfuzz to find best matches
        results = process.extract(word, candidates, scorer=fuzz.ratio, limit=self.MAX_SUGGESTIONS)
        
        # Return just the words (not scores)
        return [match[0] for match in results if match[1] >= 60]  # 60% minimum similarity

    def validate_detected_words(self):
        """Validate all detected words and store results."""
        self.word_validations = []
        self.selected_corrections = {}
        
        for i, (word_cells, is_new_flags) in enumerate(self.detected_words):
            word_string = self.get_word_string(word_cells)
            is_valid, suggestions = self.validate_word(word_string)
            self.word_validations.append((word_string, is_valid, suggestions))
            
            # If invalid and has exactly one suggestion, auto-select it
            if not is_valid and len(suggestions) == 1:
                self.selected_corrections[i] = suggestions[0]

    def draw_digital_board(self):
        """Creates a clean digital visualization of the board state with confidence coloring."""
        # Calculate total height including confirmation panel if needed
        board_height = 450
        total_height = board_height + (self.CONFIRM_PANEL_HEIGHT if self.awaiting_confirmation else 0)
        
        # Create a blank beige image (resembling a board)
        board_img = np.zeros((total_height, 450, 3), dtype=np.uint8)
        board_img[:board_height] = (220, 245, 245)  # Beige background (BGR)
        
        step_x = 450 // self.grid_size
        step_y = board_height // self.grid_size

        # Build set of cells that are part of detected words for highlighting
        word_cells_new = set()  # New tiles in detected words
        word_cells_existing = set()  # Existing tiles in detected words
        if self.awaiting_confirmation and self.detected_words:
            for word_cells, is_new_flags in self.detected_words:
                for i, (r, c) in enumerate(word_cells):
                    if is_new_flags[i]:
                        word_cells_new.add((r, c))
                    else:
                        word_cells_existing.add((r, c))

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
                    
                    # Check for manual override - display override letter instead
                    override_letter = self.manual_overrides.get((row, col))
                    if override_letter:
                        # Use override letter, display * as blank indicator
                        display_letter = override_letter if override_letter != '*' else '_'
                        display_confidence = 100  # Override is always 100% confidence
                        has_override = True
                    else:
                        display_letter = letter
                        display_confidence = confidence
                        has_override = False
                    
                    # Calculate tile background color based on confidence (0-100)
                    # High confidence (90+): Green-ish
                    # Medium confidence (60-90): Yellow-ish  
                    # Low confidence (<60): Red-ish
                    # Use cyan tint for overridden tiles, otherwise confidence-based color
                    if has_override:
                        # Cyan tint for override: BGR (255, 255, 180) - light cyan
                        tile_color = (255, 255, 180)
                    elif display_confidence >= 90:
                        # Green tint: BGR (180, 255, 200) - light green
                        tile_color = (180, 255, 200)
                    elif display_confidence >= 60:
                        # Yellow/Orange tint: BGR (150, 220, 255) - light orange
                        # Interpolate between green and orange
                        t = (display_confidence - 60) / 30.0  # 0 to 1
                        tile_color = (
                            int(150 + t * 30),   # B: 150 -> 180
                            int(220 + t * 35),   # G: 220 -> 255
                            int(255 - t * 55)    # R: 255 -> 200
                        )
                    else:
                        # Red tint: BGR (150, 150, 255) - light red
                        # Interpolate between orange and red
                        t = display_confidence / 60.0  # 0 to 1
                        tile_color = (
                            int(150),            # B: 150
                            int(150 + t * 70),   # G: 150 -> 220
                            int(255)             # R: 255
                        )
                    
                    # Draw a tile background with confidence color
                    cv2.rectangle(board_img, (x1+2, y1+2), (x1 + step_x-2, y1 + step_y-2), tile_color, -1)
                    
                    # Center the text
                    text_size = cv2.getTextSize(display_letter, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    text_x = x1 + (step_x - text_size[0]) // 2
                    text_y = y1 + (step_y + text_size[1]) // 2
                    
                    # Use different text color for overrides
                    text_color = (128, 0, 0) if has_override else (0, 0, 0)  # Dark blue for override
                    cv2.putText(board_img, display_letter, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                    
                    # Draw small confidence value or 'OVR' indicator in corner
                    if has_override:
                        conf_text = "OVR"
                        conf_color = (128, 0, 0)  # Dark blue
                    else:
                        conf_text = f"{display_confidence}"
                        conf_color = (80, 80, 80)
                    cv2.putText(board_img, conf_text, (x1 + 3, y1 + step_y - 3), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.25, conf_color, 1)
                    
                    # Draw word highlight borders if awaiting confirmation
                    if (row, col) in word_cells_new:
                        # New tile - blue thick border
                        cv2.rectangle(board_img, (x1+1, y1+1), (x1 + step_x-1, y1 + step_y-1), (255, 100, 0), 3)
                    elif (row, col) in word_cells_existing:
                        # Existing tile in word - orange border
                        cv2.rectangle(board_img, (x1+1, y1+1), (x1 + step_x-1, y1 + step_y-1), (0, 165, 255), 2)
        
        # Draw confirmation panel if awaiting confirmation
        if self.awaiting_confirmation:
            panel_y = board_height
            # Dark gray panel background
            board_img[panel_y:] = (50, 50, 50)
            
            # Layout: Word info on top (first 35px), buttons below (remaining space)
            word_area_height = 35
            button_area_y = panel_y + word_area_height
            button_height = 35
            
            # Calculate button positions (centered horizontally)
            button_gap = 20
            total_buttons_width = self.CONFIRM_BUTTON_WIDTH + self.CANCEL_BUTTON_WIDTH + button_gap
            start_x = (450 - total_buttons_width) // 2
            
            # Confirm button (green)
            confirm_x1 = start_x
            confirm_x2 = confirm_x1 + self.CONFIRM_BUTTON_WIDTH
            confirm_y1 = button_area_y + 5
            confirm_y2 = confirm_y1 + button_height
            cv2.rectangle(board_img, (confirm_x1, confirm_y1), (confirm_x2, confirm_y2), (0, 180, 0), -1)
            cv2.rectangle(board_img, (confirm_x1, confirm_y1), (confirm_x2, confirm_y2), (0, 220, 0), 2)
            
            # Confirm text
            confirm_text = "Confirm"
            text_size = cv2.getTextSize(confirm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = confirm_x1 + (self.CONFIRM_BUTTON_WIDTH - text_size[0]) // 2
            text_y = confirm_y1 + (button_height + text_size[1]) // 2
            cv2.putText(board_img, confirm_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Cancel button (red)
            cancel_x1 = confirm_x2 + button_gap
            cancel_x2 = cancel_x1 + self.CANCEL_BUTTON_WIDTH
            cancel_y1 = confirm_y1
            cancel_y2 = confirm_y2
            cv2.rectangle(board_img, (cancel_x1, cancel_y1), (cancel_x2, cancel_y2), (0, 0, 180), -1)
            cv2.rectangle(board_img, (cancel_x1, cancel_y1), (cancel_x2, cancel_y2), (0, 0, 220), 2)
            
            # Cancel text
            cancel_text = "Cancel"
            text_size = cv2.getTextSize(cancel_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = cancel_x1 + (self.CANCEL_BUTTON_WIDTH - text_size[0]) // 2
            text_y = cancel_y1 + (button_height + text_size[1]) // 2
            cv2.putText(board_img, cancel_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Store button coordinates for click detection
            self.confirm_button_rect = (confirm_x1, confirm_y1, confirm_x2, confirm_y2)
            self.cancel_button_rect = (cancel_x1, cancel_y1, cancel_x2, cancel_y2)
            
            # Display word validation info or manual input state (in top area)
            if self.manual_input_active and len(self.detected_words) > 0:
                # Show manual input mode for main word
                word_cells, _ = self.detected_words[0]
                expected_len = len(word_cells)
                detected_word = self.get_word_string(word_cells)
                input_text = f"'{detected_word}' -> {self.manual_input_text}_"
                cv2.putText(board_img, input_text, (10, panel_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                hint_text = f"Type word ({expected_len} letters) ENTER=confirm ESC=cancel"
                cv2.putText(board_img, hint_text, (10, panel_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
            elif self.word_validations:
                # Show main word validation status
                word, is_valid, suggestions = self.word_validations[0] if self.word_validations else ("", True, [])
                if 0 in self.selected_corrections:
                    word_display = f"{word} -> {self.selected_corrections[0]}"
                elif is_valid:
                    word_display = f"{word} [OK]"
                elif suggestions:
                    word_display = f"{word} ? (maybe: {suggestions[0]})"
                else:
                    word_display = f"{word} [?]"
                
                cv2.putText(board_img, word_display, (10, panel_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                hint_text = "Type to override word"
                cv2.putText(board_img, hint_text, (10, panel_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        # Draw error message if present
        if self.turn_error_message and self.turn_error_time:
            current_time = time.time()
            # Show error for 3 seconds
            if current_time - self.turn_error_time < 3.0:
                # Red background bar at top
                cv2.rectangle(board_img, (0, 0), (450, 30), (0, 0, 150), -1)
                cv2.putText(board_img, self.turn_error_message, (10, 22), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                self.turn_error_message = None
                self.turn_error_time = None
        
        # Draw scoreboard and turn info (top right area)
        if hasattr(self, 'player_scores') and hasattr(self, 'num_players'):
            # Background for scoreboard
            cv2.rectangle(board_img, (300, 0), (450, 25 + self.num_players * 18), (240, 240, 240), -1)
            cv2.rectangle(board_img, (300, 0), (450, 25 + self.num_players * 18), (150, 150, 150), 1)
            
            # Turn number
            turn_text = f"Turn {self.turn_number}"
            cv2.putText(board_img, turn_text, (310, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Player scores
            for i in range(self.num_players):
                y_pos = 32 + i * 18
                
                # Highlight current player
                if i == self.current_player:
                    cv2.rectangle(board_img, (302, y_pos - 12), (448, y_pos + 5), (180, 255, 180), -1)
                    prefix = ">"
                else:
                    prefix = " "
                
                # Mark our player
                if hasattr(self, 'our_player_index') and i == self.our_player_index:
                    player_label = f"{prefix}P{i+1} (You): {self.player_scores[i]}"
                else:
                    player_label = f"{prefix}P{i+1}: {self.player_scores[i]}"
                
                cv2.putText(board_img, player_label, (310, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        else:
            # Fallback: just show turn number
            turn_text = f"Turn: {self.turn_number}"
            cv2.putText(board_img, turn_text, (380, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return board_img

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                # Visual feedback
                cv2.circle(self.display_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('Mark 4 Corners', self.display_frame)

    def on_board_click(self, event, x, y, flags, params):
        """Handle clicks on the Digital Game State window to show cached tile images or handle buttons."""
        if event == cv2.EVENT_LBUTTONDOWN:
            current_time_ms = time.time() * 1000.0
            
            # Check if we're clicking on confirmation buttons
            if self.awaiting_confirmation:
                # Check Confirm button
                if hasattr(self, 'confirm_button_rect'):
                    x1, y1, x2, y2 = self.confirm_button_rect
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.confirm_turn()
                        return
                
                # Check Cancel button
                if hasattr(self, 'cancel_button_rect'):
                    x1, y1, x2, y2 = self.cancel_button_rect
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.cancel_turn()
                        return
            
            # Otherwise, handle cell clicks for tile debug view
            step_x = 450 // self.grid_size
            step_y = 450 // self.grid_size
            col = x // step_x
            row = y // step_y
            
            # Bounds check
            if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                cell_data = self.board_state[row][col]
                if cell_data is not None and (row, col) in self.tile_image_cache:
                    # Check for double-click to start override input
                    if (self.last_tile_click_cell == (row, col) and 
                        current_time_ms - self.last_tile_click_time < self.DOUBLE_CLICK_THRESHOLD_MS):
                        # Double-click detected - start override input mode
                        self.override_input_active = True
                        self.override_input_cell = (row, col)
                        self.override_input_text = ""
                        print(f"Override mode for ({row},{col}): Type A-Z for letter, * for blank, ESC to cancel, ENTER to confirm")
                    
                    # Update click tracking
                    self.last_tile_click_cell = (row, col)
                    self.last_tile_click_time = current_time_ms
                    
                    # Show tile debug view
                    self.show_tile_debug_view(row, col)
                else:
                    print(f"No cached image for cell ({row},{col})")
    
    def show_tile_debug_view(self, row, col):
        """Display the tile debug view window for a specific cell."""
        cell_data = self.board_state[row][col]
        if cell_data is None or (row, col) not in self.tile_image_cache:
            return
        
        letter, confidence = cell_data
        tile_img = self.tile_image_cache[(row, col)]
        original_img = self.tile_original_cache.get((row, col))
        
        # Scale up tile images for better visibility
        tile_size = (150, 150)
        processed_display = cv2.resize(tile_img, tile_size, interpolation=cv2.INTER_NEAREST)
        
        # Create combined display with original (if available) and processed
        if original_img is not None:
            original_display = cv2.resize(original_img, tile_size, interpolation=cv2.INTER_NEAREST)
            # Stack horizontally: original | processed
            combined_tiles = np.hstack([original_display, processed_display])
        else:
            # Only processed available
            combined_tiles = processed_display
        
        # Create display with info panel below tiles
        info_height = 140  # Increased for override info
        display_width = combined_tiles.shape[1]
        tile_display = np.zeros((tile_size[1] + info_height, display_width, 3), dtype=np.uint8)
        tile_display[:tile_size[1], :] = combined_tiles
        
        # Add labels above tiles
        if original_img is not None:
            cv2.putText(tile_display, "Original", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(tile_display, "Processed", (tile_size[0] + 10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(tile_display, "Processed", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add text overlay with detection info (in info panel)
        info_y_start = tile_size[1] + 20
        
        # Show override status
        override_letter = self.manual_overrides.get((row, col))
        if override_letter:
            display_letter = override_letter if override_letter != '*' else 'BLANK'
            cv2.putText(tile_display, f"Cell: ({row},{col})  OCR: {letter}  Override: {display_letter}", 
                       (10, info_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        else:
            cv2.putText(tile_display, f"Cell: ({row},{col})  Letter: {letter}  Conf: {confidence}%", 
                       (10, info_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show last attempt time
        last_attempt = self.last_ocr_attempt_time.get((row, col), 0)
        if last_attempt > 0:
            current_time_ms = time.time() * 1000.0
            seconds_ago = (current_time_ms - last_attempt) / 1000.0
            cv2.putText(tile_display, f"Last try: {seconds_ago:.1f}s ago", (10, info_y_start + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        else:
            cv2.putText(tile_display, f"Last try: N/A", (10, info_y_start + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        
        # Show if pending
        is_pending = self.ocr_service.is_pending(row, col)
        cv2.putText(tile_display, f"Pending: {is_pending}", (10, info_y_start + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255) if is_pending else (0, 255, 0), 1)
        
        # Show override input state if active
        if self.override_input_active and self.override_input_cell == (row, col):
            cv2.rectangle(tile_display, (5, info_y_start + 60), (display_width - 5, info_y_start + 95), (0, 100, 100), -1)
            input_text = f"Override: {self.override_input_text}_" if self.override_input_text else "Override: _"
            cv2.putText(tile_display, input_text, (10, info_y_start + 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(tile_display, "A-Z=letter, *=blank, ENTER=ok, ESC=cancel", (10, info_y_start + 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        else:
            cv2.putText(tile_display, "Double-click to override letter", (10, info_y_start + 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        cv2.imshow('Tile Debug View', tile_display)
    
    def confirm_turn(self):
        """Confirm the current turn and update the confirmed board state."""
        # Get pending tiles for scoring
        pending_tiles = self.get_pending_tiles()
        
        # If manual input is active and valid, apply it before confirming
        if self.manual_input_active and self.manual_input_text and len(self.detected_words) > 0:
            word_cells, _ = self.detected_words[0]
            expected_len = len(word_cells)
            
            if len(self.manual_input_text) == expected_len:
                correction = self.manual_input_text.upper()
                self.selected_corrections[0] = correction
                
                # Update board_state with the correction
                for i, (r, c) in enumerate(word_cells):
                    if i < len(correction):
                        new_letter = correction[i]
                        old_data = self.board_state[r][c]
                        if old_data is not None:
                            old_letter, _ = old_data
                            if old_letter != new_letter:
                                self.board_state[r][c] = (new_letter, 100)
                                print(f"  Applied pending input ({r},{c}): {old_letter} -> {new_letter}")
                
                print(f"Applied in-progress manual input: {correction}")
            else:
                print(f"Warning: Pending input '{self.manual_input_text}' has wrong length (expected {expected_len})")
        
        print(f"Turn {self.turn_number} confirmed by Player {self.current_player + 1}!")
        
        # Collect all cells that are part of detected words
        all_word_cells = set()
        for word_cells, is_new_flags in self.detected_words:
            for r, c in word_cells:
                all_word_cells.add((r, c))
        
        # Apply corrections from selected words
        for word_idx, correction in self.selected_corrections.items():
            if word_idx < len(self.detected_words):
                word_cells, is_new_flags = self.detected_words[word_idx]
                # Apply each letter from the correction
                for i, (r, c) in enumerate(word_cells):
                    if i < len(correction):
                        new_letter = correction[i]
                        old_data = self.board_state[r][c]
                        if old_data is not None:
                            old_letter, old_conf = old_data
                            if old_letter != new_letter:
                                # Update to corrected letter with 100% confidence
                                self.board_state[r][c] = (new_letter, 100)
                                print(f"  Corrected ({r},{c}): {old_letter} -> {new_letter}")
        
        # Set ALL letters in confirmed words to 100% confidence
        for r, c in all_word_cells:
            if self.board_state[r][c] is not None:
                letter, conf = self.board_state[r][c]
                if conf < 100:
                    self.board_state[r][c] = (letter, 100)
                    print(f"  Confirmed ({r},{c}): {letter} -> 100%")
        
        # Calculate and add score for this turn
        if hasattr(self, 'player_scores') and hasattr(self, 'current_player'):
            turn_score, word_scores = self.calculate_turn_score(pending_tiles)
            self.player_scores[self.current_player] += turn_score
            
            print(f"  Turn score: {turn_score}")
            for word, score in word_scores:
                print(f"    {word}: {score} pts")
            print(f"  Player {self.current_player + 1} total: {self.player_scores[self.current_player]}")
            
            # Advance to next player
            self.current_player = (self.current_player + 1) % self.num_players
            print(f"  Next: Player {self.current_player + 1}'s turn")
        
        # Copy current board state to confirmed state
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.board_state[row][col] is not None:
                    self.confirmed_board_state[row][col] = self.board_state[row][col]
        
        # Increment turn number
        self.turn_number += 1
        
        # Reset turn tracking state
        self.awaiting_confirmation = False
        self.detected_words = []
        self.word_validations = []
        self.selected_corrections = {}
        self.manual_input_active = False
        self.manual_input_text = ""
        self.previous_pending_tiles = set()
        self.pending_stable_since = None
        
        print(f"Ready for turn {self.turn_number}")
    
    def handle_override_input(self, key):
        """Handle keyboard input for tile override in debug view."""
        if not self.override_input_active:
            return False  # Not handled
        
        row, col = self.override_input_cell
        
        if key == 27:  # ESC - cancel override input
            self.override_input_active = False
            self.override_input_cell = None
            self.override_input_text = ""
            print(f"Override cancelled for ({row},{col})")
            # Refresh debug view
            self.show_tile_debug_view(row, col)
            return True
        
        elif key == 13 or key == 10:  # ENTER - confirm override
            if self.override_input_text:
                override_char = self.override_input_text.upper()
                if override_char == '*' or (len(override_char) == 1 and override_char.isalpha()):
                    # Apply override
                    self.manual_overrides[(row, col)] = override_char
                    display_char = 'BLANK' if override_char == '*' else override_char
                    print(f"Override applied: ({row},{col}) -> {display_char}")
                    
                    # Reset dataset timer for this cell so it saves with new label
                    self.dataset_last_save_time.pop((row, col), None)
                else:
                    print(f"Invalid override: must be A-Z or * for blank")
            else:
                # Empty input - remove override if exists
                if (row, col) in self.manual_overrides:
                    del self.manual_overrides[(row, col)]
                    print(f"Override removed for ({row},{col})")
            
            self.override_input_active = False
            self.override_input_cell = None
            self.override_input_text = ""
            # Refresh debug view
            self.show_tile_debug_view(row, col)
            return True
        
        elif key == 8:  # BACKSPACE
            self.override_input_text = self.override_input_text[:-1]
            self.show_tile_debug_view(row, col)
            return True
        
        elif key == ord('*'):  # Asterisk for blank tile
            self.override_input_text = "*"
            self.show_tile_debug_view(row, col)
            return True
        
        elif 32 <= key <= 126:  # Printable ASCII
            char = chr(key).upper()
            if char.isalpha():
                # Only allow single character
                self.override_input_text = char
                self.show_tile_debug_view(row, col)
                return True
        
        return True  # Consumed the key even if not recognized
    
    def handle_key_input(self, key):
        """Handle keyboard input for manual word correction."""
        # First check if override input is active (takes priority)
        if self.override_input_active:
            self.handle_override_input(key)
            return
        
        if not self.awaiting_confirmation:
            return
        
        # If manual input is active, handle typing
        if self.manual_input_active:
            if key == 27:  # ESC - cancel manual input
                self.manual_input_active = False
                self.manual_input_text = ""
                print("Manual input cancelled")
            elif key == 13 or key == 10:  # ENTER - confirm manual input
                if self.manual_input_text and len(self.detected_words) > 0:
                    # Apply to the main word (first word, index 0)
                    word_cells, _ = self.detected_words[0]
                    expected_len = len(word_cells)
                    
                    if len(self.manual_input_text) == expected_len:
                        correction = self.manual_input_text.upper()
                        self.selected_corrections[0] = correction
                        
                        # Immediately update board_state to show corrected letters
                        for i, (r, c) in enumerate(word_cells):
                            if i < len(correction):
                                new_letter = correction[i]
                                old_data = self.board_state[r][c]
                                if old_data is not None:
                                    old_letter, _ = old_data
                                    if old_letter != new_letter:
                                        # Update the display immediately with 100% confidence
                                        self.board_state[r][c] = (new_letter, 100)
                                        print(f"  Updated ({r},{c}): {old_letter} -> {new_letter}")
                        
                        # Update word_validations to reflect the correction
                        if self.word_validations:
                            old_word, _, suggestions = self.word_validations[0]
                            self.word_validations[0] = (old_word, True, [])  # Mark as valid now
                        
                        print(f"Manual correction applied: {correction}")
                        self.manual_input_active = False
                        self.manual_input_text = ""
                    else:
                        print(f"Word length mismatch: expected {expected_len}, got {len(self.manual_input_text)}")
                else:
                    self.manual_input_active = False
                    self.manual_input_text = ""
            elif key == 8:  # BACKSPACE
                self.manual_input_text = self.manual_input_text[:-1]
            elif 32 <= key <= 126:  # Printable ASCII
                char = chr(key).upper()
                if char.isalpha():
                    self.manual_input_text += char
        else:
            # Start manual input mode when user types any letter
            if 65 <= key <= 90 or 97 <= key <= 122:  # A-Z or a-z
                if len(self.detected_words) > 0:
                    self.manual_input_active = True
                    self.manual_input_text = chr(key).upper()
                    word_cells, _ = self.detected_words[0]
                    expected_len = len(word_cells)
                    detected_word = self.get_word_string(word_cells)
                    print(f"Manual edit for main word: '{detected_word}' ({expected_len} letters)")
                    print(f"Type the correct word and press ENTER (or ESC to cancel)")
    
    def cancel_turn(self):
        """Cancel the current turn confirmation and continue tracking."""
        print("Turn cancelled, continuing to track...")
        self.awaiting_confirmation = False
        self.detected_words = []
        self.word_validations = []
        self.selected_corrections = {}
        self.manual_input_active = False
        self.manual_input_text = ""
        # Reset stability tracking so we can re-detect
        self.pending_stable_since = None
    
    def calculate_turn_score(self, pending_tiles):
        """
        Calculate the score for the current turn based on detected words.
        
        Scoring rules:
        - Each letter has a point value (from LETTER_VALUES)
        - Letter multipliers (DL, TL) only apply to newly placed tiles
        - Word multipliers (DW, TW) only apply if a new tile is on that cell
        - If all 7 tiles are used (bingo), add 50 bonus points
        
        Returns: (total_score, word_scores_list)
        where word_scores_list is [(word_string, word_score), ...]
        """
        total_score = 0
        word_scores = []
        
        for word_cells, is_new_flags in self.detected_words:
            word_score = 0
            word_multiplier = 1
            word_string = ""
            
            for i, (r, c) in enumerate(word_cells):
                cell_data = self.board_state[r][c]
                if cell_data is None:
                    continue
                
                letter = cell_data[0]
                word_string += letter
                
                # Get base letter value
                letter_value = self.LETTER_VALUES.get(letter, 0)
                
                # Apply letter multipliers only if this is a newly placed tile
                if is_new_flags[i]:
                    if (r, c) in self.SPECIAL_CELLS['DL']:
                        letter_value *= 2
                    elif (r, c) in self.SPECIAL_CELLS['TL']:
                        letter_value *= 3
                    
                    # Check for word multipliers
                    if (r, c) in self.SPECIAL_CELLS['DW']:
                        word_multiplier *= 2
                    elif (r, c) in self.SPECIAL_CELLS['TW']:
                        word_multiplier *= 3
                
                word_score += letter_value
            
            # Apply word multiplier
            word_score *= word_multiplier
            word_scores.append((word_string, word_score))
            total_score += word_score
        
        # Bingo bonus: +50 if all 7 tiles were used
        if len(pending_tiles) == 7:
            total_score += 50
            print("BINGO! +50 bonus points!")
        
        return total_score, word_scores
    
    def get_word_string(self, word_cells):
        """Convert a list of (row, col) cells to a word string.
        Uses manual overrides if present, otherwise uses OCR result.
        '*' (blank tile) is displayed as '?' in word strings."""
        letters = []
        for r, c in word_cells:
            # Check for manual override first
            override = self.manual_overrides.get((r, c))
            if override:
                # Blank tiles (*) display as ? in words
                letters.append('?' if override == '*' else override)
            elif self.board_state[r][c] is not None:
                letters.append(self.board_state[r][c][0])
            else:
                letters.append('?')
        return ''.join(letters)
    
    def get_word_snapshot(self):
        """
        Get a snapshot of current word state for change detection.
        Returns dict with word strings and their lengths.
        """
        if not self.detected_words:
            return {}
        
        snapshot = {}
        for i, (word_cells, is_new_flags) in enumerate(self.detected_words):
            word_str = self.get_word_string(word_cells)
            # Store the word, its length, and count of '?' characters
            snapshot[i] = {
                'word': word_str,
                'length': len(word_cells),
                'unknown_count': word_str.count('?')
            }
        return snapshot
    
    def check_and_refresh_words(self, current_pending):
        """
        Check if words need to be refreshed due to:
        1. Longer word detected (more tiles on same line)
        2. OCR improvement (? replaced with letter, or higher confidence)
        """
        if not self.detected_words or not hasattr(self, 'last_word_snapshot'):
            return
        
        needs_refresh = False
        
        # Check 1: More pending tiles than before (longer word possible)
        if len(current_pending) > len(self.previous_pending_tiles):
            # New tiles added - re-extract words to get longer version
            needs_refresh = True
            print("Word refresh: More tiles detected")
        
        # Check 2: Compare current word state with snapshot
        if not needs_refresh:
            current_snapshot = self.get_word_snapshot()
            
            for i, current_data in current_snapshot.items():
                if i in self.last_word_snapshot:
                    last_data = self.last_word_snapshot[i]
                    
                    # Check if word length increased
                    if current_data['length'] > last_data['length']:
                        needs_refresh = True
                        print(f"Word refresh: Word {i} length increased {last_data['length']} -> {current_data['length']}")
                        break
                    
                    # Check if unknown count decreased (OCR improved)
                    if current_data['unknown_count'] < last_data['unknown_count']:
                        needs_refresh = True
                        print(f"Word refresh: Word {i} OCR improved (fewer ?)")
                        break
                    
                    # Check if word changed (better OCR result)
                    if current_data['word'] != last_data['word']:
                        # Only refresh if the new word is "better" (fewer ? or different letters)
                        if current_data['unknown_count'] <= last_data['unknown_count']:
                            needs_refresh = True
                            print(f"Word refresh: Word {i} changed '{last_data['word']}' -> '{current_data['word']}'")
                            break
        
        if needs_refresh:
            # Re-extract words and re-validate
            self.detected_words = self.extract_formed_words(current_pending)
            self.validate_detected_words()
            self.last_word_snapshot = self.get_word_snapshot()
            
            # Clear manual input state since word changed
            if self.manual_input_active:
                self.manual_input_active = False
                self.manual_input_text = ""
            
            print(f"Words refreshed: {[self.get_word_string(w) for w, _ in self.detected_words]}")

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

    def calculate_tile_obstruction_offset(self):
        """
        Calculate how many pixels at the bottom of each cell could be obstructed
        by a tile in the cell below, based on the board's perspective angle.
        
        The key insight is that when viewing the board at an angle from above,
        tiles have physical height that causes them to visually extend upward
        into the cell above them in the warped 2D view.
        
        We calculate this by:
        1. Using the 3D pose (rvec, tvec) to project two points on a tile's top surface
        2. Comparing where those points appear vs. where they would appear on the board surface
        3. The vertical difference tells us the pixel offset
        """
        if self.rvec is None or self.tvec is None:
            return 0
        
        # Sample point: center of cell (7, 7) - the middle of the board
        # We'll compare where a point on the board surface vs tile top surface projects
        sample_row, sample_col = 7.5, 7.5  # Center of a cell
        
        # Point on the board surface (Z=0)
        board_point = np.array([[sample_col, sample_row + 1, 0]], dtype=np.float32)  # Bottom edge of cell
        
        # Same XY point but on top of a tile (Z = -TILE_HEIGHT_RATIO, negative is "up")
        tile_top_point = np.array([[sample_col, sample_row + 1, -self.TILE_HEIGHT_RATIO]], dtype=np.float32)
        
        # Project both points to 2D
        board_proj, _ = cv2.projectPoints(board_point, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        tile_proj, _ = cv2.projectPoints(tile_top_point, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        
        board_y = board_proj[0][0][1]
        tile_y = tile_proj[0][0][1]
        
        # The difference in Y (in original image pixels) tells us how much the tile
        # appears to extend upward. We need to convert this to warped board pixels.
        
        # To convert from original image pixels to warped board pixels, we need to
        # consider the scale of the warped board relative to the original
        
        # Approximate scale: board_size / average_board_dimension_in_image
        corners = self.current_corners.reshape(-1, 2)
        # Calculate average vertical span in the original image
        left_height = np.linalg.norm(corners[3] - corners[0])   # BL - TL
        right_height = np.linalg.norm(corners[2] - corners[1])  # BR - TR
        avg_height = (left_height + right_height) / 2
        
        # Scale factor: warped height / original height
        scale = self.board_size[1] / avg_height
        
        # The offset in warped pixels
        offset_pixels = abs(tile_y - board_y) * scale
        
        # Add a small buffer for safety
        offset_pixels = int(offset_pixels * 1.2)
        
        # Clamp to reasonable range (0 to half cell height)
        max_offset = self.cell_size[1] // 2
        offset_pixels = min(max(0, offset_pixels), max_offset)
        
        return offset_pixels
    
    def get_cell_average_color(self, warped_board_lab, row, col):
        """Get the average LAB color of a specific cell, ignoring bottom portion that may be obstructed."""
        x1 = col * self.cell_size[0]
        y1 = row * self.cell_size[1]
        x2 = x1 + self.cell_size[0]
        y2 = y1 + self.cell_size[1]
        
        # Use center region of cell (avoid grid lines)
        margin = 3
        
        # Also ignore bottom portion that could be obstructed by tile in cell below
        # The obstruction only affects cells that have a cell below them (row < 14)
        bottom_margin = margin
        if row < self.grid_size - 1:
            # This cell could have its bottom obstructed by a tile in the cell below
            bottom_margin = max(margin, self.tile_obstruction_offset)
        
        cell_region = warped_board_lab[y1+margin:y2-bottom_margin, x1+margin:x2-margin]
        
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

    def get_pending_tiles(self):
        """Get tiles that are locked but not yet confirmed (new tiles this turn)."""
        pending = set()
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.locked_cells[row, col] and self.confirmed_board_state[row][col] is None:
                    # Also check that we have a recognized letter
                    if self.board_state[row][col] is not None:
                        pending.add((row, col))
        return pending
    
    def is_board_empty(self):
        """Check if the confirmed board has no tiles."""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.confirmed_board_state[row][col] is not None:
                    return False
        return True
    
    def validate_turn_placement(self, pending_tiles):
        """
        Validate that pending tiles form a valid Scrabble placement.
        Returns (is_valid, error_message_or_None).
        Does NOT check if words are valid dictionary words.
        """
        if not pending_tiles:
            return False, "No tiles placed"
        
        pending_list = list(pending_tiles)
        
        # Check 1: All tiles in a single row OR single column
        rows = set(r for r, c in pending_list)
        cols = set(c for r, c in pending_list)
        
        is_horizontal = len(rows) == 1
        is_vertical = len(cols) == 1
        
        if not is_horizontal and not is_vertical:
            return False, "Tiles must be in a single row or column"
        
        # Check 2: First turn must cover center (7, 7)
        center = (7, 7)
        if self.is_board_empty():
            if center not in pending_tiles:
                return False, "First word must cover center square"
        else:
            # Check 3: Must connect to existing tiles
            connects = False
            for (r, c) in pending_list:
                # Check all 4 neighbors
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if self.confirmed_board_state[nr][nc] is not None:
                            connects = True
                            break
                if connects:
                    break
            
            if not connects:
                # Also check if tiles fill gaps between existing tiles
                # (e.g., placing tiles that extend through existing tiles)
                if is_horizontal:
                    row = pending_list[0][0]
                    min_col = min(c for r, c in pending_list)
                    max_col = max(c for r, c in pending_list)
                    for c in range(min_col, max_col + 1):
                        if self.confirmed_board_state[row][c] is not None:
                            connects = True
                            break
                else:
                    col = pending_list[0][1]
                    min_row = min(r for r, c in pending_list)
                    max_row = max(r for r, c in pending_list)
                    for r in range(min_row, max_row + 1):
                        if self.confirmed_board_state[r][col] is not None:
                            connects = True
                            break
            
            if not connects:
                return False, "Must connect to existing tiles"
        
        # Check 4: Tiles must be contiguous (no gaps unless filled by existing tiles)
        if is_horizontal:
            row = pending_list[0][0]
            min_col = min(c for r, c in pending_list)
            max_col = max(c for r, c in pending_list)
            for c in range(min_col, max_col + 1):
                has_pending = (row, c) in pending_tiles
                has_confirmed = self.confirmed_board_state[row][c] is not None
                if not has_pending and not has_confirmed:
                    return False, "Word has gaps"
        else:
            col = pending_list[0][1]
            min_row = min(r for r, c in pending_list)
            max_row = max(r for r, c in pending_list)
            for r in range(min_row, max_row + 1):
                has_pending = (r, col) in pending_tiles
                has_confirmed = self.confirmed_board_state[r][col] is not None
                if not has_pending and not has_confirmed:
                    return False, "Word has gaps"
        
        return True, None
    
    def extract_formed_words(self, pending_tiles):
        """
        Extract all words formed by the pending tiles.
        Returns list of tuples: (word_cells, is_new_tile_flags)
        where word_cells is list of (row, col) and is_new_tile_flags is list of bools
        """
        if not pending_tiles:
            return []
        
        pending_list = list(pending_tiles)
        words = []
        
        # Determine orientation
        rows = set(r for r, c in pending_list)
        is_horizontal = len(rows) == 1
        
        # Helper to get letter at position (from board_state, includes both pending and confirmed)
        def get_letter(r, c):
            if self.board_state[r][c] is not None:
                return self.board_state[r][c][0]
            return None
        
        # Helper to extract a word in a direction
        def extract_word_line(start_r, start_c, dr, dc):
            """Extract word starting from position, going in direction, then extending backwards."""
            cells = []
            is_new = []
            
            # First, go backwards to find start of word
            r, c = start_r, start_c
            while True:
                prev_r, prev_c = r - dr, c - dc
                if 0 <= prev_r < self.grid_size and 0 <= prev_c < self.grid_size:
                    if get_letter(prev_r, prev_c) is not None:
                        r, c = prev_r, prev_c
                        continue
                break
            
            # Now go forward collecting the word
            while 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                letter = get_letter(r, c)
                if letter is None:
                    break
                cells.append((r, c))
                is_new.append((r, c) in pending_tiles)
                r, c = r + dr, c + dc
            
            return cells, is_new
        
        # 1. Extract the main word (along the line of placed tiles)
        if is_horizontal:
            # Main word is horizontal
            row = pending_list[0][0]
            main_cells, main_is_new = extract_word_line(row, pending_list[0][1], 0, 1)
            if len(main_cells) > 1:
                words.append((main_cells, main_is_new))
        else:
            # Main word is vertical
            col = pending_list[0][1]
            main_cells, main_is_new = extract_word_line(pending_list[0][0], col, 1, 0)
            if len(main_cells) > 1:
                words.append((main_cells, main_is_new))
        
        # 2. For each new tile, check for perpendicular words (cross-words)
        for (r, c) in pending_list:
            if is_horizontal:
                # Check vertical cross-word
                cross_cells, cross_is_new = extract_word_line(r, c, 1, 0)
                if len(cross_cells) > 1:
                    # Only add if not duplicate
                    if (cross_cells, cross_is_new) not in words:
                        words.append((cross_cells, cross_is_new))
            else:
                # Check horizontal cross-word
                cross_cells, cross_is_new = extract_word_line(r, c, 0, 1)
                if len(cross_cells) > 1:
                    if (cross_cells, cross_is_new) not in words:
                        words.append((cross_cells, cross_is_new))
        
        return words

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

    def process_headless(self, progress_callback=None):
        """
        Process video without display, returning detection results.
        Used for testing - runs as fast as possible.
        
        Args:
            progress_callback: Optional function(frame_num, total_frames) for progress updates
            
        Returns:
            dict with:
                - 'locked_cells': set of (row, col) tuples for detected tiles
                - 'board_state': 15x15 grid with (letter, confidence) or None
                - 'corners': the corner coordinates used
                - 'total_frames': number of frames processed
        """
        # Initialize timing
        self.last_frame_time = time.time()
        reference_captured = False
        
        # Destination corners for 2D board
        dst_corners = np.array([
            [0, 0],
            [self.board_size[0] - 1, 0],
            [self.board_size[0] - 1, self.board_size[1] - 1],
            [0, self.board_size[1] - 1]
        ], dtype=np.float32)
        
        # Smoothing parameters (same as process_video)
        ALPHA = 0.4
        MAX_SHIFT_THRESHOLD = 50.0
        
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_num += 1
            if progress_callback:
                progress_callback(frame_num, total_frames)
            
            frame_clean = frame.copy()
            
            # Calculate delta time
            current_time = time.time()
            delta_ms = (current_time - self.last_frame_time) * 1000.0
            self.last_frame_time = current_time
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Feature Matching (ORB) - same as process_video
            kp2, des2 = self.orb.detectAndCompute(frame_gray, None)
            
            if des2 is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(self.ref_descriptors, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                
                if len(matches) > 15:
                    src_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if M is not None:
                        new_corners = cv2.perspectiveTransform(self.ref_corners, M)
                        shift_metric = np.linalg.norm(new_corners - self.current_corners) / 4
                        
                        if shift_metric < MAX_SHIFT_THRESHOLD:
                            self.current_corners = (1 - ALPHA) * self.current_corners + ALPHA * new_corners
                            self.update_3d_pose()
            
            # Process 2D board
            if self.current_corners is not None:
                transform_matrix = cv2.getPerspectiveTransform(self.current_corners, dst_corners)
                warped_board = cv2.warpPerspective(frame, transform_matrix, self.board_size)
                
                if not reference_captured:
                    self.capture_reference_colors(warped_board)
                    reference_captured = True
                
                if self.reference_cell_colors is not None:
                    self.detect_cell_changes(warped_board, delta_ms)
                    
                    # Process OCR results
                    current_time_ms = current_time * 1000.0
                    
                    for (r, c, letter, confidence, tile_clean) in self.ocr_service.get_results():
                        if self.locked_cells[r, c]:
                            existing = self.board_state[r][c]
                            if existing is None:
                                self.board_state[r][c] = (letter, confidence)
                            elif confidence > existing[1]:
                                self.board_state[r][c] = (letter, confidence)
                    
                    # Submit OCR requests for locked cells
                    for r in range(self.grid_size):
                        for c in range(self.grid_size):
                            if self.locked_cells[r, c]:
                                existing = self.board_state[r][c]
                                needs_ocr = False
                                
                                if existing is None:
                                    needs_ocr = True
                                elif existing[1] < self.OCR_MIN_CONFIDENCE:
                                    last_attempt = self.last_ocr_attempt_time.get((r, c), 0)
                                    if current_time_ms - last_attempt >= self.OCR_RETRY_INTERVAL_MS:
                                        needs_ocr = True
                                
                                if needs_ocr and not self.ocr_service.is_pending(r, c):
                                    tile_face = self.extract_tile_face(frame_clean, r, c, output_size=(100, 100), margin_scale=0.1)
                                    if tile_face is not None:
                                        self.last_ocr_attempt_time[(r, c)] = current_time_ms
                                        self.ocr_service.submit(tile_face, r, c)
                            else:
                                self.ocr_service.cancel(r, c)
                                self.board_state[r][c] = None
        
        # Wait for pending OCR to complete (with timeout)
        timeout_start = time.time()
        while time.time() - timeout_start < 5.0:  # 5 second timeout
            # Check if any OCR still pending
            any_pending = False
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if self.ocr_service.is_pending(r, c):
                        any_pending = True
                        break
                if any_pending:
                    break
            
            if not any_pending:
                break
            
            # Process any completed results
            for (r, c, letter, confidence, tile_clean) in self.ocr_service.get_results():
                if self.locked_cells[r, c]:
                    existing = self.board_state[r][c]
                    if existing is None:
                        self.board_state[r][c] = (letter, confidence)
                    elif confidence > existing[1]:
                        self.board_state[r][c] = (letter, confidence)
            
            time.sleep(0.1)
        
        # Collect final results
        locked_cells_set = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.locked_cells[r, c]:
                    locked_cells_set.add((r, c))
        
        # Convert corners to list format for JSON serialization
        corners_list = [int(x) for pt in self.points for x in pt]
        
        self.cap.release()
        
        return {
            'locked_cells': locked_cells_set,
            'board_state': self.board_state,
            'corners': corners_list,
            'total_frames': frame_num
        }

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
            # Draw ORB features if enabled
            if hasattr(self, 'show_orb') and self.show_orb and kp2 is not None:
                # Draw all detected keypoints as small cyan dots
                for kp in kp2:
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                
                # If we have matches, highlight the matched keypoints as small green dots
                if des2 is not None and len(matches) > 0:
                    for m in matches[:50]:  # Top 50 matches
                        kp = kp2[m.trainIdx]
                        x, y = int(kp.pt[0]), int(kp.pt[1])
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
                # Show ORB stats
                cv2.putText(frame, f"ORB Features: {len(kp2)}", (20, 125), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, f"Ref Features: {len(self.ref_keypoints)}", (20, 145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if des2 is not None:
                    cv2.putText(frame, f"Matches: {len(matches)}", (20, 165), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
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
                                
                                # Save to dataset if enabled (use original cached tile)
                                if (r, c) in self.tile_original_cache:
                                    self.save_tile_to_dataset(r, c, letter, confidence, 
                                                             self.tile_original_cache[(r, c)], current_time_ms)
                            elif confidence > existing[1]:
                                # New result is better - update
                                old_letter, old_conf = existing
                                self.board_state[r][c] = (letter, confidence)
                                self.tile_image_cache[(r, c)] = tile_clean
                                print(f"Updated ({r},{c}): {old_letter}({old_conf}) -> {letter}({confidence})")
                                
                                # Save to dataset if enabled (use original cached tile)
                                if (r, c) in self.tile_original_cache:
                                    self.save_tile_to_dataset(r, c, letter, confidence, 
                                                             self.tile_original_cache[(r, c)], current_time_ms)
                    
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
                                        # Cache the original tile capture (before OCR processing)
                                        self.tile_original_cache[(r, c)] = tile_face.copy()
                                        self.ocr_service.submit(tile_face, r, c)
                                
                                # Dataset: Periodic re-capture every 5 seconds
                                # Also save if there's an override (regardless of OCR confidence)
                                if self.dataset_dir is not None and existing is not None:
                                    letter, confidence = existing
                                    has_override = (r, c) in self.manual_overrides
                                    if has_override or confidence >= self.DATASET_MIN_CONFIDENCE:
                                        last_save = self.dataset_last_save_time.get((r, c), 0)
                                        if current_time_ms - last_save >= self.DATASET_SAVE_INTERVAL_MS:
                                            # Re-extract fresh tile and save
                                            tile_face = self.extract_tile_face(frame_clean, r, c, output_size=(100, 100), margin_scale=0.1)
                                            if tile_face is not None:
                                                self.save_tile_to_dataset(r, c, letter, confidence, tile_face, current_time_ms)
                            else:
                                # Cell is UNLOCKED - cancel any pending OCR and clear state
                                self.ocr_service.cancel(r, c)
                                self.board_state[r][c] = None
                                self.tile_image_cache.pop((r, c), None)
                                self.tile_original_cache.pop((r, c), None)
                                self.last_ocr_attempt_time.pop((r, c), None)

                    # --- TURN DETECTION LOGIC (only in turns mode) ---
                    if hasattr(self, 'turns_mode') and self.turns_mode:
                        # Get current pending tiles (locked but not confirmed)
                        current_pending = self.get_pending_tiles()
                        
                        if not self.awaiting_confirmation:
                            # Check if pending tiles have changed
                            if current_pending != self.previous_pending_tiles:
                                # Tiles changed - reset stability timer
                                self.pending_stable_since = None
                                self.previous_pending_tiles = current_pending.copy()
                            elif current_pending:  # Tiles exist and haven't changed
                                if self.pending_stable_since is None:
                                    # Start stability timer
                                    self.pending_stable_since = current_time_ms
                                else:
                                    # Check if stable long enough
                                    stable_duration = current_time_ms - self.pending_stable_since
                                    if stable_duration >= self.PENDING_STABLE_TIME_MS:
                                        # Time to validate!
                                        is_valid, error_msg = self.validate_turn_placement(current_pending)
                                        
                                        if is_valid:
                                            # Extract formed words and show confirmation
                                            self.detected_words = self.extract_formed_words(current_pending)
                                            # Validate words against dictionary
                                            self.validate_detected_words()
                                            # Store word snapshot for change detection
                                            self.last_word_snapshot = self.get_word_snapshot()
                                            self.awaiting_confirmation = True
                                            print(f"Valid placement detected! Words: {[self.get_word_string(w) for w, _ in self.detected_words]}")
                                            for i, (word, is_valid_word, suggestions) in enumerate(self.word_validations):
                                                status = "✓" if is_valid_word else "✗"
                                                print(f"  {status} {word}: {suggestions if suggestions else 'OK'}")
                                        else:
                                            # Invalid placement - show error and reset
                                            self.turn_error_message = error_msg
                                            self.turn_error_time = current_time
                                            self.pending_stable_since = None  # Reset to try again
                                            print(f"Invalid placement: {error_msg}")
                        else:
                            # Already awaiting confirmation - check if we need to refresh
                            self.check_and_refresh_words(current_pending)
                    
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

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to quit (unless in input mode)
                if self.override_input_active:
                    # Cancel override input, don't exit
                    self.handle_override_input(27)
                elif self.manual_input_active:
                    self.handle_key_input(27)
                else:
                    break
            elif key != 255:  # Any other key
                # Check override input first (takes priority)
                if self.override_input_active:
                    self.handle_override_input(key)
                else:
                    self.handle_key_input(key)

        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final board state as 2D grid
        self.print_board_state()
    
    def print_board_state(self):
        """Print the current board state as a 2D grid to console.
        Shows override letters where present, with visual indicators."""
        # Count tiles
        tile_count = sum(1 for r in range(self.grid_size) for c in range(self.grid_size) 
                        if self.board_state[r][c] is not None)
        
        if tile_count == 0:
            print("\nNo tiles detected on board.")
            return
        
        override_count = len(self.manual_overrides)
        override_info = f", {override_count} overrides" if override_count > 0 else ""
        print(f"\n=== Final Board State ({tile_count} tiles{override_info}) ===")
        print("     " + " ".join(f"{i:2}" for i in range(self.grid_size)))
        print("    +" + "-" * 46 + "+")
        
        for row in range(self.grid_size):
            row_letters = []
            for col in range(self.grid_size):
                cell = self.board_state[row][col]
                override = self.manual_overrides.get((row, col))
                if cell:
                    if override:
                        # Show override letter (use _ for blank)
                        display = override if override != '*' else '_'
                        row_letters.append(f" {display}")
                    else:
                        row_letters.append(f" {cell[0]}")
                else:
                    row_letters.append(" .")
            print(f"  {row:2} |" + " ".join(row_letters) + " |")
        
        print("    +" + "-" * 46 + "+")
        
        # Print tile details with confidence and overrides
        print(f"\nTile details:")
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell = self.board_state[row][col]
                if cell:
                    letter, conf = cell
                    override = self.manual_overrides.get((row, col))
                    if override:
                        display = 'BLANK' if override == '*' else override
                        print(f"  ({row:2},{col:2}): {letter} -> {display} [OVERRIDE]")
                    else:
                        print(f"  ({row:2},{col:2}): {letter} (conf={conf})")

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
        
        # Update tile obstruction offset based on new pose
        old_offset = self.tile_obstruction_offset
        self.tile_obstruction_offset = self.calculate_tile_obstruction_offset()
        
        # Only print if offset changed significantly
        if abs(self.tile_obstruction_offset - old_offset) > 1:
            print(f"Tile obstruction offset: {self.tile_obstruction_offset}px (ignoring bottom {self.tile_obstruction_offset}px of each cell)")

    def calculate_tile_vertical_shift(self):
        """
        Calculate the visual vertical shift (in grid units) caused by tile height
        when viewing from an angle. This tells us how much the tile appears to shift
        upward in its cell due to perspective.
        
        Returns the shift in grid units (0-1 range typically).
        """
        if self.rvec is None or self.tvec is None:
            return 0.0
        
        # Use center of board for calculation
        sample_row, sample_col = 7.5, 7.5
        
        # Point on the board surface (Z=0)
        board_point = np.array([[sample_col, sample_row, 0]], dtype=np.float32)
        
        # Same XY point but on top of a tile
        tile_top_point = np.array([[sample_col, sample_row, -self.TILE_HEIGHT_RATIO]], dtype=np.float32)
        
        # Project both points to 2D
        board_proj, _ = cv2.projectPoints(board_point, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        tile_proj, _ = cv2.projectPoints(tile_top_point, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        
        # Get the difference in Y (original image coordinates)
        board_y = board_proj[0][0][1]
        tile_y = tile_proj[0][0][1]
        y_diff = board_y - tile_y  # Positive means tile appears higher
        
        # Convert to grid units by projecting a reference distance
        # Project a point one grid unit down from sample point
        next_row_point = np.array([[sample_col, sample_row + 1, 0]], dtype=np.float32)
        next_row_proj, _ = cv2.projectPoints(next_row_point, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        grid_unit_in_pixels = abs(next_row_proj[0][0][1] - board_proj[0][0][1])
        
        if grid_unit_in_pixels > 0:
            shift_in_grid_units = y_diff / grid_unit_in_pixels
        else:
            shift_in_grid_units = 0.0
        
        return shift_in_grid_units

    def extract_tile_face(self, frame, row, col, output_size=(100, 100), margin_scale=0.2):
        """
        Extracts the tile face accounting for height (3D) and perspective shift
        to accurately capture the visible tile surface.
        """
        if self.rvec is None or self.tvec is None:
            return None

        # Z is "up" out of the board. 
        h_val = -1.0 * self.TILE_HEIGHT_RATIO 
        
        m = margin_scale / 2.0
        
        # Calculate the visual shift caused by perspective
        # This adjusts where we look for the tile to account for the
        # upward visual displacement when viewing at an angle
        vertical_shift = self.calculate_tile_vertical_shift()
        
        # Apply the shift - move our extraction window upward to follow 
        # where the tile visually appears. We use a portion of the shift
        # to avoid over-correcting.
        row_adjusted = row - (vertical_shift * 0.7)  # Use 70% of the shift
        
        # 3D points of the tile TOP face, with adjusted row position
        top_face_3d = np.array([
            [col - m,     row_adjusted - m,     h_val], # TL
            [col + 1 + m, row_adjusted - m,     h_val], # TR
            [col + 1 + m, row_adjusted + 1 + m, h_val], # BR
            [col - m,     row_adjusted + 1 + m, h_val]  # BL
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
    parser.add_argument('--show-orb', action='store_true', help='Show ORB features for debugging')
    parser.add_argument('--turns', action='store_true', help='Enable turn management mode (validates placement, requires confirmation)')
    parser.add_argument('--players', type=int, default=2, help='Number of players (2-4)')
    parser.add_argument('--our-turn', type=int, default=1, help='Which position our player is at (1-based: 1=first, 2=second, etc.)')
    parser.add_argument('--save-dataset', type=str, default=None, help='Save detected tiles to dataset folder (e.g., dataset_raw)')
    
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
    tracker.show_orb = args.show_orb
    tracker.turns_mode = args.turns
    
    # Set up player configuration
    num_players = max(2, min(4, args.players))  # Clamp to 2-4
    our_turn = max(1, min(num_players, args.our_turn))  # Clamp to valid range
    tracker.num_players = num_players
    tracker.our_player_index = our_turn - 1  # Convert to 0-based
    tracker.current_player = 0  # Start with player 1 (index 0)
    tracker.player_scores = [0] * num_players
    
    if args.turns:
        print("Turn management mode ENABLED - tiles will require confirmation")
        print(f"Players: {num_players}, Our position: {our_turn} (0-based index: {tracker.our_player_index})")
    else:
        print("Simple OCR mode - tiles are detected and displayed without turn validation")
    
    # Enable dataset saving if requested
    if args.save_dataset:
        tracker.set_dataset_dir(args.save_dataset)
    
    if tracker.initialize():
        tracker.process_video()
