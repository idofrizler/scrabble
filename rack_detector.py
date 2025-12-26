"""
Rack and Tile Detection Module for Scrabble

This module detects the tile rack and individual tiles on it using YOLOv8,
then uses the existing OCR service to recognize the letters on each tile.
"""

import cv2
import numpy as np
import threading
import queue
import time
import os
import uuid
from pathlib import Path

# PyTorch and YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Rack detection disabled.")

# Import OCR service from main module
try:
    from main import OCRService, ScrabbleNet
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: Could not import OCRService from main.py")


class RackDetector:
    """
    Detects tile racks and the tiles on them using YOLOv8.
    Runs in a separate thread to not block the main video processing.
    """
    
    # Class indices from training
    CLASS_RACK = 0
    CLASS_TILE = 1
    
    # Detection thresholds
    RACK_CONFIDENCE = 0.5
    TILE_CONFIDENCE = 0.3
    
    # Tile locking thresholds (similar to board tiles)
    LOCK_CONFIDENCE_THRESHOLD = 80  # Lock tile at this confidence
    OCR_RETRY_INTERVAL_MS = 1000.0  # Retry OCR every 1 second for low-confidence tiles
    
    # Virtual rack display settings
    VIRTUAL_RACK_WIDTH = 500
    VIRTUAL_RACK_HEIGHT = 80
    TILE_DISPLAY_SIZE = 60
    
    # Detection rate limiting
    DETECTION_INTERVAL_MS = 1000.0  # Only detect once per second
    
    def __init__(self, model_path="training/rack_tile_yolov8.pt", use_tesseract=False):
        """
        Initialize the rack detector.
        
        Args:
            model_path: Path to the trained YOLOv8 model
            use_tesseract: Whether to use Tesseract for OCR (False = use CNN)
        """
        self.model = None
        self.ocr_service = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)  # Buffer for frames to process
        self.result_queue = queue.Queue()  # Results to display
        
        # Current rack state
        self.current_rack_letters = []  # List of (letter, confidence, tile_image)
        self.last_rack_box = None  # (x1, y1, x2, y2) of detected rack
        self.last_tile_boxes = []  # List of (x1, y1, x2, y2) for each tile
        
        # Tile locking state (persists within a turn)
        # Indexed by position (0-6), stores: (letter, confidence, tile_image, locked, last_retry_time)
        self.locked_tiles = {}  # position -> (letter, conf, tile_img, is_locked, last_retry_time)
        self.manual_overrides = {}  # position -> letter (user overrides)
        
        # Callback for when rack changes (for word solver recalculation)
        self.on_rack_change_callback = None
        
        # Thread for detection
        self.detection_thread = None
        
        # Rate limiting
        self.last_detection_time = 0
        
        # Save-to-dataset input state (no persistent overrides - tiles shift too much)
        self.save_input_active = False
        self.save_input_index = None
        self.save_input_text = ""
        
        # Double-click tracking for save action
        self.last_click_time = 0
        self.last_click_index = None
        self.DOUBLE_CLICK_THRESHOLD_MS = 500.0
        
        # Dataset directory for saving overrides
        self.dataset_dir = "data/tile_ocr"
        
        # Turn tracking (to know when to reset locks)
        self.current_turn_number = -1
        
        # Load YOLO model
        if YOLO_AVAILABLE:
            self._load_model(model_path)
        
        # Initialize OCR service
        if OCR_AVAILABLE:
            self.ocr_service = OCRService(use_tesseract=use_tesseract)
            print(f"Rack detector OCR initialized (Tesseract={use_tesseract})")
    
    def _load_model(self, model_path):
        """Load the YOLOv8 model."""
        if not Path(model_path).exists():
            print(f"Warning: Rack detection model not found at {model_path}")
            return
        
        try:
            self.model = YOLO(model_path)
            print(f"Rack detection model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading rack detection model: {e}")
            self.model = None
    
    def start(self):
        """Start the background detection thread."""
        if self.model is None:
            print("Cannot start rack detector: model not loaded")
            return False
        
        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        print("Rack detector started")
        return True
    
    def stop(self):
        """Stop the background detection thread."""
        self.running = False
        if self.detection_thread is not None:
            self.detection_thread.join(timeout=2.0)
        print("Rack detector stopped")
    
    def submit_frame(self, frame):
        """
        Submit a frame for rack detection (non-blocking).
        Drops old frames if queue is full.
        """
        try:
            # Try to clear old frame first
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass  # Skip this frame
    
    def get_results(self):
        """
        Get detection results (non-blocking).
        
        Returns:
            dict with:
                - 'rack_box': (x1, y1, x2, y2) or None
                - 'tile_boxes': list of (x1, y1, x2, y2)
                - 'letters': list of (letter, confidence)
        """
        try:
            result = self.result_queue.get_nowait()
            return result
        except queue.Empty:
            return None
    
    def _detection_loop(self):
        """Background thread that processes frames for rack/tile detection."""
        while self.running:
            try:
                # Get frame from queue (with timeout)
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Rate limiting - only detect once per second
                current_time = time.time() * 1000.0
                if current_time - self.last_detection_time < self.DETECTION_INTERVAL_MS:
                    continue
                self.last_detection_time = current_time
                
                # Run YOLO detection
                results = self.model(frame, verbose=False)[0]
                
                rack_box = None
                tile_boxes = []
                tile_images = []
                
                # Parse detections
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if cls_id == self.CLASS_RACK and conf >= self.RACK_CONFIDENCE:
                        rack_box = (x1, y1, x2, y2)
                    elif cls_id == self.CLASS_TILE and conf >= self.TILE_CONFIDENCE:
                        tile_boxes.append((x1, y1, x2, y2, conf))
                
                # If we have a rack, filter tiles to only those inside the rack
                if rack_box is not None:
                    rx1, ry1, rx2, ry2 = rack_box
                    tiles_in_rack = []
                    
                    for (tx1, ty1, tx2, ty2, conf) in tile_boxes:
                        # Check if tile center is inside rack
                        tile_cx = (tx1 + tx2) // 2
                        tile_cy = (ty1 + ty2) // 2
                        
                        if rx1 <= tile_cx <= rx2 and ry1 <= tile_cy <= ry2:
                            tiles_in_rack.append((tx1, ty1, tx2, ty2, conf))
                            # Extract tile image for OCR
                            tile_img = frame[ty1:ty2, tx1:tx2].copy()
                            tile_images.append(tile_img)
                    
                    # Sort tiles left to right (by x coordinate)
                    sorted_tiles = sorted(zip(tiles_in_rack, tile_images), 
                                         key=lambda x: x[0][0])
                    
                    if sorted_tiles:
                        tiles_in_rack, tile_images = zip(*sorted_tiles)
                        tiles_in_rack = list(tiles_in_rack)
                        tile_images = list(tile_images)
                    else:
                        tiles_in_rack = []
                        tile_images = []
                    
                    tile_boxes = tiles_in_rack
                
                # Run OCR on each tile with locking logic
                letters = []
                current_time = time.time() * 1000.0
                rack_changed = False
                
                for i, tile_img in enumerate(tile_images):
                    # Check if we have a manual override for this position
                    if i in self.manual_overrides:
                        override_letter = self.manual_overrides[i]
                        letters.append((override_letter, 100, tile_img))
                        continue
                    
                    # Check if this tile is already locked
                    if i in self.locked_tiles:
                        locked_letter, locked_conf, _, is_locked, last_retry = self.locked_tiles[i]
                        
                        if is_locked and locked_conf >= self.LOCK_CONFIDENCE_THRESHOLD:
                            # Tile is locked - use locked value
                            letters.append((locked_letter, locked_conf, tile_img))
                            continue
                        else:
                            # Low confidence locked tile - check if we should retry OCR
                            if current_time - last_retry < self.OCR_RETRY_INTERVAL_MS:
                                # Not time to retry yet - use existing value
                                letters.append((locked_letter, locked_conf, tile_img))
                                continue
                    
                    # Run OCR on this tile
                    if self.ocr_service is not None:
                        letter, conf, _ = self.ocr_service._recognize_tile(tile_img)
                    else:
                        letter, conf = '?', 0
                    
                    # Update or create locked tile entry
                    old_letter = self.locked_tiles.get(i, ('?', 0, None, False, 0))[0]
                    should_lock = conf >= self.LOCK_CONFIDENCE_THRESHOLD
                    self.locked_tiles[i] = (letter, conf, tile_img, should_lock, current_time)
                    
                    # Check if rack changed (for callback)
                    if letter != old_letter and letter not in ('?', '*'):
                        rack_changed = True
                    
                    letters.append((letter, conf, tile_img))
                
                # Clean up locked tiles for positions that no longer exist
                positions_to_remove = [pos for pos in self.locked_tiles if pos >= len(tile_images)]
                for pos in positions_to_remove:
                    del self.locked_tiles[pos]
                    if pos in self.manual_overrides:
                        del self.manual_overrides[pos]
                
                # Update current state
                self.last_rack_box = rack_box
                self.last_tile_boxes = [(t[0], t[1], t[2], t[3]) for t in tile_boxes]
                self.current_rack_letters = letters
                
                # Trigger callback if rack changed
                if rack_changed and self.on_rack_change_callback:
                    self.on_rack_change_callback()
                
                # Put result in queue
                result = {
                    'rack_box': rack_box,
                    'tile_boxes': self.last_tile_boxes,
                    'letters': [(l, c) for l, c, _ in letters],
                    'tile_images': [img for _, _, img in letters]
                }
                
                # Replace old result
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                self.result_queue.put(result)
                
            except Exception as e:
                print(f"Rack detection error: {e}")
                import traceback
                traceback.print_exc()
    
    def draw_detections(self, frame):
        """
        Draw rack and tile detections on the frame.
        
        Args:
            frame: The video frame to draw on (will be modified in place)
        
        Returns:
            The modified frame
        """
        # Draw rack box (blue)
        if self.last_rack_box is not None:
            x1, y1, x2, y2 = self.last_rack_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 3)
            cv2.putText(frame, "RACK", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        
        # Draw tile boxes (green) with letters
        for i, (x1, y1, x2, y2) in enumerate(self.last_tile_boxes):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add letter label if available
            if i < len(self.current_rack_letters):
                letter, conf, _ = self.current_rack_letters[i]
                label = f"{letter}"
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def set_turn_number(self, turn_number):
        """Update turn number and reset locks if turn changed."""
        if turn_number != self.current_turn_number:
            self.current_turn_number = turn_number
            self.reset_tile_locks()
    
    def reset_tile_locks(self):
        """Reset all tile locks and overrides (called on new turn)."""
        self.locked_tiles.clear()
        self.manual_overrides.clear()
        print("Rack tile locks reset for new turn")
    
    def set_on_rack_change(self, callback):
        """Set callback for when rack letters change (for word solver)."""
        self.on_rack_change_callback = callback
    
    def get_rack_letters(self):
        """
        Get the current letters detected on the rack.
        Uses locked/overridden values where available.
        
        Returns:
            String of letters (e.g., "ABCDEFG") or empty string
        """
        result = []
        for i, (letter, conf, _) in enumerate(self.current_rack_letters):
            # Check for manual override first
            if i in self.manual_overrides:
                override = self.manual_overrides[i]
                if override not in ('?', '*'):
                    result.append(override)
            # Then check locked tiles
            elif i in self.locked_tiles:
                locked_letter, _, _, is_locked, _ = self.locked_tiles[i]
                if is_locked and locked_letter not in ('?', '*'):
                    result.append(locked_letter)
            # Fall back to current detection
            elif letter not in ('?', '*'):
                result.append(letter)
        return ''.join(result)
    
    def override_tile(self, position, letter):
        """
        Override a tile's letter at the given position.
        Triggers word solver recalculation.
        """
        if 0 <= position < len(self.current_rack_letters):
            old_letter = self.manual_overrides.get(position, 
                         self.current_rack_letters[position][0] if position < len(self.current_rack_letters) else '?')
            self.manual_overrides[position] = letter
            
            # Also update locked_tiles to reflect the override
            if position in self.locked_tiles:
                _, conf, tile_img, _, last_retry = self.locked_tiles[position]
                self.locked_tiles[position] = (letter, 100, tile_img, True, last_retry)
            
            print(f"Tile {position} overridden: {old_letter} -> {letter}")
            
            # Trigger word solver recalculation
            if self.on_rack_change_callback:
                self.on_rack_change_callback()
            
            return True
        return False
    
    def handle_virtual_rack_click(self, x, y):
        """
        Handle a click on the virtual rack display.
        Returns the tile index if clicked on a tile, or None.
        """
        if not self.current_rack_letters:
            return None
        
        num_tiles = len(self.current_rack_letters)
        tile_spacing = 5
        total_tiles_width = num_tiles * self.TILE_DISPLAY_SIZE + (num_tiles - 1) * tile_spacing
        start_x = (self.VIRTUAL_RACK_WIDTH - total_tiles_width) // 2
        tile_y = (self.VIRTUAL_RACK_HEIGHT - self.TILE_DISPLAY_SIZE) // 2
        
        # Check if click is within tile Y bounds
        if not (tile_y <= y <= tile_y + self.TILE_DISPLAY_SIZE):
            return None
        
        # Check each tile's X bounds
        for i in range(num_tiles):
            tile_x = start_x + i * (self.TILE_DISPLAY_SIZE + tile_spacing)
            if tile_x <= x <= tile_x + self.TILE_DISPLAY_SIZE:
                return i
        
        return None
    
    def on_virtual_rack_click(self, event, x, y, flags, params):
        """Mouse callback for virtual rack window."""
        if event == cv2.EVENT_LBUTTONDOWN:
            current_time = time.time() * 1000.0
            tile_index = self.handle_virtual_rack_click(x, y)
            
            if tile_index is not None:
                # Check for double-click
                if (self.last_click_index == tile_index and 
                    current_time - self.last_click_time < self.DOUBLE_CLICK_THRESHOLD_MS):
                    # Double-click detected - start save-to-dataset mode
                    self.save_input_active = True
                    self.save_input_index = tile_index
                    self.save_input_text = ""
                    letter, conf, _ = self.current_rack_letters[tile_index]
                    print(f"Save to dataset mode for tile {tile_index} (OCR: {letter}, {conf}%)")
                    print("  Type A-Z for correct letter, * for blank, ENTER to save, ESC to cancel")
                
                self.last_click_index = tile_index
                self.last_click_time = current_time
    
    def handle_key_input(self, key):
        """
        Handle keyboard input for save-to-dataset mode.
        Returns True if key was consumed.
        """
        if not self.save_input_active:
            return False
        
        tile_index = self.save_input_index
        
        if key == 27:  # ESC - cancel
            self.save_input_active = False
            self.save_input_index = None
            self.save_input_text = ""
            print("Save cancelled")
            return True
        
        elif key == 13 or key == 10:  # ENTER - confirm and save/override
            if self.save_input_text:
                save_char = self.save_input_text.upper()
                if save_char == '*' or (len(save_char) == 1 and save_char.isalpha()):
                    # Save to dataset
                    if tile_index < len(self.current_rack_letters):
                        _, _, tile_img = self.current_rack_letters[tile_index]
                        self.save_tile_to_dataset(tile_img, save_char)
                    
                    # Also apply as override (updates locked value and triggers word solver)
                    self.override_tile(tile_index, save_char)
                    
                    display = 'BLANK' if save_char == '*' else save_char
                    print(f"Saved tile {tile_index} as '{display}' to dataset and applied override")
                else:
                    print(f"Invalid input: must be A-Z or * for blank")
            
            self.save_input_active = False
            self.save_input_index = None
            self.save_input_text = ""
            return True
        
        elif key == 8:  # BACKSPACE
            self.save_input_text = self.save_input_text[:-1]
            return True
        
        elif key == ord('*'):  # Asterisk for blank
            self.save_input_text = "*"
            return True
        
        elif 32 <= key <= 126:  # Printable ASCII
            char = chr(key).upper()
            if char.isalpha():
                self.save_input_text = char
            return True
        
        return True  # Consume key even if not recognized
    
    def save_tile_to_dataset(self, tile_img, letter):
        """Save a tile image to the dataset for training improvement."""
        if tile_img is None or tile_img.size == 0:
            return False
        
        # Determine folder
        if letter == '*':
            folder = "BLANK"
        elif letter.isalpha():
            folder = letter.upper()
        else:
            return False
        
        # Create directory if needed
        folder_path = os.path.join(self.dataset_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        
        # Generate unique filename and save
        filename = f"rack_{uuid.uuid4()}.png"
        filepath = os.path.join(folder_path, filename)
        cv2.imwrite(filepath, tile_img)
        
        print(f"  Saved to dataset: {folder}/{filename}")
        return True
    
    def create_virtual_rack_display(self):
        """
        Create a virtual rack display window showing detected tiles.
        
        Returns:
            numpy array (image) of the virtual rack display
        """
        # Create background (wood-like color)
        display = np.zeros((self.VIRTUAL_RACK_HEIGHT, self.VIRTUAL_RACK_WIDTH, 3), dtype=np.uint8)
        display[:] = (60, 100, 140)  # Wood brown-ish color (BGR)
        
        # Draw rack border
        cv2.rectangle(display, (5, 5), (self.VIRTUAL_RACK_WIDTH - 5, self.VIRTUAL_RACK_HEIGHT - 5), 
                     (40, 70, 100), 2)
        
        if not self.current_rack_letters:
            # No tiles detected
            cv2.putText(display, "No rack detected", 
                       (self.VIRTUAL_RACK_WIDTH // 2 - 80, self.VIRTUAL_RACK_HEIGHT // 2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            return display
        
        # Calculate tile positions
        num_tiles = len(self.current_rack_letters)
        tile_spacing = 5
        total_tiles_width = num_tiles * self.TILE_DISPLAY_SIZE + (num_tiles - 1) * tile_spacing
        start_x = (self.VIRTUAL_RACK_WIDTH - total_tiles_width) // 2
        tile_y = (self.VIRTUAL_RACK_HEIGHT - self.TILE_DISPLAY_SIZE) // 2
        
        for i, (letter, conf, tile_img) in enumerate(self.current_rack_letters):
            tile_x = start_x + i * (self.TILE_DISPLAY_SIZE + tile_spacing)
            
            # Check lock/override status
            is_overridden = i in self.manual_overrides
            is_locked = i in self.locked_tiles and self.locked_tiles[i][3]
            
            # Draw tile background (different colors for locked/overridden)
            if is_overridden:
                tile_color = (255, 255, 180)  # Cyan tint for override
            elif is_locked:
                tile_color = (180, 255, 200)  # Green tint for locked
            else:
                tile_color = (180, 210, 230)  # Light beige for unlocked
            
            cv2.rectangle(display, 
                         (tile_x, tile_y), 
                         (tile_x + self.TILE_DISPLAY_SIZE, tile_y + self.TILE_DISPLAY_SIZE),
                         tile_color, -1)
            
            # Highlight border if save input is active for this tile
            if self.save_input_active and self.save_input_index == i:
                border_color = (0, 255, 255)  # Yellow
                border_width = 3
            elif is_overridden:
                border_color = (200, 100, 0)  # Blue for override
                border_width = 2
            elif is_locked:
                border_color = (0, 180, 0)  # Green for locked
                border_width = 2
            else:
                border_color = (100, 130, 160)
                border_width = 2
            
            cv2.rectangle(display, 
                         (tile_x, tile_y), 
                         (tile_x + self.TILE_DISPLAY_SIZE, tile_y + self.TILE_DISPLAY_SIZE),
                         border_color, border_width)
            
            # Draw letter (or input text if in save mode)
            if self.save_input_active and self.save_input_index == i:
                display_letter = self.save_input_text + "_" if self.save_input_text else "_"
            else:
                display_letter = letter if letter != '*' else '_'
            
            text_size = cv2.getTextSize(display_letter, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            text_x = tile_x + (self.TILE_DISPLAY_SIZE - text_size[0]) // 2
            text_y = tile_y + (self.TILE_DISPLAY_SIZE + text_size[1]) // 2
            
            # Color based on confidence
            if conf >= 80:
                text_color = (0, 0, 0)  # Black for high confidence
            elif conf >= 50:
                text_color = (0, 100, 150)  # Orange for medium
            else:
                text_color = (0, 0, 200)  # Red for low
            
            cv2.putText(display, display_letter, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
            
            # Draw confidence indicator and lock status
            if is_overridden:
                status_text = "OVR"
                status_color = (200, 100, 0)
            elif is_locked:
                status_text = f"{conf}% L"
                status_color = (0, 120, 0)
            else:
                status_text = f"{conf}%"
                status_color = (80, 80, 80)
            
            cv2.putText(display, status_text, 
                       (tile_x + 2, tile_y + self.TILE_DISPLAY_SIZE - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, status_color, 1)
        
        # Show hint at bottom
        if not self.save_input_active:
            hint = "Double-click tile to save to dataset"
            cv2.putText(display, hint, (10, self.VIRTUAL_RACK_HEIGHT - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        else:
            hint = "A-Z=letter, *=blank, ENTER=save, ESC=cancel"
            cv2.putText(display, hint, (10, self.VIRTUAL_RACK_HEIGHT - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        
        return display


def run_standalone_demo(video_path, model_path="training/rack_tile_yolov8.pt"):
    """
    Run a standalone demo of rack detection on a video.
    
    Args:
        video_path: Path to video file
        model_path: Path to the trained YOLOv8 model
    """
    if not YOLO_AVAILABLE:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    detector = RackDetector(model_path=model_path)
    
    if not detector.start():
        print("Failed to start rack detector")
        return
    
    print("Rack detection demo running. Press ESC to quit.")
    print("  SPACE: Pause/Resume video")
    print("  1/2/3/4: Set playback speed (1x/2x/3x/4x)")
    print("  Double-click a tile on the virtual rack to save it to dataset")
    print("  Detection runs once per second (rate limited)")
    
    # Create windows and set mouse callback
    cv2.namedWindow('Virtual Rack')
    cv2.setMouseCallback('Virtual Rack', detector.on_virtual_rack_click)
    
    last_letters = ""
    paused = False
    playback_speed = 1  # 1x, 2x, 3x, 4x
    frame_skip_counter = 0
    
    # Get total frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    try:
        frame = None
        while True:
            if not paused:
                # Skip frames based on playback speed
                for _ in range(playback_speed):
                    ret, frame = cap.read()
                    if not ret:
                        break
                
                if not ret:
                    print("\nEnd of video reached.")
                    break
                
                # Submit frame for detection (only when not paused)
                if frame is not None:
                    detector.submit_frame(frame)
            
            # If we have no frame yet, wait
            if frame is None:
                ret, frame = cap.read()
                if not ret:
                    continue
            
            # Get latest results (non-blocking)
            result = detector.get_results()
            
            # Draw detections on frame
            frame = detector.draw_detections(frame)
            
            # Create virtual rack display
            virtual_rack = detector.create_virtual_rack_display()
            
            # Get video progress
            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            progress_pct = (current_frame_num / total_frames * 100) if total_frames > 0 else 0
            
            # Add status overlay and progress bar to frame
            display_frame = frame.copy()
            
            # Draw progress bar at bottom
            bar_height = 6
            bar_y = display_frame.shape[0] - bar_height - 2
            bar_width = display_frame.shape[1] - 20
            bar_x = 10
            
            # Background (dark gray)
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
            # Progress (green)
            progress_width = int(bar_width * progress_pct / 100)
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 200, 0), -1)
            # Border
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (150, 150, 150), 1)
            
            # Status text (above progress bar)
            status_y = display_frame.shape[0] - 20
            if paused:
                cv2.putText(display_frame, f"PAUSED ({progress_pct:.0f}%) - Press SPACE to resume", 
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif playback_speed > 1:
                cv2.putText(display_frame, f"{playback_speed}x Speed ({progress_pct:.0f}%)", 
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(display_frame, f"{progress_pct:.0f}%", 
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show windows
            cv2.imshow('Rack Detection', display_frame)
            cv2.imshow('Virtual Rack', virtual_rack)
            
            # Show detected letters (only when changed)
            letters = detector.get_rack_letters()
            if letters and letters != last_letters:
                print(f"\rRack: {letters}  ", end='', flush=True)
                last_letters = letters
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            
            # First check if override mode consumes the key
            if detector.handle_key_input(key):
                continue
            
            if key == 27:  # ESC (not in save mode)
                break
            elif key == ord(' '):  # SPACE - toggle pause
                paused = not paused
                if paused:
                    print("\n[PAUSED] - Double-click tiles to save, press SPACE to resume")
                else:
                    print("[RESUMED]")
            elif key == ord('1'):
                playback_speed = 1
                print(f"\n[Speed: 1x]")
            elif key == ord('2'):
                playback_speed = 2
                print(f"\n[Speed: 2x]")
            elif key == ord('3'):
                playback_speed = 3
                print(f"\n[Speed: 3x]")
            elif key == ord('4'):
                playback_speed = 4
                print(f"\n[Speed: 4x]")
    
    finally:
        detector.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("\nDemo finished.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Rack Detection Demo')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, default='training/rack_tile_yolov8.pt',
                       help='Path to YOLOv8 model')
    
    args = parser.parse_args()
    
    run_standalone_demo(args.video_path, args.model)
