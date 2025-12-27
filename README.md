# Scrabble Vision Tracker

A computer vision system for tracking Scrabble games in real-time. Uses a camera or video feed to detect the board, recognize tiles, suggest optimal moves, and keep score.

## Features

- **Real-time board tracking** using ORB feature matching
- **Tile detection** with CNN-based OCR (custom trained model)
- **Rack detection** using YOLOv8 object detection
- **Word solver** that suggests optimal moves
- **Turn management** with validation and scoring
- **Interactive mode** for live camera play

## Installation

```bash
# Clone the repository
git clone https://github.com/idofrizler/scrabble.git
cd scrabble

# Install dependencies
pip install -r requirements.txt

# For rack detection, also install ultralytics
pip install ultralytics
```

## Quick Start (Interactive Mode)

The primary use case is interactive mode with a live camera:

```bash
python main.py -i
```

This will:
1. Open your camera feed
2. Prompt you to position the camera (use +/- to zoom)
3. Ask you to mark the 4 corners of the board
4. Start tracking the game with turns, rack detection, and word suggestions

---

## 1. Running the Game

### Interactive Mode (Live Camera)

```bash
# Basic interactive mode (recommended)
python main.py -i

# Specify camera (if you have multiple)
python main.py -i --camera 1

# Record session to video file
python main.py -i --record game_session.mp4

# With pre-defined corners (skip corner marking)
python main.py -i --corners "x1,y1,x2,y2,x3,y3,x4,y4"
```

### Video File Mode

```bash
# Basic video processing
python main.py ./videos/game.mp4

# With turn management
python main.py ./videos/game.mp4 --turns --detect-rack

# With manual corners and 4 players
python main.py ./videos/game.mp4 --corners "100,50,900,50,900,700,100,700" --turns --players 4

# Show ORB features for debugging
python main.py ./videos/game.mp4 --show-orb
```

### Command Line Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-i`, `--interactive` | Use live camera instead of video file | Off |
| `--camera N` | Camera index for interactive mode | 0 |
| `--corners "x1,y1,..."` | Pre-defined board corners (8 values) | None (manual) |
| `--turns` | Enable turn management mode | Off (auto in interactive) |
| `--players N` | Number of players (2-4) | 2 |
| `--our-turn N` | Which player position we are (1-based) | 1 |
| `--detect-rack` | Enable rack/tile detection | Off (auto in interactive) |
| `--rack-model PATH` | Path to YOLOv8 rack model | `training/rack_tile_yolov8.pt` |
| `--use-tesseract` | Use Tesseract instead of CNN for OCR | Off (CNN default) |
| `--save-dataset DIR` | Save detected tiles to folder for training | None |
| `--record FILE` | Record interactive session to MP4 | None |
| `--show-orb` | Show ORB feature points for debugging | Off |

### Keyboard Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume |
| `1`, `2`, `3` | Set playback speed (video mode only) |
| `ESC` | Exit |
| `+`, `-` | Zoom in/out (camera positioning only) |

---

## 2. Training the OCR Model

The OCR model recognizes letters on Scrabble tiles. Training data is organized in folders by letter.

### Prepare Training Data

```bash
# Directory structure expected:
data/tile_ocr/
├── A/          # Images of 'A' tiles
│   ├── img1.png
│   └── ...
├── B/
├── ...
├── Z/
├── BLANK/      # Blank/joker tiles
└── NOISE/      # Non-tile images (false positives)
```

### Collect Training Data

During gameplay, double-click tiles in the debug view to save them:
1. Run the game with `--save-dataset dataset_raw`
2. Click a tile to open debug view
3. Double-click and type the correct letter (A-Z, * for blank, ! for noise)
4. Press ENTER to save

### Train the Model

```bash
python training/train_scrabble.py
```

The training script will:
- Load images from `data/tile_ocr/`
- Train a CNN classifier
- Save the model to `training/scrabble_net.pth`

---

## 3. Training the Rack Detection Model

The rack detector uses YOLOv8 to find the tile rack and individual tiles.

### Prepare Training Data

```bash
# Directory structure:
data/rack_tile/
├── images/
│   ├── frame_000.jpg
│   └── ...
└── yolo_annotations/
    ├── frame_000.txt    # YOLO format: class x_center y_center width height
    └── ...
```

Annotation format (YOLO):
- Class 0 = rack
- Class 1 = tile

### Train the Model

```bash
python training/train_rack_tile.py
```

The script will:
- Set up data splits (train/val)
- Train YOLOv8 for object detection
- Save the model to `training/rack_tile_yolov8.pt`

---

## 4. Running the Test Suite

Tests verify tile detection accuracy against known game states.

### Run All Tests

```bash
pytest tests/
```

### Run Specific Tests

```bash
# Just the detection accuracy tests
pytest tests/test_detection.py

# With verbose output
pytest tests/test_detection.py -v

# Generate new test fixtures from videos
python tests/generate_fixtures.py
```

### Test Fixtures

Test fixtures are JSON files in `tests/fixtures/` containing:
- Video path and corners
- Expected board state at end
- Metadata (expected cells, tolerance)

---

## 5. Accuracy Dashboard

The accuracy dashboard shows detection results across all test videos.

### Generate Results

```bash
# Run tests and generate results
pytest tests/test_detection.py

# Results saved to tests/accuracy_results.json
```

### View Dashboard

```bash
# Start a simple HTTP server
cd tests
python -m http.server 8000
```

Then open in browser: http://localhost:8000/accuracy_dashboard.html

The dashboard shows:
- Per-video accuracy metrics
- Cell-by-cell comparison (expected vs detected)
- Overall statistics

---

## Project Structure

```
scrabble/
├── main.py                 # Main application
├── word_solver.py          # Word suggestion engine
├── rack_detector.py        # YOLOv8 rack/tile detector
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── data/
│   ├── words.txt           # Scrabble dictionary
│   ├── tile_ocr/           # OCR training images
│   └── rack_tile/          # Rack detection training data
│
├── training/
│   ├── train_scrabble.py   # OCR model training
│   ├── train_rack_tile.py  # Rack detection training
│   ├── scrabble_net.pth    # Trained OCR model
│   └── rack_tile_yolov8.pt # Trained YOLO model
│
├── tests/
│   ├── test_detection.py   # Detection accuracy tests
│   ├── conftest.py         # Test configuration
│   ├── generate_fixtures.py# Create test fixtures
│   ├── accuracy_dashboard.html
│   └── fixtures/           # Test data files
│
└── videos/                 # Sample game videos
```

---

## Troubleshooting

### Camera not detected
```bash
# List available cameras (macOS)
system_profiler SPCameraDataType

# Try different camera index
python main.py -i --camera 1
```

### OCR accuracy issues
- Ensure good lighting on the board
- Train with more examples: `--save-dataset dataset_raw`
- Check for NOISE class examples (false positives)

### Rack not detected
- Ensure the rack is in the camera frame
- The YOLO model needs clear view of tiles
- Retrain with more examples if needed

### Corner tracking lost
- Use `--show-orb` to debug feature matching
- Ensure the board has enough texture/features
- Avoid reflections and shadows
