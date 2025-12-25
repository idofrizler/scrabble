"""
Tests for Scrabble board tile detection and OCR.

These tests use fixture files generated from test videos to verify
that the detection logic correctly identifies:
1. Which cells have tiles (cell coverage detection)
2. What letters are on those tiles (OCR accuracy)

Tests are designed to ALWAYS PASS but report accuracy scores.
This allows tracking improvement over time without blocking CI.
"""

import pytest
import json
import os
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import load_fixture, get_video_path, parse_corners, FIXTURES_DIR
from main import ScrabbleTracker

# Results file to track accuracy over time
RESULTS_FILE = Path(__file__).parent / "accuracy_results.json"

# Run ID file to persist run ID across test invocations in the same session
RUN_ID_FILE = Path(__file__).parent / ".current_run_id"

def get_current_run_id():
    """
    Get the current run ID. 
    Run IDs are generated once per test session and shared across all tests.
    Format: YYYYMMDD-HHMMSS
    """
    # Check environment variable first (set by conftest)
    env_run_id = os.environ.get('SCRABBLE_TEST_RUN_ID')
    if env_run_id:
        return env_run_id
    
    # Fallback: generate new run ID
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def board_to_cells(board):
    """Convert 2D board array to set of (row, col) tuples for non-empty cells."""
    cells = set()
    for row, row_str in enumerate(board):
        for col, char in enumerate(row_str):
            if char != '.':
                cells.add((row, col))
    return cells

def board_to_letters(board):
    """Convert 2D board array to dict of (row, col) -> letter."""
    letters = {}
    for row, row_str in enumerate(board):
        for col, char in enumerate(row_str):
            if char != '.':
                letters[(row, col)] = char
    return letters

def result_to_board(result):
    """Convert process_headless result to 2D board array."""
    board = []
    for row in range(15):
        row_str = ""
        for col in range(15):
            cell_data = result['board_state'][row][col]
            if cell_data and (row, col) in result['locked_cells']:
                row_str += cell_data[0]
            else:
                row_str += "."
        board.append(row_str)
    return board

def calculate_scores(fixture, result):
    """
    Calculate accuracy scores for cell detection and OCR.
    
    Returns dict with:
        - cell_precision: % of detected cells that are correct
        - cell_recall: % of expected cells that were detected  
        - cell_f1: harmonic mean of precision and recall
        - ocr_accuracy: % of detected cells with correct letter
        - overall_accuracy: % of expected tiles correctly detected AND recognized
    """
    expected_cells = board_to_cells(fixture['board'])
    expected_letters = board_to_letters(fixture['board'])
    detected_cells = result['locked_cells']
    
    # Cell detection metrics
    true_positives = expected_cells & detected_cells
    false_positives = detected_cells - expected_cells
    false_negatives = expected_cells - detected_cells
    
    precision = len(true_positives) / len(detected_cells) * 100 if detected_cells else 100
    recall = len(true_positives) / len(expected_cells) * 100 if expected_cells else 100
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # OCR accuracy (only for correctly detected cells)
    ocr_correct = 0
    ocr_errors = []
    for (row, col) in true_positives:
        expected_letter = expected_letters[(row, col)]
        cell_data = result['board_state'][row][col]
        if cell_data and cell_data[0] == expected_letter:
            ocr_correct += 1
        else:
            detected_letter = cell_data[0] if cell_data else None
            ocr_errors.append(((row, col), expected_letter, detected_letter))
    
    ocr_accuracy = ocr_correct / len(true_positives) * 100 if true_positives else 100
    
    # Overall accuracy: tiles that are both detected AND correctly recognized
    overall_correct = ocr_correct
    overall_accuracy = overall_correct / len(expected_cells) * 100 if expected_cells else 100
    
    # False positive tracking metrics
    tiles_ever_locked = result.get('tiles_ever_locked', set())
    tiles_removed = result.get('tiles_removed', set())
    
    # False positives that were removed: locked transiently then unlocked, and NOT expected tiles
    # These are cells that were wrongly detected as tiles and later correctly cleared
    false_detections_removed = tiles_removed - expected_cells
    
    # All transient false positives: cells ever locked that shouldn't have been
    transient_false_positives = tiles_ever_locked - expected_cells
    
    return {
        'cell_precision': round(precision, 1),
        'cell_recall': round(recall, 1),
        'cell_f1': round(f1, 1),
        'ocr_accuracy': round(ocr_accuracy, 1),
        'overall_accuracy': round(overall_accuracy, 1),
        'expected_tiles': len(expected_cells),
        'detected_tiles': len(detected_cells),
        'correct_cells': len(true_positives),
        'extra_cells': len(false_positives),
        'missing_cells': len(false_negatives),
        'ocr_correct': ocr_correct,
        'ocr_errors': ocr_errors,
        'false_positives': sorted(false_positives),
        'false_negatives': sorted(false_negatives),
        # New false detection tracking metrics
        'tiles_ever_locked': len(tiles_ever_locked),
        'tiles_removed': len(tiles_removed),
        'transient_false_positives': len(transient_false_positives),
        'false_detections_removed': len(false_detections_removed),
        'false_detections_removed_list': sorted(false_detections_removed),
    }

def save_results(results, run_id=None):
    """Save test results to JSON file for tracking over time."""
    # Load existing results
    all_results = {}
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, 'r') as f:
                all_results = json.load(f)
        except:
            pass
    
    # Add timestamp and run_id
    timestamp = datetime.now().isoformat()
    if run_id is None:
        run_id = get_current_run_id()
    
    # Update with new results
    for video_name, scores in results.items():
        if video_name not in all_results:
            all_results[video_name] = []
        all_results[video_name].append({
            'timestamp': timestamp,
            'run_id': run_id,
            **scores
        })
        # Keep only last 10 results per video
        all_results[video_name] = all_results[video_name][-10:]
    
    # Save
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

def print_score_report(video_name, scores):
    """Print a formatted accuracy report."""
    print(f"\n{'='*60}")
    print(f"ACCURACY REPORT: {video_name}")
    print(f"{'='*60}")
    print(f"  Overall Accuracy: {scores['overall_accuracy']:.1f}%")
    print(f"  ├─ Cell Detection:")
    print(f"  │    Precision: {scores['cell_precision']:.1f}% ({scores['correct_cells']}/{scores['detected_tiles']} detected are correct)")
    print(f"  │    Recall:    {scores['cell_recall']:.1f}% ({scores['correct_cells']}/{scores['expected_tiles']} expected found)")
    print(f"  │    F1 Score:  {scores['cell_f1']:.1f}%")
    print(f"  └─ OCR Accuracy: {scores['ocr_accuracy']:.1f}% ({scores['ocr_correct']}/{scores['correct_cells']} letters correct)")
    
    # False detection tracking (transient false positives)
    print(f"\n  False Detection Tracking:")
    print(f"    Tiles ever locked:       {scores['tiles_ever_locked']}")
    print(f"    Tiles removed (unlocked): {scores['tiles_removed']}")
    print(f"    Transient false positives: {scores['transient_false_positives']} (locked but shouldn't be)")
    print(f"    False detections removed:  {scores['false_detections_removed']} (wrongly locked, then correctly cleared)")
    
    if scores['false_detections_removed_list']:
        print(f"    └─ Cells: {scores['false_detections_removed_list']}")
    
    if scores['missing_cells']:
        print(f"\n  Missing cells: {scores['missing_cells']}")
    if scores['false_positives']:
        print(f"  Extra cells (still locked at end): {scores['false_positives']}")
    if scores['ocr_errors']:
        print(f"  OCR errors:")
        for (row, col), expected, detected in scores['ocr_errors']:
            print(f"    ({row},{col}): expected '{expected}', got '{detected}'")
    print(f"{'='*60}")


class TestAccuracy:
    """
    Accuracy scoring tests - these always pass but report scores.
    
    Run with: pytest tests/test_detection.py -v -s
    """
    
    def test_video_accuracy(self, video_fixture, tracker_factory):
        """
        Process video and report accuracy scores.
        
        This test always passes - it's for tracking accuracy, not enforcement.
        """
        fixture = video_fixture['data']
        video_name = video_fixture['name']
        
        tracker = tracker_factory(video_fixture['video_path'], fixture['corners'])
        result = tracker.process_headless()
        
        # Calculate scores
        scores = calculate_scores(fixture, result)
        
        # Print report
        print_score_report(video_name, scores)
        
        # Get run ID for this test session
        run_id = get_current_run_id()
        
        # Save results for tracking
        save_results({video_name: {
            'overall_accuracy': scores['overall_accuracy'],
            'cell_precision': scores['cell_precision'],
            'cell_recall': scores['cell_recall'],
            'cell_f1': scores['cell_f1'],
            'ocr_accuracy': scores['ocr_accuracy'],
            'expected_tiles': scores['expected_tiles'],
            'ocr_correct': scores['ocr_correct'],
            # New false detection tracking metrics
            'tiles_ever_locked': scores['tiles_ever_locked'],
            'tiles_removed': scores['tiles_removed'],
            'transient_false_positives': scores['transient_false_positives'],
            'false_detections_removed': scores['false_detections_removed'],
        }}, run_id=run_id)
        
        # Store scores for summary
        if not hasattr(self.__class__, '_all_scores'):
            self.__class__._all_scores = {}
        self.__class__._all_scores[video_name] = scores
        
        # Always pass - this is for scoring, not enforcement
        assert True


def test_summary_report():
    """
    Print summary of all video accuracy scores.
    
    This runs last and summarizes overall performance.
    """
    fixtures = list(FIXTURES_DIR.glob("*.json")) if FIXTURES_DIR.exists() else []
    
    if len(fixtures) == 0:
        pytest.skip("No fixtures found. Run 'python generate_fixtures.py' first.")
    
    # Load results from file
    if not RESULTS_FILE.exists():
        print("\nNo results yet - run individual tests first.")
        return
    
    with open(RESULTS_FILE, 'r') as f:
        all_results = json.load(f)
    
    print(f"\n{'='*70}")
    print("SUMMARY: Overall Accuracy Across All Videos")
    print(f"{'='*70}")
    print(f"{'Video':<15} {'Overall':>10} {'Cell F1':>10} {'OCR':>10} {'Tiles':>10}")
    print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    total_expected = 0
    total_correct = 0
    
    for fixture_path in sorted(fixtures):
        video_name = fixture_path.stem
        if video_name in all_results and all_results[video_name]:
            latest = all_results[video_name][-1]
            overall = latest.get('overall_accuracy', 0)
            cell_f1 = latest.get('cell_f1', 0)
            ocr = latest.get('ocr_accuracy', 0)
            tiles = latest.get('expected_tiles', 0)
            ocr_correct = latest.get('ocr_correct', 0)
            
            total_expected += tiles
            total_correct += ocr_correct
            
            print(f"{video_name:<15} {overall:>9.1f}% {cell_f1:>9.1f}% {ocr:>9.1f}% {tiles:>10}")
        else:
            print(f"{video_name:<15} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
    
    print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    if total_expected > 0:
        overall_pct = total_correct / total_expected * 100
        print(f"{'TOTAL':<15} {overall_pct:>9.1f}% {'-':>10} {'-':>10} {total_expected:>10}")
    
    print(f"{'='*70}")


# Individual video tests (for more specific assertions if needed)

def test_fixtures_exist():
    """Meta-test: verify we have at least one fixture to test."""
    fixtures = list(FIXTURES_DIR.glob("*.json")) if FIXTURES_DIR.exists() else []
    
    if len(fixtures) == 0:
        pytest.skip(
            "No fixtures found. Run 'python generate_fixtures.py' first to create test fixtures."
        )
    
    print(f"\nFound {len(fixtures)} fixture(s): {[f.stem for f in fixtures]}")
    assert len(fixtures) > 0


# Helper function for manual testing
def run_single_video_test(video_name):
    """
    Helper to run tests on a single video manually.
    
    Usage (from Python):
        from tests.test_detection import run_single_video_test
        run_single_video_test("test1")
    """
    fixture = load_fixture(video_name)
    if fixture is None:
        print(f"Fixture not found: {video_name}")
        return
    
    video_path = get_video_path(fixture)
    corners = parse_corners(fixture['corners'])
    
    tracker = ScrabbleTracker(video_path, corners)
    if not tracker.initialize():
        print(f"Failed to initialize tracker for {video_path}")
        return
    
    print(f"Processing {video_name}...")
    result = tracker.process_headless()
    
    # Calculate and print scores
    scores = calculate_scores(fixture, result)
    print_score_report(video_name, scores)
    
    # Print boards for comparison
    detected_board = result_to_board(result)
    
    print(f"\nExpected Board:")
    for i, row in enumerate(fixture['board']):
        print(f"  {i:2}: {row}")
    
    print(f"\nDetected Board:")
    for i, row in enumerate(detected_board):
        print(f"  {i:2}: {row}")
    
    return scores
