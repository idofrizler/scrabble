#!/usr/bin/env python3
"""
Fixture Generator for Scrabble Tracker Tests

This script runs each test video, allows the user to mark corners (if not saved),
and saves the detected tiles and letters to JSON fixture files.

Usage:
    python generate_fixtures.py                    # Process all videos
    python generate_fixtures.py videos/test1.mp4  # Process specific video
    python generate_fixtures.py --reprocess       # Reprocess all, ignoring saved corners
"""

import argparse
import json
import os
import sys
from pathlib import Path

from main import ScrabbleTracker


def get_fixture_path(video_path):
    """Get the fixture JSON path for a video file."""
    video_name = Path(video_path).stem  # e.g., "test1"
    fixture_dir = Path(__file__).parent / "tests" / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    return fixture_dir / f"{video_name}.json"


def load_existing_fixture(fixture_path):
    """Load existing fixture if it exists."""
    if fixture_path.exists():
        with open(fixture_path, 'r') as f:
            return json.load(f)
    return None


def save_fixture(fixture_path, data):
    """Save fixture data to JSON file."""
    with open(fixture_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved fixture to {fixture_path}")


def process_video(video_path, reprocess=False):
    """
    Process a video and generate/update its fixture file.
    
    Args:
        video_path: Path to the video file
        reprocess: If True, ignore saved corners and prompt for new ones
        
    Returns:
        dict with fixture data
    """
    fixture_path = get_fixture_path(video_path)
    existing = load_existing_fixture(fixture_path)
    
    # Use saved corners if available and not reprocessing
    manual_corners = None
    if existing and 'corners' in existing and not reprocess:
        coords = existing['corners']
        manual_corners = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        print(f"Using saved corners: {coords}")
    else:
        print("No saved corners - you'll need to click the 4 corners (TL, TR, BR, BL)")
    
    # Create tracker
    tracker = ScrabbleTracker(video_path, manual_corners)
    
    if not tracker.initialize():
        print(f"Error: Failed to initialize tracker for {video_path}")
        return None
    
    # Process video headlessly with progress updates
    print(f"Processing {video_path}...")
    
    def progress_callback(frame_num, total_frames):
        if frame_num % 100 == 0 or frame_num == total_frames:
            pct = (frame_num / total_frames) * 100
            print(f"  Progress: {frame_num}/{total_frames} ({pct:.1f}%)")
    
    result = tracker.process_headless(progress_callback)
    
    # Convert results to 2D board format (much easier to read/edit)
    # Each row is a string of 15 characters, '.' for empty cells
    board_2d = []
    for row in range(15):
        row_str = ""
        for col in range(15):
            cell_data = result['board_state'][row][col]
            if cell_data and (row, col) in result['locked_cells']:
                row_str += cell_data[0]  # letter
            else:
                row_str += "."
        board_2d.append(row_str)
    
    fixture_data = {
        'video': video_path,
        'corners': result['corners'],
        'board': board_2d
    }
    
    # Save fixture
    save_fixture(fixture_path, fixture_data)
    
    # Count tiles
    tile_count = sum(1 for row in board_2d for c in row if c != '.')
    
    # Print summary
    print(f"\nResults for {video_path}:")
    print(f"  Corners: {result['corners']}")
    print(f"  Detected tiles: {tile_count}")
    
    # Print 2D board visualization
    print("\n  Board (15x15):")
    print("      " + "".join(f"{i:3}" for i in range(15)))
    print("     +" + "-" * 46 + "+")
    
    for row_idx, row_str in enumerate(board_2d):
        row_display = "  ".join(c for c in row_str)
        print(f"  {row_idx:2} |  {row_display}  |")
    
    print("     +" + "-" * 46 + "+")
    
    return fixture_data


def find_all_videos():
    """Find all test videos in the videos directory."""
    videos_dir = Path(__file__).parent / "videos"
    if not videos_dir.exists():
        return []
    
    videos = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        videos.extend(videos_dir.glob(ext))
    
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(description='Generate test fixtures from videos')
    parser.add_argument('video', nargs='?', help='Specific video to process (optional)')
    parser.add_argument('--reprocess', action='store_true', 
                        help='Reprocess all videos, ignoring saved corners')
    parser.add_argument('--list', action='store_true',
                        help='List all videos and their fixture status')
    
    args = parser.parse_args()
    
    if args.list:
        videos = find_all_videos()
        print(f"Found {len(videos)} videos:")
        for v in videos:
            fixture_path = get_fixture_path(str(v))
            status = "✓ fixture exists" if fixture_path.exists() else "✗ no fixture"
            print(f"  {v.name}: {status}")
        return
    
    if args.video:
        # Process single video
        if not os.path.exists(args.video):
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)
        process_video(args.video, args.reprocess)
    else:
        # Process all videos
        videos = find_all_videos()
        if not videos:
            print("No videos found in videos/ directory")
            sys.exit(1)
        
        print(f"Processing {len(videos)} videos...")
        for i, video in enumerate(videos, 1):
            print(f"\n=== Video {i}/{len(videos)}: {video.name} ===")
            process_video(str(video), args.reprocess)
        
        print("\n=== All videos processed ===")


if __name__ == "__main__":
    main()
