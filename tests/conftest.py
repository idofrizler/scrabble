"""
Pytest configuration and fixtures for Scrabble Tracker tests.
"""

import json
import sys
from pathlib import Path

import pytest

# Add parent directory to path so we can import main
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import ScrabbleTracker


FIXTURES_DIR = Path(__file__).parent / "fixtures"
VIDEOS_DIR = Path(__file__).parent.parent / "videos"


def get_available_fixtures():
    """Get list of available fixture files."""
    if not FIXTURES_DIR.exists():
        return []
    return list(FIXTURES_DIR.glob("*.json"))


def load_fixture(fixture_name):
    """
    Load a fixture by name (without .json extension).
    
    Returns:
        dict with fixture data or None if not found
    """
    fixture_path = FIXTURES_DIR / f"{fixture_name}.json"
    if not fixture_path.exists():
        return None
    
    with open(fixture_path, 'r') as f:
        return json.load(f)


def get_video_path(fixture_data):
    """Get the video path from fixture data, handling relative paths."""
    video_path = fixture_data.get('video', '')
    
    # Handle relative paths
    if not Path(video_path).is_absolute():
        video_path = Path(__file__).parent.parent / video_path
    
    return str(video_path)


def parse_corners(corners_list):
    """Convert corners list [x1,y1,x2,y2,...] to [(x1,y1), (x2,y2), ...]."""
    return [(corners_list[i], corners_list[i+1]) for i in range(0, len(corners_list), 2)]


@pytest.fixture
def fixture_loader():
    """Fixture that provides a function to load test fixtures."""
    return load_fixture


@pytest.fixture(params=[f.stem for f in get_available_fixtures()])
def video_fixture(request):
    """
    Parametrized fixture that yields each available video fixture.
    
    This automatically creates a test for each fixture file found.
    """
    fixture_data = load_fixture(request.param)
    if fixture_data is None:
        pytest.skip(f"Fixture {request.param} not found")
    
    video_path = get_video_path(fixture_data)
    if not Path(video_path).exists():
        pytest.skip(f"Video not found: {video_path}")
    
    return {
        'name': request.param,
        'data': fixture_data,
        'video_path': video_path
    }


@pytest.fixture
def tracker_factory():
    """
    Fixture that provides a factory function to create configured trackers.
    
    Usage:
        def test_something(tracker_factory):
            tracker = tracker_factory("videos/test1.mp4", corners=[...])
            result = tracker.process_headless()
    """
    def factory(video_path, corners=None):
        manual_corners = None
        if corners:
            manual_corners = parse_corners(corners)
        
        tracker = ScrabbleTracker(video_path, manual_corners)
        # Initialize OCR service (required for process_headless)
        tracker.set_ocr_backend(use_tesseract=False)
        
        if not tracker.initialize():
            raise RuntimeError(f"Failed to initialize tracker for {video_path}")
        
        return tracker
    
    return factory
