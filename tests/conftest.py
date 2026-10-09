"""Pytest configuration: ensure Review 3 is on sys.path for test discovery."""

import sys
from pathlib import Path

review3_dir = Path(__file__).resolve().parent.parent / "Review 3"
if str(review3_dir) not in sys.path:
    sys.path.insert(0, str(review3_dir))
