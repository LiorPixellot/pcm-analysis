#!/usr/bin/env python3
"""
Shared camera discovery utilities for multi-camera (2 or 3 camera) setups.

Camera naming conventions:
  - S2 (2-cam): CAM0_1.jpg, CAM1_1.jpg (or _1_rot.jpg variants)
  - S3 (3-cam): CAM0_0.jpg, CAM1_0.jpg, CAM2_0.jpg (or _0_rot.jpg variants)

Camera ordering: CAM0=right, CAM1=center (3-cam) or left (2-cam), CAM2=left (3-cam only).
"""

from pathlib import Path
from typing import Optional


# Suffixes to try, in order of preference
_CAM_SUFFIXES = ['_1.jpg', '_1_rot.jpg', '_0.jpg', '_0_rot.jpg']


def find_cam_image(directory: Path, cam: str) -> Optional[Path]:
    """Find a camera image file in directory, trying multiple suffix fallbacks.

    Args:
        directory: Focus directory to search in.
        cam: Camera prefix, e.g. 'CAM0', 'CAM1', 'CAM2'.

    Returns:
        Path to the first existing image file, or None if not found.
    """
    for suffix in _CAM_SUFFIXES:
        path = directory / f'{cam}{suffix}'
        if path.exists():
            return path
    return None


def detect_camera_count(focus_dir: Path) -> int:
    """Detect whether a focus directory has 2 or 3 cameras.

    Returns 3 if CAM2 is found, otherwise 2.
    """
    if find_cam_image(focus_dir, 'CAM2') is not None:
        return 3
    return 2


def detect_dataset_camera_count(data_dir: Path) -> int:
    """Check the first venue with images to determine if the dataset is 2-cam or 3-cam.

    Returns 3 if any venue has CAM2, otherwise 2.
    """
    data_path = Path(data_dir)
    for venue_dir in sorted(data_path.iterdir()):
        if not venue_dir.is_dir():
            continue
        for event_dir in sorted(venue_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            focus_dir = event_dir / 'focus'
            if focus_dir.is_dir() and find_cam_image(focus_dir, 'CAM0') is not None:
                return detect_camera_count(focus_dir)
    return 2
