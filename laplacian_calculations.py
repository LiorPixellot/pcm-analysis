#!/usr/bin/env python3
"""
Calculate focus metrics for all focus folders in data_13 and validate against existing focus.json files.
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

import cv2


def image_focus(image_view):
    """
    Calculate focus metrics for an image region.
    Returns (mean_intensity, variance).
    """
    laplacian_image = cv2.Laplacian(image_view, cv2.CV_64F)
    mean_l, stddev = cv2.meanStdDev(laplacian_image)
    variance = stddev[0].mean() * stddev[0].mean()
    mean_i = image_view.mean()
    return mean_i, variance


def extract_region(image, side, joined_frame_size=800):
    """
    Extract 800x800 region from center of image.
    side: 'right' (CAM0) extracts left edge, 'left' (CAM1) extracts right edge.
    """
    center_y, size_x, _ = image.shape
    center_y = center_y // 2

    if side == 'right':
        # CAM0 (right camera) - extract from left side of image
        region = image[center_y - joined_frame_size // 2:center_y + joined_frame_size // 2,
                       0:joined_frame_size]
    else:
        # CAM1 (left camera) - extract from right side of image
        region = image[center_y - joined_frame_size // 2:center_y + joined_frame_size // 2,
                       size_x - joined_frame_size:size_x]
    return region


def validate_value(calculated, expected, tolerance=0.001):
    """Check if calculated value matches expected within relative tolerance."""
    if expected == 0:
        return abs(calculated) < tolerance
    return abs(calculated - expected) / abs(expected) < tolerance


def process_focus_folder(focus_folder):
    """
    Process a single focus folder and return calculated metrics.
    Returns dict with metrics or None if images are missing.
    """
    cam0_path = focus_folder / 'CAM0_1.jpg'
    cam1_path = focus_folder / 'CAM1_1.jpg'
    if not cam0_path.exists() or not cam1_path.exists():
        cam0_path = focus_folder / 'CAM0_1_rot.jpg'
        cam1_path = focus_folder / 'CAM1_1_rot.jpg'
    if not cam0_path.exists() or not cam1_path.exists():
        return None

    # Load images
    img_right = cv2.imread(str(cam0_path))
    img_left = cv2.imread(str(cam1_path))

    if img_right is None or img_left is None:
        return None

    # Extract center regions
    region_right = extract_region(img_right, 'right')
    region_left = extract_region(img_left, 'left')

    # Calculate focus metrics
    right_mean, right_mid_focus = image_focus(region_right)
    left_mean, left_mid_focus = image_focus(region_left)

    # Calculate relative difference
    focus_abs_dif = abs(right_mid_focus - left_mid_focus)
    focus_abs_dif_rel = focus_abs_dif / (right_mid_focus + left_mid_focus) * 2

    return {
        'focus_right_mean': right_mean,
        'focus_left_mean': left_mean,
        'focus_right_mid': right_mid_focus,
        'focus_left_mid': left_mid_focus,
        'focus_abs_dif_rel': focus_abs_dif_rel
    }


def validate_against_json(calculated, focus_json_path, tolerance=0.001):
    """
    Validate calculated values against focus.json.
    Returns (status, details) tuple.
    """
    if not focus_json_path.exists():
        return 'NO_JSON', 'focus.json not found'

    try:
        with open(focus_json_path) as f:
            expected = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return 'JSON_ERROR', str(e)

    mismatches = []
    fields = ['focus_right_mean', 'focus_left_mean', 'focus_right_mid', 'focus_left_mid', 'focus_abs_dif_rel']

    for field in fields:
        if field not in expected:
            mismatches.append(f'{field}: missing in JSON')
            continue

        calc_val = calculated[field]
        exp_val = expected[field]

        if not validate_value(calc_val, exp_val, tolerance):
            diff_pct = abs(calc_val - exp_val) / abs(exp_val) * 100 if exp_val != 0 else float('inf')
            mismatches.append(f'{field}: calc={calc_val:.6f} exp={exp_val:.6f} ({diff_pct:.2f}% diff)')

    if mismatches:
        return 'FAIL', '; '.join(mismatches)
    return 'PASS', ''


def process_entry(venue_id, event_id, focus_folder, tolerance):
    """
    Worker function: process a single entry and return the result dict.
    Runs in a separate process.
    """
    focus_folder = Path(focus_folder)
    metrics = process_focus_folder(focus_folder)
    if metrics is None:
        return {'venue_id': venue_id, 'event_id': event_id, 'skipped': True}

    focus_json_path = focus_folder / 'focus.json'
    status, details = validate_against_json(metrics, focus_json_path, tolerance)

    joined_path = focus_folder / 'joined_0_1.jpg'
    joined_image = f'=HYPERLINK("file:///{joined_path.resolve()}")' if joined_path.exists() else ''

    return {
        'venue_id': venue_id,
        'event_id': event_id,
        'skipped': False,
        'focus_right_mean': metrics['focus_right_mean'],
        'focus_left_mean': metrics['focus_left_mean'],
        'focus_right_mid': metrics['focus_right_mid'],
        'focus_left_mid': metrics['focus_left_mid'],
        'focus_abs_dif_rel': metrics['focus_abs_dif_rel'],
        'validation_status': status,
        'validation_details': details,
        'joined_image': joined_image,
    }


def find_focus_folders(data_dir):
    """
    Find all focus folders under data_dir following structure:
    data_dir/{venue_id}/{event_id}/focus/
    """
    data_path = Path(data_dir)
    for venue_dir in sorted(data_path.iterdir()):
        if not venue_dir.is_dir():
            continue
        for event_dir in sorted(venue_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            focus_folder = event_dir / 'focus'
            if focus_folder.exists() and focus_folder.is_dir():
                yield venue_dir.name, event_dir.name, focus_folder


def main():
    parser = argparse.ArgumentParser(
        description='Calculate focus metrics and validate against focus.json files'
    )
    parser.add_argument('data_dir', help='Path to data directory (e.g., data_13)')
    parser.add_argument('-o', '--output', default='laplacian_calculations.csv',
                        help='Output CSV file (default: laplacian_calculations.csv)')
    parser.add_argument('-t', '--tolerance', type=float, default=0.001,
                        help='Tolerance for validation (default: 0.001)')
    parser.add_argument('-w', '--workers', type=int, default=cpu_count(),
                        help='Number of parallel workers (default: number of CPUs)')
    parser.add_argument('--output-dir', default=None,
                        help='Parent output directory (creates laplacian/ subdir). If omitted, creates timestamped dir.')
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f'Error: {args.data_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    # Create output directory
    script_dir = Path(__file__).parent.resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir) / "laplacian"
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_dir = script_dir / "output_dir" / f"laplacian_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output CSV path: use user-provided path or default inside output_dir
    if args.output != 'laplacian_calculations.csv':
        output_path = args.output
    else:
        output_path = str(output_dir / "laplacian_calculations.csv")

    # CSV columns
    fieldnames = [
        'venue_id',
        'event_id',
        'focus_right_mean',
        'focus_left_mean',
        'focus_right_mid',
        'focus_left_mid',
        'focus_abs_dif_rel',
        'validation_status',
        'validation_details',
        'joined_image'
    ]

    # Collect all focus folders first for progress tracking
    print('Scanning for focus folders...')
    all_folders = list(find_focus_folders(args.data_dir))
    total = len(all_folders)
    print(f'Found {total} focus folders to process.')

    processed = 0
    skipped = 0
    passed = 0
    failed = 0
    done = 0

    print(f'Using {args.workers} workers.')

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_entry, venue_id, event_id, str(focus_folder), args.tolerance): (venue_id, event_id)
                for venue_id, event_id, focus_folder in all_folders
            }

            for future in as_completed(futures):
                done += 1
                if done % 100 == 0 or done == total:
                    print(f'  Completed {done}/{total} ({done * 100 // total}%)', flush=True)

                result = future.result()
                if result['skipped']:
                    skipped += 1
                    continue

                row = {
                    'venue_id': result['venue_id'],
                    'event_id': result['event_id'],
                    'focus_right_mean': result['focus_right_mean'],
                    'focus_left_mean': result['focus_left_mean'],
                    'focus_right_mid': result['focus_right_mid'],
                    'focus_left_mid': result['focus_left_mid'],
                    'focus_abs_dif_rel': result['focus_abs_dif_rel'],
                    'validation_status': result['validation_status'],
                    'validation_details': result['validation_details'],
                    'joined_image': result['joined_image'],
                }
                writer.writerow(row)
                processed += 1

                if result['validation_status'] == 'PASS':
                    passed += 1
                elif result['validation_status'] == 'FAIL':
                    failed += 1

    print(f'Processed: {processed}')
    print(f'Skipped (missing images): {skipped}')
    print(f'Passed validation: {passed}')
    print(f'Failed validation: {failed}')
    print(f'Output dir: {output_dir}')
    print(f'Output written to: {output_path}')


if __name__ == '__main__':
    main()
