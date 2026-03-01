#!/usr/bin/env python3
"""
Consolidate all setup.json results from a dataset into a single xlsx.

Produces one row per venue using the MAX field_area entry, matching the
setup columns from enrich_blur_with_setup.py.

Usage:
    python consolidate_setup_results.py /path/to/dataset
    python consolidate_setup_results.py /path/to/dataset -o output.xlsx
"""

import json
import sys
import argparse
from pathlib import Path

import openpyxl
from openpyxl.styles import Font


def find_setup_json(venue_dir: Path) -> Path | None:
    """Find the first event with a setup/setup.json for the given venue."""
    for event_path in sorted(venue_dir.iterdir(), reverse=True):
        if not event_path.is_dir():
            continue
        setup_json = event_path / "setup" / "setup.json"
        if setup_json.is_file():
            return setup_json
    return None


def extract_max_setup(setup_json_path: Path) -> tuple[dict | None, bool, str, list[str]]:
    """Parse setup.json and return (max_entry, used_fallback, setup_joined_image, all_sport_types)."""
    with open(setup_json_path) as f:
        data = json.load(f)

    setup_joined_image = data.get("setup_joined_image", "")
    setups = data.get("setups", [])
    all_sport_types = [e.get("sport_type", "") for e in setups]

    if not setups:
        return None, False, setup_joined_image, all_sport_types

    for entry in setups:
        if entry.get("max_field_area") == "MAX":
            return entry, False, setup_joined_image, all_sport_types

    best = max(setups, key=lambda e: e.get("field_area", 0))
    return best, True, setup_joined_image, all_sport_types


def compute_setup_severity(all_spares) -> str:
    if all_spares == "" or all_spares is None:
        return "Aborted"
    try:
        value = float(all_spares)
    except (ValueError, TypeError):
        return "Aborted"
    if value > 3000:
        return "Error"
    if value >= 2000:
        return "Warning"
    return "Ok"


def compute_can_improve_by_zoom(left_spare, right_spare, max_camera_overlap) -> str:
    try:
        ls = float(left_spare)
        rs = float(right_spare)
        mco = float(max_camera_overlap)
    except (ValueError, TypeError):
        return ""
    if mco == 0:
        return ""
    ratio = (ls + rs) / mco / 2
    return "Yes" if 0.7 <= ratio <= 1.3 else "No"


def compute_decision(setup_severity: str, can_improve_by_zoom: str) -> str:
    if setup_severity == "Error":
        return "maintain_remote" if can_improve_by_zoom == "Yes" else "maintain_on_site"
    if setup_severity == "Warning":
        return "watch"
    if setup_severity == "Ok":
        return "ok"
    return "NA"


def main():
    parser = argparse.ArgumentParser(description="Consolidate setup.json results into xlsx")
    parser.add_argument("dataset", help="Dataset directory containing venue subdirs")
    parser.add_argument("-o", "--output", default=None, help="Output xlsx path")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_dir():
        print(f"Error: {dataset_dir} not found")
        sys.exit(1)

    output_path = Path(args.output) if args.output else dataset_dir / "setup_results.xlsx"

    headers = [
        "venue_id", "active_calibrations",
        "setup_sport_type", "num_of_cam", "max_camera_overlap", "all_spares",
        "setup_severity", "left_spare", "right_spare", "s2_mode_calc",
        "X_of_center", "Y_from_line", "height", "focals_diff", "cam_angle_diff",
        "field_area", "up_side_down",
        "can_improve_by_zoom", "decision", "setup_join_image",
    ]

    venues = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    rows = []
    matched = 0
    no_setup = 0

    for venue_dir in venues:
        venue_id = venue_dir.name
        setup_json_path = find_setup_json(venue_dir)

        if setup_json_path is None:
            no_setup += 1
            continue

        max_entry, used_fallback, setup_joined_image, all_sport_types = extract_max_setup(setup_json_path)

        if max_entry is None:
            no_setup += 1
            continue

        all_spares = max_entry.get("all_spares", "")
        left_spare = max_entry.get("left_spare", "")
        right_spare = max_entry.get("right_spare", "")
        max_camera_overlap = max_entry.get("max_camera_overlap", "")
        severity = compute_setup_severity(all_spares)
        can_improve = compute_can_improve_by_zoom(left_spare, right_spare, max_camera_overlap)
        decision = compute_decision(severity, can_improve)

        row = {
            "venue_id": venue_id,
            "active_calibrations": ",".join(all_sport_types),
            "setup_sport_type": max_entry.get("sport_type", ""),
            "num_of_cam": max_entry.get("num_of_cam", ""),
            "max_camera_overlap": max_camera_overlap,
            "all_spares": all_spares,
            "setup_severity": severity,
            "left_spare": left_spare,
            "right_spare": right_spare,
            "s2_mode_calc": max_entry.get("s2_mode_calc", ""),
            "X_of_center": max_entry.get("X_of_center", ""),
            "Y_from_line": max_entry.get("Y_from_line", ""),
            "height": max_entry.get("height", ""),
            "focals_diff": max_entry.get("focals_diff", ""),
            "cam_angle_diff": max_entry.get("cam_angle_diff", ""),
            "field_area": max_entry.get("field_area", ""),
            "up_side_down": max_entry.get("up-side-down", ""),
            "can_improve_by_zoom": can_improve,
            "decision": decision,
            "setup_join_image": str(setup_json_path.parent / "setup_join.jpg"),
        }
        rows.append(row)
        matched += 1

    # Write xlsx
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(headers)

    setup_jpg_col = headers.index("setup_join_image") + 1

    for row_idx, row in enumerate(rows, start=2):  # row 1 is header
        ws.append([row.get(h, "") for h in headers])
        # Make the setup_join_image cell a clickable file hyperlink
        cell = ws.cell(row=row_idx, column=setup_jpg_col)
        jpg_path = row.get("setup_join_image", "")
        if jpg_path and Path(jpg_path).is_file():
            cell.hyperlink = Path(jpg_path).resolve().as_uri()
            cell.value = Path(jpg_path).name
            cell.font = Font(color="0000FF", underline="single")

    wb.save(output_path)

    print(f"Matched: {matched}")
    print(f"No setup.json: {no_setup}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
