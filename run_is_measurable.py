#!/usr/bin/env python3
"""
Indoor/Outdoor Measurability Router

Reads PQS_blur_by_venue.xlsx to classify venues as indoor/outdoor based on
ACTIVE_CALIBRATIONS, then routes each venue to the appropriate Gemini-based
measurability analyzer (indoor or outdoor).

Usage:
    python run_is_measurable.py --dataset all_data_02_09
    python run_is_measurable.py --dataset all_data_02_09 --limit 10 --max-workers 5
"""

import os
import csv
import argparse
import traceback
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Set Google Cloud credentials if not already set
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    script_dir = Path(__file__).parent.resolve()
    credentials_path = script_dir / "service-account-key.json"
    if credentials_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

import openpyxl
from google import genai
from google.genai import types

from is_measurable import (
    load_config,
    load_venues,
    load_model_pricing,
    calculate_pricing,
    get_token_usage,
    load_images,
)
from is_measurable_indoor import analyze_indoor, load_examples as load_indoor_examples
from is_measurable_outdoor import analyze_outdoor, load_examples as load_outdoor_examples


# Calibration types that indicate an indoor venue (case-insensitive)
INDOOR_CALIBRATIONS = {
    "basketball", "waterpolo", "volleyball", "wrestling",
}


def classify_venue(active_calibrations: str) -> str:
    """Classify a venue as indoor or outdoor based on ACTIVE_CALIBRATIONS.

    A venue is indoor if at least one calibration type matches the indoor set.
    Otherwise outdoor.
    """
    if not active_calibrations or not active_calibrations.strip():
        return "outdoor"

    calibrations = [c.strip().lower() for c in active_calibrations.split(",")]
    for cal in calibrations:
        if cal in INDOOR_CALIBRATIONS:
            return "indoor"
    return "outdoor"


def read_blur_xlsx(xlsx_path: Path) -> list:
    """Read PQS_blur_by_venue.xlsx and return list of row dicts.

    Each dict has keys matching column headers.
    """
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        wb.close()
        return []

    headers = [str(h).strip() if h is not None else "" for h in rows[0]]
    data = []
    for row in rows[1:]:
        row_dict = {}
        for i, val in enumerate(row):
            if i < len(headers):
                row_dict[headers[i]] = val if val is not None else ""
        data.append(row_dict)

    wb.close()
    return data


def find_venue_focus_dir(dataset_dir: Path, venue_id: str) -> Optional[Path]:
    """Find the first event with a focus/ directory for the given venue.

    Returns the focus_dir path or None if not found.
    """
    venue_dir = dataset_dir / str(venue_id)
    if not venue_dir.is_dir():
        return None

    for event_path in sorted(venue_dir.iterdir()):
        if not event_path.is_dir():
            continue
        focus_dir = event_path / "focus"
        if focus_dir.is_dir():
            # Check that at least CAM0 and CAM1 exist (standard or _rot variants)
            if (focus_dir / "CAM0_1.jpg").exists() and (focus_dir / "CAM1_1.jpg").exists():
                return focus_dir
            if (focus_dir / "CAM0_1_rot.jpg").exists() and (focus_dir / "CAM1_1_rot.jpg").exists():
                return focus_dir

    return None


def process_venue(index: int, venue_row: dict, venue_type: str,
                  focus_dir: Path, client: genai.Client, model_name: str,
                  media_resolution: Optional[str] = None,
                  indoor_examples: Optional[dict] = None,
                  outdoor_examples: Optional[dict] = None) -> dict:
    """Load images and call the appropriate analyzer. Used as parallel worker."""
    venue_id = str(venue_row.get("PIXELLOT_VENUE_ID", ""))

    images = load_images(focus_dir)
    if images is None:
        return {
            "index": index,
            "is_measurable": "N/A",
            "reason": "Could not load images",
            "token_usage": None,
            "processed": False,
            "skipped": True,
            "skip_reason": "could not load images",
        }

    cam0_bytes, cam1_bytes, _ = images

    try:
        if venue_type == "indoor":
            result, token_usage = analyze_indoor(
                client, model_name, cam0_bytes, cam1_bytes, None, media_resolution,
                indoor_examples,
            )
        else:
            result, token_usage = analyze_outdoor(
                client, model_name, cam0_bytes, cam1_bytes, None, media_resolution,
                outdoor_examples,
            )

        is_measurable = result.get("is_measurable", "Unknown")
        reason = result.get("reason", "")
        reason = reason.replace(",", ";") if reason else ""

        return {
            "index": index,
            "is_measurable": is_measurable,
            "reason": reason,
            "token_usage": token_usage,
            "processed": True,
            "skipped": False,
        }

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "index": index,
            "is_measurable": "Error",
            "reason": str(e),
            "token_usage": None,
            "processed": False,
            "skipped": True,
            "error": str(e),
            "traceback": tb,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Route venues to indoor/outdoor measurability analysis based on calibration type"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Data directory path (e.g. all_data_02_09)")
    parser.add_argument("--blur-xlsx", type=str, default="PQS_blur_by_venue.xlsx", help="Path to PQS blur xlsx (default: PQS_blur_by_venue.xlsx)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: auto in output_dir/)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of venues to process")
    parser.add_argument("--max-workers", type=int, default=None, help="Max parallel Gemini API calls (default: from config.yaml or 10)")
    parser.add_argument("--venues", type=str, default=None, help="YAML file with venue filter (default: venues.yaml if exists)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Parent output directory (creates run_is_measurable/ subdir). If omitted, creates timestamped dir.")
    args = parser.parse_args()

    # Load config
    config = load_config()
    model_name = config.get("model", "gemini-3-flash-preview")
    project = config.get("project", "pixellot-ai")
    location = config.get("location", "global")
    max_workers = args.max_workers if args.max_workers else config.get("max_workers", 10)
    media_resolution = config.get("media_resolution")

    # Load venue filter
    is_full, venue_filter = load_venues(args.venues)

    script_dir = Path(__file__).parent.resolve()

    # Read xlsx
    xlsx_path = Path(args.blur_xlsx) if Path(args.blur_xlsx).is_absolute() else script_dir / args.blur_xlsx
    if not xlsx_path.exists():
        print(f"  Error: xlsx file not found: {xlsx_path}")
        return

    venue_rows = read_blur_xlsx(xlsx_path)
    print(f"  Loaded {len(venue_rows)} venues from {xlsx_path.name}")

    # Dataset directory
    dataset_dir = Path(args.dataset) if Path(args.dataset).is_absolute() else script_dir / args.dataset
    if not dataset_dir.is_dir():
        print(f"  Error: dataset directory not found: {dataset_dir}")
        return

    # Classify venues and find their focus dirs
    work_items = []  # (index_in_venue_rows, venue_row, venue_type, focus_dir_or_None)

    for i, row in enumerate(venue_rows):
        venue_id = str(row.get("PIXELLOT_VENUE_ID", "")).strip()
        if not venue_id:
            continue

        # Apply venue filter
        if not is_full and venue_filter and venue_id not in venue_filter:
            continue

        active_cals = str(row.get("ACTIVE_CALIBRATIONS", ""))
        venue_type = classify_venue(active_cals)
        focus_dir = find_venue_focus_dir(dataset_dir, venue_id)

        work_items.append((i, row, venue_type, focus_dir))

    if args.limit:
        work_items = work_items[:args.limit]

    # Count classifications
    indoor_count = sum(1 for _, _, vt, _ in work_items if vt == "indoor")
    outdoor_count = sum(1 for _, _, vt, _ in work_items if vt == "outdoor")
    found_count = sum(1 for _, _, _, fd in work_items if fd is not None)
    not_found_count = len(work_items) - found_count

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir) / "run_is_measurable"
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_dir = script_dir / "output_dir" / f"run_is_measurable_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output) if Path(args.output).is_absolute() else script_dir / args.output
    else:
        output_path = output_dir / "run_is_measurable.csv"

    cost_path = output_dir / "cost.txt"

    print("=" * 60)
    print("  RUN IS MEASURABLE — INDOOR/OUTDOOR ROUTER")
    print("=" * 60)
    print(f"  Output dir: {output_dir}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  XLSX: {xlsx_path.name}")
    print(f"  Model: {model_name}")
    print(f"  Total venues: {len(work_items)}")
    print(f"    Indoor: {indoor_count}")
    print(f"    Outdoor: {outdoor_count}")
    print(f"    Found in dataset: {found_count}")
    print(f"    Not in dataset: {not_found_count}")
    if is_full:
        print("  Venue filter: ALL (full mode)")
    else:
        print(f"  Venue filter: {len(venue_filter)} filtered")
    print(f"  Max workers: {max_workers}")
    if media_resolution:
        print(f"  Media resolution: {media_resolution}")
    if args.limit:
        print(f"  Limit: {args.limit}")

    if not work_items:
        print("  No venues to process. Exiting.")
        return

    # Initialize Gemini client
    retry_attempts = config.get("retry_attempts", 10)
    retry_initial_delay = config.get("retry_initial_delay", 2.0)
    retry_max_delay = config.get("retry_max_delay", 16.0)

    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=retry_attempts,
                initial_delay=retry_initial_delay,
                max_delay=retry_max_delay,
            )
        ),
    )

    # Load few-shot examples for both analyzers
    indoor_examples = load_indoor_examples()
    outdoor_examples = load_outdoor_examples()
    indoor_ex_count = sum(len(imgs) for imgs in indoor_examples.values())
    outdoor_ex_count = sum(len(imgs) for imgs in outdoor_examples.values())
    print(f"  Indoor examples: {indoor_ex_count} images from {len(indoor_examples)} categories")
    for category, imgs in indoor_examples.items():
        print(f"    {category}: {len(imgs)} images")
    print(f"  Outdoor examples: {outdoor_ex_count} images from {len(outdoor_examples)} categories")
    for category, imgs in outdoor_examples.items():
        print(f"    {category}: {len(imgs)} images")
    print("=" * 60)

    # Determine xlsx column names for output
    sample_row = venue_rows[0] if venue_rows else {}
    xlsx_columns = [k for k in sample_row.keys() if k]

    # Output fieldnames: all xlsx columns + our additions
    extra_columns = ["venue_id", "venue_type", "is_measurable", "reason"]
    fieldnames = xlsx_columns + extra_columns

    # Track totals
    total_input_tokens = 0
    total_output_tokens = 0
    total_thinking_tokens = 0
    processed_count = 0
    skipped_count = 0

    model_pricing = load_model_pricing()

    # Prepare results array (same order as work_items)
    total_items = len(work_items)
    results = [None] * total_items
    completed = 0

    # Separate items: those with focus dirs (need Gemini) vs those without
    gemini_items = []  # (work_index, venue_row, venue_type, focus_dir)
    for wi, (orig_idx, venue_row, venue_type, focus_dir) in enumerate(work_items):
        if focus_dir is None:
            # No data in dataset — mark as N/A immediately
            result_row = {k: venue_row.get(k, "") for k in xlsx_columns}
            result_row["venue_id"] = str(venue_row.get("PIXELLOT_VENUE_ID", ""))
            result_row["venue_type"] = venue_type
            result_row["is_measurable"] = "N/A"
            result_row["reason"] = "Venue not found in dataset"
            results[wi] = result_row
            skipped_count += 1
            completed += 1
            venue_id = str(venue_row.get("PIXELLOT_VENUE_ID", ""))
            print(f"[{completed}/{total_items}] {venue_id} ({venue_type}) — N/A: not in dataset")
        else:
            gemini_items.append((wi, venue_row, venue_type, focus_dir))

    # Process venues with images in parallel
    if gemini_items:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_wi = {}
            for wi, venue_row, venue_type, focus_dir in gemini_items:
                future = executor.submit(
                    process_venue, wi, venue_row, venue_type, focus_dir,
                    client, model_name, media_resolution,
                    indoor_examples, outdoor_examples,
                )
                future_to_wi[future] = (wi, venue_row, venue_type)

            for future in concurrent.futures.as_completed(future_to_wi):
                wi, venue_row, venue_type = future_to_wi[future]
                res = future.result()
                completed += 1

                venue_id = str(venue_row.get("PIXELLOT_VENUE_ID", ""))

                result_row = {k: venue_row.get(k, "") for k in xlsx_columns}
                result_row["venue_id"] = str(venue_row.get("PIXELLOT_VENUE_ID", ""))
                result_row["venue_type"] = venue_type
                result_row["is_measurable"] = res["is_measurable"]
                result_row["reason"] = res.get("reason", "")
                results[wi] = result_row

                if res.get("skipped"):
                    reason = res.get("skip_reason") or res.get("error", "unknown")
                    print(f"[{completed}/{total_items}] {venue_id} ({venue_type}) — skipped: {reason}")
                    if res.get("traceback"):
                        print(res["traceback"])
                    skipped_count += 1
                elif res["processed"]:
                    token_usage = res["token_usage"]
                    total_input_tokens += token_usage["input_tokens"]
                    total_output_tokens += token_usage["output_tokens"]
                    total_thinking_tokens += token_usage["thinking_tokens"]
                    processed_count += 1

                    print(f"[{completed}/{total_items}] {venue_id} ({venue_type})"
                          f" — measurable: {res['is_measurable']}"
                          f", tokens: {token_usage['total_tokens']:,}")

                if completed % 100 == 0:
                    pricing = calculate_pricing(model_name, total_input_tokens, total_output_tokens + total_thinking_tokens, model_pricing)
                    print(f"  --- Cost after {completed} venues: ${pricing['total_price']:.6f} ---")

    # Write output CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            if row is not None:
                writer.writerow(row)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Output dir: {output_dir}")
    print(f"  Total venues: {total_items}")
    print(f"    Indoor: {indoor_count}")
    print(f"    Outdoor: {outdoor_count}")
    print(f"  Processed (Gemini): {processed_count}")
    print(f"  Skipped/N/A: {skipped_count}")
    print(f"  Output: {output_path}")

    # Calculate and display cost
    pricing = calculate_pricing(
        model_name,
        total_input_tokens,
        total_output_tokens + total_thinking_tokens,
        model_pricing,
    )

    print("\n  TOKEN USAGE:")
    print(f"    Input tokens:    {total_input_tokens:,}")
    print(f"    Thinking tokens: {total_thinking_tokens:,}")
    print(f"    Output tokens:   {total_output_tokens:,}")
    print(f"    Total tokens:    {total_input_tokens + total_output_tokens + total_thinking_tokens:,}")

    print("\n  COST:")
    print(f"    Input cost:  ${pricing['input_price']:.6f}")
    print(f"    Output cost: ${pricing['output_price']:.6f}")
    print(f"    Total cost:  ${pricing['total_price']:.6f}")

    # Write cost report
    with open(cost_path, 'w') as f:
        f.write("RUN IS MEASURABLE - INDOOR/OUTDOOR ROUTER - COST REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_dir}\n")
        f.write(f"XLSX: {xlsx_path.name}\n")
        f.write(f"Total venues: {total_items}\n")
        f.write(f"  Indoor: {indoor_count}\n")
        f.write(f"  Outdoor: {outdoor_count}\n")
        f.write(f"Processed (Gemini): {processed_count} venues\n")
        f.write(f"Skipped/N/A: {skipped_count} venues\n\n")
        f.write("TOKEN USAGE:\n")
        f.write(f"  Input tokens:    {total_input_tokens:,}\n")
        f.write(f"  Thinking tokens: {total_thinking_tokens:,}\n")
        f.write(f"  Output tokens:   {total_output_tokens:,}\n")
        f.write(f"  Total tokens:    {total_input_tokens + total_output_tokens + total_thinking_tokens:,}\n\n")
        f.write("COST:\n")
        f.write(f"  Input cost:  ${pricing['input_price']:.6f}\n")
        f.write(f"  Output cost: ${pricing['output_price']:.6f}\n")
        f.write(f"  Total cost:  ${pricing['total_price']:.6f}\n")

    print(f"\n  Cost report: {cost_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
