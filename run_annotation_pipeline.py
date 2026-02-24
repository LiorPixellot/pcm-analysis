#!/usr/bin/env python3
"""
Annotation Pipeline Runner

Iterates complete_annotation.csv, runs the Gemini AI analysis (measurability +
issue detection) and Laplacian severity per row, writes a predictions CSV for
evaluation by confusion_metrics.py.
"""

import os
import csv
import argparse
import traceback
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Set Google Cloud credentials if not already set
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    script_dir = Path(__file__).parent.resolve()
    credentials_path = script_dir / "service-account-key.json"
    if credentials_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

from google import genai
from google.genai import types

from is_measurable import (
    load_config,
    load_model_pricing,
    calculate_pricing,
    get_token_usage,
    load_images,
)
from is_measurable_indoor import (
    analyze_indoor,
    load_examples as load_indoor_examples,
)
from is_measurable_outdoor import (
    analyze_outdoor,
    load_examples as load_outdoor_examples,
)
from detect_issues import (
    analyze_venue,
    load_examples as load_issues_examples,
)
from laplacian_calculations import process_focus_folder
from laplacian_th_calculations import calc_focus_severity


# Calibration types indicating indoor (same as run_is_measurable.py)
INDOOR_CALIBRATIONS = {
    "basketball", "waterpolo", "volleyball", "wrestling",
}

OUTPUT_FIELDNAMES = [
    # Identity
    "System ID", "Checkup date", "Calibration", "Indoor / Outdoor", "source_sheet",
    # Ground truth (normalized)
    "gt_measurable", "gt_has_issue", "gt_issue_type", "gt_severity", "gt_which_camera",
    # Predictions - measurability
    "pred_measurable", "pred_measurable_reason",
    "pred_has_quality_issue", "pred_quality_issue_type", "pred_which_camera",
    # Predictions - issue detection
    "pred_has_issue", "pred_issue_type", "pred_observation",
    # Predictions - severity
    "pred_severity",
    # Laplacian metrics
    "focus_right_mean", "focus_left_mean", "focus_right_mid", "focus_left_mid",
    "focus_abs_dif_rel",
    # Image paths
    "cam0_image", "cam1_image", "joined_image",
    # Cost tracking
    "measurable_tokens", "issues_tokens",
]


def normalize_gt(row: dict) -> dict:
    """Normalize ground truth columns from annotation CSV."""
    measurable = str(row.get("Is Measurable Focus Image (Yes/No)", "")).strip().lower()
    has_issue = str(row.get("Has Quality Issue (Yes/No)", "")).strip().lower()
    issue_type = str(row.get("Quality Issue Type", "")).strip().lower()
    if issue_type in ("-", "", "none"):
        issue_type = "none"
    severity = str(row.get("Focus severity", "")).strip().upper()
    which_camera = str(row.get("which camera (left/right/none/both)", "")).strip().lower()
    if which_camera in ("-", ""):
        which_camera = "none"

    return {
        "gt_measurable": measurable,
        "gt_has_issue": has_issue,
        "gt_issue_type": issue_type,
        "gt_severity": severity,
        "gt_which_camera": which_camera,
    }


def is_indoor(row: dict) -> bool:
    """Determine if venue is indoor based on Indoor / Outdoor column, falling back to Calibration."""
    io_val = str(row.get("Indoor / Outdoor", "")).strip().lower()
    if io_val == "indoor":
        return True
    if io_val == "outdoor":
        return False
    # Fallback to calibration type
    calib = str(row.get("Calibration", "")).strip().lower()
    return calib in INDOOR_CALIBRATIONS


def process_row(index: int, row: dict,
                client: genai.Client, model_name: str,
                indoor_examples: Optional[Dict],
                outdoor_examples: Optional[Dict],
                issues_examples: Optional[Dict],
                media_resolution: Optional[str],
                skip_gemini: bool) -> dict:
    """Process a single annotation row. Returns result dict."""
    cam0_path = row.get("cam0_image", "").strip()
    cam1_path = row.get("cam1_image", "").strip()
    joined_path = row.get("joined_image", "").strip()

    focus_dir = Path(cam0_path).parent if cam0_path else None

    gt = normalize_gt(row)

    result = {
        "System ID": row.get("System ID", ""),
        "Checkup date": row.get("Checkup date", ""),
        "Calibration": row.get("Calibration", ""),
        "Indoor / Outdoor": row.get("Indoor / Outdoor", ""),
        "source_sheet": row.get("source_sheet", ""),
        **gt,
        "pred_measurable": "", "pred_measurable_reason": "",
        "pred_has_quality_issue": "", "pred_quality_issue_type": "", "pred_which_camera": "",
        "pred_has_issue": "", "pred_issue_type": "", "pred_observation": "",
        "pred_severity": "",
        "focus_right_mean": "", "focus_left_mean": "",
        "focus_right_mid": "", "focus_left_mid": "",
        "focus_abs_dif_rel": "",
        "cam0_image": cam0_path,
        "cam1_image": cam1_path,
        "joined_image": joined_path,
        "measurable_tokens": 0, "issues_tokens": 0,
    }

    # Check images exist
    if not focus_dir or not focus_dir.exists():
        result["pred_measurable"] = "N/A"
        result["pred_has_issue"] = "N/A"
        result["pred_severity"] = "N/A"
        return {"index": index, "result": result, "measurable_usage": None,
                "issues_usage": None, "error": "focus dir not found"}

    # --- Laplacian severity ---
    try:
        metrics = process_focus_folder(focus_dir)
        if metrics:
            result["focus_right_mean"] = metrics["focus_right_mean"]
            result["focus_left_mean"] = metrics["focus_left_mean"]
            result["focus_right_mid"] = metrics["focus_right_mid"]
            result["focus_left_mid"] = metrics["focus_left_mid"]
            result["focus_abs_dif_rel"] = metrics["focus_abs_dif_rel"]
            result["pred_severity"] = calc_focus_severity(
                metrics["focus_right_mean"], metrics["focus_left_mean"],
                metrics["focus_abs_dif_rel"],
                metrics["focus_right_mid"], metrics["focus_left_mid"],
            )
        else:
            result["pred_severity"] = "N/A"
    except Exception as e:
        result["pred_severity"] = "Error"
        print(f"  Laplacian error for {row.get('System ID', '?')}: {e}")

    if skip_gemini:
        result["pred_measurable"] = ""
        result["pred_has_issue"] = ""
        return {"index": index, "result": result, "measurable_usage": None,
                "issues_usage": None}

    # --- Load images for Gemini ---
    images = load_images(focus_dir)
    if images is None:
        result["pred_measurable"] = "N/A"
        result["pred_has_issue"] = "N/A"
        return {"index": index, "result": result, "measurable_usage": None,
                "issues_usage": None, "error": "could not load images"}

    cam0_bytes, cam1_bytes, joined_bytes = images
    measurable_usage = None
    issues_usage = None

    # --- Measurability (indoor/outdoor) ---
    # Pipeline sends only CAM0+CAM1 (no joined) for measurability
    try:
        indoor = is_indoor(row)
        if indoor:
            meas_result, measurable_usage = analyze_indoor(
                client, model_name, cam0_bytes, cam1_bytes,
                joined_bytes=None, media_resolution=media_resolution,
                examples=indoor_examples,
            )
        else:
            meas_result, measurable_usage = analyze_outdoor(
                client, model_name, cam0_bytes, cam1_bytes,
                joined_bytes=None, media_resolution=media_resolution,
                examples=outdoor_examples,
            )

        result["pred_measurable"] = meas_result.get("is_measurable", "Unknown")
        reason = meas_result.get("reason", "") or meas_result.get("not_measurable_reason", "")
        result["pred_measurable_reason"] = reason.replace(",", ";") if reason else ""
        # Indoor/outdoor analyzers don't return quality fields — leave blank
        result["pred_has_quality_issue"] = ""
        result["pred_quality_issue_type"] = ""
        result["pred_which_camera"] = ""

    except Exception as e:
        tb = traceback.format_exc()
        result["pred_measurable"] = "Error"
        result["pred_measurable_reason"] = str(e)
        print(f"  Measurability error for {row.get('System ID', '?')}: {e}")
        print(tb)

    # If not measurable, severity is not applicable
    if result["pred_measurable"].lower() == "no":
        result["pred_severity"] = "NA"

    # --- Issue detection (only if measurable) ---
    if result["pred_measurable"].lower() == "yes":
        try:
            issue_result, issues_usage = analyze_venue(
                client, model_name, cam0_bytes, cam1_bytes, joined_bytes,
                examples=issues_examples, media_resolution=media_resolution,
            )
            result["pred_has_issue"] = issue_result.get("has_issue", "Unknown")
            issue_type = issue_result.get("issue_type", "None")
            result["pred_issue_type"] = issue_type if issue_type else "None"
            obs = issue_result.get("observation", "")
            result["pred_observation"] = obs.replace(",", ";") if obs else ""

        except Exception as e:
            tb = traceback.format_exc()
            result["pred_has_issue"] = "Error"
            result["pred_issue_type"] = "None"
            result["pred_observation"] = str(e)
            print(f"  Issue detection error for {row.get('System ID', '?')}: {e}")
            print(tb)
    else:
        result["pred_has_issue"] = "N/A"
        result["pred_issue_type"] = "N/A"
        result["pred_observation"] = "Skipped — not measurable"

    # Token tracking
    if measurable_usage:
        result["measurable_tokens"] = measurable_usage.get("total_tokens", 0)
    if issues_usage:
        result["issues_tokens"] = issues_usage.get("total_tokens", 0)

    return {"index": index, "result": result,
            "measurable_usage": measurable_usage,
            "issues_usage": issues_usage}


def main():
    parser = argparse.ArgumentParser(
        description="Run annotation pipeline: Laplacian + Gemini measurability + issue detection"
    )
    parser.add_argument("--annotations", type=str, default="complete_annotation.csv",
                        help="Path to annotation CSV (default: complete_annotation.csv)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (creates timestamped subdir if not specified)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process first N rows only")
    parser.add_argument("--skip-gemini", action="store_true",
                        help="Skip Gemini calls, run only Laplacian severity prediction")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Parallel Gemini workers (default from config.yaml)")
    parser.add_argument("--examples-dir", type=str, default="examples/",
                        help="Examples directory for detect_issues (default: examples/)")
    args = parser.parse_args()

    # Load config
    config = load_config()
    model_name = config.get("model", "gemini-3-flash-preview")
    project = config.get("project", "pixellot-ai")
    location = config.get("location", "global")
    max_workers = args.max_workers if args.max_workers else config.get("max_workers", 10)
    media_resolution = config.get("media_resolution")

    script_dir = Path(__file__).parent.resolve()

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_dir = script_dir / "output_dir" / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "annotation_predictions.csv"
    cost_path = output_dir / "cost.txt"

    # Read annotation CSV
    ann_path = Path(args.annotations) if Path(args.annotations).is_absolute() else script_dir / args.annotations
    with open(ann_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.limit:
        rows = rows[:args.limit]

    print("=" * 60)
    print("  ANNOTATION PIPELINE")
    print("=" * 60)
    print(f"  Annotations: {ann_path} ({len(rows)} rows)")
    print(f"  Output dir:  {output_dir}")
    print(f"  Model:       {model_name}")
    print(f"  Max workers: {max_workers}")
    print(f"  Skip Gemini: {args.skip_gemini}")
    if media_resolution:
        print(f"  Media res:   {media_resolution}")
    if args.limit:
        print(f"  Limit:       {args.limit}")

    # Load examples
    indoor_examples = None
    outdoor_examples = None
    issues_examples = None
    client = None

    if not args.skip_gemini:
        indoor_examples = load_indoor_examples()
        outdoor_examples = load_outdoor_examples()
        issues_examples = load_issues_examples(Path(args.examples_dir))

        print(f"  Indoor examples:  {sum(len(v) for v in indoor_examples.values())} images")
        print(f"  Outdoor examples: {sum(len(v) for v in outdoor_examples.values())} images")
        print(f"  Issues examples:  {sum(len(v) for v in issues_examples.values())} images")

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

    print("=" * 60)

    # Track totals
    total_measurable_input = 0
    total_measurable_output = 0
    total_measurable_thinking = 0
    total_issues_input = 0
    total_issues_output = 0
    total_issues_thinking = 0
    processed_count = 0
    skipped_count = 0
    error_count = 0

    model_pricing = load_model_pricing()

    total_rows = len(rows)
    results = [None] * total_rows
    completed = 0

    if args.skip_gemini:
        # Sequential processing (Laplacian is CPU-bound, already fast)
        for i, row in enumerate(rows):
            res = process_row(i, row, client, model_name,
                              indoor_examples, outdoor_examples, issues_examples,
                              media_resolution, skip_gemini=True)
            results[i] = res["result"]
            completed += 1
            severity = res["result"]["pred_severity"]
            venue = res["result"]["System ID"]
            if completed % 50 == 0 or completed == total_rows:
                print(f"  [{completed}/{total_rows}] {venue} — severity: {severity}")
    else:
        # Parallel processing for Gemini calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    process_row, i, row, client, model_name,
                    indoor_examples, outdoor_examples, issues_examples,
                    media_resolution, False
                ): i
                for i, row in enumerate(rows)
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                res = future.result()
                idx = res["index"]
                results[idx] = res["result"]
                completed += 1

                r = res["result"]
                venue = r["System ID"]
                meas = r["pred_measurable"]
                issue = r["pred_has_issue"]
                sev = r["pred_severity"]

                if res.get("error"):
                    error_count += 1
                    print(f"[{completed}/{total_rows}] {venue} — error: {res['error']}")
                else:
                    processed_count += 1
                    tokens_m = r.get("measurable_tokens", 0)
                    tokens_i = r.get("issues_tokens", 0)
                    print(f"[{completed}/{total_rows}] {venue}"
                          f" — meas: {meas}, issue: {issue}, sev: {sev}"
                          f", tokens: {tokens_m + tokens_i:,}")

                # Accumulate token usage
                mu = res.get("measurable_usage")
                if mu:
                    total_measurable_input += mu.get("input_tokens", 0)
                    total_measurable_output += mu.get("output_tokens", 0)
                    total_measurable_thinking += mu.get("thinking_tokens", 0)

                iu = res.get("issues_usage")
                if iu:
                    total_issues_input += iu.get("input_tokens", 0)
                    total_issues_output += iu.get("output_tokens", 0)
                    total_issues_thinking += iu.get("thinking_tokens", 0)

                if completed % 100 == 0:
                    total_in = total_measurable_input + total_issues_input
                    total_out = (total_measurable_output + total_measurable_thinking +
                                 total_issues_output + total_issues_thinking)
                    pricing = calculate_pricing(model_name, total_in, total_out, model_pricing)
                    print(f"  --- Cost after {completed} rows: ${pricing['total_price']:.6f} ---")

    # Write output CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        for r in results:
            if r is not None:
                writer.writerow(r)

    # Quality issue coverage stats
    issue_evaluated = sum(1 for r in results if r and r.get("pred_has_issue") not in ("N/A", ""))
    issue_skipped_not_meas = sum(1 for r in results if r and r.get("pred_observation") == "Skipped — not measurable")
    issue_skipped_no_data = sum(1 for r in results if r and r.get("pred_has_issue") == "N/A"
                                and r.get("pred_observation") != "Skipped — not measurable")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Output:    {output_path}")
    print(f"  Total:     {total_rows}")
    print(f"  Processed: {processed_count}")
    print(f"  Errors:    {error_count}")
    print(f"\n  QUALITY ISSUE COVERAGE:")
    print(f"    Evaluated:                {issue_evaluated}/{total_rows} ({100*issue_evaluated/total_rows:.1f}%)")
    print(f"    Skipped (not measurable): {issue_skipped_not_meas}")
    print(f"    Skipped (no data):        {issue_skipped_no_data}")

    if not args.skip_gemini:
        total_in = total_measurable_input + total_issues_input
        total_out = (total_measurable_output + total_measurable_thinking +
                     total_issues_output + total_issues_thinking)
        total_think = total_measurable_thinking + total_issues_thinking

        pricing_meas = calculate_pricing(
            model_name, total_measurable_input,
            total_measurable_output + total_measurable_thinking, model_pricing)
        pricing_issues = calculate_pricing(
            model_name, total_issues_input,
            total_issues_output + total_issues_thinking, model_pricing)
        pricing_total = calculate_pricing(model_name, total_in, total_out, model_pricing)

        print("\n  TOKEN USAGE:")
        print(f"    Measurability — input: {total_measurable_input:,}"
              f"  thinking: {total_measurable_thinking:,}"
              f"  output: {total_measurable_output:,}")
        print(f"    Issues        — input: {total_issues_input:,}"
              f"  thinking: {total_issues_thinking:,}"
              f"  output: {total_issues_output:,}")
        print(f"    Total         — input: {total_in:,}"
              f"  thinking: {total_think:,}"
              f"  output: {total_measurable_output + total_issues_output:,}")

        print("\n  COST:")
        print(f"    Measurability: ${pricing_meas['total_price']:.6f}")
        print(f"    Issues:        ${pricing_issues['total_price']:.6f}")
        print(f"    Total:         ${pricing_total['total_price']:.6f}")

        # Write cost report
        with open(cost_path, 'w') as f:
            f.write("ANNOTATION PIPELINE - COST REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Annotations: {ann_path}\n")
            f.write(f"Total rows: {total_rows}\n")
            f.write(f"Processed: {processed_count}\n")
            f.write(f"Errors: {error_count}\n\n")
            f.write("TOKEN USAGE:\n")
            f.write(f"  Measurability — input: {total_measurable_input:,}"
                    f"  thinking: {total_measurable_thinking:,}"
                    f"  output: {total_measurable_output:,}\n")
            f.write(f"  Issues        — input: {total_issues_input:,}"
                    f"  thinking: {total_issues_thinking:,}"
                    f"  output: {total_issues_output:,}\n")
            f.write(f"  Total         — input: {total_in:,}"
                    f"  thinking: {total_think:,}"
                    f"  output: {total_measurable_output + total_issues_output:,}\n\n")
            f.write("COST:\n")
            f.write(f"  Measurability: ${pricing_meas['total_price']:.6f}\n")
            f.write(f"  Issues:        ${pricing_issues['total_price']:.6f}\n")
            f.write(f"  Total:         ${pricing_total['total_price']:.6f}\n")

        print(f"\n  Cost report: {cost_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
