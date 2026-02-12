#!/usr/bin/env python3
"""
Detect Issues Script

Reads PQS_blur_by_venue.xlsx and processes one event per venue (first event with
a focus dir), using Gemini to detect camera issues: focus problems, condensation,
obstructions, and dark field conditions.
Optionally uses few-shot example images from a specified directory.
"""

import os
import csv
import json
import argparse
import traceback
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set Google Cloud credentials if not already set
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    script_dir = Path(__file__).parent.resolve()
    credentials_path = script_dir / "service-account-key.json"
    if credentials_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

from PIL import Image
from google import genai
from google.genai import types

from is_measurable import (
    load_config,
    load_venues,
    load_model_pricing,
    calculate_pricing,
    get_token_usage,
    pil_to_bytes,
    load_images,
)
from run_is_measurable import read_blur_xlsx, find_venue_focus_dir


DETECT_ISSUES_PROMPT = """You are analyzing camera images from a sports venue to detect issues.

The images are from a paired camera setup:
- CAM0 = right camera
- CAM1 = left camera
- joined = side-by-side overlap composition (left half from CAM1, right half from CAM0)

TASK: Check for issues: Focus / Object / None.

IMPORTANT — Focus detection:
A key indicator of a focus problem is a LARGE DIFFERENCE IN SHARPNESS between CAM0 and CAM1.
Compare edges, field lines, text, and fine details between the two cameras. If one camera is
noticeably sharper/crisper than the other, that is a focus issue. Use the joined overlap image
to directly compare the same scene area from both cameras side by side.

another key identicator 
Identify segments in the video where there is a change in camera focus.
Some areas appear sharp and clear, while other areas appear blurred or smoothed. Flag any transitions between sharp and soft focus.

Return ONLY valid JSON with exactly this structure:
{
    "observation": "Brief description of what you see",
    "has_issue": "Yes" or "No",
    "issue_type": "Focus" or "Object" or "None"
}

Rules:
- If the field is too dark to analyze or field is covered havely by snow (not only on sides), set has_issue to "No" and issue_type to "None"
- Return ONLY valid JSON, no markdown formatting or extra text"""


EXAMPLE_JOINED_LABEL = """This is an example of a joined image (left half = CAM1, right half = CAM0). Notice the difference in sharpness between the two cameras."""

EXAMPLE_CHANGE_FOCUS_LABEL = """This is an example of a change in camera focus. You can see the change inside the camera — some areas appear sharp while others are blurred or smoothed."""

EXAMPLE_COMPLETE_FOCUS_LABEL = """This is an example of a completely smoothed camera with bad focus all over the lens. The entire image lacks sharpness."""

# Ordered list of (subdirectory_name, section_title, label_constant)
EXAMPLE_CATEGORIES = [
    ("joined", "Joined image (sharpness difference between cameras)", EXAMPLE_JOINED_LABEL),
    ("change_focus", "Change in camera focus", EXAMPLE_CHANGE_FOCUS_LABEL),
    ("complete_focus", "Complete focus problem", EXAMPLE_COMPLETE_FOCUS_LABEL),
]


def load_examples(examples_dir: Path) -> Dict[str, List[bytes]]:
    """Load example images from categorized subdirectories.

    Walks known subdirectories (joined/, change_focus/, complete_focus/) and
    loads all *.jpg/*.png files from each.

    Returns dict keyed by category name, each value a list of image bytes.
    """
    examples: Dict[str, List[bytes]] = {}
    if not examples_dir.is_dir():
        print(f"  Warning: examples directory not found: {examples_dir}")
        return examples

    for category, _, _ in EXAMPLE_CATEGORIES:
        cat_dir = examples_dir / category
        if not cat_dir.is_dir():
            print(f"  Warning: example category dir not found: {cat_dir}")
            continue

        images = []
        for img_path in sorted(cat_dir.glob("*.jpg")) + sorted(cat_dir.glob("*.png")):
            try:
                img_bytes = pil_to_bytes(Image.open(img_path))
                images.append(img_bytes)
            except Exception as e:
                print(f"  Warning: failed to load example {img_path.name}: {e}")

        if images:
            examples[category] = images

    return examples


def analyze_venue(client: genai.Client, model_name: str,
                  cam0_bytes: bytes, cam1_bytes: bytes, joined_bytes: Optional[bytes],
                  examples: Optional[Dict[str, List[bytes]]] = None,
                  media_resolution: Optional[str] = None) -> Tuple[dict, dict]:
    """Build parts list with prompt + optional categorized examples + venue images, call Gemini.

    Returns (result_dict, token_usage_dict).
    """
    image_kwargs = {}
    if media_resolution:
        image_kwargs["media_resolution"] = media_resolution

    parts = [types.Part.from_text(text=DETECT_ISSUES_PROMPT)]

    # Insert categorized examples
    if examples:
        parts.append(types.Part.from_text(
            text="\n\nReference examples of known focus issues:"
        ))
        for category, title, label in EXAMPLE_CATEGORIES:
            if category not in examples:
                continue
            parts.append(types.Part.from_text(text=f"\nExample — {title}:"))
            for img_bytes in examples[category]:
                parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=img_bytes, **image_kwargs))
            parts.append(types.Part.from_text(text=label))

        parts.append(types.Part.from_text(
            text="\n\nNow analyze the following venue images:"
        ))

    # Append venue images
    parts.append(types.Part.from_text(text="CAM0 (right camera):"))
    parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=cam0_bytes, **image_kwargs))
    parts.append(types.Part.from_text(text="CAM1 (left camera):"))
    parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=cam1_bytes, **image_kwargs))

    if joined_bytes is not None:
        parts.append(types.Part.from_text(text="Joined overlap image (left half = CAM1, right half = CAM0):"))
        parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=joined_bytes, **image_kwargs))

    contents = [types.Content(role="user", parts=parts)]

    generation_config = types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        max_output_tokens=65536,
        labels={"job": "detect-issues", "pipeline": "pcm"},
    )

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=generation_config,
    )

    # Parse JSON response - extract text from candidates when thinking is enabled
    response_text = None
    if response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    response_text = part.text
                    break

    if response_text is None:
        response_text = response.text

    if response_text is None:
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            reason = str(candidate.finish_reason) if candidate.finish_reason else "Empty response from model"
        else:
            reason = "No candidates in response"

        token_usage = get_token_usage(response)
        return {
            "has_issue": "Error",
            "issue_type": "None",
            "observation": reason,
        }, token_usage

    response_text = response_text.strip()
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        result = {
            "has_issue": "Error",
            "issue_type": "None",
            "observation": f"Failed to parse response: {response_text[:200]}",
        }

    token_usage = get_token_usage(response)
    return result, token_usage


def process_entry(index: int, entry: dict, client: genai.Client, model_name: str,
                  examples: Optional[Dict[str, List[bytes]]] = None,
                  media_resolution: Optional[str] = None) -> dict:
    """Load images and call analyze_venue. Used as parallel worker.

    Returns dict with keys: index, result_row, token_usage, processed, skipped.
    """
    venue_id = entry["venue_id"]
    event_id = entry["event_id"]
    focus_dir = Path(entry["focus_dir"])

    result_row = {
        "venue_id": venue_id,
        "event_id": event_id,
        "has_issue": "",
        "issue_type": "",
    }

    images = load_images(focus_dir)
    if images is None:
        result_row["has_issue"] = "N/A"
        result_row["issue_type"] = "None"
        return {"index": index, "result_row": result_row, "token_usage": None,
                "processed": False, "skipped": True, "skip_reason": "could not load images"}

    cam0_bytes, cam1_bytes, joined_bytes = images

    try:
        result, token_usage = analyze_venue(
            client, model_name, cam0_bytes, cam1_bytes, joined_bytes,
            examples, media_resolution,
        )

        has_issue = result.get("has_issue", "No")
        issue_type = result.get("issue_type", "None")
        result_row["has_issue"] = has_issue
        result_row["issue_type"] = issue_type if issue_type else "None"

        return {"index": index, "result_row": result_row, "token_usage": token_usage,
                "processed": True, "skipped": False}

    except Exception as e:
        tb = traceback.format_exc()
        result_row["has_issue"] = "Error"
        result_row["issue_type"] = "None"
        return {"index": index, "result_row": result_row, "token_usage": None,
                "processed": False, "skipped": True, "error": str(e), "traceback": tb}


def main():
    parser = argparse.ArgumentParser(description="Detect camera issues using Gemini (xlsx-driven, one event per venue)")
    parser.add_argument("--dataset", type=str, required=True, help="Data directory path (e.g. all_data_02_09)")
    parser.add_argument("--blur-xlsx", type=str, default="PQS_blur_by_venue.xlsx", help="Path to PQS blur xlsx (default: PQS_blur_by_venue.xlsx)")
    parser.add_argument("--use-examples", type=str, default="examples/", metavar="DIR", help="Directory containing example images of focus problems (default: examples/)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of venues to process")
    parser.add_argument("--max-workers", type=int, default=None, help="Max parallel Gemini API calls (default: from config.yaml or 10)")
    parser.add_argument("--venues", type=str, default=None, help="YAML file with venue filter (default: venues.yaml if exists)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: auto in output_dir/)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Parent output directory (creates detect_issues/ subdir). If omitted, creates timestamped dir.")
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

    # Build work items: one event per venue, driven by xlsx
    work_items = []  # list of (venue_id, event_id, focus_dir_or_None)

    for row in venue_rows:
        venue_id = str(row.get("PIXELLOT_VENUE_ID", "")).strip()
        if not venue_id:
            continue

        # Apply venue filter
        if not is_full and venue_filter and venue_id not in venue_filter:
            continue

        focus_dir = find_venue_focus_dir(dataset_dir, venue_id)
        if focus_dir is not None:
            event_id = focus_dir.parent.name
        else:
            event_id = ""

        work_items.append((venue_id, event_id, focus_dir))

    if args.limit:
        work_items = work_items[:args.limit]

    found_count = sum(1 for _, _, fd in work_items if fd is not None)
    not_found_count = len(work_items) - found_count

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir) / "detect_issues"
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_dir = script_dir / "output_dir" / f"detect_issues_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output CSV path
    if args.output:
        output_path = Path(args.output) if Path(args.output).is_absolute() else script_dir / args.output
    else:
        output_path = output_dir / "detect_issues.csv"

    cost_path = output_dir / "cost.txt"

    print("=" * 60)
    print("  DETECT ISSUES")
    print("=" * 60)
    print(f"  Output dir: {output_dir}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  XLSX: {xlsx_path.name}")
    print(f"  Model: {model_name}")
    print(f"  Total venues: {len(work_items)}")
    print(f"    Found in dataset: {found_count}")
    print(f"    Not in dataset: {not_found_count}")
    if is_full:
        print("  Venues: ALL (full mode)")
    else:
        print(f"  Venues: {len(venue_filter)} filtered")
    print(f"  Max workers: {max_workers}")
    if media_resolution:
        print(f"  Media resolution: {media_resolution}")
    if args.limit:
        print(f"  Limit: {args.limit}")
    print(f"  Use examples: {args.use_examples}")

    # Load categorized examples if enabled
    examples = None
    if args.use_examples:
        examples = load_examples(Path(args.use_examples))
        total_example_images = sum(len(imgs) for imgs in examples.values())
        print(f"  Examples loaded: {total_example_images} images from {len(examples)} categories")
        for category, imgs in examples.items():
            print(f"    {category}: {len(imgs)} images")
    print("=" * 60)

    if not work_items:
        print("  No venues to process. Exiting.")
        return

    # Retry settings
    retry_attempts = config.get("retry_attempts", 10)
    retry_initial_delay = config.get("retry_initial_delay", 2.0)
    retry_max_delay = config.get("retry_max_delay", 16.0)

    # Initialize Gemini client
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

    # Track totals for cost report
    total_input_tokens = 0
    total_output_tokens = 0
    total_thinking_tokens = 0
    processed_count = 0
    skipped_count = 0

    # Load pricing for running cost updates
    model_pricing = load_model_pricing()

    # Process venues
    total_items = len(work_items)
    results = [None] * total_items
    completed = 0

    fieldnames = ["venue_id", "event_id", "has_issue", "issue_type"]

    # Separate items: those with focus dirs (need Gemini) vs those without
    gemini_items = []  # (work_index, entry_dict)
    for wi, (venue_id, event_id, focus_dir) in enumerate(work_items):
        if focus_dir is None:
            result_row = {
                "venue_id": venue_id,
                "event_id": "",
                "has_issue": "N/A",
                "issue_type": "None",
            }
            results[wi] = result_row
            skipped_count += 1
            completed += 1
            print(f"[{completed}/{total_items}] {venue_id} — N/A: not in dataset")
        else:
            entry = {
                "venue_id": venue_id,
                "event_id": event_id,
                "focus_dir": str(focus_dir),
            }
            gemini_items.append((wi, entry))

    # Process venues with images in parallel
    if gemini_items:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_wi = {}
            for wi, entry in gemini_items:
                future = executor.submit(
                    process_entry, wi, entry, client, model_name, examples, media_resolution,
                )
                future_to_wi[future] = (wi, entry)

            for future in concurrent.futures.as_completed(future_to_wi):
                wi, entry = future_to_wi[future]
                res = future.result()
                result_row = res["result_row"]
                results[wi] = result_row
                completed += 1

                venue_id = result_row["venue_id"]
                event_id = result_row["event_id"]

                if res.get("skipped"):
                    reason = res.get("skip_reason") or res.get("error", "unknown")
                    print(f"[{completed}/{total_items}] {venue_id}/{event_id} — skipped: {reason}")
                    if res.get("traceback"):
                        print(res["traceback"])
                    skipped_count += 1
                elif res["processed"]:
                    token_usage = res["token_usage"]
                    total_input_tokens += token_usage["input_tokens"]
                    total_output_tokens += token_usage["output_tokens"]
                    total_thinking_tokens += token_usage["thinking_tokens"]
                    processed_count += 1

                    print(f"[{completed}/{total_items}] {venue_id}/{event_id}"
                          f" — issue: {result_row['has_issue']}"
                          f", type: {result_row['issue_type']}"
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
    print(f"    Found in dataset: {found_count}")
    print(f"    Not in dataset: {not_found_count}")
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
        f.write("DETECT ISSUES - COST REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_dir}\n")
        f.write(f"XLSX: {xlsx_path.name}\n")
        f.write(f"Examples dir: {args.use_examples or 'None'}\n")
        if examples:
            total_ex = sum(len(imgs) for imgs in examples.values())
            f.write(f"Examples: {total_ex} images from {len(examples)} categories\n")
            for cat, imgs in examples.items():
                f.write(f"  {cat}: {len(imgs)} images\n")
        f.write(f"Total venues: {total_items}\n")
        f.write(f"  Found in dataset: {found_count}\n")
        f.write(f"  Not in dataset: {not_found_count}\n")
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
