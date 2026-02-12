#!/usr/bin/env python3
"""
Detect Issues Script

Scans a data directory directly (no CSV input needed) and uses Gemini to detect
camera issues: focus problems, condensation, obstructions, and dark field conditions.
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


DETECT_ISSUES_PROMPT = """You are analyzing camera images from a sports venue to detect issues.

The images are from a paired camera setup:
- CAM0 = right camera
- CAM1 = left camera
- joined = side-by-side overlap composition (left half from CAM1, right half from CAM0)

TASK:
1. Is the scene measurable? Answer "No" if: field is too dark (main lights OFF).
2. If measurable, check for issues: Focus / Object / Condensation / None.

IMPORTANT — Focus detection:
many times, A key indicator of a focus problem is a
LARGE DIFFERENCE IN SHARPNESS between CAM0 and CAM1. Compare edges, field lines, text,
and fine details between the two cameras. If one camera is noticeably sharper/crisper than
the other, that is a focus issue. Use the joined overlap image to directly compare the
same scene area from both cameras side by side.

Return ONLY valid JSON with exactly this structure:
{
    "observation": "Brief description of what you see",
    "is_measurable": "Yes" or "No",
    "has_issue": "Yes" or "No",
    "issue_type": "Focus" or "Object" or "Condensation" or "None"
}

Rules:
- If is_measurable is "No", set has_issue to "No" and issue_type to "None"
- Condensation on lens: is_measurable = "Yes", has_issue = "Yes", issue_type = "Condensation"
- Return ONLY valid JSON, no markdown formatting or extra text"""


def scan_database(data_dir: Path, venue_filter: Optional[set] = None) -> List[Dict[str, str]]:
    """Walk <data_dir>/<venue_id>/<event_id>/focus/ and return list of entry dicts.

    Returns list of {"venue_id", "event_id", "focus_dir"} dicts.
    Filters by venue_filter if provided.
    """
    entries = []
    if not data_dir.is_dir():
        print(f"  Error: data directory not found: {data_dir}")
        return entries

    for venue_path in sorted(data_dir.iterdir()):
        if not venue_path.is_dir():
            continue
        venue_id = venue_path.name
        if venue_filter and venue_id not in venue_filter:
            continue
        for event_path in sorted(venue_path.iterdir()):
            if not event_path.is_dir():
                continue
            focus_dir = event_path / "focus"
            if focus_dir.is_dir():
                entries.append({
                    "venue_id": venue_id,
                    "event_id": event_path.name,
                    "focus_dir": str(focus_dir),
                })
    return entries


def load_examples(examples_dir: Path) -> List[Tuple[str, List[bytes]]]:
    """Load example images from a flat directory of .jpg files.

    Each image is loaded as bytes using pil_to_bytes().

    Returns list of (filename_stem, [image_bytes]) tuples.
    """
    examples = []
    if not examples_dir.is_dir():
        print(f"  Warning: examples directory not found: {examples_dir}")
        return examples

    for img_path in sorted(examples_dir.glob("*.jpg")):
        try:
            img_bytes = pil_to_bytes(Image.open(img_path))
            examples.append((img_path.stem, [img_bytes]))
        except Exception as e:
            print(f"  Warning: failed to load example {img_path.name}: {e}")

    return examples


def analyze_venue(client: genai.Client, model_name: str,
                  cam0_bytes: bytes, cam1_bytes: bytes, joined_bytes: Optional[bytes],
                  hard_examples: Optional[List[Tuple[str, List[bytes]]]] = None,
                  media_resolution: Optional[str] = None) -> Tuple[dict, dict]:
    """Build parts list with prompt + optional hard examples + venue images, call Gemini.

    Returns (result_dict, token_usage_dict).
    """
    image_kwargs = {}
    if media_resolution:
        image_kwargs["media_resolution"] = media_resolution

    parts = [types.Part.from_text(text=DETECT_ISSUES_PROMPT)]

    # Insert hard examples as a batch
    if hard_examples:
        parts.append(types.Part.from_text(
            text="\n\nReference: Examples of Venues with Known Issues\n"
                 "The following are examples of venues where issues were confirmed:"
        ))
        for i, (venue_id, image_list) in enumerate(hard_examples, 1):
            parts.append(types.Part.from_text(text=f"\nExample venue {i}:"))
            for img_bytes in image_list:
                parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=img_bytes, **image_kwargs))

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
            "is_measurable": "Error",
            "has_issue": "No",
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
            "is_measurable": "Unknown",
            "has_issue": "No",
            "issue_type": "None",
            "observation": f"Failed to parse response: {response_text[:200]}",
        }

    token_usage = get_token_usage(response)
    return result, token_usage


def process_entry(index: int, entry: dict, client: genai.Client, model_name: str,
                  hard_examples: Optional[List[Tuple[str, List[bytes]]]] = None,
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
        "is_measurable": "",
        "has_issue": "",
        "issue_type": "",
    }

    images = load_images(focus_dir)
    if images is None:
        result_row["is_measurable"] = "N/A"
        result_row["has_issue"] = "No"
        result_row["issue_type"] = "None"
        return {"index": index, "result_row": result_row, "token_usage": None,
                "processed": False, "skipped": True, "skip_reason": "could not load images"}

    cam0_bytes, cam1_bytes, joined_bytes = images

    try:
        result, token_usage = analyze_venue(
            client, model_name, cam0_bytes, cam1_bytes, joined_bytes,
            hard_examples, media_resolution,
        )

        is_measurable = result.get("is_measurable", "Unknown")
        result_row["is_measurable"] = is_measurable

        if is_measurable == "No":
            result_row["has_issue"] = "No"
            result_row["issue_type"] = "None"
        else:
            has_issue = result.get("has_issue", "No")
            issue_type = result.get("issue_type", "None")
            result_row["has_issue"] = has_issue
            result_row["issue_type"] = issue_type if issue_type else "None"

        return {"index": index, "result_row": result_row, "token_usage": token_usage,
                "processed": True, "skipped": False}

    except Exception as e:
        tb = traceback.format_exc()
        result_row["is_measurable"] = "Error"
        result_row["has_issue"] = "No"
        result_row["issue_type"] = "None"
        return {"index": index, "result_row": result_row, "token_usage": None,
                "processed": False, "skipped": True, "error": str(e), "traceback": tb}


def main():
    parser = argparse.ArgumentParser(description="Scan data directory and detect camera issues using Gemini")
    parser.add_argument("--database", type=str, required=True, help="Data directory with <venue_id>/<event_id>/focus/ structure")
    parser.add_argument("--use-examples", type=str, default=None, metavar="DIR", help="Directory containing example images of focus problems")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of entries to process")
    parser.add_argument("--max-workers", type=int, default=None, help="Max parallel Gemini API calls (default: from config.yaml or 10)")
    parser.add_argument("--venues", type=str, default=None, help="YAML file with venue filter (default: venues.yaml if exists)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: auto in output_dir/)")
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

    # Create timestamped output directory
    script_dir = Path(__file__).parent.resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = script_dir / "output_dir" / f"detect_issues_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output CSV path
    if args.output:
        output_path = Path(args.output) if Path(args.output).is_absolute() else script_dir / args.output
    else:
        output_path = output_dir / "detect_issues.csv"

    cost_path = output_dir / "cost.txt"

    # Scan database
    data_dir = script_dir / args.database
    entries = scan_database(data_dir, venue_filter if not is_full else None)

    if args.limit:
        entries = entries[:args.limit]

    print("=" * 60)
    print("  DETECT ISSUES")
    print("=" * 60)
    print(f"  Output dir: {output_dir}")
    print(f"  Database: {data_dir}")
    print(f"  Model: {model_name}")
    print(f"  Entries: {len(entries)}")
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

    # Load hard examples if enabled
    hard_examples = None
    if args.use_examples:
        hard_examples = load_examples(Path(args.use_examples))
        print(f"  Examples loaded: {len(hard_examples)} from {args.use_examples}")
    print("=" * 60)

    if not entries:
        print("  No entries found. Exiting.")
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

    # Process entries in parallel
    total_entries = len(entries)
    results = [None] * total_entries
    completed = 0

    fieldnames = ["venue_id", "event_id", "is_measurable", "has_issue", "issue_type"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_entry, i, entry, client, model_name, hard_examples, media_resolution): i
            for i, entry in enumerate(entries)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            res = future.result()
            idx = res["index"]
            result_row = res["result_row"]
            results[idx] = result_row
            completed += 1

            venue_id = result_row["venue_id"]
            event_id = result_row["event_id"]

            if res.get("skipped"):
                reason = res.get("skip_reason") or res.get("error", "unknown")
                print(f"[{completed}/{total_entries}] {venue_id}/{event_id} — skipped: {reason}")
                if res.get("traceback"):
                    print(res["traceback"])
                skipped_count += 1
            elif res["processed"]:
                token_usage = res["token_usage"]
                total_input_tokens += token_usage["input_tokens"]
                total_output_tokens += token_usage["output_tokens"]
                total_thinking_tokens += token_usage["thinking_tokens"]
                processed_count += 1

                print(f"[{completed}/{total_entries}] {venue_id}/{event_id}"
                      f" — measurable: {result_row['is_measurable']}"
                      f", issue: {result_row['has_issue']}"
                      f", type: {result_row['issue_type']}"
                      f", tokens: {token_usage['total_tokens']:,}")

            if completed % 100 == 0:
                pricing = calculate_pricing(model_name, total_input_tokens, total_output_tokens + total_thinking_tokens, model_pricing)
                print(f"  --- Cost after {completed} entries: ${pricing['total_price']:.6f} ---")

    # Write output CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Output dir: {output_dir}")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
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
        f.write(f"Database: {data_dir}\n")
        f.write(f"Examples dir: {args.use_examples or 'None'}\n")
        if hard_examples:
            f.write(f"Hard examples: {len(hard_examples)}\n")
        f.write(f"Processed: {processed_count} entries\n")
        f.write(f"Skipped: {skipped_count} entries\n\n")
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
