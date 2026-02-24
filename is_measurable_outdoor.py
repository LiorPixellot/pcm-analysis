#!/usr/bin/env python3
"""
Outdoor Measurability Analysis Script

Analyzes camera frames from OUTDOOR sports venues using Gemini to determine
if environmental conditions allow valid focus measurement.

Outdoor-specific checks: lights off/too dark, snow covering field, sun directed
at camera, object obstructing view, condensation.
"""

import os
import json
import csv
import argparse
import traceback
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# Set Google Cloud credentials if not already set
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    script_dir = Path(__file__).parent.resolve()
    credentials_path = script_dir / "service-account-key.json"
    if credentials_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

from google import genai
from google.genai import types

from PIL import Image

from is_measurable import (
    load_config,
    load_venues,
    load_model_pricing,
    calculate_pricing,
    get_token_usage,
    pil_to_bytes,
    load_images,
)


OUTDOOR_PROMPT = """You are analyzing camera images from an OUTDOOR sports venue to determine if conditions allow valid focus measurement.

The images are from a paired camera setup:
- CAM0 = right camera
- CAM1 = left camera
- joined = side-by-side overlap composition (left half from CAM1, right half from CAM0), if provided

## Check Environmental Conditions

Answer "No" for is_measurable if you see ANY of these:

**Lighting problems:**
- TOO DARK: Main field lights are OFF or insufficient. Even if some details are visible from ambient/twilight light, if the venue's main lights are clearly not on, it is NOT measurable.
- SUN ON CAMERA: Direct sunlight hitting the camera lens causing severe glare, flare artifacts, or washed-out images that prevent reliable measurement.

**Weather conditions:**
- SNOW: Snow covering the playing surface MASSIVELY (white/bright field, heavy snow accumulation on the playing area). Light dusting or snow only outside field boundaries does NOT affect measurability.

**Visibility issues:**
- Object blocking or obstructing the camera view too much such that field can't be seen (lens covered, equipment in front, etc.)
- Lens heavily dirty or obscured

**NOT a reason for "No":** Condensation/moisture ON the lens — this is still measurable but is a quality issue.

## Response Format

Return ONLY valid JSON with exactly this structure:
{
    "observation": "Brief description of what you see in the images",
    "is_measurable": "Yes" or "No",
    "reason": "reason if not measurable, empty string if measurable"
}

Rules:
- If is_measurable is "Yes", set reason to ""
- Return ONLY valid JSON, no markdown formatting or extra text"""


EXAMPLE_SUN_LABEL = """This is an example of direct sunlight hitting the camera lens. The severe glare and flare artifacts make focus measurement impossible.
Expected response: {"observation": "Direct sun on camera lens causing severe glare", "is_measurable": "No", "reason": "Sun directed at camera — glare prevents reliable measurement"}"""

EXAMPLE_SNOW_LABEL = """This is an example of massive snow covering the entire playing field. The field is uniformly white with heavy snow accumulation — this prevents focus measurement.
Expected response: {"observation": "Heavy snow fully covers the playing field", "is_measurable": "No", "reason": "Snow massively covers the field — insufficient contrast for measurement"}"""

EXAMPLE_DARK_LABEL = """This is an example of an outdoor venue where the main field lights are OFF. Even though some ambient light is visible, the field itself is too dark for reliable focus measurement.
Expected response: {"observation": "Main field lights are off — only ambient light visible", "is_measurable": "No", "reason": "Field lights off — ambient light insufficient for measurement"}"""

EXAMPLE_SMALL_SNOW_LABEL = """This is an example of snow around the venue (track, surroundings) but the playing field itself is clearly visible and NOT covered by snow. This IS measurable — light snow outside the field does not prevent focus measurement.
Expected response: {"observation": "Snow on surrounding track and grounds, but playing field is clear and fully visible", "is_measurable": "Yes", "reason": ""}"""

# Ordered list of (subdirectory_name, section_title, label_constant, is_positive)
EXAMPLE_CATEGORIES = [
    ("outdoor_sun", "Direct sun on camera", EXAMPLE_SUN_LABEL, False),
    ("outdor_snow", "Snow covering field", EXAMPLE_SNOW_LABEL, False),
    ("outdor_dark", "Dark field with ambient light", EXAMPLE_DARK_LABEL, False),
    ("outdor_small_snow", "Snow around venue but field is clear", EXAMPLE_SMALL_SNOW_LABEL, True),
]


def load_examples(examples_dir: Path = None) -> Dict[str, list]:
    """Load example images from categorized subdirectories.

    Returns dict keyed by category name, each value a list of image bytes.
    """
    if examples_dir is None:
        examples_dir = Path(__file__).parent.resolve() / "examples"

    examples: Dict[str, list] = {}
    if not examples_dir.is_dir():
        print(f"  Warning: examples directory not found: {examples_dir}")
        return examples

    for category, _, _, _ in EXAMPLE_CATEGORIES:
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


def analyze_outdoor(client: genai.Client, model_name: str,
                    cam0_bytes: bytes, cam1_bytes: bytes,
                    joined_bytes: Optional[bytes] = None,
                    media_resolution: Optional[str] = None,
                    examples: Optional[Dict[str, list]] = None) -> Tuple[dict, dict]:
    """Analyze outdoor venue images for measurability.

    Returns (result_dict, token_usage_dict).
    """
    image_kwargs = {}
    if media_resolution:
        image_kwargs["media_resolution"] = media_resolution

    parts = [types.Part.from_text(text=OUTDOOR_PROMPT)]

    # Insert categorized examples
    if examples:
        # Negative examples (NOT measurable)
        negative = [(c, t, l) for c, t, l, pos in EXAMPLE_CATEGORIES if not pos and c in examples]
        if negative:
            parts.append(types.Part.from_text(
                text="\n\nReference examples of conditions that are NOT measurable:"
            ))
            for category, title, label in negative:
                parts.append(types.Part.from_text(text=f"\nExample — {title}:"))
                for img_bytes in examples[category]:
                    parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=img_bytes, **image_kwargs))
                parts.append(types.Part.from_text(text=label))

        # Positive examples (IS measurable)
        positive = [(c, t, l) for c, t, l, pos in EXAMPLE_CATEGORIES if pos and c in examples]
        if positive:
            parts.append(types.Part.from_text(
                text="\n\nReference examples of conditions that ARE measurable:"
            ))
            for category, title, label in positive:
                parts.append(types.Part.from_text(text=f"\nExample — {title}:"))
                for img_bytes in examples[category]:
                    parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=img_bytes, **image_kwargs))
                parts.append(types.Part.from_text(text=label))

        parts.append(types.Part.from_text(
            text="\n\nNow analyze the following venue images:"
        ))

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
        labels={"job": "is-measurable-outdoor", "pipeline": "pcm"},
    )

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=generation_config,
    )

    # Parse JSON response
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
            "observation": reason,
            "is_measurable": "Error",
            "reason": reason,
        }, token_usage

    response_text = response_text.strip()
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        result = {
            "observation": f"Failed to parse response: {response_text[:200]}",
            "is_measurable": "Unknown",
            "reason": f"Failed to parse response: {response_text[:200]}",
        }

    token_usage = get_token_usage(response)
    return result, token_usage


def scan_database(data_dir: Path, venue_filter: Optional[set] = None):
    """Walk <data_dir>/<venue_id>/<event_id>/focus/ and return list of entry dicts."""
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


def process_entry(index: int, entry: dict, client: genai.Client, model_name: str,
                  media_resolution: Optional[str] = None,
                  examples: Optional[Dict[str, list]] = None) -> dict:
    """Load images and call analyze_outdoor. Used as parallel worker."""
    venue_id = entry["venue_id"]
    event_id = entry["event_id"]
    focus_dir = Path(entry["focus_dir"])

    result_row = {
        "venue_id": venue_id,
        "event_id": event_id,
        "is_measurable": "",
        "reason": "",
    }

    images = load_images(focus_dir)
    if images is None:
        result_row["is_measurable"] = "N/A"
        result_row["reason"] = "Could not load images"
        return {"index": index, "result_row": result_row, "token_usage": None,
                "processed": False, "skipped": True, "skip_reason": "could not load images"}

    cam0_bytes, cam1_bytes, joined_bytes = images

    try:
        result, token_usage = analyze_outdoor(
            client, model_name, cam0_bytes, cam1_bytes, joined_bytes, media_resolution,
            examples,
        )

        result_row["is_measurable"] = result.get("is_measurable", "Unknown")
        reason = result.get("reason", "")
        result_row["reason"] = reason.replace(",", ";") if reason else ""

        return {"index": index, "result_row": result_row, "token_usage": token_usage,
                "processed": True, "skipped": False}

    except Exception as e:
        tb = traceback.format_exc()
        result_row["is_measurable"] = "Error"
        result_row["reason"] = str(e)
        return {"index": index, "result_row": result_row, "token_usage": None,
                "processed": False, "skipped": True, "error": str(e), "traceback": tb}


def main():
    parser = argparse.ArgumentParser(description="Analyze outdoor venue camera frames for measurability")
    parser.add_argument("--database", type=str, required=True, help="Data directory with <venue_id>/<event_id>/focus/ structure")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of entries to process")
    parser.add_argument("--max-workers", type=int, default=None, help="Max parallel Gemini API calls (default: from config.yaml or 10)")
    parser.add_argument("--venues", type=str, default=None, help="YAML file with venue filter (default: venues.yaml if exists)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: auto in output_dir/)")
    args = parser.parse_args()

    config = load_config()
    model_name = config.get("model", "gemini-3-flash-preview")
    project = config.get("project", "pixellot-ai")
    location = config.get("location", "global")
    max_workers = args.max_workers if args.max_workers else config.get("max_workers", 10)
    media_resolution = config.get("media_resolution")

    is_full, venue_filter = load_venues(args.venues)

    script_dir = Path(__file__).parent.resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = script_dir / "output_dir" / f"is_measurable_outdoor_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output) if Path(args.output).is_absolute() else script_dir / args.output
    else:
        output_path = output_dir / "is_measurable_outdoor.csv"

    cost_path = output_dir / "cost.txt"

    data_dir = script_dir / args.database
    entries = scan_database(data_dir, venue_filter if not is_full else None)

    if args.limit:
        entries = entries[:args.limit]

    print("=" * 60)
    print("  IS MEASURABLE — OUTDOOR")
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

    if not entries:
        print("  No entries found. Exiting.")
        return

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

    # Load few-shot examples
    examples = load_examples()
    total_example_images = sum(len(imgs) for imgs in examples.values())
    print(f"  Examples loaded: {total_example_images} images from {len(examples)} categories")
    for category, imgs in examples.items():
        print(f"    {category}: {len(imgs)} images")
    print("=" * 60)

    total_input_tokens = 0
    total_output_tokens = 0
    total_thinking_tokens = 0
    processed_count = 0
    skipped_count = 0

    model_pricing = load_model_pricing()

    total_entries = len(entries)
    results = [None] * total_entries
    completed = 0

    fieldnames = ["venue_id", "event_id", "is_measurable", "reason"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_entry, i, entry, client, model_name, media_resolution, examples): i
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
                      f", tokens: {token_usage['total_tokens']:,}")

            if completed % 100 == 0:
                pricing = calculate_pricing(model_name, total_input_tokens, total_output_tokens + total_thinking_tokens, model_pricing)
                print(f"  --- Cost after {completed} entries: ${pricing['total_price']:.6f} ---")

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

    with open(cost_path, 'w') as f:
        f.write("IS MEASURABLE OUTDOOR - COST REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Database: {data_dir}\n")
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
