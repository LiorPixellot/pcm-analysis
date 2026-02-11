#!/usr/bin/env python3
"""
Find Condensation Script

Scans all venue/event image pairs in a data directory, asks Gemini
whether there is condensation on the camera lens, outputs a CSV of
results, and copies predicted-positive images into an output folder.
"""

import os
import io
import csv
import json
import yaml
import shutil
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


# ---------------------------------------------------------------------------
# Config helpers (mirrored from is_measurable.py)
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load configuration from config.yaml file in the script directory."""
    script_dir = Path(__file__).parent.resolve()
    config_path = script_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def load_venues(venues_file: Optional[str]) -> Tuple[bool, set]:
    """Load venue filter from venues.yaml file.

    Returns:
        Tuple of (is_full, venue_set):
        - is_full: True if 'full: true' is set (process all venues)
        - venue_set: Set of venue IDs to filter (empty if full mode)
    """
    script_dir = Path(__file__).parent.resolve()

    if venues_file:
        venues_path = Path(venues_file)
        if not venues_path.is_absolute():
            venues_path = script_dir / venues_file
    else:
        venues_path = script_dir / "venues.yaml"

    if not venues_path.exists():
        return True, set()  # No file = process all

    with open(venues_path, 'r') as f:
        data = yaml.safe_load(f) or {}

    if data.get("full", False):
        return True, set()

    if isinstance(data, list):
        venues = set(data)
    elif isinstance(data, dict):
        venues = set(data.get("venues", []) or [])
    else:
        venues = set()

    return False, venues


def load_model_pricing() -> dict:
    """Load model pricing from cost.yaml in the script directory."""
    script_dir = Path(__file__).parent.resolve()
    cost_path = script_dir / "cost.yaml"
    if not cost_path.exists():
        print(f"  Warning: {cost_path} not found, pricing will be unavailable")
        return {}
    with open(cost_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    return data.get("models", {})


def calculate_pricing(model_name: str, input_tokens: int, output_tokens: int,
                      model_pricing: dict = None) -> Dict[str, float]:
    """Calculate pricing for a request based on token counts."""
    if model_pricing is None:
        model_pricing = load_model_pricing()
    pricing = model_pricing.get(model_name)
    if not pricing:
        return {"input_price": 0.0, "output_price": 0.0, "total_price": 0.0}

    threshold = pricing["threshold"]
    input_low = pricing["input_low"] / 1e6
    input_high = pricing["input_high"] / 1e6
    output_low = pricing["output_low"] / 1e6
    output_high = pricing["output_high"] / 1e6 if pricing["output_high"] else None

    if input_tokens <= threshold:
        input_price = input_tokens * input_low
    else:
        input_price = threshold * input_low + (input_tokens - threshold) * input_high

    if output_high is None:
        output_price = output_tokens * output_low
    elif input_tokens <= threshold:
        output_price = output_tokens * output_low
    else:
        output_price = output_tokens * output_high

    return {
        "input_price": input_price,
        "output_price": output_price,
        "total_price": input_price + output_price,
    }


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def pil_to_bytes(image: Image.Image) -> bytes:
    """Convert PIL Image to JPEG bytes."""
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, (0, 0), image)
        image = background
    elif image.mode not in ('RGB', 'L'):
        image = image.convert('RGB')

    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=95)
    return buf.getvalue()


def load_images(focus_dir: Path) -> Optional[Tuple[bytes, bytes]]:
    """Load CAM0_1.jpg and CAM1_1.jpg from the focus directory."""
    cam0_path = focus_dir / "CAM0_1.jpg"
    cam1_path = focus_dir / "CAM1_1.jpg"

    if not cam0_path.exists() or not cam1_path.exists():
        return None

    try:
        cam0_image = Image.open(cam0_path)
        cam1_image = Image.open(cam1_path)
        return pil_to_bytes(cam0_image), pil_to_bytes(cam1_image)
    except Exception as e:
        print(f"  Error loading images: {e}")
        return None


def get_token_usage(response: types.GenerateContentResponse) -> Dict[str, int]:
    """Extract token usage from a Gemini response."""
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage.prompt_token_count is not None else 0
    output_tokens = usage.candidates_token_count if usage.candidates_token_count is not None else 0
    thinking_tokens = usage.thoughts_token_count if usage.thoughts_token_count is not None else 0
    return {
        "input_tokens": input_tokens,
        "thinking_tokens": thinking_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens + thinking_tokens,
    }


# ---------------------------------------------------------------------------
# Few-shot example
# ---------------------------------------------------------------------------

def load_condensation_examples() -> List[bytes]:
    """Load all condensation example images from examples/Condensation/ at original resolution."""
    script_dir = Path(__file__).parent.resolve()
    examples_dir = script_dir / "examples" / "Condensation"

    if not examples_dir.is_dir():
        print(f"  Warning: Condensation examples directory not found: {examples_dir}")
        return []

    paths = sorted(
        p for p in examples_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if not paths:
        print(f"  Warning: No images found in {examples_dir}")
        return []

    images = []
    for path in paths:
        try:
            img = Image.open(path)
            images.append(pil_to_bytes(img))
        except Exception as e:
            print(f"  Warning: Failed to load condensation example {path.name}: {e}")

    return images


# ---------------------------------------------------------------------------
# Prompt (condensation-focused)
# ---------------------------------------------------------------------------

CONDENSATION_PROMPT_PART1 = """You are analyzing two images from adjacent cameras at a sports venue.

The first image is from the RIGHT camera (CAM0), the second image is from the LEFT camera (CAM1).

Your task is to determine whether there is **condensation** (moisture) on either camera lens.

## What Condensation Looks Like

Condensation on a camera lens typically appears as:
- **Foggy or hazy patches** — parts of the image look blurred or milky while the background scene is still partially visible
- **Moisture droplets** — small water droplets visible on the lens surface
- **Uneven haziness** — some areas are clear while others are fogged, unlike atmospheric fog which is uniform
- **Soft glow around lights** — bright light sources produce exaggerated halos or blooming due to moisture on the lens

Condensation is ON THE LENS, not in the air. The key distinction is that parts of the image may be clear while other parts are fogged, or the entire image has a characteristic milky/hazy quality that differs from atmospheric conditions.

## Reference Example

Here are examples of condensation on a camera lens:"""

CONDENSATION_EXAMPLE_LABEL = """These images show condensation/moisture on the camera lens. Notice the foggy, hazy quality — the field is partially visible but obscured by moisture on the lens surface.

## Now Analyze the Following Images

Look at both camera images below and determine if either or both have condensation on the lens."""

CONDENSATION_PROMPT_PART2 = """
## Response Format

Provide a JSON response with exactly this structure:
{
    "observation": "Brief description of what you see regarding moisture/condensation",
    "has_condensation": "Yes" or "No",
    "which_camera": "Left" or "Right" or "Both" or "None"
}

**Rules:**
- "has_condensation": "Yes" if ANY camera has condensation, "No" otherwise
- "which_camera": "Left" = CAM1, "Right" = CAM0, "Both" = both cameras, "None" = no condensation
- If has_condensation is "No", set which_camera to "None"
- Return ONLY valid JSON, no markdown formatting or extra text"""


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def discover_venue_events(data_dir: Path) -> list[dict]:
    """Walk data_dir/{venue_id}/{event_id}/focus/ and return venue/event pairs."""
    pairs = []
    if not data_dir.exists():
        return pairs

    for venue_entry in sorted(data_dir.iterdir()):
        if not venue_entry.is_dir():
            continue
        venue_id = venue_entry.name
        for event_entry in sorted(venue_entry.iterdir()):
            if not event_entry.is_dir():
                continue
            event_id = event_entry.name
            focus_dir = event_entry / "focus"
            if focus_dir.is_dir():
                pairs.append({"venue_id": venue_id, "event_id": event_id})

    return pairs


# ---------------------------------------------------------------------------
# Gemini analysis
# ---------------------------------------------------------------------------

def analyze_condensation(client: genai.Client, model_name: str,
                         cam0_bytes: bytes, cam1_bytes: bytes,
                         example_images: Optional[List[bytes]] = None,
                         media_resolution: Optional[str] = None) -> Tuple[dict, dict]:
    """Send images to Gemini and get condensation analysis."""
    image_kwargs = {}
    if media_resolution:
        image_kwargs["media_resolution"] = media_resolution

    parts = [types.Part.from_text(text=CONDENSATION_PROMPT_PART1)]

    if example_images:
        for img_bytes in example_images:
            parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=img_bytes, **image_kwargs))

    parts.append(types.Part.from_text(text=CONDENSATION_EXAMPLE_LABEL))
    parts.append(types.Part.from_text(text=CONDENSATION_PROMPT_PART2))

    # Actual camera images
    parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=cam0_bytes, **image_kwargs))
    parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=cam1_bytes, **image_kwargs))

    contents = [types.Content(role="user", parts=parts)]

    generation_config = types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        max_output_tokens=65536,
        labels={"job": "find-condensation", "pipeline": "pcm"},
    )

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=generation_config,
    )

    # Parse response — handle thinking mode
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
            "has_condensation": "Error",
            "which_camera": "None",
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
            "has_condensation": "Unknown",
            "which_camera": "None",
            "observation": f"Failed to parse response: {response_text[:200]}",
        }

    token_usage = get_token_usage(response)
    return result, token_usage


# ---------------------------------------------------------------------------
# Single-item processing
# ---------------------------------------------------------------------------

def process_single(index: int, venue_id: str, event_id: str, data_base: Path,
                   client: genai.Client, model_name: str,
                   example_images: Optional[List[bytes]] = None,
                   media_resolution: Optional[str] = None) -> dict:
    """Process a single venue/event pair."""
    result_row = {
        "venue_id": venue_id,
        "event_id": event_id,
        "has_condensation": "",
        "which_camera": "",
        "observation": "",
    }

    focus_dir = data_base / venue_id / event_id / "focus"

    if not focus_dir.exists():
        result_row["has_condensation"] = "N/A"
        result_row["which_camera"] = "None"
        result_row["observation"] = "Focus directory not found"
        return {"index": index, "result_row": result_row, "token_usage": None,
                "processed": False, "skipped": True, "skip_reason": "focus directory not found"}

    images = load_images(focus_dir)
    if images is None:
        result_row["has_condensation"] = "N/A"
        result_row["which_camera"] = "None"
        result_row["observation"] = "Could not load images"
        return {"index": index, "result_row": result_row, "token_usage": None,
                "processed": False, "skipped": True, "skip_reason": "could not load images"}

    cam0_bytes, cam1_bytes = images

    try:
        result, token_usage = analyze_condensation(
            client, model_name, cam0_bytes, cam1_bytes, example_images, media_resolution
        )

        has_cond = result.get("has_condensation", "Unknown")
        result_row["has_condensation"] = has_cond
        result_row["which_camera"] = result.get("which_camera", "None") or "None"

        observation = result.get("observation", "")
        result_row["observation"] = observation.replace(",", ";") if observation else ""

        if has_cond == "No":
            result_row["which_camera"] = "None"

        return {"index": index, "result_row": result_row, "token_usage": token_usage,
                "processed": True, "skipped": False}

    except Exception as e:
        tb = traceback.format_exc()
        result_row["has_condensation"] = "Error"
        result_row["which_camera"] = "None"
        result_row["observation"] = str(e)
        return {"index": index, "result_row": result_row, "token_usage": None,
                "processed": False, "skipped": True, "error": str(e), "traceback": tb}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detect condensation on camera lenses using Gemini")
    parser.add_argument("data_dir", type=str, help="Data directory to scan (e.g. data_12_13/)")
    parser.add_argument("--output", type=str, default="condensation_results.csv", help="Output CSV file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of venue/event pairs to process")
    parser.add_argument("--venues", type=str, default=None, help="YAML file with venue filter (default: venues.yaml if exists)")
    parser.add_argument("--max-workers", type=int, default=None, help="Max parallel Gemini API calls (default: from config.yaml or 10)")
    parser.add_argument("--cost-file", type=str, default=None, help="Output path for cost report (default: cost.txt in script dir)")
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
    data_base = Path(args.data_dir)
    if not data_base.is_absolute():
        data_base = script_dir / args.data_dir

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = script_dir / "output_dir" / f"condensation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output CSV path: use user-provided path or default inside output_dir
    if args.output != "condensation_results.csv":
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = script_dir / args.output
    else:
        output_path = output_dir / "condensation_results.csv"

    print("=" * 60)
    print("  CONDENSATION DETECTION")
    print("=" * 60)
    print(f"  Output dir: {output_dir}")
    print(f"  Data dir:   {data_base}")
    print(f"  Output CSV: {output_path}")
    print(f"  Model:      {model_name}")
    if is_full:
        print("  Venues:     ALL (full mode)")
    else:
        print(f"  Venues:     {len(venue_filter)} filtered")
    print(f"  Max workers:  {max_workers}")
    if media_resolution:
        print(f"  Media resolution: {media_resolution}")
    if args.limit:
        print(f"  Limit: {args.limit} pairs")
    print("=" * 60)

    # Discover venue/event pairs
    pairs = discover_venue_events(data_base)
    print(f"  Discovered: {len(pairs)} venue/event pairs")

    # Filter by venues
    if not is_full and venue_filter:
        original_count = len(pairs)
        pairs = [p for p in pairs if p["venue_id"] in venue_filter]
        print(f"  Filtered: {original_count} → {len(pairs)} pairs (by venue)")

    if args.limit:
        pairs = pairs[:args.limit]

    if not pairs:
        print("  No venue/event pairs found. Exiting.")
        return

    # Load few-shot examples
    example_images = load_condensation_examples()
    if example_images:
        print(f"  Examples: {len(example_images)} condensation image(s) loaded")
    else:
        print("  Examples: not available (no condensation examples found)")

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

    # Track totals
    total_input_tokens = 0
    total_output_tokens = 0
    total_thinking_tokens = 0
    processed_count = 0
    skipped_count = 0
    condensation_count = 0

    total_pairs = len(pairs)
    results = [None] * total_pairs
    completed = 0

    # Process in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                process_single, i, p["venue_id"], p["event_id"],
                data_base, client, model_name, example_images, media_resolution
            ): i
            for i, p in enumerate(pairs)
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
                print(f"[{completed}/{total_pairs}] {venue_id}/{event_id} — skipped: {reason}")
                if res.get("traceback"):
                    print(res["traceback"])
                skipped_count += 1
            elif res["processed"]:
                token_usage = res["token_usage"]
                total_input_tokens += token_usage["input_tokens"]
                total_output_tokens += token_usage["output_tokens"]
                total_thinking_tokens += token_usage["thinking_tokens"]
                processed_count += 1

                has_cond = result_row["has_condensation"]
                if has_cond == "Yes":
                    condensation_count += 1

                print(f"[{completed}/{total_pairs}] {venue_id}/{event_id}"
                      f" — condensation: {has_cond}"
                      f", camera: {result_row['which_camera']}"
                      f", tokens: {token_usage['total_tokens']:,}")

    # Write output CSV
    fieldnames = ["venue_id", "event_id", "has_condensation", "which_camera", "observation"]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Copy condensation-positive images to output folder
    images_dir = output_dir / "condensation_images"
    copied_files = 0
    for row in results:
        if row and row["has_condensation"] == "Yes":
            venue_id = row["venue_id"]
            event_id = row["event_id"]
            focus_dir = data_base / venue_id / event_id / "focus"
            for cam_file in ("CAM0_1.jpg", "CAM1_1.jpg"):
                src = focus_dir / cam_file
                if src.exists():
                    images_dir.mkdir(parents=True, exist_ok=True)
                    dst = images_dir / f"{venue_id}_{cam_file}"
                    shutil.copy2(src, dst)
                    copied_files += 1

    # Summary
    no_condensation_count = processed_count - condensation_count
    pct = (condensation_count / processed_count * 100) if processed_count > 0 else 0.0

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Output dir:        {output_dir}")
    print(f"  Total pairs:       {total_pairs}")
    print(f"  Processed:         {processed_count}")
    print(f"  Skipped:           {skipped_count}")
    print(f"  Condensation:      {condensation_count} ({pct:.1f}%)")
    print(f"  No condensation:   {no_condensation_count}")
    if copied_files > 0:
        print(f"  Images copied:     {copied_files} files to {images_dir}/")
    print(f"  Output CSV:        {output_path}")

    # Cost calculation
    model_pricing = load_model_pricing()
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
    cost_path = Path(args.cost_file) if args.cost_file else output_dir / "cost.txt"
    with open(cost_path, 'w') as f:
        f.write("CONDENSATION DETECTION - COST REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Processed: {processed_count} pairs\n")
        f.write(f"Skipped: {skipped_count} pairs\n")
        f.write(f"Condensation found: {condensation_count}\n\n")
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
