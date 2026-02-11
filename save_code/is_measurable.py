#!/usr/bin/env python3
"""
Is Measurable Analysis Script

Analyzes camera frames using Gemini to determine:
1. Whether environmental conditions allow valid focus measurement
2. Quality issues (focus problems, condensation) detected in the images
"""

import os
import io
import csv
import json
import yaml
import argparse
import traceback
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Set Google Cloud credentials if not already set
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    script_dir = Path(__file__).parent.resolve()
    credentials_path = script_dir / "service-account-key.json"
    if credentials_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

from PIL import Image
from google import genai
from google.genai import types


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


def load_examples() -> Dict[str, Union[bytes, List[bytes]]]:
    """Load and resize few-shot example images from examples/ directory.

    Returns a dict mapping example keys to JPEG bytes resized to 512x512
    at quality 70 to minimize token cost. The 'condensation' key maps to
    a list of bytes (all images in examples/Condensation/), while other
    keys map to single bytes.
    """
    script_dir = Path(__file__).parent.resolve()
    example_paths = {
        "dark_field": script_dir / "examples" / "examples_of_dark_field" / "CAM0_1.jpg",
        "dark_field_ambient": script_dir / "examples" / "examples_of_dark_field_ambient" / "CAM0_1.jpg",
        "object_in_field": script_dir / "examples" / "examples_of_object_in_field" / "5bfac2507a75205380d5b04b_CAM1_1.jpg",
    }

    examples: Dict[str, Union[bytes, List[bytes]]] = {}
    for key, path in example_paths.items():
        if not path.exists():
            print(f"  Warning: Example image not found: {path}")
            continue
        try:
            img = Image.open(path)
            img = img.resize((512, 512), Image.LANCZOS)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=70)
            examples[key] = buf.getvalue()
        except Exception as e:
            print(f"  Warning: Failed to load example {key}: {e}")

    # Load all condensation examples from the directory
    condensation_dir = script_dir / "examples" / "Condensation"
    if condensation_dir.is_dir():
        cond_images = []
        for path in sorted(
            p for p in condensation_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ):
            try:
                img = Image.open(path)
                img = img.resize((512, 512), Image.LANCZOS)
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=70)
                cond_images.append(buf.getvalue())
            except Exception as e:
                print(f"  Warning: Failed to load condensation example {path.name}: {e}")
        if cond_images:
            examples["condensation"] = cond_images
    else:
        print(f"  Warning: Condensation examples directory not found: {condensation_dir}")

    return examples


# Prompt for measurability analysis — split into parts for few-shot example interleaving
MEASURABILITY_PROMPT_PART1 = """You are analyzing two images from adjacent cameras at a sports venue.

The first image is from the RIGHT camera (CAM0), the second image is from the LEFT camera (CAM1).

## STEP 1: Check Environmental Conditions FIRST

Before assessing focus, check if environmental conditions allow reliable measurement.

Answer "No" for is_measurable if you see ANY of these:

**Weather conditions:**
- SNOW: Snow covering the playing surface INSIDE the field (white/bright field, reduced contrast, snow accumulation on field). Snow visible only OUTSIDE the field boundaries does NOT affect measurability.
- FOG/MIST: Hazy appearance, reduced visibility, objects fade into gray/white background
- RAIN: Water droplets on lens, streaks, wet reflective surfaces, reduced clarity

**Lighting problems:**
- TOO DARK: Main overhead/arena lights are OFF or insufficient. Even if some details are visible from ambient light, window light, or emergency lighting — if the venue's main lights are clearly not on, it is NOT measurable.
- TOO BRIGHT: Overexposed areas, washed out whites, sun glare, lens flare
- SUN ON CAMERA: Direct sunlight causing glare or flare artifacts

**Visibility issues:**
- Smoke, dust, or haze reducing visibility
- Lens obstructed, dirty, or covered
- Objects blocking the camera view

**NOT a reason for "No":** Condensation/moisture ON the lens — this is still measurable but is a quality issue (see Step 2).

## Reference Examples

The following are examples of conditions that make measurement NOT possible:

**Example 1 - Dark Field (Too Dark):**"""

EXAMPLE_DARK_FIELD_LABEL = """This field is too dark to measure — the lighting is insufficient.

**Example 1b - Dark Field with Ambient Light (Still Too Dark):**"""

EXAMPLE_DARK_FIELD_AMBIENT_LABEL = """The main overhead lights are OFF. Even though some court details are visible from window/ambient light, the venue lighting is insufficient for measurement. This is NOT measurable.

**Example 2 - Object Interfering:**"""

EXAMPLE_OBJECT_IN_FIELD_LABEL = """An object is blocking/interfering with the camera view.

**Example 3 - Condensation on Lens (STILL MEASURABLE):**"""

EXAMPLE_CONDENSATION_LABEL = """This image shows condensation/moisture on the camera lens. The field is still visible through the condensation, so is_measurable = "Yes", but quality_issue_type = "Condensation"."""

MEASURABILITY_PROMPT_PART2 = """
## STEP 2: Assess Quality Issues (only if measurable)

If environmental conditions are clear (is_measurable = "Yes"), check for these quality issues:

**Focus issues:**
- Soft or blurry edges on players, field lines, or objects
- Out of focus areas that should be sharp
- Compare sharpness between the two cameras

**Condensation/moisture on lens:**
- Foggy or hazy patches ON the lens (not atmospheric fog)
- Moisture droplets visible on camera housing
- Parts of image obscured by lens moisture while background is still partially visible

## Response Format

Provide a JSON response with exactly this structure:
{
    "observation": "Brief description of what you see in the images (weather, lighting, field condition)",
    "is_measurable": "Yes" or "No",
    "not_measurable_reason": "reason if not measurable, empty string if measurable",
    "has_quality_issue": "Yes" or "No",
    "quality_issue_type": "Focus" or "Condensation" or "None",
    "which_camera": "Left" or "Right" or "Both" or "None"
}

**Rules:**
- If is_measurable is "No", set has_quality_issue to "No", quality_issue_type to "None", which_camera to "None"
- quality_issue_type: "Focus" for focus problems, "Condensation" for lens moisture, "None" if no issues
- For which_camera: "Left" = CAM1, "Right" = CAM0, "Both" = both cameras, "None" = no issues
- Return ONLY valid JSON, no markdown formatting or extra text"""


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

    # Check for "full: true" option
    if data.get("full", False):
        return True, set()

    # Get venue list (supports both "venues:" key and plain list)
    if isinstance(data, list):
        venues = set(data)
    elif isinstance(data, dict):
        venues = set(data.get("venues", []) or [])
    else:
        venues = set()

    return False, venues


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


def calculate_pricing(model_name: str, input_tokens: int, output_tokens: int, model_pricing: dict = None) -> Dict[str, float]:
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


def analyze_images(client: genai.Client, model_name: str, cam0_bytes: bytes, cam1_bytes: bytes,
                    examples: Optional[Dict[str, bytes]] = None,
                    media_resolution: Optional[str] = None) -> Tuple[dict, dict]:
    """Send images to Gemini and get measurability analysis."""
    # Build parts list, interleaving example images within the prompt
    parts = [types.Part.from_text(text=MEASURABILITY_PROMPT_PART1)]

    # Build kwargs for Part.from_bytes, including media_resolution if set
    image_kwargs = {}
    if media_resolution:
        image_kwargs["media_resolution"] = media_resolution

    if examples and "dark_field" in examples:
        parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=examples["dark_field"], **image_kwargs))
    parts.append(types.Part.from_text(text=EXAMPLE_DARK_FIELD_LABEL))

    if examples and "dark_field_ambient" in examples:
        parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=examples["dark_field_ambient"], **image_kwargs))
    parts.append(types.Part.from_text(text=EXAMPLE_DARK_FIELD_AMBIENT_LABEL))

    if examples and "object_in_field" in examples:
        parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=examples["object_in_field"], **image_kwargs))
    parts.append(types.Part.from_text(text=EXAMPLE_OBJECT_IN_FIELD_LABEL))

    if examples and "condensation" in examples:
        cond = examples["condensation"]
        cond_list = cond if isinstance(cond, list) else [cond]
        for cond_bytes in cond_list:
            parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=cond_bytes, **image_kwargs))
    parts.append(types.Part.from_text(text=EXAMPLE_CONDENSATION_LABEL))

    parts.append(types.Part.from_text(text=MEASURABILITY_PROMPT_PART2))

    # Append the actual camera images to analyze
    parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=cam0_bytes, **image_kwargs))
    parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=cam1_bytes, **image_kwargs))

    contents = [
        types.Content(
            role="user",
            parts=parts
        )
    ]

    generation_config = types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        max_output_tokens=65536,  # Increased to prevent MAX_TOKENS errors with thinking models
        labels={"job": "is-measurable", "pipeline": "pcm"},
    )

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=generation_config,
    )

    # Parse JSON response - extract text from candidates when thinking is enabled
    response_text = None

    # Try to get text from candidates' parts (needed when thinking mode is active)
    if response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                # Skip thinking parts, get only text parts
                if hasattr(part, 'text') and part.text:
                    response_text = part.text
                    break

    # Fallback to response.text
    if response_text is None:
        response_text = response.text

    if response_text is None:
        # Check if there's a block reason or other issue
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.finish_reason:
                reason = str(candidate.finish_reason)
            else:
                reason = "Empty response from model"
        else:
            reason = "No candidates in response"

        token_usage = get_token_usage(response)
        return {
            "is_measurable": "Error",
            "not_measurable_reason": reason,
            "has_quality_issue": "No",
            "quality_issue_type": "None",
            "which_camera": "None"
        }, token_usage

    response_text = response_text.strip()
    # Remove markdown code blocks if present
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        result = {
            "is_measurable": "Unknown",
            "not_measurable_reason": f"Failed to parse response: {response_text[:100]}",
            "has_quality_issue": "No",
            "quality_issue_type": "None",
            "which_camera": "None"
        }

    token_usage = get_token_usage(response)

    return result, token_usage


def process_single_row(index: int, row: dict, data_base: Path, client: genai.Client,
                        model_name: str, examples: Optional[Dict[str, bytes]],
                        media_resolution: Optional[str] = None) -> dict:
    """Process a single row: load images, call Gemini, return results.

    Returns a dict with keys: index, result_row, token_usage, processed, skipped.
    """
    venue_id = row["venue_id"]
    event_id = row["event_id"]

    result_row = {
        "venue_id": venue_id,
        "event_id": event_id,
        "is_measurable": "",
        "not_measurable_reason": "",
        "has_quality_issue": "",
        "quality_issue_type": "",
        "which_camera": ""
    }

    focus_dir = data_base / venue_id / event_id / "focus"

    if not focus_dir.exists():
        result_row["is_measurable"] = "N/A"
        result_row["not_measurable_reason"] = "Images not found"
        result_row["has_quality_issue"] = "None"
        result_row["quality_issue_type"] = "None"
        result_row["which_camera"] = "None"
        return {"index": index, "result_row": result_row, "token_usage": None,
                "processed": False, "skipped": True, "skip_reason": "focus directory not found"}

    images = load_images(focus_dir)
    if images is None:
        result_row["is_measurable"] = "N/A"
        result_row["not_measurable_reason"] = "Could not load images"
        result_row["has_quality_issue"] = "None"
        result_row["quality_issue_type"] = "None"
        result_row["which_camera"] = "None"
        return {"index": index, "result_row": result_row, "token_usage": None,
                "processed": False, "skipped": True, "skip_reason": "could not load images"}

    cam0_bytes, cam1_bytes = images

    try:
        result, token_usage = analyze_images(client, model_name, cam0_bytes, cam1_bytes, examples, media_resolution)

        is_measurable = result.get("is_measurable", "Unknown")
        result_row["is_measurable"] = is_measurable

        reason = result.get("not_measurable_reason", "")
        result_row["not_measurable_reason"] = reason.replace(",", ";") if reason else ""

        if is_measurable == "No":
            result_row["has_quality_issue"] = "None"
            result_row["quality_issue_type"] = "None"
            result_row["which_camera"] = "None"
        else:
            has_quality = result.get("has_quality_issue", "No")
            quality_type = result.get("quality_issue_type", "None")
            if has_quality == "Yes":
                result_row["has_quality_issue"] = "Yes"
                result_row["quality_issue_type"] = quality_type if quality_type else "Focus"
            else:
                result_row["has_quality_issue"] = "No"
                result_row["quality_issue_type"] = "None"
            result_row["which_camera"] = result.get("which_camera", "None") or "None"

        return {"index": index, "result_row": result_row, "token_usage": token_usage,
                "processed": True, "skipped": False}

    except Exception as e:
        tb = traceback.format_exc()
        result_row["is_measurable"] = "Error"
        result_row["not_measurable_reason"] = str(e)
        result_row["has_quality_issue"] = "None"
        result_row["quality_issue_type"] = "None"
        result_row["which_camera"] = "None"
        return {"index": index, "result_row": result_row, "token_usage": None,
                "processed": False, "skipped": True, "error": str(e), "traceback": tb}


def main():
    parser = argparse.ArgumentParser(description="Analyze camera frames for measurability")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to process")
    parser.add_argument("--input", type=str, default="laplacian_th.csv", help="Input CSV file")
    parser.add_argument("--output", type=str, default="is_measurable.csv", help="Output CSV file")
    parser.add_argument("--data-dir", type=str, default=None, help="Base data directory (default: from config.yaml)")
    parser.add_argument("--venues", type=str, default=None, help="YAML file with venue filter (default: venues.yaml if exists)")
    parser.add_argument("--cost-file", type=str, default=None, help="Output path for cost report (default: cost.txt in script dir)")
    parser.add_argument("--max-workers", type=int, default=None, help="Max parallel Gemini API calls (default: from config.yaml or 10)")
    args = parser.parse_args()

    # Load config
    config = load_config()
    model_name = config.get("model", "gemini-3-flash-preview")
    project = config.get("project", "pixellot-ai")
    location = config.get("location", "global")

    # Use data_dir from args or config
    data_dir = args.data_dir if args.data_dir else config.get("data_dir", "data_13")

    # Max parallel workers
    max_workers = args.max_workers if args.max_workers else config.get("max_workers", 10)

    # Media resolution for image parts (e.g., MEDIA_RESOLUTION_ULTRA_HIGH)
    media_resolution = config.get("media_resolution")

    # Load venue filter
    is_full, venue_filter = load_venues(args.venues)

    # Create timestamped output directory
    script_dir = Path(__file__).parent.resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = script_dir / "output_dir" / f"is_measurable_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output CSV path: use user-provided path or default inside output_dir
    if args.output != "is_measurable.csv":
        output_path = Path(args.output) if Path(args.output).is_absolute() else script_dir / args.output
    else:
        output_path = output_dir / "is_measurable.csv"

    # Determine cost file path: use user-provided path or default inside output_dir
    cost_path = Path(args.cost_file) if args.cost_file else output_dir / "cost.txt"

    print("=" * 60)
    print("  IS MEASURABLE ANALYSIS")
    print("=" * 60)
    print(f"  Output dir: {output_dir}")
    print(f"  Input: {args.input}")
    print(f"  Output: {output_path}")
    print(f"  Model: {model_name}")
    if is_full:
        print("  Venues: ALL (full mode)")
    else:
        print(f"  Venues: {len(venue_filter)} filtered")
    print(f"  Max workers: {max_workers}")
    if media_resolution:
        print(f"  Media resolution: {media_resolution}")
    if args.limit:
        print(f"  Limit: {args.limit} rows")
    print("=" * 60)

    # Load few-shot example images (once, reused for all requests)
    examples = load_examples()
    if examples:
        print(f"  Examples: {len(examples)} loaded ({', '.join(examples.keys())})")
    else:
        print("  Examples: none (few-shot examples not available)")

    # Retry settings for rate limiting (429 RESOURCE_EXHAUSTED)
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

    # Read input CSV
    input_path = script_dir / args.input
    data_base = script_dir / data_dir

    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        # Output only venue_id, event_id + Gemini results (no redundant focus columns)
        fieldnames = [
            "venue_id",
            "event_id",
            "is_measurable",
            "not_measurable_reason",
            "has_quality_issue",
            "quality_issue_type",
            "which_camera"
        ]

    # Filter by venues if not in full mode
    if not is_full and venue_filter:
        original_count = len(rows)
        rows = [r for r in rows if r["venue_id"] in venue_filter]
        print(f"  Filtered: {original_count} → {len(rows)} rows (by venue)")

    if args.limit:
        rows = rows[:args.limit]

    # Track totals for cost report
    total_input_tokens = 0
    total_output_tokens = 0
    total_thinking_tokens = 0
    processed_count = 0
    skipped_count = 0

    # Process rows in parallel
    total_rows = len(rows)
    results = [None] * total_rows
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_single_row, i, row, data_base, client, model_name, examples, media_resolution): i
            for i, row in enumerate(rows)
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
                print(f"[{completed}/{total_rows}] {venue_id}/{event_id} — skipped: {reason}")
                if res.get("traceback"):
                    print(res["traceback"])
                skipped_count += 1
            elif res["processed"]:
                token_usage = res["token_usage"]
                total_input_tokens += token_usage["input_tokens"]
                total_output_tokens += token_usage["output_tokens"]
                total_thinking_tokens += token_usage["thinking_tokens"]
                processed_count += 1

                print(f"[{completed}/{total_rows}] {venue_id}/{event_id}"
                      f" — measurable: {result_row['is_measurable']}"
                      f", quality_issue: {result_row['has_quality_issue']}"
                      f", camera: {result_row['which_camera']}"
                      f", tokens: {token_usage['total_tokens']:,}")

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
    model_pricing = load_model_pricing()
    pricing = calculate_pricing(
        model_name,
        total_input_tokens,
        total_output_tokens + total_thinking_tokens,
        model_pricing
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
        f.write("IS MEASURABLE ANALYSIS - COST REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Processed: {processed_count} images\n")
        f.write(f"Skipped: {skipped_count} images\n\n")
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
