# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

PCM Push is a camera focus quality analysis pipeline for sports venues. It detects focus problems, condensation, and environmental issues (dark field, obstructions) across paired camera setups (CAM0/CAM1 = right/left cameras).

## Pipeline Scripts (run in order)

```bash
# Step 1: Compute Laplacian focus metrics from camera images
python laplacian_calculations.py <data_dir>

# Step 2: Classify severity (Ok/Warning/Error/NA) from Step 1 output
python laplacian_th_calculations.py <step1_output.csv>

# Step 3: AI measurability + quality issue detection (reads Step 2 CSV, needs GCP credentials)
python is_measurable.py --input <step2_output.csv> --data-dir <data_dir>

# Alternative to Step 3: Xlsx-driven issue detection (one event per venue, like run_is_measurable.py)
python detect_issues.py --dataset <data_dir> --blur-xlsx <blur_xlsx>

# Post-pipeline: Consolidate two pipeline runs into best-per-venue xlsx
# Picks the best row per venue, generates joined CAM0+CAM1 images, and adds image hyperlink columns
python consolidate_runs.py <run1_xlsx> <run2_xlsx> -o <output.xlsx>

# Setup analysis: Enrich blur xlsx with camera geometry data (setup.json)
# Adds 15 columns: severity, coverage spares, zoom improvability, decision
python enrich_blur_with_setup.py --dataset <data_dir> --blur-xlsx <blur_xlsx> -o <output.xlsx>

# Setup analysis: Standalone setup xlsx (one row per venue, 20 columns, no blur join)
python consolidate_setup_results.py <data_dir>
```

Setup analysis is a cross-repo process — see `new_dataset_runbook_setup.md` for the full runbook including `batch_setup.py` (in `proactive-camera-monitoring` repo).

Each script creates a timestamped subdirectory under `output_dir/` (e.g., `output_dir/laplacian_2025-02-12_14-30/`).

## Architecture

**`is_measurable.py` is the shared utility module.** `detect_issues.py` and `run_is_measurable.py` import from it:
- `load_config()`, `load_venues()`, `load_model_pricing()`, `calculate_pricing()`
- `get_token_usage()`, `pil_to_bytes()`, `load_images()`

**`run_is_measurable.py` provides xlsx helpers.** `detect_issues.py` imports from it:
- `read_blur_xlsx()`, `find_venue_focus_dir()`

Both `run_is_measurable.py` and `detect_issues.py` are xlsx-driven (one event per venue) and follow the same pattern:
1. Read `PQS_blur_by_venue.xlsx` → apply venue filter → `find_venue_focus_dir()` per venue
2. Load config from `config.yaml`, venues from `venues.yaml`
3. Initialize `genai.Client` with Vertex AI + retry settings
4. Process venues in parallel via `concurrent.futures.ThreadPoolExecutor`
5. Each worker: load images → build multimodal prompt with few-shot examples → call Gemini → parse JSON response
6. Write results CSV + cost report

**Gemini prompt structure**: Both scripts build `types.Content` with interleaved text and image `types.Part` objects. `is_measurable.py` uses a two-part prompt (`MEASURABILITY_PROMPT_PART1` / `PART2`) with labeled example images between them. `detect_issues.py` uses a single `DETECT_ISSUES_PROMPT` with categorized few-shot examples (joined, change_focus, complete_focus) each with a descriptive label constant.

**`run_is_measurable.py`** sends only CAM0 and CAM1 images to the indoor/outdoor analyzers (joined image is skipped — CAM0+CAM1 have all info needed for measurability).

## Configuration Files

- `config.yaml` — Gemini model name, GCP project/location, `max_workers`, retry settings, `media_resolution`
- `venues.yaml` — Venue filter: list venue IDs to process a subset, or set `full: true` for all venues
- `cost.yaml` — Token pricing per model (threshold-based tiered pricing per 1M tokens)
- `service-account-key.json` — GCP credentials (auto-detected by scripts if `GOOGLE_APPLICATION_CREDENTIALS` not set)

## Data Directory Structure

```
<data_dir>/<venue_id>/<event_id>/focus/
    CAM0_1.jpg       # Right camera image
    CAM1_1.jpg       # Left camera image
    joined_0_1.jpg   # Side-by-side overlap composition (optional)
    focus.json        # Ground truth focus metrics from production
```

Data directories: `all_data_02_09/` (~5100 entries), `data_11_2/`, `data/16_2_linux_s2/`, `data/18_2_linux_s2/`.

## Focus Severity Thresholds (laplacian_th_calculations.py)

- **NA**: either camera mean intensity < 70 (too dark)
- **Error**: `focus_abs_dif_rel` > 1.25, or either camera `mid_focus` <= 10
- **Warning**: `focus_abs_dif_rel` >= 0.7, or either camera `mid_focus` <= 20
- **Ok**: all values within acceptable ranges

## Setup Severity Thresholds (enrich_blur_with_setup.py / consolidate_setup_results.py)

- **Error**: `all_spares` > 3000
- **Warning**: `all_spares` >= 2000
- **Ok**: `all_spares` < 2000
- **can_improve_by_zoom**: `"Yes"` if `(left_spare + right_spare) / max_camera_overlap / 2` is in [0.7, 1.3]
- **decision**: Error+Yes → `maintain_remote`, Error+No → `maintain_on_site`, Warning → `watch`, Ok → `ok`

## Few-Shot Examples

`is_measurable.py` loads example images from subdirectories under `examples/`:
- `examples/examples_of_dark_field/` — dark field reference
- `examples/examples_of_dark_field_ambient/` — ambient-lit but still too dark
- `examples/examples_of_object_in_field/` — obstructed camera view
- `examples/Condensation/` — all `.jpg`/`.png` files loaded as condensation examples

`detect_issues.py` loads categorized examples from subdirectories under `examples/` (override with `--use-examples <dir>`):
- `examples/joined/` — joined image showing sharpness difference between cameras
- `examples/change_focus/` — focus change visible within a single camera image
- `examples/complete_focus/` — completely smoothed camera with bad focus all over the lens

Each category has a label constant (`EXAMPLE_JOINED_LABEL`, `EXAMPLE_CHANGE_FOCUS_LABEL`, `EXAMPLE_COMPLETE_FOCUS_LABEL`) and is registered in `EXAMPLE_CATEGORIES`.

To add a new category to `is_measurable.py`: add image to `examples/examples_of_<category>/`, register the path in `load_examples()`, create an `EXAMPLE_<CATEGORY>_LABEL` prompt constant, and wire it into `analyze_images()`.

To add a new category to `detect_issues.py`: create `examples/<category>/` with images, add an `EXAMPLE_<CATEGORY>_LABEL` constant, and append to `EXAMPLE_CATEGORIES`.

## Dependencies

Python 3.12 with venv. No `requirements.txt` — key packages:
- **opencv-python** (`cv2`) — `laplacian_calculations.py`
- **openpyxl** — `concat_blur.py`, `concat_with_is_measurable.py`, `enrich_blur_with_setup.py`, `consolidate_setup_results.py` (Excel output with clickable hyperlinks)
- **Pillow** (`PIL`) — `is_measurable.py`, `detect_issues.py`, `consolidate_runs.py`
- **google-genai** (`from google import genai`) — Gemini API client
- **pandas** + **plotly** — `analyze_blur_severity.py` (reads both CSV and XLSX input)
- **PyYAML** — config parsing
