# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

PCM Push is a camera quality analysis pipeline for sports venues. It has **3 separate flows**:

1. **Focus Flow** — Detects focus problems, condensation, and environmental issues (dark field, obstructions) using Laplacian metrics + Gemini AI
2. **Movement Flow** — Compares current camera images vs reference calibration images using optical flow to detect camera movement
3. **Setup Flow** — Analyzes camera geometry/coverage to determine if zoom or repositioning is needed

Each flow produces an xlsx with per-venue results. All flows auto-detect 2-cam (S2) vs 3-cam (S3) setups via `camera_utils.py`.


---

## Flow 1: Focus (`run_full_pipeline.py`)

**Runbook:** `focus_finel.md`

Orchestrates 7 steps, auto-detects 2 vs 3 cameras:

1. `run_laplacian_pipeline.py` → `laplacian_calculations.py` + `laplacian_th_calculations.py` (focus metrics + severity)
2. `concat_blur.py` — joins with PQS blur xlsx
3. `run_is_measurable.py` — Gemini AI measurability (skippable with `--is-measurable-csv`)
4. `concat_with_is_measurable.py` — joins measurability; **overrides `Focus_severity` to `NA` when `is_measurable=No`**
5. `detect_issues.py` (2-cam) or `detect_issues_3cam.py` (3-cam) — Gemini AI issue detection (skippable with `--detect-issues-csv`)
6. `concat_with_detect_issues.py` — joins detect_issues results
7. `analyze_blur_severity.py` — plots + stats

```bash
# Run the full focus analysis pipeline
python run_full_pipeline.py <data_dir> <blur_xlsx>

# With pre-computed AI results (skip expensive Gemini calls)
python run_full_pipeline.py <data_dir> <blur_xlsx> \
  --is-measurable-csv <path> --detect-issues-csv <path>

# With options
python run_full_pipeline.py <data_dir> <blur_xlsx> \
  --output-dir output_dir/<NAME> \
  --examples-dir examples/ \
  --limit 10 \
  --num-cameras 3
```

### Focus Severity Thresholds (laplacian_th_calculations.py)

Metrics are computed on an 800x800 crop from each camera's overlap edge. For 3-cam setups, `max_dif_rel` = worst of pairs (0,1) and (1,2), and `min_mid`/`max_mid` consider all 3 cameras.

- **NA**: any camera mean intensity < 50 (too dark)
- **Error** (any one triggers):
  - `max_dif_rel` > 1.0 (large difference between camera pairs)
  - `min_mid` <= 15 (one camera very blurry)
  - `avg_mean` < 95 AND `min_mid` <= 25 (dark + bad mid)
  - `max_mid` <= 50 AND `avg_mean` < 100 (all cameras bad)
- **Warning**: `max_dif_rel` >= 0.5 OR `min_mid` <= 25
- **Ok**: all values within acceptable ranges

### Few-Shot Examples

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

---

## Flow 2: Movement

**Runbook:** `new_dataset_runbook_for_movment.md`

Compares current camera images vs calibration reference images using optical flow.

### Steps

1. **`batch_movement.py`** — Optical flow computation (runs from scrapanyzer repo, uses pcm_push venv)
   - Output: `<venue>/<event>/movement/movement.json`
   - Idempotent — skips venues with existing `movement.json`

2. **`add_movement_to_blur.py`** — Joins movement data with `PQS_blur_by_venue.xlsx`, classifies movement severity
   - Output: `output_dir/PQS_blur_by_venue_with_movement_<DATASET>.xlsx`
   - New columns: `calibration_movement`, `movement_indicator`, `movement_length_cam0/1/2`, `movement_severity`

3. **`batch_blend.py`** — Creates panoramic blend images (current vs reference, 50/50 alpha blend)
   - Output: `<venue>/<event>/movement/movement_<venue>_pano_<sport>_{current,reference,blend}.jpg`
   - Idempotent

4. **`add_blend_links_to_xlsx.py`** — Adds clickable `file:///` hyperlinks to the movement xlsx (modifies in-place)

5. **(Optional) `consolidate_runs.py`** — Merges two dataset runs, picks best row per venue. Uses hardcoded paths at top of file — edit lines 21-27 before running.

```bash
# Step 1: Movement analysis (from scrapanyzer repo, use pcm_push venv)
cd /home/liorb/work/proactive-camera-monitoring/venue/scrapanyzer
/home/liorb/work/pcm_push/venv/bin/python3 batch_movement.py /home/liorb/work/pcm_push/data/<DATASET>

# Steps 2-4: From pcm_push repo
cd /home/liorb/work/pcm_push
python3 add_movement_to_blur.py data/<DATASET>
python3 batch_blend.py data/<DATASET>
python3 add_blend_links_to_xlsx.py data/<DATASET>
```

---

## Flow 3: Setup

**Runbook:** `new_dataset_runbook_setup.md`

Analyzes camera geometry/coverage (field area, spare pixels, overlap) to determine if zoom adjustment or on-site repositioning is needed.

### Steps

1. **`batch_setup.py`** — Runs setup analysis on all venues (from proactive-camera-monitoring repo)
   - Output: `<venue>/<ts>/setup/setup.json`, `setup.csv`, `setup_join.jpg`
   - Idempotent — skips venues with existing `setup.json`
   - Requires `collected/` + `multisportcalibration/` directories
   - **Note:** The setup code (in proactive-camera-monitoring) was designed for production and includes S3 uploads. `batch_setup.py` automatically sets `PRO_CAM_MON_OFFLINE=1` to skip S3 uploads — no manual action needed. The Movement flow does NOT use this variable.

2. **`enrich_blur_with_setup.py`** — Joins setup data with blur xlsx, adds 15 columns (severity, decision, spare metrics, zoom mode, etc.)
   - Output: enriched xlsx

3. **(Optional) `consolidate_setup_results.py`** — Standalone setup xlsx without joining to blur data
   - Output: `<DATASET>/setup_results.xlsx`

```bash
# Step 1: Run setup analysis (from proactive-camera-monitoring repo)
cd /home/liorb/work/proactive-camera-monitoring
PYTHONPATH=venue/scrapanyzer python venue/scrapanyzer/batch_setup.py <DATASET>

# Step 2a: Enrich blur xlsx with setup data
cd /home/liorb/work/pcm_push
python enrich_blur_with_setup.py --dataset <DATASET> --blur-xlsx <BLUR_XLSX> -o output_dir/pqs_blur_by_venue_with_setup.xlsx

# Step 2b (alternative): Standalone setup xlsx
python consolidate_setup_results.py <DATASET>
```

### Setup Severity & Decision Logic

- **Error**: `all_spares` > 3000
- **Warning**: `all_spares` >= 2000
- **Ok**: `all_spares` < 2000
- **can_improve_by_zoom**: `"Yes"` if `(left_spare + right_spare) / max_camera_overlap / 2` is in [0.7, 1.3]
- **decision**: Error+Yes → `maintain_remote`, Error+No → `maintain_on_site`, Warning → `watch`, Ok → `ok`

---

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

## 2-Camera vs 3-Camera Support

**`camera_utils.py`** provides `detect_dataset_camera_count()` and `find_cam_image()` — all scripts auto-detect camera count.

| Aspect | 2-cam (S2/Windows) | 3-cam (S3) |
|---|---|---|
| **Images** | `CAM0_1.jpg`, `CAM1_1.jpg` | `CAM0_0.jpg`, `CAM1_0.jpg`, `CAM2_0.jpg` |
| **Camera order** | CAM0=right, CAM1=left | CAM0=right, CAM1=center, CAM2=left |
| **Panorama** | CAM1 + CAM0 | CAM2 + CAM1 + CAM0 |
| **Issue detection** | `detect_issues.py` | `detect_issues_3cam.py` |

### Directory usage per flow

| Flow | Reads from (in venue) | Writes to (in venue) | Excel output |
|---|---|---|---|
| **Focus** | `focus/` | — | `output_dir/` |
| **Movement** | `collected/`, calibrations | `movement/` | `output_dir/` |
| **Setup** | `collected/`, `multisportcalibration/` | `setup/` | `output_dir/` |

linux datasets have `collected/` directories instead of `focus/`. Only the **Focus flow** needs symlinks — Movement and Setup read `collected/` directly:
```bash
# Only needed for Focus flow on S3 datasets
find data/<S3_DATASET>/ -type d -name collected -exec sh -c \
  'ln -sf collected "$(dirname "$1")/focus"' _ {} \;
```
this will create Data Directory Structure



```
<data_dir>/<venue_id>/<event_id>/focus/
    CAM0_1.jpg       # Right camera image
    CAM1_1.jpg       # Left camera image
    joined_0_1.jpg   # Side-by-side overlap composition (optional)
    focus.json        # Ground truth focus metrics from production
```


## Configuration Files

- `config.yaml` — Gemini model name, GCP project/location, `max_workers`, retry settings
- `venues.yaml` — Venue filter: list venue IDs to process a subset, or set `full: true` for all venues
- `cost.yaml` — Token pricing per model (threshold-based tiered pricing per 1M tokens)
- `service-account-key.json` — GCP credentials (auto-detected by scripts if `GOOGLE_APPLICATION_CREDENTIALS` not set)

## Dependencies

Python 3.12 with venv. No `requirements.txt` — key packages:
- **opencv-python** (`cv2`) — `laplacian_calculations.py`, `batch_movement.py`, `batch_blend.py`
- **openpyxl** — Excel output with clickable hyperlinks (multiple scripts)
- **Pillow** (`PIL`) — `is_measurable.py`, `detect_issues.py`, `consolidate_runs.py`
- **google-genai** (`from google import genai`) — Gemini API client
- **pandas** + **plotly** — `analyze_blur_severity.py` (reads both CSV and XLSX input)
- **PyYAML** — config parsing
- **boto3** — setup analysis (S3 access)

## Related Documentation

- `focus_finel.md` — full focus pipeline documentation with output column descriptions
- `new_dataset_runbook_for_movment.md` — movement flow runbook with troubleshooting
- `new_dataset_runbook_setup.md` — setup flow runbook with troubleshooting

## Cross-Repository Dependencies

pcm_push calls scripts from `/home/liorb/work/proactive-camera-monitoring/venue/scrapanyzer/` as subprocesses. No Python imports cross the repo boundary — pcm_push reads the JSON outputs these scripts produce.

### Movement Flow → `batch_movement.py`
- **Script**: `venue/scrapanyzer/batch_movement.py`
- **Dependencies**: `image_movement_calc.py`, `execution_context.py`
- **Output read by pcm_push**: `<venue>/<event>/movement/movement.json`
- **How to run**: from scrapanyzer dir, using pcm_push venv

### Setup Flow → `batch_setup.py`
- **Script**: `venue/scrapanyzer/batch_setup.py`
- **Dependencies**: `run_all_setup.py`, `camera_calibration.py`, `execution_context.py`, `aws_utils.py`
- **Output read by pcm_push**: `<venue>/<event>/setup/setup.json`, `setup.csv`, `setup_join.jpg`
- **How to run**: from proactive-camera-monitoring root, with `PYTHONPATH=venue/scrapanyzer`
- **Env var**: `PRO_CAM_MON_OFFLINE=1` (set automatically by batch_setup.py)
