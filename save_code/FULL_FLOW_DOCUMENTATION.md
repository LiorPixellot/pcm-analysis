# Full Flow Pipeline Documentation

## Camera Focus Analysis System

**Version:** 1.2
**Author:** PCM Team
**Last Updated:** February 2026

**Changes in v1.2:**
- Switched default model to `gemini-3-flash-preview` (faster, cheaper, less prone to MAX_TOKENS errors)
- Increased `max_output_tokens` to 65536 to prevent truncation with thinking models
- Moved config to `config.yaml` in main directory (was `use_gemini/config.yaml`)
- Refactored `full_flow.py` to call `is_measurable.py` as subprocess (removed ~300 lines of duplicated code)

**Changes in v1.1:**
- Simplified Gemini analysis to check **focus issues only** (not all quality issues)
- `quality_issue_type` now outputs only `Focus` or `None`

---

## Overview

`full_flow.py` is a unified pipeline that automates the entire camera focus quality analysis process for sports venue camera systems. It combines four processing stages into a single executable workflow.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FULL FLOW PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│   │   STEP 1     │    │   STEP 2     │    │   STEP 3     │    │  STEP 4  │ │
│   │  Laplacian   │───▶│  Threshold   │───▶│ Measurable   │───▶│  Concat  │ │
│   │ Calculations │    │ Calculations │    │   (Gemini)   │    │ Results  │ │
│   └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘ │
│         │                   │                   │                   ▲       │
│         ▼                   ▼                   ▼                   │       │
│   laplacian_           laplacian_th.csv   is_measurable.csv        │       │
│   calculations.csv     (focus_* cols)     (quality_* cols)         │       │
│                              │                   │                  │       │
│                              └───────────────────┴──────────────────┘       │
│                                    Both merged into                         │
│                                 concatenated_results.csv                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Input Data Structure

The pipeline expects a data directory with the following structure:

```
{data_dir}/
├── {venue_id_1}/
│   ├── {event_id_1}/
│   │   └── focus/
│   │       ├── CAM0_1.jpg    (Right camera image)
│   │       ├── CAM1_1.jpg    (Left camera image)
│   │       └── focus.json    (Optional: validation data)
│   └── {event_id_2}/
│       └── focus/
│           └── ...
├── {venue_id_2}/
│   └── ...
```

---

## Pipeline Stages

### Stage 1: Laplacian Calculations

**Purpose:** Calculate focus quality metrics using Laplacian variance analysis.

**Process:**
1. Load CAM0_1.jpg (right camera) and CAM1_1.jpg (left camera)
2. Extract 800x800 pixel regions from image centers
3. Apply Laplacian filter to detect edges/sharpness
4. Calculate variance as focus quality metric
5. Validate against existing focus.json if available

**Output Metrics:**

| Metric | Description |
|--------|-------------|
| `focus_right_mean` | Mean intensity of right camera region |
| `focus_left_mean` | Mean intensity of left camera region |
| `focus_right_mid` | Laplacian variance of right camera (focus score) |
| `focus_left_mid` | Laplacian variance of left camera (focus score) |
| `focus_abs_dif_rel` | Relative difference between cameras |
| `validation_status` | PASS/FAIL against focus.json |

**Output File:** `laplacian_calculations.csv`

---

### Stage 2: Threshold Calculations

**Purpose:** Classify focus quality into severity levels based on thresholds.

**Severity Classification Logic:**

```
┌─────────────────────────────────────────────────────────────┐
│                    SEVERITY DECISION TREE                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  IF focus_right_mean < 70 OR focus_left_mean < 70           │
│     └─▶ "NA" (image too dark to measure)                    │
│                                                             │
│  ELSE calculate from two factors:                           │
│                                                             │
│  Factor 1: Relative Difference (focus_abs_dif_rel)          │
│     • > 1.25  ──▶ "Error"                                   │
│     • 0.70-1.25 ──▶ "Warning"                               │
│     • < 0.70  ──▶ "Ok"                                      │
│                                                             │
│  Factor 2: Mid Focus Values (per camera)                    │
│     • ≤ 10   ──▶ "Error"                                    │
│     • 10-20  ──▶ "Warning"                                  │
│     • > 20   ──▶ "Ok"                                       │
│                                                             │
│  Final = WORST(Factor 1, Factor 2)                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Output File:** `laplacian_th.csv` (adds `Focus_severity` column)

---

### Stage 3: Measurability Analysis (Gemini AI)

**Purpose:** Use AI vision to determine if environmental conditions allow valid measurement and detect **focus issues only**.

**AI Model:** Gemini 3 Flash Preview (via Vertex AI) - configurable in `config.yaml`

**Implementation:** `full_flow.py` calls `is_measurable.py` as a subprocess, which handles all Gemini API communication. This keeps the Gemini logic centralized in one file.

**Analysis Categories:**

#### 1. Is Measurable (Yes/No)

Conditions that make measurement **unreliable**:

| Category | Examples |
|----------|----------|
| Weather | Fog, snow, rain, hail, mist |
| Lighting | Too dark, too bright, sun glare |
| Visibility | Smoke, dust, lens obstructed |

#### 2. Focus Issues Only

The AI checks **only for focus/sharpness problems**:
- Soft images
- Blurry images
- Out of focus images

**Note:** Other quality issues (chromatic aberration, white balance, etc.) are not checked.

#### 3. Which Camera

Identifies if focus issues are on: `Left` / `Right` / `Both` / `None`

**Output Columns (is_measurable.csv):**

| Column | Description | Values |
|--------|-------------|--------|
| `venue_id` | Venue identifier | string |
| `event_id` | Event identifier | string |
| `is_measurable` | Can focus be measured? | Yes / No / N/A / Error |
| `not_measurable_reason` | Why not measurable | fog, dark, sun glare, etc. |
| `has_quality_issue` | Has focus problem? | Yes / No / None |
| `quality_issue_type` | Type of issue | **Focus** / **None** |
| `which_camera` | Which camera affected | Left / Right / Both / None |

**Output Files:**
- `is_measurable.csv` - Gemini focus analysis only (7 columns)
- `cost.txt` - Token usage and API cost report

**Note:** This file contains only Gemini results. Laplacian focus metrics are in `laplacian_th.csv`.

**Cost Estimate:** ~$0.005 per image pair (Flash model)

---

### Stage 4: Concatenate with Annotations

**Purpose:** Merge all analysis results with existing manual annotations.

**Process:**
1. Read Excel file: `Annotators - Working Table 2025.xlsx` (POC sheet)
2. Read `laplacian_th.csv` for focus metrics (focus_* columns)
3. Read `is_measurable.csv` for Gemini results (quality_* columns)
4. Join both on `System ID` = `venue_id`
5. Add new columns with category prefixes

**Important:** The Excel file contains multiple rows per `System ID` (one row per calibration).
The analysis is performed **once per venue** using the first event found in the data directory.
All calibration rows for the same venue will receive the same focus and quality metrics.

```
┌─────────────────────────────────────────────────────────────────┐
│                    VENUE → EXCEL MATCHING                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Analysis (one per venue):          Excel (multiple per venue): │
│  ┌─────────────────────┐            ┌─────────────────────────┐ │
│  │ venue_A → metrics   │ ──────────▶│ venue_A, Calib #1       │ │
│  │                     │            │ venue_A, Calib #2       │ │
│  │                     │            │ venue_A, Calib #3       │ │
│  └─────────────────────┘            └─────────────────────────┘ │
│                                                                 │
│  Same metrics applied to all calibration rows for that venue    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**New Columns Added (with category prefix):**

| Column | Source | Description |
|--------|--------|-------------|
| `focus_right_mean` | Laplacian | Right camera mean intensity |
| `focus_left_mean` | Laplacian | Left camera mean intensity |
| `focus_right_mid` | Laplacian | Right camera variance (focus score) |
| `focus_left_mid` | Laplacian | Left camera variance (focus score) |
| `focus_abs_dif_rel` | Laplacian | Relative difference between cameras |
| `focus_severity` | Threshold | Ok / Warning / Error / NA |
| `quality_is_measurable` | Gemini | Yes / No |
| `quality_not_measurable_reason` | Gemini | fog, dark, sun glare, etc. |
| `quality_has_issue` | Gemini | Yes / No |
| `quality_issue_type` | Gemini | **Focus** / **None** |
| `quality_which_camera` | Gemini | Left / Right / Both / None |

**Output File:** `concatenated_results.csv`

---

## Usage

### Basic Commands

```bash
# Activate virtual environment
source /home/lior/Work/pcm_new/venv/bin/activate

# Test run (1 event)
python3 full_flow.py /path/to/data_dir --limit 1

# Test run (5 events)
python3 full_flow.py /path/to/data_dir --limit 5

# Full production run
python3 full_flow.py /path/to/data_dir
```

### Running is_measurable.py Standalone

You can also run `is_measurable.py` directly for Gemini analysis only:

```bash
# Basic usage (reads from laplacian_th.csv, outputs to is_measurable.csv)
python3 is_measurable.py

# With custom input/output files
python3 is_measurable.py --input my_input.csv --output my_output.csv --data-dir data_13

# Limit number of rows to process
python3 is_measurable.py --limit 10

# Process only venues from a specific file
python3 is_measurable.py --venues venues.yaml

# Process ALL venues (set full: true in venues.yaml)
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `data_dir` | Path to data directory (required) |
| `--limit N` | Process only first N events |
| `--venues FILE` | YAML file with venue IDs to process (default: `venues.yaml` if exists) |
| `--skip-gemini` | Skip AI analysis (Stage 3) |
| `--skip-concat` | Skip concatenation (Stage 4) |

### Venue Filtering

By default, if `venues.yaml` exists in the script directory, only venues listed in it will be processed.

**File:** `venues.yaml`

```yaml
# Set "full: true" to process ALL venues (ignores venue list)
full: false

# List of venue IDs (System IDs) to process
venues:
  - 5bed37433e663c077bc36cef
  - 5bed375c3e663c077bc36cf1
  - 5bed84843e663c077bc36daf
```

**Alternative simple list format:**
```yaml
- 5bed37433e663c077bc36cef
- 5bed375c3e663c077bc36cf1
- 5bed84843e663c077bc36daf
```

**Behavior:**
- `full: true` → processes ALL venues in data directory (ignores venue list)
- `full: false` (or omitted) → filters to venues listed in the file
- If `venues.yaml` doesn't exist → processes all venues in data directory
- Use `--venues other.yaml` to specify a different file

### Example Workflows

```bash
# Quick validation (no AI cost)
python3 full_flow.py /path/to/data_dir --limit 10 --skip-gemini

# Full AI analysis without Excel merge
python3 full_flow.py /path/to/data_dir --skip-concat

# Complete pipeline (uses venues.yaml if exists)
python3 full_flow.py /path/to/data_dir

# Process specific venues from custom file
python3 full_flow.py /path/to/data_dir --venues my_venues.yaml
```

---

## Input/Output Files Summary

### Input Files

| File | Required | Description |
|------|----------|-------------|
| `data_dir` | Yes | Data directory with venue/event folders (user must specify) |
| `venues.yaml` | No | List of venue IDs to filter (auto-loaded if exists) |
| `Annotators - Working Table 2025.xlsx` | No | Excel file for Step 4 concatenation |
| `config.yaml` | No | Gemini model configuration (in main directory) |
| `service-account-key.json` | Yes* | GCP credentials (*required for Step 3) |

### Output Files

| File | Stage | Description |
|------|-------|-------------|
| `laplacian_calculations.csv` | 1 | Raw focus metrics |
| `laplacian_th.csv` | 2 | Focus metrics + severity classification (focus_* columns) |
| `is_measurable.csv` | 3 | Gemini quality analysis only (quality_* columns) |
| `cost.txt` | 3 | Gemini API usage report |
| `concatenated_results.csv` | 4 | **Final output**: Excel annotations + focus_* + quality_* columns |

**Data Flow:**
- Steps 1-2 produce focus metrics → `laplacian_th.csv`
- Step 3 produces quality metrics → `is_measurable.csv`
- Step 4 merges both into `concatenated_results.csv`

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐      ┌─────────────────┐                              │
│  │   Data Source   │      │  venues.yaml    │                              │
│  │   (data_dir/)   │◀────▶│  (Filter)       │                              │
│  └────────┬────────┘      └─────────────────┘                              │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐      ┌─────────────────┐                              │
│  │    OpenCV       │      │    NumPy        │                              │
│  │  (cv2.Laplacian)│◀────▶│  (Statistics)   │                              │
│  └────────┬────────┘      └─────────────────┘                              │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │   Threshold     │                                                        │
│  │    Engine       │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐      ┌─────────────────┐                              │
│  │  Gemini 3 Flash │◀────▶│   Vertex AI     │                              │
│  │   (Vision AI)   │      │   (GCP)         │                              │
│  └────────┬────────┘      └─────────────────┘                              │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐      ┌─────────────────┐                              │
│  │     Pandas      │◀────▶│   Excel/CSV     │                              │
│  │  (Data Merge)   │      │   (Annotations) │                              │
│  └────────┬────────┘      └─────────────────┘                              │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │  Final Reports  │                                                        │
│  │    (CSV)        │                                                        │
│  └─────────────────┘                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.x | Image processing, Laplacian filter |
| Pillow | 10.x | Image loading for Gemini |
| google-genai | latest | Gemini AI client |
| pandas | 2.x | Data manipulation, Excel reading |
| openpyxl | 3.x | Excel file support |
| PyYAML | 6.x | Configuration loading |

---

## Configuration

**File:** `config.yaml` (in main directory)

```yaml
# Model settings
model: "gemini-3-flash-preview"  # Flash is faster and less prone to MAX_TOKENS errors
temperature: 0
top_p: 0.95

# GCP settings
project: "pixellot-ai"
location: "global"
```

**Credentials:** `service-account-key.json` (in main directory)

---

## Cost Estimation

| Events | Estimated Cost (Flash) | Processing Time |
|--------|------------------------|-----------------|
| 10 | ~$0.05 | ~1 min |
| 100 | ~$0.50 | ~10 min |
| 500 | ~$2.50 | ~45 min |
| 1000 | ~$5.00 | ~1.5 hours |

*Based on ~$0.005 per image pair with Gemini 3 Flash*

---

## Error Handling

| Error Type | Handling |
|------------|----------|
| Missing images | Skipped, logged as N/A |
| Gemini API error | Logged, continues to next |
| Invalid JSON response | Marked as "Unknown" |
| Missing Excel file | Step 4 skipped |

---

## Sample Output

### laplacian_th.csv (focus metrics)

```csv
venue_id,event_id,focus_right_mean,focus_left_mean,focus_right_mid,focus_left_mid,focus_abs_dif_rel,validation_status,validation_details,Focus_severity
5bed374...,42a0c6c...,104.66,96.21,55.16,33.74,0.48,PASS,,Ok
5bed375...,e097172...,109.98,113.38,42.11,38.22,0.10,PASS,,Ok
```

### is_measurable.csv (Gemini focus analysis only)

```csv
venue_id,event_id,is_measurable,not_measurable_reason,has_quality_issue,quality_issue_type,which_camera
5bed374...,42a0c6c...,Yes,,Yes,Focus,Left
5bed375...,e097172...,No,too dark,None,None,None
5bed376...,f123abc...,Yes,,No,None,None
```

**Note:** `quality_issue_type` is now only `Focus` or `None` (no longer checks for other issues).

### concatenated_results.csv (new columns)

```csv
...,focus_right_mean,focus_left_mean,focus_right_mid,focus_left_mid,focus_abs_dif_rel,focus_severity,quality_is_measurable,quality_not_measurable_reason,quality_has_issue,quality_issue_type,quality_which_camera
...,104.66,96.21,55.16,33.74,0.48,Ok,Yes,,Yes,Focus,Left
```

### cost.txt

```
IS MEASURABLE ANALYSIS - COST REPORT
==================================================

Model: gemini-3-flash-preview
Processed: 100 images
Skipped: 5 images

TOKEN USAGE:
  Input tokens:    249,800
  Thinking tokens: 0
  Output tokens:   8,500
  Total tokens:    258,300

COST:
  Input cost:  $0.124900
  Output cost: $0.025500
  Total cost:  $0.150400
```

---

## Future Enhancements

1. **Parallel Processing** - Process multiple events concurrently
2. **Caching** - Skip already-processed events
3. **Progress Bar** - Visual progress indicator
4. **Retry Logic** - Automatic retry on API failures

---

## Contact

For questions or issues, contact the PCM team.
