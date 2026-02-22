# Runbook: Processing a New Linux Dataset

How to run the full movement + blend pipeline on a new Linux-collected dataset.

## Prerequisites

### Dataset directory structure

Your dataset must be at `data/<DATASET_NAME>/` (or any path) with this layout:

```
data/<DATASET_NAME>/<venue_id>/<event_id>/
    collected/
        CAM0_1.jpg (or CAM0_0.jpg)      # Right camera
        CAM1_1.jpg (or CAM1_0.jpg)      # Left camera
        CAM2_1.jpg (or CAM2_0.jpg)      # Third camera (optional, S3 datasets)
        multisportcalibration/<sport>/
            0.jpg, 1.jpg, 2.jpg         # Calibration reference images
            Cam00_Calib.xml, Cam01_Calib.xml, ...
    focus/
        CAM0_1.jpg, CAM1_1.jpg          # Focus images (used by blur pipeline)
        joined_0_1.jpg                  # Optional joined image
```

### Environment

```bash
# Activate the pcm_push venv (has cv2, openpyxl, Pillow, pandas)
source /home/liorb/work/pcm_push/venv/bin/activate
```

### Source xlsx

`PQS_blur_by_venue.xlsx` must exist in the pcm_push root (hardcoded by `add_movement_to_blur.py`).

---

## Pipeline Steps

### Step 1: batch_movement.py — Compute optical flow

**Repo:** scrapanyzer (different repo from pcm_push)
**Must use pcm_push venv** — system python3 lacks cv2.

For each venue: compares current camera images against calibration reference images using optical flow, writes `movement.json`.

```bash
cd /home/liorb/work/proactive-camera-monitoring/venue/scrapanyzer
/home/liorb/work/pcm_push/venv/bin/python3 batch_movement.py /home/liorb/work/pcm_push/data/<DATASET_NAME>
```

**Output:** `<venue_dir>/<event_id>/movement/movement.json`

Idempotent — skips venues that already have `movement.json`. Delete the file to regenerate.

### Step 2: add_movement_to_blur.py — Join movement data with PQS blur xlsx

Reads all `movement.json` files from the dataset, joins with `PQS_blur_by_venue.xlsx`, classifies movement severity. One output row per venue × calibration.

```bash
cd /home/liorb/work/pcm_push
python3 add_movement_to_blur.py data/<DATASET_NAME>
```

**Output:** `output_dir/PQS_blur_by_venue_with_movement_<DATASET_NAME>.xlsx`

**New columns:** `calibration_movement`, `movement_indicator`, `movement_length_cam0`, `movement_length_cam1`, `movement_length_cam2`, `movement_severity`

### Step 3: batch_blend.py — Create panoramic blend images

For each venue with movement data: builds a current camera panorama, a reference panorama, and a 50/50 alpha blend. Ghosting in the blend = camera movement.

```bash
cd /home/liorb/work/pcm_push
python3 batch_blend.py data/<DATASET_NAME>
```

**Output per venue × calibration:**
```
<venue_dir>/<event_id>/movement/
    movement_<venue_id>_pano_<sport>_current.jpg
    movement_<venue_id>_pano_<sport>_reference.jpg
    movement_<venue_id>_pano_<sport>_blend.jpg
```

Idempotent — skips venues where all three images already exist.

### Step 4: add_blend_links_to_xlsx.py — Add clickable hyperlinks

Adds `file:///` hyperlink columns (`current_image`, `reference_image`, `blend_image`) to the movement xlsx from Step 2.

```bash
cd /home/liorb/work/pcm_push
python3 add_blend_links_to_xlsx.py data/<DATASET_NAME>
```

This modifies the xlsx from Step 2 in-place (overwrites it with hyperlink columns added).

You can also specify a custom xlsx path:
```bash
python3 add_blend_links_to_xlsx.py data/<DATASET_NAME> --xlsx output_dir/PQS_blur_by_venue_with_movement_<DATASET_NAME>.xlsx
```

### Step 5 (optional): consolidate_runs.py — Merge two dataset runs

Picks the best row per venue across two runs. Selection: measurable > non-measurable, then better severity, ties go to run1.

**Note:** This script uses hardcoded paths at the top of the file. Edit lines 21-27 before running:

```python
RUN1_PATH = "output_dir/<run1_xlsx>"
RUN2_PATH = "output_dir/<run2_xlsx>"
OUTPUT_PATH = "output_dir/consolidated_<name>.xlsx"
RUN1_DATA_DIR = "data/<DATASET_1>"
RUN2_DATA_DIR = "data/<DATASET_2>"
```

Then run:
```bash
cd /home/liorb/work/pcm_push
python3 consolidate_runs.py
```

---

## S2 vs S3 Differences

| | S2 (2-camera) | S3 (3-camera) |
|---|---|---|
| Cameras | CAM0, CAM1 | CAM0, CAM1, CAM2 |
| Image naming | Usually `CAM0_1.jpg` | Usually `CAM0_0.jpg` |
| Panorama | CAM1 + CAM0 (left to right) | CAM2 + CAM1 + CAM0 (left to right) |
| Extra column | `movement_length_cam2` is empty | `movement_length_cam2` is populated |

**No flags needed.** All scripts auto-detect:
- Camera image naming: tries `_1.jpg` first, falls back to `_0.jpg`
- CAM2 presence: included in movement/blend only when both `CAM2_*.jpg` and calibration ref `2.jpg` exist
- Severity uses `max(cam0, cam1)` or `max(cam0, cam1, cam2)` depending on what's available

---

## Copy-Paste Command Block

Replace `<DATASET_NAME>` with your dataset directory name (e.g., `linux_s4_22_2`).

```bash
# Step 1: Movement analysis (run from scrapanyzer repo, use pcm_push venv)
cd /home/liorb/work/proactive-camera-monitoring/venue/scrapanyzer
/home/liorb/work/pcm_push/venv/bin/python3 batch_movement.py /home/liorb/work/pcm_push/data/<DATASET_NAME>

# Steps 2-4: Run from pcm_push repo
cd /home/liorb/work/pcm_push

# Step 2: Join movement with blur xlsx
python3 add_movement_to_blur.py data/<DATASET_NAME>

# Step 3: Create blend images
python3 batch_blend.py data/<DATASET_NAME>

# Step 4: Add clickable hyperlinks to xlsx
python3 add_blend_links_to_xlsx.py data/<DATASET_NAME>
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`
`batch_movement.py` must run with the pcm_push venv, not system python. Use the full venv path:
```bash
/home/liorb/work/pcm_push/venv/bin/python3 batch_movement.py ...
```

### `movement.json already exists` / want to regenerate
All scripts are idempotent — they skip existing outputs. To regenerate:
```bash
# Delete all movement.json files in a dataset
find data/<DATASET_NAME> -name "movement.json" -delete

# Delete all blend images
find data/<DATASET_NAME> -path "*/movement/movement_*_pano_*" -delete
```

### Calibration symlink errors (broken symlinks)
`batch_movement.py` creates temporary mixed-case symlinks for calibration files (Linux has lowercase, code expects `Cam00_Calib.xml`). If you see errors about symlinks already existing, leftover symlinks from a previous interrupted run may be present. Clean them:
```bash
find data/<DATASET_NAME> -name "Cam*_Calib.xml" -type l -delete
```

### Venue not matched in xlsx
`add_movement_to_blur.py` matches by `PIXELLOT_VENUE_ID` column in `PQS_blur_by_venue.xlsx`. Venues not in the xlsx are silently dropped. Venues in the xlsx without movement data are kept as single rows with empty movement columns.

### No blend images for some calibrations
Blend requires both the camera image AND the calibration reference image (`<sport>/0.jpg`, `1.jpg`, optionally `2.jpg`). Missing reference images = no blend for that calibration.

### Movement severity is NA
Indicator values of `2` or `-1` produce NA severity. Indicator `2` means the optical flow algorithm couldn't compute movement (typically due to insufficient features in the image). Indicator `-1` means calibration data was incomplete.
