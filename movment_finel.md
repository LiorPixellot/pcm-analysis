# Movement XLSX Pipeline

How to create the movement xlsx — what it contains, the 4-step pipeline, and commands for each system.

## What the Movement XLSX Contains

The output is `PQS_blur_by_venue_with_movement_<dataset>.xlsx` — the original `PQS_blur_by_venue.xlsx` enriched with these new columns:

| Column | Description |
|---|---|
| `calibration_movement` | Sport calibration name (e.g., soccer, basketball) |
| `movement_indicator` | `0` = computed, `2` = couldn't compute, `-1` = incomplete calibration |
| `movement_length_cam0` | Optical flow pixel distance for right camera |
| `movement_length_cam1` | Optical flow pixel distance for left camera |
| `movement_length_cam2` | Pixel distance for third camera (S3 only, empty on S2) |
| `movement_severity` | **OK** (<10px), **WARNING** (10-40px), **ERROR** (>40px), **NA** |
| `current_image` | Clickable `file:///` link to current panorama |
| `reference_image` | Clickable `file:///` link to reference panorama |
| `blend_image` | Clickable `file:///` link to 50/50 blend (ghosting = movement) |

One row per **venue x calibration** (a venue with 4 sport calibrations = 4 rows).

---

## The 4-Step Pipeline

| Step | Script | Repo | What it does |
|---|---|---|---|
| 1 | `batch_movement.py` | scrapanyzer | Optical flow: compares camera images vs calibration references -> `movement.json` |
| 2 | `add_movement_to_blur.py` | pcm_push | Joins `movement.json` data with `PQS_blur_by_venue.xlsx` -> movement xlsx |
| 3 | `batch_blend.py` | pcm_push | Creates panorama images (current, reference, blend) on disk |
| 4 | `add_blend_links_to_xlsx.py` | pcm_push | Adds clickable `file:///` hyperlinks to the xlsx |

---

## Commands by System

### Linux S2 (2-camera)

Dataset: `data/16_2_linux_s2` (or similar S2 dataset)

```bash
# Step 1: Movement analysis (from scrapanyzer repo, using pcm_push venv)
cd /home/liorb/work/proactive-camera-monitoring/venue/scrapanyzer
/home/liorb/work/pcm_push/venv/bin/python3 batch_movement.py /home/liorb/work/pcm_push/data/16_2_linux_s2

# Steps 2-4: From pcm_push repo
cd /home/liorb/work/pcm_push

# Step 2: Join movement with blur xlsx
python3 add_movement_to_blur.py data/16_2_linux_s2

# Step 3: Create blend images
python3 batch_blend.py data/16_2_linux_s2

# Step 4: Add clickable hyperlinks
python3 add_blend_links_to_xlsx.py data/16_2_linux_s2
```

**Output:** `output_dir/PQS_blur_by_venue_with_movement_16_2_linux_s2.xlsx`

- 2 cameras: CAM0 + CAM1, panorama is `[CAM1, CAM0]` (left-to-right)
- `movement_length_cam2` column will be empty
- Image naming: usually `CAM0_1.jpg`

### Linux S3 (3-camera)

Dataset: `data/linux_s3_16_2` (or similar S3 dataset)

```bash
# Step 1
cd /home/liorb/work/proactive-camera-monitoring/venue/scrapanyzer
/home/liorb/work/pcm_push/venv/bin/python3 batch_movement.py /home/liorb/work/pcm_push/data/linux_s3_16_2

# Steps 2-4
cd /home/liorb/work/pcm_push
python3 add_movement_to_blur.py data/linux_s3_16_2
python3 batch_blend.py data/linux_s3_16_2
python3 add_blend_links_to_xlsx.py data/linux_s3_16_2
```

**Output:** `output_dir/PQS_blur_by_venue_with_movement_linux_s3_16_2.xlsx`

- 3 cameras: CAM0 + CAM1 + CAM2, panorama is `[CAM2, CAM1, CAM0]`
- `movement_length_cam2` is populated
- Severity uses `max(cam0, cam1, cam2)`
- Image naming: usually `CAM0_0.jpg` (note the `_0` difference)

### Windows

Dataset: `data/17_2_windows` (or similar Windows dataset)

```bash
# Same pipeline commands, just different dataset path
cd /home/liorb/work/proactive-camera-monitoring/venue/scrapanyzer
/home/liorb/work/pcm_push/venv/bin/python3 batch_movement.py /home/liorb/work/pcm_push/data/17_2_windows

cd /home/liorb/work/pcm_push
python3 add_movement_to_blur.py data/17_2_windows
python3 batch_blend.py data/17_2_windows
python3 add_blend_links_to_xlsx.py data/17_2_windows
```

**Output:** `output_dir/PQS_blur_by_venue_with_movement_17_2_windows.xlsx`

**Note:** Blend images (steps 3-4) were previously skipped for Windows — only steps 1-2 were done.

### Consolidating S2 + S3 (optional Step 5)

After running both Linux datasets, merge them into one best-per-venue file:

```bash
python3 consolidate_runs.py \
  output_dir/PQS_blur_by_venue_with_movement_16_2_linux_s2.xlsx \
  output_dir/PQS_blur_by_venue_with_movement_linux_s3_16_2.xlsx \
  -o output_dir/consolidated_best_per_venue_s2_s3.xlsx
```

This picks the best row per venue (prefers measurable, then better severity).

---

## Key Differences Summary

| | S2 (Linux) | S3 (Linux) | Windows |
|---|---|---|---|
| Cameras | CAM0 + CAM1 | CAM0 + CAM1 + CAM2 | CAM0 + CAM1 |
| Image suffix | `_1.jpg` | `_0.jpg` | `_1.jpg` |
| Panorama order | CAM1 + CAM0 | CAM2 + CAM1 + CAM0 | CAM1 + CAM0 |
| cam2 column | Empty | Populated | Empty |
| Auto-detect? | Yes, no flags needed | Yes, no flags needed | Yes, no flags needed |

---

## Movement Severity Classification

| Indicator | Condition | Severity |
|---|---|---|
| 2 or -1 | — | NA |
| 0 | max(cam0, cam1[, cam2]) > 40 | ERROR |
| 0 | max(cam0, cam1[, cam2]) between 10-40 | WARNING |
| 0 | max(cam0, cam1[, cam2]) < 10 | OK |
| None (no data) | — | None |

---

## Prerequisites

- `PQS_blur_by_venue.xlsx` must exist in the pcm_push root directory before running step 2
- Must use pcm_push venv for step 1 (`batch_movement.py`) — system python lacks cv2
- All scripts are idempotent — they skip existing outputs. Delete `movement.json` or blend images to regenerate

## Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`
`batch_movement.py` must run with the pcm_push venv, not system python. Use the full venv path:
```bash
/home/liorb/work/pcm_push/venv/bin/python3 batch_movement.py ...
```

### Venue not matched in xlsx
`add_movement_to_blur.py` matches by `PIXELLOT_VENUE_ID` column. Venues not in the xlsx are still included (added in a second pass with empty PQS columns).

### Movement severity is NA
Indicator values of `2` or `-1` produce NA severity. Indicator `2` means optical flow couldn't compute movement. Indicator `-1` means calibration data was incomplete.

### Calibration symlink errors
`batch_movement.py` creates temporary mixed-case symlinks. Clean leftover symlinks with:
```bash
find data/<DATASET_NAME> -name "Cam*_Calib.xml" -type l -delete
```
