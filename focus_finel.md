## What Is the Focus XLSX?

The final focus xlsx (`PQS_blur_by_venue_with_movement_<name>.xlsx`) is a per-venue report combining:

1. **PQS blur data** — from the source `PQS_blur_by_venue.xlsx`
2. **Laplacian focus metrics** — computed from camera images
3. **Focus severity classification** — Ok / Warning / Error / NA
4. **AI measurability** — Gemini determines if camera images are measurable
5. **AI issue detection** — Gemini detects focus quality issues (condensation, dark field, obstruction, etc.)
6. **Movement data** — optical flow comparing current vs reference calibration images
7. **Blend images** — panoramic current/reference/blend images with clickable hyperlinks
8. **Setup geometry** (optional) — camera coverage, spare pixels, zoom improvability

---

## Pipeline Steps to Create It

### Phase 1: Focus Analysis (runs in `pcm_push` repo)

```bash
cd /home/liorb/work/pcm_push
source venv/bin/activate

# Step 1: Laplacian metrics + severity (CPU only, no API)
python run_full_pipeline.py data/<DATASET>/ PQS_blur_by_venue.xlsx \
  --output-dir output_dir/<NAME>
```

This runs internally:
1. **`laplacian_calculations.py`** — computes focus metrics per camera (Laplacian variance, mid_focus, etc.)
2. **`laplacian_th_calculations.py`** — classifies severity based on thresholds
3. **`concat_blur.py`** — joins with PQS blur xlsx
4. **`run_is_measurable.py`** — Gemini AI measurability (needs GCP credentials)
5. **`detect_issues.py`** (2-cam) or **`detect_issues_3cam.py`** (3-cam) — Gemini AI issue detection
6. **`concat_with_is_measurable.py`** — joins measurability results
7. **`analyze_blur_severity.py`** — produces plots and stats

### Phase 2: Movement Analysis

```bash
# Step 1: Optical flow (from scrapanyzer repo, needs pcm_push venv)
cd /home/liorb/work/proactive-camera-monitoring/venue/scrapanyzer
/home/liorb/work/pcm_push/venv/bin/python3 batch_movement.py /home/liorb/work/pcm_push/data/<DATASET>

# Steps 2-4: From pcm_push repo
cd /home/liorb/work/pcm_push

# Step 2: Join movement with blur xlsx
python3 add_movement_to_blur.py data/<DATASET>

# Step 3: Create blend images (current/reference/blend panoramas)
python3 batch_blend.py data/<DATASET>

# Step 4: Add clickable file:// hyperlinks to xlsx
python3 add_blend_links_to_xlsx.py data/<DATASET>
```

### Phase 3 (Optional): Setup Geometry

```bash
# Step 1: Run setup analysis (from proactive-camera-monitoring repo)
cd /home/liorb/work/proactive-camera-monitoring
PYTHONPATH=venue/scrapanyzer python venue/scrapanyzer/batch_setup.py <DATASET>

# Step 2: Enrich blur xlsx with setup data
cd /home/liorb/work/pcm_push
python enrich_blur_with_setup.py --dataset <DATASET> --blur-xlsx <BLUR_XLSX> -o <OUTPUT>
```

---

## What Columns Does the Final XLSX Include?

| Category | Columns |
|---|---|
| **PQS blur** | All original `PQS_blur_by_venue.xlsx` columns (venue ID, blur metrics, etc.) |
| **Laplacian focus** | `focus_mean_cam0`, `focus_mean_cam1`, `mid_focus_cam0`, `mid_focus_cam1`, `focus_abs_dif_rel`, `Focus_severity` (Ok/Warning/Error/NA) |
| **3-cam extra** | `num_cameras`, `focus_center_mean`, `focus_center_mid`, `focus_abs_dif_rel_12` |
| **AI measurability** | `is_measurable` (Yes/No), `explanation` |
| **AI issue detection** | `has_issue`, `issue_type`, `confidence`, token counts |
| **Movement** | `calibration_movement`, `movement_indicator`, `movement_length_cam0/cam1/cam2`, `movement_severity` |
| **Blend images** | `current_image`, `reference_image`, `blend_image` (clickable `file:///` hyperlinks) |
| **Setup** (optional) | `setup_severity`, `all_spares`, `max_camera_overlap`, `left_spare`, `right_spare`, `can_improve_by_zoom`, `decision`, `setup_join_image`, etc. (15 columns) |

---

## Windows vs Linux, S2 vs S3

| Aspect | Windows | Linux S2 | Linux S3 |
|---|---|---|---|
| **Cameras** | 2 (CAM0, CAM1) | 2 (CAM0, CAM1) | 3 (CAM0, CAM1, CAM2) |
| **Image naming** | `CAM0_1.jpg` | `CAM0_1.jpg` | `CAM0_0.jpg` |
| **Data dir examples** | `data/17_2_windows` | `data/16_2_linux_s2`, `data/s2_23` | `data/linux_s3_16_2`, `data/s3_23` |
| **Panorama order** | CAM1 + CAM0 (left to right) | CAM1 + CAM0 | CAM2 + CAM1 + CAM0 |
| **Movement severity** | `max(cam0, cam1)` | `max(cam0, cam1)` | `max(cam0, cam1, cam2)` |
| **Issue detection script** | `detect_issues.py` | `detect_issues.py` | `detect_issues_3cam.py` (auto-detected) |
| **Setup cam_angle_diff** | `abs(pan[0]-pan[1])` | `abs(pan[0]-pan[1])` | `abs(pan[0]-pan[2])` full rig span |
| **Setup overlap** | 1 pair | 1 pair | 2 pairs (max taken) |
| **Blend images** | Not generated yet | Generated | Generated |

**No flags needed** — all scripts auto-detect:
- Camera naming: tries `_1.jpg` first, falls back to `_0.jpg`
- CAM2 presence: included only when both `CAM2_*.jpg` and calibration ref `2.jpg` exist
- `run_full_pipeline.py` auto-detects 2 vs 3 cameras via `detect_dataset_camera_count()`

---

## S3 Data Prep Note

S3 datasets have `collected/` directories instead of `focus/`. You need to create symlinks once before running the focus pipeline:

```bash
find data/<S3_DATASET>/ -type d -name collected -exec sh -c \
  'ln -sf collected "$(dirname "$1")/focus"' _ {} \;
```

---

## Consolidating S2 + S3 Runs

If you have both an S2 and S3 run for the same venues, merge them:

```bash
python3 consolidate_runs.py \
  output_dir/PQS_blur_by_venue_with_movement_<S2>.xlsx \
  output_dir/PQS_blur_by_venue_with_movement_<S3>.xlsx \
  -o output_dir/consolidated_best_per_venue.xlsx
```

This picks the best row per venue (measurable > non-measurable, then better severity).
