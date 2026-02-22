# Runbook: Setup Analysis for New Datasets

Run camera geometry/setup analysis on a new Linux-collected dataset and produce an xlsx with coverage metrics, severity, and actionable decisions.

## Prerequisites

- **Python packages:** `boto3`, `opencv-python`, `numpy`, `openpyxl`
- **Repos:**
  - `proactive-camera-monitoring` — contains `batch_setup.py` and `run_all_setup.py`
  - `pcm_push` — contains `enrich_blur_with_setup.py` and `consolidate_setup_results.py`

### Dataset directory structure

```
<DATASET>/
└── <venue_id>/
    └── <timestamp>/
        └── collected/
            ├── CAM0_1.jpg (S2) or CAM0_0.jpg (S3)
            ├── CAM1_1.jpg (S2) or CAM1_0.jpg (S3)
            ├── CAM2_0.jpg (S3 only)
            └── multisportcalibration/
                ├── soccer/
                │   ├── cam00_calib.xml
                │   ├── cam01_calib.xml
                │   └── fieldparameters.cfg
                ├── basketball/
                └── ...
```

Both `collected/` and `multisportcalibration/` must exist. Venues missing either are skipped automatically.

---

## Step 1: `batch_setup.py` — Run setup analysis on all venues

```bash
cd /home/liorb/work/proactive-camera-monitoring
PYTHONPATH=venue/scrapanyzer python venue/scrapanyzer/batch_setup.py <DATASET>
```

**What it does** for each venue:
1. Finds the latest `<timestamp>/collected/` that has both `multisportcalibration/` and a CAM0 image
2. Creates a temp dir at `/tmp/pcm_setup_batch/` with `extract/<venue_id>` symlinked to `collected/`
3. Creates case-fix symlinks (`cam00_calib.xml` → `Cam00_Calib.xml`, etc.) in each sport dir
4. Runs `run_all_setup.py` in a subprocess with `PRO_CAM_MON_OFFLINE=1`
5. Copies output (`setup.json`, `setup.csv`, `setup_join.jpg`) to `<venue>/<timestamp>/setup/`
6. Cleans up temp files and symlinks

**Idempotent:** Skips venues where `setup/setup.json` already exists. To re-run a venue, delete its `setup/` directory first.

**Output per venue:**

| File | Description |
|---|---|
| `setup.json` | Per-sport-type setup measurements (the MAX field_area entry is used downstream) |
| `setup.csv` | Same data in tabular format |
| `setup_join.jpg` | Side-by-side scaled camera view (2 panels for S2, 3 for S3) |

**Timing:** ~2 min for ~400 venues.

---

## Step 2: `enrich_blur_with_setup.py` — Join setup data with blur xlsx

```bash
cd /home/liorb/work/pcm_push
python enrich_blur_with_setup.py --dataset <DATASET> --blur-xlsx <BLUR_XLSX> -o <OUTPUT_XLSX>
```

**Example:**
```bash
python enrich_blur_with_setup.py \
    --dataset data/16_2_linux_s2 \
    --blur-xlsx output_dir/PQS_blur_by_venue_with_movement.xlsx \
    -o output_dir/pqs_blur_by_venue_with_setup.xlsx
```

Reads each venue's `setup/setup.json`, picks the entry with `max_field_area: "MAX"`, and appends 15 columns to the existing blur xlsx.

### 15 columns added

| Column | Source |
|---|---|
| `setup_sport_type` | Sport type of the MAX field_area entry (e.g., soccer) |
| `max_camera_overlap` | Max overlap between adjacent camera pairs (pixels) |
| `all_spares` | `max_camera_overlap + left_spare + right_spare` |
| `setup_severity` | Derived from `all_spares` thresholds (see below) |
| `left_spare` | How far the leftmost camera sees past the field left edge |
| `right_spare` | How far the rightmost camera sees past the field right edge |
| `s2_mode_calc` | Zoom mode 1-6 (1=widest ~90deg, 6=narrowest ~38deg, -1=out of range) |
| `X_of_center` | Camera X position relative to field center |
| `Y_from_line` | Camera Y distance from the sideline |
| `height` | Camera mounting height |
| `focals_diff` | `max - min` focal length across all cameras |
| `cam_angle_diff` | Pan angle diff between outermost cameras (degrees) |
| `can_improve_by_zoom` | Yes/No — whether zooming in can fix coverage issues (see below) |
| `decision` | Action recommendation (see below) |
| `setup_join_image` | Clickable hyperlink to the `setup_join.jpg` image |

---

## Step 3 (optional): `consolidate_setup_results.py` — Standalone setup xlsx

Use this if you want setup data independently, without joining to blur data.

```bash
cd /home/liorb/work/pcm_push
python consolidate_setup_results.py <DATASET>
python consolidate_setup_results.py <DATASET> -o custom_output.xlsx
```

**Output:** `<DATASET>/setup_results.xlsx` (default) — one row per venue, 20 columns.

### 20 output columns

`venue_id`, `active_calibrations`, `setup_sport_type`, `num_of_cam`, `max_camera_overlap`, `all_spares`, `setup_severity`, `left_spare`, `right_spare`, `s2_mode_calc`, `X_of_center`, `Y_from_line`, `height`, `focals_diff`, `cam_angle_diff`, `field_area`, `up_side_down`, `can_improve_by_zoom`, `decision`, `setup_join_image`

Extra columns vs. enrich: `venue_id`, `active_calibrations`, `num_of_cam`, `field_area`, `up_side_down`.

---

## Severity & Decision Logic

### setup_severity (from `all_spares`)

| Severity | Condition |
|---|---|
| **Error** | `all_spares > 3000` |
| **Warning** | `2000 <= all_spares <= 3000` |
| **Ok** | `all_spares < 2000` |
| **Aborted** | No setup data available |

### can_improve_by_zoom

```
ratio = (left_spare + right_spare) / max_camera_overlap / 2
can_improve_by_zoom = "Yes" if 0.7 <= ratio <= 1.3 else "No"
```

- **Balanced (ratio ~1.0) → Yes:** spare and overlap shrink equally when zooming in — fixable remotely
- **Too much spare (ratio > 1.3) → No:** overlap disappears before spare is gone → gap in the middle, needs on-site repositioning
- **Too little spare (ratio < 0.7) → No:** spare disappears before overlap is reduced → field edges uncovered

**Note:** This formula was designed for S2 (2 cameras). For S3 it's a rough heuristic — the middle camera affects both overlaps but not spare.

### decision matrix

| setup_severity | can_improve_by_zoom | decision |
|---|---|---|
| Error | Yes | `maintain_remote` |
| Error | No | `maintain_on_site` |
| Warning | — | `watch` |
| Ok | — | `ok` |
| Aborted | — | `NA` |

---

## S2 vs S3 Differences

Auto-detected by `batch_setup.py` and `run_all_setup.py` — no flags needed.

| Aspect | S2 | S3 |
|---|---|---|
| Camera images | `CAM0_1.jpg`, `CAM1_1.jpg` | `CAM0_0.jpg`, `CAM1_0.jpg`, `CAM2_0.jpg` |
| Calib files per sport | `cam00_calib.xml`, `cam01_calib.xml` | + `cam02_calib.xml` |
| `cam_angle_diff` | `abs(pan[0] - pan[1])` ~65deg | `abs(pan[0] - pan[2])` full rig span |
| `max_camera_overlap` | 1 pair: CAM0-CAM1 | 2 pairs: CAM0-CAM1, CAM1-CAM2 (takes max) |
| `left_spare` | From `calibs[1]` (leftmost of 2) | From `calibs[2]` (leftmost of 3) |
| `setup_join.jpg` | 2-panel image | 3-panel image |
| Excluded sport types | `other`, `scoreboard` | + `center`, `leftside`, `rightside`, `fieldhockey` |

---

## Copy-Paste Command Block

Replace `<DATASET>` and `<BLUR_XLSX>` with actual paths.

```bash
# Step 1: Run setup analysis on all venues
cd /home/liorb/work/proactive-camera-monitoring
PYTHONPATH=venue/scrapanyzer python venue/scrapanyzer/batch_setup.py <DATASET>

# Step 2a: Enrich blur xlsx with setup data
cd /home/liorb/work/pcm_push
python enrich_blur_with_setup.py --dataset <DATASET> --blur-xlsx <BLUR_XLSX> -o output_dir/pqs_blur_by_venue_with_setup.xlsx

# Step 2b (alternative): Standalone setup xlsx
python consolidate_setup_results.py <DATASET>
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Venue skipped — "missing collected" | No `<timestamp>/collected/` dir, or no `multisportcalibration/` + camera image inside | Verify data was collected correctly for that venue |
| `setup.json` already exists (venue skipped) | `batch_setup.py` is idempotent | Delete `<venue>/<ts>/setup/` to force re-run |
| Sport type excluded | `other`, `scoreboard` (S2/S3), `center`, `leftside`, `rightside`, `fieldhockey` (S3) are excluded by `run_all_setup.py` | Expected — these don't have standard calibration data |
| Case-fix symlink error | Symlink target already exists (e.g., from a previous interrupted run) | Delete stale `Cam00_Calib.xml` etc. symlinks from the sport dirs |
| Timeout (60s per venue) | Large calibration data or slow disk | Venue is skipped and counted as failed; re-run will retry it |
| `enrich_blur_with_setup.py` shows "Venue not in dataset" | Blur xlsx has venues not present in the dataset directory | Expected if blur xlsx covers more venues than the dataset |
| `setup_severity: Aborted` | No setup.json found, or empty setups array in setup.json | Check if Step 1 succeeded for that venue |
