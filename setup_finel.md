# Setup xlsx — What It Is and How to Create It

The **setup xlsx** analyzes camera geometry/coverage for each venue — whether the cameras cover the full playing field, severity of coverage gaps, and whether problems can be fixed remotely (by adjusting zoom) or require on-site maintenance.

## What the Setup xlsx Contains

### Standalone (`consolidate_setup_results.py`) — 20 columns

| Column | Meaning |
|---|---|
| `venue_id` | Venue identifier |
| `active_calibrations` | Which sport calibrations exist |
| `setup_sport_type` | Sport of the MAX field_area entry (e.g., soccer) |
| `num_of_cam` | 2 (S2) or 3 (S3) |
| `max_camera_overlap` | Max overlap between adjacent cameras (pixels) |
| `all_spares` | `overlap + left_spare + right_spare` — total "wasted" coverage |
| `setup_severity` | **Error** (>3000), **Warning** (2000-3000), **Ok** (<2000) |
| `left_spare` / `right_spare` | How far cameras see past field edges |
| `s2_mode_calc` | Zoom mode 1-6 (1=widest, 6=narrowest) |
| `X_of_center`, `Y_from_line`, `height` | Camera position relative to field |
| `focals_diff` | Max-min focal length across cameras |
| `cam_angle_diff` | Pan angle diff between outermost cameras |
| `field_area`, `up_side_down` | Field size and orientation flag |
| `can_improve_by_zoom` | Yes/No — can zooming fix the issue? |
| `decision` | `ok`, `watch`, `maintain_remote`, or `maintain_on_site` |
| `setup_join_image` | Clickable hyperlink to the side-by-side camera image |

### Enriched (`enrich_blur_with_setup.py`)

Takes an existing blur xlsx and adds 15 of the above columns (minus `venue_id`, `active_calibrations`, `num_of_cam`, `field_area`, `up_side_down` which are already known from blur data).

---

## Steps to Create

### Step 1: Run `batch_setup.py` (in `proactive-camera-monitoring` repo)

This computes `setup.json` for every venue. ~2 min for ~400 venues.

```bash
cd /home/liorb/work/proactive-camera-monitoring
PYTHONPATH=venue/scrapanyzer python venue/scrapanyzer/batch_setup.py <DATASET>
```

It's **idempotent** — skips venues where `setup/setup.json` already exists. Delete `<venue>/<ts>/setup/` to force re-run.

### Step 2: Create the xlsx (in `pcm_push` repo)

**Option A — Standalone setup xlsx** (no blur data needed):
```bash
cd /home/liorb/work/pcm_push
venv/bin/python consolidate_setup_results.py <DATASET>
# Output: <DATASET>/setup_results.xlsx
```

**Option B — Enrich an existing blur xlsx with setup columns:**
```bash
cd /home/liorb/work/pcm_push
venv/bin/python enrich_blur_with_setup.py --dataset <DATASET> --blur-xlsx <BLUR_XLSX> -o <OUTPUT_XLSX>
```

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

### Decision matrix

| setup_severity | can_improve_by_zoom | decision |
|---|---|---|
| Error | Yes | `maintain_remote` |
| Error | No | `maintain_on_site` |
| Warning | — | `watch` |
| Ok | — | `ok` |
| Aborted | — | `NA` |

---

## Windows vs Linux (S2 vs S3)

**There is no setup analysis for Windows datasets.** Setup requires `multisportcalibration/` data which is only available in Linux-collected datasets. Windows datasets (e.g., `data/17_2_windows/`) don't have this calibration data.

For Linux datasets, S2 vs S3 is **auto-detected** — no flags needed:

| Aspect | Linux S2 | Linux S3 |
|---|---|---|
| **Dataset example** | `data/s2_23/`, `data/16_2_linux_s2` | `data/s3_23/`, `data/linux_s3_16_2` |
| **Cameras** | CAM0, CAM1 (`_1.jpg`) | CAM0, CAM1, CAM2 (`_0.jpg`) |
| **Directory layout** | `<venue>/<ts>/collected/` | Same |
| **Calibration files** | `cam00_calib.xml`, `cam01_calib.xml` | + `cam02_calib.xml` |
| **`cam_angle_diff`** | Between 2 cameras (~65deg) | Between outermost 2 (CAM0, CAM2) |
| **`max_camera_overlap`** | 1 pair (CAM0-CAM1) | Max of 2 pairs (CAM0-CAM1, CAM1-CAM2) |
| **`left_spare`** | From CAM1 (leftmost) | From CAM2 (leftmost) |
| **`setup_join.jpg`** | 2-panel image | 3-panel image |
| **Excluded sport types** | `other`, `scoreboard` | + `center`, `leftside`, `rightside`, `fieldhockey` |
| **`can_improve_by_zoom`** | Accurate formula | Rough heuristic (middle camera complicates it) |

### Concrete commands for each dataset

**Linux S2:**
```bash
# Step 1
cd /home/liorb/work/proactive-camera-monitoring
PYTHONPATH=venue/scrapanyzer python venue/scrapanyzer/batch_setup.py /home/liorb/work/pcm_push/data/s2_23

# Step 2
cd /home/liorb/work/pcm_push
venv/bin/python consolidate_setup_results.py data/s2_23
# or with blur:
venv/bin/python enrich_blur_with_setup.py --dataset data/s2_23 --blur-xlsx <BLUR_XLSX> -o output_dir/pqs_blur_by_venue_with_setup_s2.xlsx
```

**Linux S3:**
```bash
# Step 1
cd /home/liorb/work/proactive-camera-monitoring
PYTHONPATH=venue/scrapanyzer python venue/scrapanyzer/batch_setup.py /home/liorb/work/pcm_push/data/s3_23

# Step 2
cd /home/liorb/work/pcm_push
venv/bin/python consolidate_setup_results.py data/s3_23
# or with blur:
venv/bin/python enrich_blur_with_setup.py --dataset data/s3_23 --blur-xlsx <BLUR_XLSX> -o output_dir/pqs_blur_by_venue_with_setup_s3.xlsx
```

**Key prerequisite:** Both `collected/` and `multisportcalibration/` must exist under each venue's timestamp directory. Venues missing either are silently skipped.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Venue skipped — "missing collected" | No `<timestamp>/collected/` dir, or no `multisportcalibration/` + camera image inside | Verify data was collected correctly for that venue |
| `setup.json` already exists (venue skipped) | `batch_setup.py` is idempotent | Delete `<venue>/<ts>/setup/` to force re-run |
| Sport type excluded | `other`, `scoreboard` (S2/S3), `center`, `leftside`, `rightside`, `fieldhockey` (S3) are excluded | Expected — these don't have standard calibration data |
| `enrich_blur_with_setup.py` shows "Venue not in dataset" | Blur xlsx has venues not present in the dataset directory | Expected if blur xlsx covers more venues than the dataset |
| `setup_severity: Aborted` | No setup.json found, or empty setups array | Check if Step 1 succeeded for that venue |
