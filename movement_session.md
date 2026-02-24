# Movement Session Summary

## 1. Setup Columns Added to pqs_blur xlsx

Added 5 setup geometry columns from `data_11_2/<venue_id>/<event_id>/setup/setup.json` to `output_dir/pqs_blur_by_venue_with_setup_change_gal.xlsx`:
- `X_of_center`, `Y_from_line`, `height`, `focals_diff`, `cam_angle_diff`
- Matched by `(PIXELLOT_VENUE_ID, setup_sport_type)`
- **8,922 matched**, 4,043 missed (no sport_type)

## 2. Setup Join Image Hyperlinks

Added clickable `setup_join_image` column with local file links (`file:///...`) to `setup_join.jpg` per venue.
- **8,941 matched**, 4,024 missed

## 3. Error Image Dumps

Created two filtered image dump directories from `laplacian_th_with_blur_measurable_issues.xlsx`:

| Directory | Filter | Venues |
|-----------|--------|--------|
| `output_dir/error_less_400/` | Focus_severity=Error AND AVERAGE_BLUR_AVG < 400 | 823 |
| `output_dir/error_more_1000/` | Focus_severity=Error AND AVERAGE_BLUR_AVG > 1000 | 1,017 |

## 4. Movement Annotation xlsx

Created `complete_annotation_movement.xlsx` from `Annotators - Working Table 2025.xlsx` sheets POC_1.0, POC_1.1, POC_1.2.
- **1,612 rows** total (161 + 721 + 730)
- Columns: Checkup date, System ID, Calibration, Setup quality, Indoor / Outdoor, Far / Close, source_sheet, Movement image (clickable S3 hyperlinks), Movement severity, Is Measurable Movement Image (Yes/No), Movement issues (ERROR/OK/WARNING), Pixels_distance, Movement length (pixels)

### How it works (`create_annotation_movement.py`)

The source workbook has 46 sheets; three contain movement annotations (POC_1.0, POC_1.1, POC_1.2) but with slightly different column names:

| Output column | POC_1.0 | POC_1.1 | POC_1.2 |
|---|---|---|---|
| Setup quality | `Setup quality` | `Setup severity` | `Setup quality` |
| Is Measurable Movement Image | `Is Measurable \nMov(Yes/No)ement Image` | `Is Measurable \nMovement Image\n(Yes/No)` | same as 1.1 |
| Movement issues | `Movement issues\n(ERROR/OK/WARNING)` | same | same |

POC_1.1 and POC_1.2 also have an extra `None` column at index 13 that shifts later columns.

The script:
1. Opens the source xlsx in **non-read-only mode** (required to access cell hyperlinks)
2. For each sheet, builds a column name → index map, skipping `None` headers
3. Iterates rows, skipping empties (no System ID). For each row:
   - Extracts the 12 relevant columns using `COL_ALIASES` to resolve name differences per sheet
   - Adds `source_sheet` to track origin
   - **Copies hyperlinks** from the "Movement image" cell (S3 URLs to blend images)
4. Writes all rows to a single output sheet

```bash
python create_annotation_movement.py
```

**Input:** `Annotators - Working Table 2025.xlsx`
**Output:** `complete_annotation_movement.xlsx` — 1,612 rows, 13 columns, with clickable S3 image links preserved.

### System vs Human Agreement Analysis

Comparing `Movement severity` (system) vs `Movement issues` (human annotation) on the 993 annotated rows (619 unannotated rows excluded). 64 system NA rows excluded from precision/recall (no human equivalent).

**Overall (excluding NA):** 929 rows — **91.4% agree**, 8.6% disagree (80 rows).

#### Precision — "When system says X, is it right?"

| System says | Correct | Wrong |
|---|---|---|
| **OK** (505) | 97.0% | 3.0% — 15 were actually WARNING |
| **WARNING** (257) | 89.9% | 10.1% — 24 actually OK, 2 actually ERROR |
| **ERROR** (167) | 76.6% | 23.4% — 30 actually OK, 9 actually WARNING |

#### Recall — "When human says X, did system catch it?"

| Human says | System correct | System missed |
|---|---|---|
| **OK** (544) | 90.1% | 9.9% — 54 flagged as WARNING/ERROR unnecessarily |
| **WARNING** (255) | 90.6% | 9.4% — 15 missed as OK, 9 over-escalated to ERROR |
| **ERROR** (130) | 98.5% | 1.5% — only 2 errors missed as WARNING |

#### Precision & Recall summary

| Class | Precision | Recall |
|---|---|---|
| **OK** | 97.0% | 90.1% |
| **WARNING** | 89.9% | 90.6% |
| **ERROR** | 76.6% | 98.5% |

#### Error direction

- **78.8%** of disagreements are **over-alerting** (system more severe than human)
- **21.2%** are **under-alerting** (system less severe than human)

#### System NA rows (64)

- 84.4% are OK per human, 10.9% warning, 4.7% error
- Hides 10 real issues (7 warnings + 3 errors)

**Key takeaway:** The system almost never misses real errors (98.5% recall on ERROR), but over-alerts — 23.4% of system ERRORs are false alarms. The bias is strongly toward over-alerting, which is the safer direction for a monitoring system.

## 5. PQS Blur with Movement Data

Created `add_movement_to_blur.py` script and generated `output_dir/PQS_blur_by_venue_with_movement.xlsx`.
- Source: `PQS_blur_by_venue.xlsx` (12,965 venues)
- Movement data from: `data_11_2/<venue>/<event>/movement/movement.json`
- One row per venue×calibration → **52,959 total rows**
- New columns: `calibration_movement`, `movement_indicator`, `movement_length_cam0`, `movement_length_cam1`, `movement_severity`

### Movement Indicator Stats

| Group | Venues |
|-------|--------|
| No movement data at all | 4,053 |
| Has indicator 0 or 2 (at least one calibration) | 8,912 |
| Only -1 across all calibrations | 0 |
| **Total** | **12,965** |

Indicator value distribution (with overlap across calibrations):
- `-1`: 6,121 venues
- `0`: 7,516 venues
- `2`: 1,419 venues
- `None`: 4,053 venues

**Key finding**: Zero venues have only `-1`. Every venue with movement data always has at least one calibration with indicator `0` or `2`.

### Movement Severity Classification

The `movement_severity` column is derived from `movement_indicator` and `max(movement_length_cam0, movement_length_cam1)`:

| Indicator | Condition | Severity |
|---|---|---|
| 2 or -1 | — | NA |
| 0 | max(cam0, cam1) > 40 | ERROR |
| 0 | max(cam0, cam1) between 10–40 | WARNING |
| 0 | max(cam0, cam1) < 10 | OK |
| None (no data) | — | None |

**Severity distribution (52,959 rows):**

| Severity | Count | % |
|---|---|---|
| NA | 27,761 | 52.4% |
| OK | 11,131 | 21.0% |
| WARNING | 6,144 | 11.6% |
| None | 4,053 | 7.7% |
| ERROR | 3,870 | 7.3% |

## How to Run `add_movement_to_blur.py`

```bash
python add_movement_to_blur.py <data_dir>
```

Example:
```bash
python add_movement_to_blur.py data_11_2
python add_movement_to_blur.py /path/to/other_data
```

**Inputs:**
- `PQS_blur_by_venue.xlsx` — source venue blur data (hardcoded)
- `<data_dir>` — configurable data directory containing `<venue_id>/<event_id>/movement/movement.json`

**Output:**
- `output_dir/PQS_blur_by_venue_with_movement_<data_dir_name>.xlsx`
  - e.g. `data_11_2` → `PQS_blur_by_venue_with_movement_data_11_2.xlsx`

The script reads each venue row from the source xlsx, finds its `movement.json`, and creates one output row per calibration with 5 new columns: `calibration_movement`, `movement_indicator`, `movement_length_cam0`, `movement_length_cam1`, `movement_severity`. Venues without movement data are kept as a single row with empty movement columns.

## End-to-End: Steps to Get Movement Data into PQS Blur

1. **Collect movement data** — run PCM on-device (Windows) or batch (Linux) to produce `movement.json` files per venue under `<data_dir>/<venue_id>/<event_id>/movement/movement.json`
2. **Add movement to blur** — `python add_movement_to_blur.py <data_dir>` joins `PQS_blur_by_venue.xlsx` with movement JSONs, classifies severity, outputs `output_dir/PQS_blur_by_venue_with_movement_<name>.xlsx`
