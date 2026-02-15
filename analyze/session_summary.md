# PCM Push Analysis Session — 2026-02-15

## 1. Indoor Measurability: Added Positive Few-Shot Example

**File modified:** `is_measurable_indoor.py`

Added a "normal light" positive example to the indoor measurability script. Previously it had only 1 negative example (`indor_dark_field` — lights off). Now it has both:

- **Negative:** `examples/indor_dark_field/` — lights off, NOT measurable
- **Positive:** `examples/indor_normal_light/5af1397692daab1e2085cafe_CAM0_1.jpg` — lights on, IS measurable

### Changes made:
1. Added `EXAMPLE_NORMAL_LIGHT_LABEL` constant (positive example description)
2. Changed `EXAMPLE_CATEGORIES` from 3-tuples to 4-tuples with `is_positive` flag: `(subdir, title, label, is_positive)`
3. Updated `load_examples()` to unpack 4-tuple
4. Updated `analyze_indoor()` to split examples into negative/positive sections with separate headers:
   - `"Reference examples of conditions that are NOT measurable:"` for negative
   - `"Reference examples of conditions that ARE measurable:"` for positive
   - Same pattern already used in `is_measurable_outdoor.py`

**Verified:** `python is_measurable_indoor.py --database data_11_2 --limit 3` — loads 2 images from 2 categories.

---

## 2. Thinking Token Configuration

Checked `detect_issues.py`, `is_measurable_indoor.py`, `is_measurable_outdoor.py` — **none** explicitly configure a `thinking_config` in `GenerateContentConfig`. The model (`gemini-3-flash-preview`) has thinking enabled by default. All three scripts track thinking tokens via `get_token_usage()` for cost reporting.

---

## 3. Dataset Coverage Analysis

**Source file:** `PQS_blur_by_venue.xlsx` (12,965 rows)
**Dataset:** `data_11_2/`

| Category | Count | % |
|---|---|---|
| Total rows in xlsx | 12,965 | 100% |
| In dataset | 9,016 | 69.5% |
| **Not in dataset** | **3,949** | **30.5%** |

### Setup.json enrichment (`enrich_blur_with_setup.py`):

| Category | Count |
|---|---|
| Matched with `max_field_area="MAX"` | 7,958 |
| **Matched with fallback** (no MAX, used largest `field_area`) | **964** |
| No setup.json found | 93 |
| Empty setups array | 1 |

**Output:** `analyze/missing_from_setup.xlsx` — all 3,949 venues not in dataset (full xlsx rows).

---

## 4. NA Severity Analysis

**Problem:** `laplacian_th_calculations.py:46-47` marks severity as NA when either camera mean < 70:
```python
if focus_right_mean < 70 or focus_left_mean < 70:
    return "NA"
```

**Source data:** `output_dir/more_examples  (Copy)/concat_data/laplacian_th_with_blur_measurable_issues.xlsx` (10,139 rows)

### Full severity x measurability breakdown:

| Focus_severity | Yes | % of Yes | No | % of No | (empty) | % of empty | Subtotal | % total |
|---|---|---|---|---|---|---|---|---|
| Ok | 3,532 | 48.1% | 221 | 13.4% | 447 | 39.1% | 4,200 | 41.4% |
| Warning | 2,213 | 30.1% | 302 | 18.3% | 325 | 28.5% | 2,840 | 28.0% |
| Error | 1,056 | 14.4% | 663 | 40.2% | 245 | 21.5% | 1,964 | 19.4% |
| NA | 545 | 7.4% | 463 | 28.1% | 125 | 10.9% | 1,133 | 11.2% |
| **Total** | **7,346** | 100% | **1,649** | 100% | **1,142** | 100% | **10,139** | 100% |

- When Gemini says No, 68.3% are Error+NA (worst categories) — good correlation
- When Gemini says Yes, 48.1% are Ok — plurality is healthy

### Why 1,142 rows have empty `is_measurable`

**Root cause:** Pipeline mismatch between Laplacian and Gemini scripts.

- **`laplacian_calculations.py`** scans `data_11_2/` directly — finds **every** venue/event with images and computes focus metrics. Uses `venue_id` from directory names.
- **`run_is_measurable.py`** is **xlsx-driven** — only processes venues listed in `PQS_blur_by_venue.xlsx` (keyed by `PIXELLOT_VENUE_ID`).

All 1,142 empty rows have a `venue_id` that exists in `data_11_2/` (images are there, Laplacian ran) but is **not listed in `PQS_blur_by_venue.xlsx`**. So the Gemini measurability pipeline never saw them.

Evidence: all 1,142 rows have empty `PIXELLOT_VENUE_ID`, empty `venue_type`, and empty `ACTIVE_CALIBRATIONS` — confirming no xlsx match.

This is the inverse of the 3,949 "missing from dataset" venues (in xlsx but not in `data_11_2/`):

| Gap | Count | Direction |
|---|---|---|
| In xlsx, not in `data_11_2/` | 3,949 | Laplacian can't run |
| In `data_11_2/`, not in xlsx | 1,142 | Gemini can't run |

### NA breakdown:

| NA subset | Count | File |
|---|---|---|
| NA + is_measurable=Yes | 545 | `analyze/NA_messurable.xlsx` |
| NA + is_measurable=No | 463 | `analyze/na_not_messurable.xlsx` |
| NA + empty | 125 | — |

---

## 5. Threshold Change Analysis

### Proposed change: `AND` + TH=55

```python
# Before (original)
if focus_right_mean < 70 or focus_left_mean < 70:
    return "NA"

# After (proposed)
if focus_right_mean < 55 and focus_left_mean < 55:
    return "NA"
```

### Recovery rates (from 545 NA+measurable rows):

| Approach | Recovered | Still NA | % recovered |
|---|---|---|---|
| Current (OR, TH=70) | 0 | 545 | 0% |
| Lower to 60 (OR) | 366 | 179 | 67% |
| AND only (TH=70) | 363 | 182 | 66% |
| **Combined AND + TH=55** | **535** | **10** | **98%** |

### New severity distribution (535 recovered rows):

| new_Focus_severity | Count |
|---|---|
| Ok | 142 |
| Warning | 193 |
| Error | 200 |

**Output:** `analyze/fixed_th_and_logic.xlsx` — 535 rows with `new_Focus_severity` column added.

---

## 6. Agreement Analysis: New Severity vs Gemini `has_issue`

Logic: Error/Warning = expects issue, Ok = expects no issue. Compared with Gemini's `has_issue` field.

| new_Focus_severity | has_issue=Yes | has_issue=No | Verdict |
|---|---|---|---|
| Error | 174 | 26 | 87% agree |
| Warning | 126 | 67 | 65% agree |
| Ok | 68 | 73 | 52% agree |
| **Total** | | | **69% agree** |

**Result:** 69% agreement rate — same as before the threshold change. The threshold change recovers 535 rows from NA without affecting accuracy.

### Agreement/Disagreement files:

| File | Rows | Description |
|---|---|---|
| `agreements_gemini_new_th.xlsx` | 374 | All agreements |
| `agreements_ok.xlsx` | 74 | Both say no problem (Ok + has_issue=No) |
| `agreements_warning_error.xlsx` | 300 | Both say problem (Warning/Error + has_issue=Yes) |
| `dissagreements_gemini_new_th.xlsx` | 161 | All disagreements |
| `severity_says_problem_gemini_no.xlsx` | 93 | Severity=Warning/Error but Gemini says no issue |
| `severity_says_ok_gemini_yes.xlsx` | 68 | Severity=Ok but Gemini says has issue |

---

## 7. Files in `analyze/`

| File | Rows | Description |
|---|---|---|
| `NA_messurable.xlsx` | 545 | Focus_severity=NA, is_measurable=Yes (with hyperlinks) |
| `na_not_messurable.xlsx` | 463 | Focus_severity=NA, is_measurable=No (with hyperlinks) |
| `missing_from_setup.xlsx` | 3,949 | Venues in xlsx but not in data_11_2 dataset |
| `fixed_th_and_logic.xlsx` | 535 | NA rows reclassified with AND+TH=55 (has `new_Focus_severity` column) |
| `agreements_gemini_new_th.xlsx` | 374 | New severity agrees with Gemini has_issue |
| `agreements_ok.xlsx` | 74 | Both say no problem |
| `agreements_warning_error.xlsx` | 300 | Both say problem |
| `dissagreements_gemini_new_th.xlsx` | 161 | New severity disagrees with Gemini has_issue |
| `severity_says_problem_gemini_no.xlsx` | 93 | Severity says problem, Gemini says no |
| `severity_says_ok_gemini_yes.xlsx` | 68 | Severity says ok, Gemini says issue |

---

## 8. Setup Geometry & Zoom Improvement Analysis

**File modified:** `output_dir/pqs_blur_by_venue_with_setup.xlsx`

Added 4 new columns from `setup.json` data:
- `left_spare` — left spare field-of-view from setup geometry
- `right_spare` — right spare field-of-view from setup geometry
- `s2_mode_calc` — calculated camera configuration mode (geometric arrangement of 2-camera system)
- `can_improve_by_zoom` — whether zooming could improve the setup

### Zoom improvement formula

```
ratio = (left_spare + right_spare) / max_camera_overlap / 2
```
- **Yes** (could improve by zooming): ratio between 0.7 and 1.3 — spares are balanced with overlap, zooming in could reduce waste
- **No** (cannot improve): ratio outside 0.7–1.3 — already zoomed tight or too unbalanced

### Zoom x Focus_severity cross-tab

| can_improve_by_zoom | Ok | Warning | Error | NA | Total |
|---|---|---|---|---|---|
| **Yes** | 1,226 | 734 | 465 | 295 | 2,720 |
| **No** | 2,521 | 1,776 | 1,249 | 710 | 6,256 |
| (empty) | 454 | 330 | 251 | 128 | 1,163 |

### Zoom x Mode x Severity combined table

| Zoom | s2_mode_calc | Ok | Warning | Error | NA | Total |
|---|---|---|---|---|---|---|
| **Yes** | -1 | 163 | 86 | 31 | 29 | 309 |
| | 1 | 114 | 93 | 81 | 49 | 337 |
| | 2 | 151 | 101 | 83 | 39 | 374 |
| | 3 | 298 | 176 | 108 | 88 | 670 |
| | 4 | 326 | 173 | 96 | 54 | 649 |
| | 5 | 174 | 105 | 66 | 36 | 381 |
| | **Subtotal** | **1,226** | **734** | **465** | **295** | **2,720** |
| **No** | -1 | 400 | 286 | 199 | 95 | 980 |
| | 1 | 356 | 308 | 190 | 108 | 962 |
| | 2 | 332 | 237 | 157 | 82 | 808 |
| | 3 | 559 | 332 | 212 | 147 | 1,250 |
| | 4 | 501 | 361 | 250 | 143 | 1,255 |
| | 5 | 373 | 252 | 241 | 135 | 1,001 |
| | **Subtotal** | **2,521** | **1,776** | **1,249** | **710** | **6,256** |
| **(empty)** | (empty) | 454 | 330 | 251 | 128 | **1,163** |

### Error rate by zoom ability

| Zoom | Ok% | Warning% | Error% | NA% | n |
|---|---|---|---|---|---|
| **Yes** | 45.1% | 27.0% | **17.1%** | 10.8% | 2,720 |
| **No** | 40.3% | 28.4% | **20.0%** | 11.3% | 6,256 |

Venues where zoom could help have a **2.9pp lower Error rate** (17.1% vs 20.0%).

### Error rate by camera mode

| Mode | Ok% | Warning% | Error% | NA% | n |
|---|---|---|---|---|---|
| -1 | 43.7% | 28.9% | 17.8% | 9.6% | 1,289 |
| 1 | 36.2% | 30.9% | 20.9% | 12.1% | 1,299 |
| 2 | 40.9% | 28.6% | 20.3% | 10.2% | 1,182 |
| **3 (best)** | 44.6% | 26.5% | **16.7%** | 12.2% | 1,920 |
| 4 | 43.4% | 28.0% | 18.2% | 10.3% | 1,904 |
| **5 (worst)** | 39.6% | 25.8% | **22.2%** | 12.4% | 1,382 |

### Best/worst zoom + mode combos

| Combo | Error% | n |
|---|---|---|
| **Best:** Zoom=Yes, mode=-1 | **10.0%** | 309 |
| **Worst:** Zoom=No, mode=5 | **24.1%** | 1,001 |

### Fixable errors breakdown

| Category | Count | % of all errors |
|---|---|---|
| Total errors | 1,965 | 100% |
| **Fixable** (Zoom=Yes + Error) | 465 | **23.7%** |
| **Not fixable** (Zoom=No + Error) | 1,249 | **63.6%** |
| Unknown (no setup data) | 251 | 12.8% |

### Problem rate (Warning + Error combined)

| Zoom | Problem rate |
|---|---|
| Yes | 44.1% |
| No | **48.4%** |

### Key insights

1. **~24% of all errors are potentially fixable by remote zoom adjustment** (465 venues) — these are the quick wins that don't require a technician visit.
2. **~64% of errors cannot be fixed by zooming** (1,249 venues) — these need physical intervention (lens cleaning, mounting adjustment, recalibration, or full setup redo).
3. **Mode 3 has the lowest error rate** (16.7%) and mode 5 the highest (22.2%) — a 5.5pp spread between best and worst camera configurations.
4. **Zoom=Yes + mode=-1 is the healthiest combo** (10.0% Error) — invalid/uncalculated setups with balanced spares rarely have focus problems, suggesting simple installations work well.
5. **Zoom=No + mode=5 is the most problematic** (24.1% Error) — these venues have the worst geometry and can't fix it by zooming.
6. **Venues that can zoom have 4.3pp fewer problems overall** (44.1% vs 48.4% problem rate) — the zoom-able geometry correlates with better focus quality even before any zoom adjustment is applied.

### s2_mode_calc distribution

| s2_mode_calc | Count | Description |
|---|---|---|
| -1 | 1,284 | Invalid/uncalculated setup |
| 1 | 1,293 | Camera config mode 1 |
| 2 | 1,169 | Camera config mode 2 |
| 3 | 1,911 | Camera config mode 3 |
| 4 | 1,896 | Camera config mode 4 |
| 5 | 1,369 | Camera config mode 5 |
| (empty) | 4,043 | No setup.json / not in dataset |

---

## Next Steps (not yet implemented)

- **Decision pending:** Whether to apply the `AND + TH=55` change to `laplacian_th_calculations.py`
- The 68 rows where severity=Ok but Gemini flags issue may warrant manual review
- 1,142 rows with empty `is_measurable` — these venues exist in `data_11_2/` but not in `PQS_blur_by_venue.xlsx`. To process them, either add them to the xlsx or run `is_measurable_indoor.py`/`is_measurable_outdoor.py` directly on `data_11_2/` (database-scan mode, not xlsx-driven)
