# PCM Push — Full Analysis Pipeline

## Overview

The pipeline computes focus quality metrics from camera images, classifies severity,
joins external blur and measurability data, then produces statistical analysis and plots.

## Steps

### Step 1: Focus metrics + severity classification

```bash
python run_laplacian_pipeline.py <data_dir>
```

Runs two scripts internally:
1. `laplacian_calculations.py` — computes Laplacian focus metrics per camera pair
2. `laplacian_th_calculations.py` — classifies severity (Ok / Warning / Error / NA)

**Output**: `output_dir/laplacian_th_<timestamp>/laplacian_th_calculations.csv`

### Step 2: Join blur data

```bash
python concat_blur.py \
  output_dir/laplacian_th_<ts>/laplacian_th_calculations.csv \
  PQS_blur_by_venue.xlsx \
  [-o output.xlsx]
```

Joins PQS blur-by-venue data onto the Step 1 CSV by `venue_id`.
Outputs Excel (`.xlsx`) with clickable hyperlinks in the `joined_image` column.

**Output**: `concat_data/laplacian_th_with_blur.xlsx`

### Step 3: AI measurability analysis (requires GCP)

Two scripts can produce measurability data:

**`run_is_measurable.py`** — venue-level analysis, routes to indoor/outdoor analyzer:
```bash
python run_is_measurable.py --dataset <data_dir> --blur-xlsx PQS_blur_by_venue.xlsx
```
**Output**: `output_dir/run_is_measurable_<timestamp>/run_is_measurable.csv`

**`is_measurable.py`** — event-level analysis:
```bash
python is_measurable.py \
  --input output_dir/laplacian_th_<ts>/laplacian_th_calculations.csv \
  --data-dir <data_dir>
```
**Output**: `output_dir/is_measurable_<timestamp>/is_measurable.csv`

Both require GCP credentials (`service-account-key.json`).

### Step 4: Join measurability data

```bash
python concat_with_is_measurable.py \
  concat_data/laplacian_th_with_blur.xlsx \
  <measurable_csv> \
  [-o output.xlsx]
```

Joins measurability columns onto the Step 2 Excel by `venue_id`.
Auto-detects which columns to add (skips columns already present in the blur file).
Outputs Excel (`.xlsx`), preserving clickable hyperlinks from Step 2.

**Output**: `concat_data/laplacian_th_with_blur_and_measurable.xlsx`

### Step 5: Analyze blur vs severity

```bash
python analyze_blur_severity.py \
  concat_data/laplacian_th_with_blur_and_measurable.xlsx
```

Accepts both `.csv` and `.xlsx` input.

Produces box plots, histograms, statistical tests (Kruskal-Wallis, Mann-Whitney U,
Spearman correlation), and a comparison of all-data vs measurable-only subsets.

**Output**: `output_dir/analyze_blur_<timestamp>/` (HTML plots + CSV stats)

## Automated run

**Full pipeline (runs all steps including Gemini measurability, requires GCP):**
```bash
python run_full_pipeline.py <data_dir> PQS_blur_by_venue.xlsx
```

**Limit Gemini calls (e.g., first 100 venues only):**
```bash
python run_full_pipeline.py <data_dir> PQS_blur_by_venue.xlsx --limit 100
```

**With pre-computed measurable CSV (skips Step 3):**
```bash
python run_full_pipeline.py <data_dir> PQS_blur_by_venue.xlsx --is-measurable-csv <path>
```

Chains Steps 1 → 2 → 3 → 4 → 5 automatically. `--limit` applies only to Step 3 (`run_is_measurable.py`).

## Output directory layout

When run via `run_full_pipeline.py`, all outputs go under a single timestamped master directory.
Concat outputs are Excel files (`.xlsx`) with clickable image hyperlinks, stored in a dedicated `concat_data/` subdirectory.

```
output_dir/<YYYY-MM-DD_HH-MM>/
├── source_files/                            # Copy of pipeline scripts for reproducibility
├── laplacian/
│   └── laplacian_calculations.csv           # Step 1a
├── laplacian_th/
│   └── laplacian_th_calculations.csv        # Step 1b
├── concat_data/
│   ├── laplacian_th_with_blur.xlsx          # Step 2 (Excel with clickable hyperlinks)
│   └── laplacian_th_with_blur_and_measurable.xlsx  # Step 4 (Excel with clickable hyperlinks)
├── run_is_measurable/
│   ├── run_is_measurable.csv                # Step 3
│   └── cost.txt
└── analyze_blur/
    ├── all_boxplot.html                     # Step 5
    ├── all_histogram.html
    ├── all_summary_stats.csv
    ├── all_statistical_tests.csv
    ├── measurable_boxplot.html
    ├── measurable_histogram.html
    ├── measurable_summary_stats.csv
    ├── measurable_statistical_tests.csv
    └── comparison_summary.csv
```
