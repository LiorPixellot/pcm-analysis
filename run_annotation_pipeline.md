# Annotation Pipeline Flow

## Overview

Evaluate Gemini AI + Laplacian predictions against human-annotated ground truth for camera focus quality analysis.

## Prerequisites

- `complete_annotation.xlsx` — human-annotated ground truth with image hyperlinks
- `annotation_data/` — downloaded venue images (from `download_annotation_data.py`)
- `service-account-key.json` — GCP credentials for Gemini API
- `config.yaml` — model name, project, workers, retry settings

## Pipeline Steps

```
Step 1: Export xlsx → csv
        xlsx_to_csv.py
        ↓
Step 2: Run predictions
        run_annotation_pipeline.py --annotations complete_annotation.csv
        ↓
        output_dir/eval_YYYY-MM-DD_HH-MM/annotation_predictions.csv
        ↓
Step 3: Compute confusion metrics
        confusion_metrics.py --predictions <predictions.csv> --output-dir <metrics_dir>
        ↓
        metrics_report.txt + plots/ + image_dumps/
```

## Step 1: Export xlsx to csv

```bash
python3 xlsx_to_csv.py
```

Converts `complete_annotation.xlsx` → `complete_annotation.csv`, resolving image hyperlinks to full file paths.

## Step 2: Run annotation pipeline

```bash
python3 run_annotation_pipeline.py --annotations complete_annotation.csv
```

For each row in the CSV, runs three analyses:

1. **Laplacian severity** (CPU, no API) — computes focus metrics from CAM0/CAM1 images, classifies as OK/Warning/Error/NA
2. **Gemini measurability** (API) — routes to indoor/outdoor analyzer based on calibration type, returns Yes/No
3. **Gemini issue detection** (API, only if measurable=Yes) — detects focus quality issues, returns has_issue + issue_type

**Options:**
- `--skip-gemini` — run only Laplacian severity (no API cost)
- `--limit N` — process first N rows only
- `--max-workers N` — parallel Gemini workers
- `--output-dir <path>` — custom output directory

**Output:** `output_dir/eval_YYYY-MM-DD_HH-MM/annotation_predictions.csv`

Columns: ground truth (`gt_*`), predictions (`pred_*`), Laplacian metrics, image paths, token counts.

## Step 3: Compute confusion metrics

```bash
python3 confusion_metrics.py \
  --predictions output_dir/eval_YYYY-MM-DD_HH-MM/annotation_predictions.csv \
  --output-dir output_dir/eval_YYYY-MM-DD_HH-MM/metrics
```

Evaluates three dimensions:

| Dimension | GT column | Pred column | Type |
|---|---|---|---|
| Measurability | gt_measurable | pred_measurable | Binary (yes/no) |
| Quality issue (Gemini) | gt_has_issue | pred_has_issue | Binary (yes/no) |
| Quality issue (Laplacian) | gt_has_issue | pred_severity → binary | Binary (OK→no, WARNING/ERROR→yes, NA→excluded) |

Each dimension reports: TP/FP/FN/TN, precision/recall/F1/accuracy, breakdowns by venue type and source_sheet.

**Output:**
- `metrics_report.txt` — text report with all metrics
- `plots/` — confusion matrices, PRF bar charts, distribution comparisons
- `image_dumps/` — symlinked images organized by:
  - `measurability/TP`, `TN`, `FP`, `FN`
  - `quality_issue/TP`, `TN`, `FP`, `FN`, `skipped_not_measurable`
  - `severity/TP`, `TN`, `FP`, `FN`, `NA`

**Options:**
- `--skip-image-dumps` — compute metrics and plots only

## Fixing Annotations

After reviewing image dumps (especially FP/FN), fix ground truth in the xlsx and re-run:

```bash
# 1. Fix annotations in complete_annotation.xlsx (manually or via fix_annotations.py)
# 2. Re-export
python3 xlsx_to_csv.py
# 3. Re-run pipeline
python3 run_annotation_pipeline.py --annotations complete_annotation.csv
# 4. Re-evaluate
python3 confusion_metrics.py --predictions <new_predictions.csv> --output-dir <new_metrics_dir>
```

## Helper Scripts

| Script | Purpose |
|---|---|
| `download_annotation_data.py` | Download venue images from S3 for CSV venues |
| `xlsx_to_csv.py` | Export xlsx to csv with resolved hyperlinks |
| `add_image_links.py` | Add image hyperlink columns to csv, output as xlsx |
| `fix_annotations.py` | Batch-fix GT values in xlsx from FN/FP review |
