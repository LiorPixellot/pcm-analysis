# PCM Push — Pipeline Summary

## 1. Production Pipeline (`run_full_pipeline.py`)

7-step flow from raw camera images to final analysis report:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `run_laplacian_pipeline.py` | Compute Laplacian focus metrics from CAM0/CAM1 images + classify severity (OK/Warning/Error/NA) |
| 2 | `concat_blur.py` | Join blur data from `PQS_blur_by_venue.xlsx` with Laplacian output → `.xlsx` |
| 3 | `run_is_measurable.py` | Gemini AI measurability analysis (indoor/outdoor routing, returns Yes/No per venue) |
| 4 | `concat_with_is_measurable.py` | Join measurability results with blur data → `.xlsx` |
| 5 | `detect_issues.py` | Gemini AI issue detection (focus/object/condensation) with few-shot examples |
| 6 | `concat_with_detect_issues.py` | Join detect_issues results → `.xlsx` |
| 7 | `analyze_blur_severity.py` | Generate plots + stats from the fully joined dataset |

**Shortcuts:** Steps 3 and 5 can be skipped with `--is-measurable-csv` / `--detect-issues-csv` to use pre-computed results.

**Output:** `output_dir/YYYY-MM-DD_HH-MM/` with subdirs for each step + `concat_data/` + `source_files/` (copied for reproducibility).

## 2. Annotation Evaluation Pipeline (`run_annotation_pipeline.md`)

3-step flow evaluating predictions against human-annotated ground truth:

```
Step 1: xlsx_to_csv.py
        Export complete_annotation.xlsx → complete_annotation.csv (resolves image hyperlinks)

Step 2: run_annotation_pipeline.py --annotations complete_annotation.csv
        Per row runs:
          a) Laplacian severity (CPU, no API) → OK/Warning/Error/NA
          b) Gemini measurability (API) → Yes/No
          c) Gemini issue detection (API, only if measurable=Yes) → has_issue + issue_type

Step 3: confusion_metrics.py --predictions <predictions.csv> --output-dir <metrics_dir>
        Evaluate predictions vs GT → metrics_report.txt + plots/ + image_dumps/
```

**Output:** `output_dir/eval_YYYY-MM-DD_HH-MM/annotation_predictions.csv` with GT columns (`gt_*`) and prediction columns (`pred_*`).

## 3. Confusion Metrics (`confusion_metrics.py`)

Evaluates 3 binary dimensions, each with TP/FP/FN/TN, precision/recall/F1/accuracy, breakdowns by venue type and source_sheet:

| Dimension | GT column | Pred column | Mapping |
|-----------|-----------|-------------|---------|
| Measurability | `gt_measurable` | `pred_measurable` | Direct yes/no |
| Quality issue (Gemini) | `gt_has_issue` | `pred_has_issue` | Direct yes/no |
| Quality issue (Laplacian) | `gt_has_issue` | `pred_severity` → binary | OK→no, WARNING/ERROR→yes, NA→excluded |

**Outputs:**
- `metrics_report.txt` — text report with all metrics + breakdowns
- `plots/` — confusion matrices, PRF bar charts, GT vs predicted distribution comparisons (9 PNGs)
- `image_dumps/` — symlinked images organized into:
  - `measurability/` → TP, TN, FP, FN
  - `quality_issue/` → TP, TN, FP, FN, skipped_not_measurable
  - `severity/` → TP, TN, FP, FN, NA
