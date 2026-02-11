# Analysis Scripts Documentation

## Overview

Three analysis scripts evaluate different aspects of the camera quality predictions against human annotations. All scripts:
- Read from `concatenated_results.xlsx` in the output directory
- Read `data_dir` from `run_config.json` for image access
- Generate metrics, PDFs, and image dumps to subdirectories

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ANALYSIS SCRIPTS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────┐  ┌────────────────────────┐  ┌───────────────┐ │
│  │ analyze_is_measurable  │  │ analyze_focus_         │  │ analyze_      │ │
│  │         .py            │  │    calculations.py     │  │ quality_issue │ │
│  └───────────┬────────────┘  └───────────┬────────────┘  └──────┬────────┘ │
│              │                           │                       │          │
│              ▼                           ▼                       ▼          │
│  ┌────────────────────────┐  ┌────────────────────────┐  ┌───────────────┐ │
│  │ Is Measurable          │  │ Focus Severity         │  │ Quality Issue │ │
│  │ (Gemini vs Human)      │  │ (Laplacian vs Human)   │  │ (Gemini vs    │ │
│  │                        │  │                        │  │  Human)       │ │
│  └────────────────────────┘  └────────────────────────┘  └───────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. analyze_is_measurable.py

**Purpose:** Evaluate if Gemini correctly identifies whether camera images are measurable (not affected by weather/lighting conditions).

### Comparison

| Role | Column | Values |
|------|--------|--------|
| **Annotation (Human)** | `Is Measurable camera problems (Yes/No)` | Yes / No |
| **Prediction (Gemini)** | `quality_is_measurable` | Yes / No |

### Filter
- None (all rows with both values)

### Output Directory
```
output_dir/
└── analysis_is_measurable/
    ├── analysis_metrics.json
    ├── fp/                      # False Positive images
    ├── fn/                      # False Negative images
    ├── false_positives.pdf
    └── false_negatives.pdf
```

### Usage
```bash
python analyze_is_measurable.py output_2026-02-08_20-53/
```

### Interpretation
- **True Positive (TP):** Both say "Yes" (measurable)
- **False Positive (FP):** Gemini says "Yes", Human says "No"
- **False Negative (FN):** Gemini says "No", Human says "Yes"
- **True Negative (TN):** Both say "No" (not measurable)

---

## 2. analyze_focus_calculations.py

**Purpose:** Evaluate if Laplacian-based focus severity correctly identifies quality issues.

### Comparison

| Role | Column | Values |
|------|--------|--------|
| **Annotation (Human)** | `Has Quality Issue (Yes/No)` | Yes / No |
| **Prediction (Laplacian)** | `Focus severity` | Error / Warning / Ok / NA |

### Prediction Modes

Two reports are generated with different thresholds:

| Mode | Prediction Logic |
|------|------------------|
| **Error Only** | `Focus_severity = Error` → Yes, else → No |
| **Error + Warning** | `Focus_severity = Error OR Warning` → Yes, else → No |

### Filter
- Only rows where `quality_is_measurable = Yes`

### Output Directory
```
output_dir/
└── analysis_focus_calculations/
    ├── focus_analysis_metrics.json
    ├── focus_fp_error_only/
    ├── focus_fn_error_only/
    ├── focus_fp_error_warning/
    ├── focus_fn_error_warning/
    ├── focus_false_positives_error_only.pdf
    ├── focus_false_negatives_error_only.pdf
    ├── focus_false_positives_error_warning.pdf
    └── focus_false_negatives_error_warning.pdf
```

### Usage
```bash
python analyze_focus_calculations.py output_2026-02-08_20-53/
```

### Interpretation
- **Error Only mode:** Strict - only "Error" severity indicates quality issue
- **Error + Warning mode:** Lenient - both "Error" and "Warning" indicate quality issue

Compare both to find optimal threshold for your use case.

---

## 3. analyze_quality_issue.py

**Purpose:** Evaluate if Gemini correctly identifies quality issues (focus problems).

### Comparison

| Role | Column | Values |
|------|--------|--------|
| **Annotation (Human)** | `Has Quality Issue (Yes/No)` | Yes / No |
| **Prediction (Gemini)** | `quality_has_issue` | Yes / No / None |

### Filter
- Only rows where `quality_is_measurable = Yes`

### Output Directory
```
output_dir/
└── analysis_quality_issue/
    ├── quality_issue_metrics.json
    ├── fp/
    ├── fn/
    ├── false_positives.pdf
    └── false_negatives.pdf
```

### Usage
```bash
python analyze_quality_issue.py output_2026-02-08_20-53/
```

### Interpretation
- **True Positive (TP):** Both say "Yes" (has quality issue)
- **False Positive (FP):** Gemini says "Yes", Human says "No"
- **False Negative (FN):** Gemini says "No", Human says "Yes"
- **True Negative (TN):** Both say "No" (no quality issue)

---

## Metrics Explained

All scripts calculate the same metrics:

### Confusion Matrix
```
                      Predicted
                    Yes      No
              +--------+--------+
  Actual Yes  |   TP   |   FN   |
              +--------+--------+
  Actual No   |   FP   |   TN   |
              +--------+--------+
```

### Classification Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Of predicted Yes, how many are correct |
| **Recall** | TP / (TP + FN) | Of actual Yes, how many were found |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |

### Which Metric Matters?

| Use Case | Focus On |
|----------|----------|
| Minimize false alarms | **Precision** (reduce FP) |
| Catch all issues | **Recall** (reduce FN) |
| Balanced performance | **F1-Score** |

---

## PDF Reports

Each PDF contains:

1. **Title page** with total cases and type
2. **Per-case pages** with:
   - Venue ID
   - Data table (annotation, prediction, metadata)
   - Focus images (CAM0, CAM1) side-by-side

### Example PDF Page
```
┌─────────────────────────────────────────────────┐
│  Case 1/5: 5bed37433e663c077bc36cef             │
├─────────────────────────────────────────────────┤
│  Field                    │ Value               │
│  ─────────────────────────┼─────────────────────│
│  System ID                │ 5bed3743...         │
│  Calibration              │ BASKETBALL          │
│  Indoor/Outdoor           │ Indoor              │
│  Annotation (Human)       │ Yes                 │
│  Prediction (Gemini)      │ No                  │
├─────────────────────────────────────────────────┤
│  Focus Images:                                  │
│  ┌──────────────┐  ┌──────────────┐            │
│  │   CAM0_1     │  │   CAM1_1     │            │
│  │   (Right)    │  │   (Left)     │            │
│  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────┘
```

---

## Running All Analyses

```bash
# Run all three analyses on an output directory
python analyze_is_measurable.py output_2026-02-08_20-53/
python analyze_focus_calculations.py output_2026-02-08_20-53/
python analyze_quality_issue.py output_2026-02-08_20-53/
```

### Final Output Structure
```
output_2026-02-08_20-53/
├── run_config.json
├── concatenated_results.xlsx
├── ...
├── analysis_is_measurable/
│   ├── analysis_metrics.json
│   ├── fp/
│   ├── fn/
│   └── *.pdf
├── analysis_focus_calculations/
│   ├── focus_analysis_metrics.json
│   ├── focus_fp_*/
│   ├── focus_fn_*/
│   └── *.pdf
└── analysis_quality_issue/
    ├── quality_issue_metrics.json
    ├── fp/
    ├── fn/
    └── *.pdf
```

---

## Summary Table

| Script | Annotation | Prediction | Filter | Modes |
|--------|------------|------------|--------|-------|
| `analyze_is_measurable.py` | Is Measurable (Yes/No) | quality_is_measurable | None | 1 |
| `analyze_focus_calculations.py` | Has Quality Issue | Focus severity | measurable=Yes | 2 |
| `analyze_quality_issue.py` | Has Quality Issue | quality_has_issue | measurable=Yes | 1 |
