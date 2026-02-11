# Complete Pipeline Documentation

## Camera Focus Analysis & Evaluation System

**Version:** 1.0
**Last Updated:** February 2026

---

## Overview

This system provides end-to-end camera focus quality analysis for sports venue camera systems. It combines:

1. **Data Processing Pipeline** (`full_flow.py`) - Calculates focus metrics and AI quality analysis
2. **Analysis Scripts** - Evaluates predictions against human annotations

```
                           COMPLETE SYSTEM OVERVIEW

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                           PHASE 1: DATA PROCESSING                          │
  │                              (full_flow.py)                                 │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
  │   │   STEP 1     │    │   STEP 2     │    │   STEP 3     │    │  STEP 4  │ │
  │   │  Laplacian   │───▶│  Threshold   │───▶│ Measurable   │───▶│  Concat  │ │
  │   │ Calculations │    │ Calculations │    │   (Gemini)   │    │ Results  │ │
  │   └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘ │
  │                                                                      │      │
  │                                                                      ▼      │
  │                                               concatenated_results.xlsx     │
  └─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                           PHASE 2: EVALUATION                               │
  │                          (Analysis Scripts)                                 │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │                                                                             │
  │   ┌────────────────────────┐  ┌────────────────────────┐  ┌───────────────┐│
  │   │ analyze_is_measurable  │  │ analyze_focus_         │  │ analyze_      ││
  │   │         .py            │  │    calculations.py     │  │ quality_issue ││
  │   │                        │  │                        │  │    .py        ││
  │   │  Gemini is_measurable  │  │  Laplacian severity    │  │  Gemini focus ││
  │   │  vs Human annotation   │  │  vs Human annotation   │  │  vs Human     ││
  │   └────────────────────────┘  └────────────────────────┘  └───────────────┘│
  │                                                                             │
  │   Output: Metrics (Precision, Recall, F1), PDFs, Image dumps               │
  │                                                                             │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

# PHASE 1: DATA PROCESSING

## full_flow.py Pipeline

### Input Data Structure

```
{data_dir}/
├── {venue_id_1}/
│   ├── {event_id_1}/
│   │   └── focus/
│   │       ├── CAM0_1.jpg    (Right camera image)
│   │       ├── CAM1_1.jpg    (Left camera image)
│   │       └── focus.json    (Optional: validation data)
│   └── {event_id_2}/
│       └── focus/
│           └── ...
├── {venue_id_2}/
│   └── ...
```

### Pipeline Stages

#### Stage 1: Laplacian Calculations

**Purpose:** Calculate focus quality metrics using Laplacian variance analysis.

**Process:**
1. Load CAM0_1.jpg (right camera) and CAM1_1.jpg (left camera)
2. Extract 800x800 pixel regions from image centers
3. Apply Laplacian filter to detect edges/sharpness
4. Calculate variance as focus quality metric

**Output Metrics:**

| Metric | Description |
|--------|-------------|
| `focus_right_mean` | Mean intensity of right camera region |
| `focus_left_mean` | Mean intensity of left camera region |
| `focus_right_mid` | Laplacian variance of right camera (focus score) |
| `focus_left_mid` | Laplacian variance of left camera (focus score) |
| `focus_abs_dif_rel` | Relative difference between cameras |

---

#### Stage 2: Threshold Calculations

**Purpose:** Classify focus quality into severity levels.

```
                    SEVERITY DECISION TREE

  IF focus_right_mean < 70 OR focus_left_mean < 70
     └─▶ "NA" (image too dark to measure)

  ELSE calculate from two factors:

  Factor 1: Relative Difference (focus_abs_dif_rel)
     • > 1.25  ──▶ "Error"
     • 0.70-1.25 ──▶ "Warning"
     • < 0.70  ──▶ "Ok"

  Factor 2: Mid Focus Values (per camera)
     • ≤ 10   ──▶ "Error"
     • 10-20  ──▶ "Warning"
     • > 20   ──▶ "Ok"

  Final = WORST(Factor 1, Factor 2)
```

**Output:** Adds `Focus_severity` column (Ok / Warning / Error / NA)

---

#### Stage 3: Measurability Analysis (Gemini AI)

**Purpose:** Use AI vision to determine:
1. If environmental conditions allow valid measurement
2. If there are focus issues

**Output Columns:**

| Column | Description | Values |
|--------|-------------|--------|
| `quality_is_measurable` | Can focus be measured? | Yes / No |
| `quality_not_measurable_reason` | Why not measurable | fog, dark, etc. |
| `quality_has_issue` | Has focus problem? | Yes / No / None |
| `quality_issue_type` | Type of issue | Focus / None |
| `quality_which_camera` | Which camera affected | Left / Right / Both / None |

---

#### Stage 4: Concatenate with Annotations

**Purpose:** Merge analysis results with human annotations from Excel.

**Input:** `Annotators - Working Table 2025.xlsx` (POC sheet)

**Output:** `concatenated_results.xlsx` - Contains:
- All original Excel columns (human annotations)
- Focus metrics (focus_* columns)
- Gemini quality results (quality_* columns)

---

### Output Directory Structure

Each run creates a timestamped directory:

```
output_YYYY-MM-DD_HH-MM/
├── run_config.json              # Execution parameters (data_dir, etc.)
├── laplacian_calculations.csv   # Stage 1 output
├── laplacian_th.csv            # Stage 2 output
├── is_measurable.csv           # Stage 3 output
├── cost.txt                    # Gemini API cost report
└── concatenated_results.xlsx   # Stage 4 output (final)
```

---

### Configuration

#### flow.yaml - Control which steps run

```yaml
steps:
  laplacian: true       # Step 1: Calculate Laplacian focus metrics
  threshold: true       # Step 2: Add Focus_severity thresholds
  is_measurable: true   # Step 3: Gemini measurability analysis
  concatenate: true     # Step 4: Concatenate with Excel annotations
```

#### config.yaml - Global settings

```yaml
data_dir: "data_12_13"           # Default data directory
model: "gemini-3-flash-preview"   # Gemini model
temperature: 0
top_p: 0.95
project: "pixellot-ai"
location: "global"
```

---

### Usage

```bash
# Activate virtual environment
source /home/lior/Work/pcm_new/venv/bin/activate

# Test run (2 events)
python3 full_flow.py --limit 2

# Full production run
python3 full_flow.py

# Re-run specific steps in existing output directory
python3 full_flow.py --output-dir output_2026-02-08_20-53/
```

---

# PHASE 2: EVALUATION

## Analysis Scripts Overview

Three analysis scripts evaluate different aspects of predictions against human annotations:

| Script | Prediction Source | Annotation Column |
|--------|-------------------|-------------------|
| `analyze_is_measurable.py` | Gemini `quality_is_measurable` | `Is Measurable camera problems (Yes/No)` |
| `analyze_focus_calculations.py` | Laplacian `Focus severity` | `Has Quality Issue (Yes/No)` |
| `analyze_quality_issue.py` | Gemini `quality_has_issue` | `Has Quality Issue (Yes/No)` |

All scripts:
- Read from `concatenated_results.xlsx` in the output directory
- Read `data_dir` from `run_config.json` for image access
- Generate metrics, PDFs, and image dumps

---

## 1. analyze_is_measurable.py

**Purpose:** Evaluate if Gemini correctly identifies whether images are measurable.

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

---

## 2. analyze_focus_calculations.py

**Purpose:** Evaluate if Laplacian-based severity correctly identifies quality issues.

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

---

## 3. analyze_quality_issue.py

**Purpose:** Evaluate if Gemini correctly identifies focus quality issues.

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

---

## Metrics Explained

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

---

# COMPLETE WORKFLOW

## Step-by-Step Guide

### 1. Run the Data Processing Pipeline

```bash
# Activate environment
source /home/lior/Work/pcm_new/venv/bin/activate

# Configure steps in flow.yaml
# Configure data_dir in config.yaml

# Run full flow
python3 full_flow.py --limit 10  # Test first

# Full run
python3 full_flow.py
```

Output: Creates `output_YYYY-MM-DD_HH-MM/` with all results.

### 2. Run Analysis Scripts

```bash
# Run all three analyses on the output directory
python analyze_is_measurable.py output_2026-02-08_20-53/
python analyze_focus_calculations.py output_2026-02-08_20-53/
python analyze_quality_issue.py output_2026-02-08_20-53/
```

### 3. Review Results

```
output_2026-02-08_20-53/
├── run_config.json
├── laplacian_calculations.csv
├── laplacian_th.csv
├── is_measurable.csv
├── cost.txt
├── concatenated_results.xlsx          # Main results
│
├── analysis_is_measurable/
│   ├── analysis_metrics.json          # Precision, Recall, F1
│   ├── false_positives.pdf            # Visual review
│   └── false_negatives.pdf
│
├── analysis_focus_calculations/
│   ├── focus_analysis_metrics.json    # Compare error_only vs error_warning
│   └── *.pdf
│
└── analysis_quality_issue/
    ├── quality_issue_metrics.json
    └── *.pdf
```

---

## Summary Table

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| `full_flow.py` | Camera images | concatenated_results.xlsx | Generate predictions |
| `analyze_is_measurable.py` | concatenated_results.xlsx | metrics + PDFs | Evaluate measurability |
| `analyze_focus_calculations.py` | concatenated_results.xlsx | metrics + PDFs | Evaluate Laplacian |
| `analyze_quality_issue.py` | concatenated_results.xlsx | metrics + PDFs | Evaluate Gemini quality |

---

## Dependencies

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| pandas | Data manipulation |
| openpyxl | Excel support |
| opencv-python | Image processing |
| Pillow | Image loading |
| google-genai | Gemini AI |
| PyYAML | Configuration |
| fpdf2 | PDF generation |

---

## Contact

For questions or issues, contact the PCM team.
