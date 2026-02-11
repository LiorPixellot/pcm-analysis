#!/usr/bin/env python3
"""
Analyze Quality Issue: Compare Gemini quality_has_issue predictions vs human annotations.

Compares "Has Quality Issue (Yes/No)" annotations vs "quality_has_issue" predictions.
Only analyzes rows where quality_is_measurable = Yes.

Usage:
    python analyze_quality_issue.py output_2026-02-08_15-30/
    python analyze_quality_issue.py /path/to/output_dir
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

# Try to import fpdf for PDF generation
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False


# Default column names (overridden by run_config.json annotations)
DEFAULT_ANNOTATION_COL = "Has Quality Issue (Yes/No)"
PREDICTION_COL = "quality_has_issue"
MEASURABLE_COL = "quality_is_measurable"
DEFAULT_VENUE_ID_COL = "System ID"
DEFAULT_ANNOTATION_TYPE_COL = "Quality Issue Type"
PREDICTION_TYPE_COL = "quality_issue_type"
DEFAULT_CALIBRATION_COL = "Calibration"
DEFAULT_INDOOR_OUTDOOR_COL = "Indoor / Outdoor"
DEFAULT_FAR_CLOSE_COL = "Far / Close"


def get_column_config(run_config):
    """Extract column names from run_config annotations, with defaults."""
    cols = {}
    if run_config:
        ann_cols = run_config.get("annotations", {}).get("columns", {})
    else:
        ann_cols = {}

    cols["annotation"] = ann_cols.get("has_quality_issue_annotation", DEFAULT_ANNOTATION_COL)
    cols["annotation_type"] = ann_cols.get("quality_issue_type_annotation", DEFAULT_ANNOTATION_TYPE_COL)
    cols["venue_id"] = run_config.get("annotations", {}).get("join_key", DEFAULT_VENUE_ID_COL) if run_config else DEFAULT_VENUE_ID_COL
    cols["calibration"] = ann_cols.get("calibration", DEFAULT_CALIBRATION_COL)
    cols["indoor_outdoor"] = ann_cols.get("indoor_outdoor", DEFAULT_INDOOR_OUTDOOR_COL)
    cols["far_close"] = ann_cols.get("far_close", DEFAULT_FAR_CLOSE_COL)
    return cols


def normalize_yes_no(val) -> str:
    """Normalize Yes/No values to lowercase 'yes' or 'no'."""
    if pd.isna(val) or val == "":
        return ""

    val_str = str(val).strip().lower()

    if val_str in ("yes", "y", "true", "1"):
        return "yes"
    elif val_str in ("no", "n", "false", "0", "none"):
        return "no"
    else:
        return ""


def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """Calculate precision, recall, F1, and confusion matrix.

    Treats 'yes' (has quality issue) as positive class.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for actual, pred in zip(y_true, y_pred):
        if actual == "yes" and pred == "yes":
            tp += 1
        elif actual == "no" and pred == "no":
            tn += 1
        elif actual == "no" and pred == "yes":
            fp += 1
        elif actual == "yes" and pred == "no":
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    precision_no = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_no = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_no = 2 * precision_no * recall_no / (precision_no + recall_no) if (precision_no + recall_no) > 0 else 0

    return {
        "total_samples": total,
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn
        },
        "accuracy": accuracy,
        "positive_class_yes": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": tp + fn
        },
        "negative_class_no": {
            "precision": precision_no,
            "recall": recall_no,
            "f1_score": f1_no,
            "support": tn + fp
        }
    }


def print_confusion_matrix(metrics: Dict):
    """Print a visual confusion matrix."""
    cm = metrics["confusion_matrix"]
    tp, tn, fp, fn = cm["true_positive"], cm["true_negative"], cm["false_positive"], cm["false_negative"]

    print("\n  CONFUSION MATRIX")
    print("  " + "-" * 40)
    print("                      Predicted")
    print("                    Yes      No")
    print("              +--------+--------+")
    print(f"  Actual Yes  |  {tp:4}   |  {fn:4}   |  (TP, FN)")
    print("              +--------+--------+")
    print(f"  Actual No   |  {fp:4}   |  {tn:4}   |  (FP, TN)")
    print("              +--------+--------+")


def print_metrics(metrics: Dict):
    """Print metrics in a formatted way."""
    print("\n  CLASSIFICATION METRICS")
    print("  " + "-" * 40)
    print(f"  Total samples: {metrics['total_samples']}")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")

    print("\n  Class 'Yes' (Has Quality Issue):")
    pos = metrics["positive_class_yes"]
    print(f"    Precision: {pos['precision']:.4f}")
    print(f"    Recall:    {pos['recall']:.4f}")
    print(f"    F1-Score:  {pos['f1_score']:.4f}")
    print(f"    Support:   {pos['support']}")

    print("\n  Class 'No' (No Quality Issue):")
    neg = metrics["negative_class_no"]
    print(f"    Precision: {neg['precision']:.4f}")
    print(f"    Recall:    {neg['recall']:.4f}")
    print(f"    F1-Score:  {neg['f1_score']:.4f}")
    print(f"    Support:   {neg['support']}")


def load_run_config(output_dir: Path) -> Optional[Dict]:
    """Load run configuration from run_config.json."""
    config_path = output_dir / "run_config.json"
    if not config_path.exists():
        return None

    with open(config_path, 'r') as f:
        return json.load(f)


def find_focus_images(data_dir: Path, venue_id: str) -> List[Path]:
    """Find focus images for a venue in the data directory."""
    images = []

    venue_dir = data_dir / venue_id
    if not venue_dir.exists():
        for subdir in data_dir.iterdir():
            if subdir.is_dir() and subdir.name == venue_id:
                venue_dir = subdir
                break

    if not venue_dir.exists():
        return images

    for event_dir in venue_dir.iterdir():
        if event_dir.is_dir():
            focus_dir = event_dir / "focus"
            if focus_dir.exists():
                for img in focus_dir.glob("*.jpg"):
                    images.append(img)
                if images:
                    break

    return images


def clean_column_name(col_name: str) -> str:
    """Clean column name for display."""
    clean = str(col_name).replace('\n', ' ').replace('\\', '/')
    while '  ' in clean:
        clean = clean.replace('  ', ' ')
    return clean.strip()[:50]


def get_display_value(row: pd.Series, col_name: str) -> str:
    """Get display value for a column, handling missing columns."""
    if col_name in row.index:
        val = row[col_name]
        if pd.isna(val):
            return "-"
        return str(val)[:60]

    clean_target = clean_column_name(col_name).lower()
    for col in row.index:
        if clean_column_name(col).lower() == clean_target:
            val = row[col]
            if pd.isna(val):
                return "-"
            return str(val)[:60]

    return "-"


def resize_image_for_pdf(img_path: Path, max_width: int = 250, max_height: int = 200) -> Optional[Path]:
    """Resize image for PDF and return path to temp file."""
    try:
        img = Image.open(img_path)
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img.save(temp_file.name, 'JPEG', quality=85)
        return Path(temp_file.name)
    except Exception as e:
        print(f"    Warning: Could not resize {img_path}: {e}")
        return None


def generate_mismatch_pdf(df: pd.DataFrame, data_dir: Path, output_path: Path,
                          title: str, mismatch_type: str, col_cfg: dict = None):
    """Generate PDF report for mismatched cases."""
    if not HAS_FPDF:
        print(f"  Warning: fpdf2 not installed, skipping PDF generation")
        return

    if col_cfg is None:
        col_cfg = get_column_config(None)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 24)
    pdf.cell(0, 20, title, ln=True, align='C')
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f"Total cases: {len(df)}", ln=True, align='C')
    pdf.cell(0, 10, f"Type: {mismatch_type}", ln=True, align='C')
    pdf.ln(10)

    # Find venue ID column
    venue_id_col = col_cfg["venue_id"]
    venue_col = None
    for col in df.columns:
        if col.lower() == venue_id_col.lower() or col == venue_id_col:
            venue_col = col
            break

    if venue_col is None:
        pdf.cell(0, 10, "Error: Could not find venue ID column", ln=True)
        pdf.output(output_path)
        return

    temp_files = []

    # Generate page for each case
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        pdf.add_page()

        venue_id = str(row[venue_col]).strip()

        # Header
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, f"Case {idx}/{len(df)}: {venue_id}", ln=True)
        pdf.ln(5)

        # Data table
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(70, 8, "Field", border=1)
        pdf.cell(0, 8, "Value", border=1, ln=True)

        pdf.set_font('Helvetica', '', 9)

        display_cols = [
            ("System ID", venue_col),
            ("Calibration", col_cfg["calibration"]),
            ("Indoor/Outdoor", col_cfg["indoor_outdoor"]),
            ("Far/Close", col_cfg["far_close"]),
            ("Is Measurable (Gemini)", "quality_is_measurable"),
            ("Annotation: Has Quality Issue", col_cfg["annotation"]),
            ("Prediction: quality_has_issue", "quality_has_issue"),
            ("Prediction: quality_issue_type", "quality_issue_type"),
            ("Prediction: which_camera", "quality_which_camera"),
        ]

        for display_name, col_name in display_cols:
            value = get_display_value(row, col_name)
            pdf.cell(70, 7, display_name, border=1)
            pdf.cell(0, 7, value, border=1, ln=True)

        pdf.ln(10)

        # Images
        if data_dir and data_dir.exists():
            images = find_focus_images(data_dir, venue_id)
            if images:
                pdf.set_font('Helvetica', 'B', 11)
                pdf.cell(0, 8, "Focus Images:", ln=True)
                pdf.ln(3)

                x_start = pdf.get_x()
                y_start = pdf.get_y()
                x_offset = 0

                for img_path in images[:2]:
                    resized = resize_image_for_pdf(img_path)
                    if resized:
                        temp_files.append(resized)
                        try:
                            pdf.image(str(resized), x=x_start + x_offset, y=y_start, w=90)
                            x_offset += 95
                        except Exception as e:
                            print(f"    Warning: Could not add image to PDF: {e}")

    pdf.output(output_path)

    for temp_file in temp_files:
        try:
            temp_file.unlink()
        except:
            pass


def dump_mismatch_images(df: pd.DataFrame, data_dir: Path, output_dir: Path, col_cfg: dict = None):
    """Copy focus images for TP, TN, FP and FN cases to separate directories and generate PDFs."""
    print("\n  DUMPING CLASSIFICATION IMAGES & PDFs")
    print("  " + "-" * 40)

    if col_cfg is None:
        col_cfg = get_column_config(None)

    # Create directories
    tp_dir = output_dir / "tp"
    tn_dir = output_dir / "tn"
    fp_dir = output_dir / "fp"
    fn_dir = output_dir / "fn"
    tp_dir.mkdir(exist_ok=True)
    tn_dir.mkdir(exist_ok=True)
    fp_dir.mkdir(exist_ok=True)
    fn_dir.mkdir(exist_ok=True)

    # Find venue ID column
    venue_id_col = col_cfg["venue_id"]
    venue_col = None
    for col in df.columns:
        if col.lower() == venue_id_col.lower() or col == venue_id_col:
            venue_col = col
            break

    if venue_col is None:
        print("  Warning: Could not find venue ID column, skipping image dump")
        return

    # Filter TP, TN, FP and FN rows
    df_tp = df[(df["_annotation"] == "yes") & (df["_prediction"] == "yes")].copy()
    df_tn = df[(df["_annotation"] == "no") & (df["_prediction"] == "no")].copy()
    df_fp = df[(df["_annotation"] == "no") & (df["_prediction"] == "yes")].copy()
    df_fn = df[(df["_annotation"] == "yes") & (df["_prediction"] == "no")].copy()

    print(f"  True Positives  (pred=Yes, actual=Yes): {len(df_tp)}")
    print(f"  True Negatives  (pred=No, actual=No):   {len(df_tn)}")
    print(f"  False Positives (pred=Yes, actual=No):   {len(df_fp)}")
    print(f"  False Negatives (pred=No, actual=Yes):   {len(df_fn)}")

    # Copy images for each category
    for category, cat_df, cat_dir in [
        ("TP", df_tp, tp_dir),
        ("TN", df_tn, tn_dir),
        ("FP", df_fp, fp_dir),
        ("FN", df_fn, fn_dir),
    ]:
        copied = 0
        for _, row in cat_df.iterrows():
            venue_id = str(row[venue_col]).strip()
            if not venue_id or pd.isna(row[venue_col]):
                continue

            images = find_focus_images(data_dir, venue_id)
            for img in images:
                dest = cat_dir / f"{venue_id}_{img.name}"
                try:
                    shutil.copy2(img, dest)
                    copied += 1
                except Exception as e:
                    pass

        print(f"  Copied {copied} images to {cat_dir.name}/")

    # Generate PDFs
    if HAS_FPDF:
        print("\n  Generating PDF reports...")

        if len(df_tp) > 0:
            tp_pdf = output_dir / "true_positives.pdf"
            generate_mismatch_pdf(df_tp, data_dir, tp_pdf,
                                  "True Positives Report",
                                  "Predicted quality issue (Yes), Actual has issue (Yes)",
                                  col_cfg)
            print(f"  Generated: {tp_pdf.name}")

        if len(df_tn) > 0:
            tn_pdf = output_dir / "true_negatives.pdf"
            generate_mismatch_pdf(df_tn, data_dir, tn_pdf,
                                  "True Negatives Report",
                                  "Predicted no issue (No), Actual no issue (No)",
                                  col_cfg)
            print(f"  Generated: {tn_pdf.name}")

        if len(df_fp) > 0:
            fp_pdf = output_dir / "false_positives.pdf"
            generate_mismatch_pdf(df_fp, data_dir, fp_pdf,
                                  "False Positives Report",
                                  "Predicted quality issue (Yes), Actual no issue (No)",
                                  col_cfg)
            print(f"  Generated: {fp_pdf.name}")

        if len(df_fn) > 0:
            fn_pdf = output_dir / "false_negatives.pdf"
            generate_mismatch_pdf(df_fn, data_dir, fn_pdf,
                                  "False Negatives Report",
                                  "Predicted no issue (No), Actual has quality issue (Yes)",
                                  col_cfg)
            print(f"  Generated: {fn_pdf.name}")


def analyze_quality_issue(output_dir: Path, col_cfg: dict = None) -> Tuple[Optional[Dict], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Analyze quality issue predictions from concatenated_results.xlsx.

    Returns (metrics, df_valid, df_measurable).
    """
    xlsx_path = output_dir / "concatenated_results.xlsx"

    if col_cfg is None:
        col_cfg = get_column_config(None)

    if not xlsx_path.exists():
        print(f"Error: File not found: {xlsx_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  ANALYZE QUALITY ISSUE (Gemini)")
    print("=" * 60)
    print(f"  Input: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    print(f"  Total rows: {len(df)}")

    # Find columns using configured names
    annotation_col = None
    prediction_col = None
    measurable_col = None
    target_annotation = col_cfg["annotation"].lower().replace("\n", " ")

    for col in df.columns:
        col_clean = str(col).lower().replace("\n", " ")
        if col_clean == target_annotation or "has quality issue" in col_clean:
            annotation_col = col
        elif col == PREDICTION_COL or col == "quality_has_issue":
            prediction_col = col
        elif col == MEASURABLE_COL or col == "quality_is_measurable":
            measurable_col = col

    if annotation_col is None:
        print(f"Error: Annotation column 'Has Quality Issue' not found.", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    if prediction_col is None:
        print(f"Error: Prediction column 'quality_has_issue' not found.", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    if measurable_col is None:
        print(f"Error: Measurable column 'quality_is_measurable' not found.", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    print(f"  Annotation column: {annotation_col}")
    print(f"  Prediction column: {prediction_col}")
    print(f"  Measurable column: {measurable_col}")

    # Filter to only measurable rows (quality_is_measurable = yes)
    df["_measurable"] = df[measurable_col].apply(normalize_yes_no)
    df_measurable = df[df["_measurable"] == "yes"].copy()
    print(f"\n  Filtered to measurable rows: {len(df_measurable)} (excluded {len(df) - len(df_measurable)} non-measurable)")

    if len(df_measurable) == 0:
        print("Error: No measurable rows found", file=sys.stderr)
        return None, None, None

    # Normalize values
    df_measurable["_annotation"] = df_measurable[annotation_col].apply(normalize_yes_no)
    df_measurable["_prediction"] = df_measurable[prediction_col].apply(normalize_yes_no)

    # Value distributions
    print("\n  VALUE DISTRIBUTION (measurable only):")
    print(f"  Annotations (Has Quality Issue): {df_measurable['_annotation'].value_counts().to_dict()}")
    print(f"  Predictions (quality_has_issue): {df_measurable['_prediction'].value_counts().to_dict()}")

    # Filter to valid rows
    df_valid = df_measurable[(df_measurable["_annotation"] != "") & (df_measurable["_prediction"] != "")]
    print(f"\n  Valid rows (both annotation and prediction): {len(df_valid)}")

    if len(df_valid) == 0:
        print("Error: No valid rows found", file=sys.stderr)
        return None, None, df_measurable

    # Calculate metrics
    y_true = df_valid["_annotation"].tolist()
    y_pred = df_valid["_prediction"].tolist()

    metrics = calculate_metrics(y_true, y_pred)

    print_confusion_matrix(metrics)
    print_metrics(metrics)

    metrics["metadata"] = {
        "input_file": str(xlsx_path),
        "total_rows": len(df),
        "measurable_rows": len(df_measurable),
        "valid_rows": len(df_valid),
        "annotation_column": annotation_col,
        "prediction_column": prediction_col
    }

    return metrics, df_valid, df_measurable


def save_metrics(metrics: Dict, output_dir: Path):
    """Save metrics to JSON file."""
    output_path = output_dir / "quality_issue_metrics.json"

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Metrics saved: {output_path}")


def analyze_condensation(df_measurable: pd.DataFrame, col_cfg: dict = None) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """Analyze condensation detection: annotation Quality Issue Type vs prediction quality_issue_type.

    Returns (metrics, df_valid_condensation).
    """
    if col_cfg is None:
        col_cfg = get_column_config(None)

    print("\n" + "=" * 60)
    print("  ANALYZE CONDENSATION DETECTION")
    print("=" * 60)

    # Find annotation type column using configured name
    annotation_type_col = None
    prediction_type_col = None
    target_type = col_cfg["annotation_type"].lower().replace("\n", " ")
    for col in df_measurable.columns:
        col_clean = str(col).lower().replace("\n", " ")
        if col_clean == target_type or "quality issue type" in col_clean:
            annotation_type_col = col
        elif col == PREDICTION_TYPE_COL or col == "quality_issue_type":
            prediction_type_col = col

    if annotation_type_col is None:
        print(f"  Error: Annotation column '{col_cfg['annotation_type']}' not found.", file=sys.stderr)
        print(f"  Available columns: {list(df_measurable.columns)}", file=sys.stderr)
        return None, None

    if prediction_type_col is None:
        print(f"  Error: Prediction column '{PREDICTION_TYPE_COL}' not found.", file=sys.stderr)
        print(f"  Available columns: {list(df_measurable.columns)}", file=sys.stderr)
        return None, None

    print(f"  Annotation column: {annotation_type_col}")
    print(f"  Prediction column: {prediction_type_col}")

    # Derive binary condensation columns
    df_measurable["_annotation_condensation"] = df_measurable[annotation_type_col].apply(
        lambda v: "yes" if pd.notna(v) and str(v).strip().lower() == "condensation" else "no"
    )
    df_measurable["_prediction_condensation"] = df_measurable[prediction_type_col].apply(
        lambda v: "yes" if pd.notna(v) and str(v).strip().lower() == "condensation" else "no"
    )

    # Filter to valid rows (both columns non-empty, i.e. original values are present)
    df_valid = df_measurable[
        df_measurable[annotation_type_col].apply(lambda v: pd.notna(v) and str(v).strip() != "")
        & df_measurable[prediction_type_col].apply(lambda v: pd.notna(v) and str(v).strip() != "")
    ].copy()

    print(f"\n  Valid rows (both annotation type and prediction type): {len(df_valid)}")

    # Value distributions
    print(f"\n  VALUE DISTRIBUTION (measurable, valid):")
    print(f"  Annotation type values: {df_valid[annotation_type_col].value_counts().to_dict()}")
    print(f"  Prediction type values: {df_valid[prediction_type_col].value_counts().to_dict()}")
    print(f"  Annotation condensation (binary): {df_valid['_annotation_condensation'].value_counts().to_dict()}")
    print(f"  Prediction condensation (binary): {df_valid['_prediction_condensation'].value_counts().to_dict()}")

    if len(df_valid) == 0:
        print("  Error: No valid rows found for condensation analysis", file=sys.stderr)
        return None, None

    y_true = df_valid["_annotation_condensation"].tolist()
    y_pred = df_valid["_prediction_condensation"].tolist()

    metrics = calculate_metrics(y_true, y_pred)

    print_confusion_matrix(metrics)
    print_metrics(metrics)

    metrics["metadata"] = {
        "analysis_type": "condensation",
        "total_measurable_rows": len(df_measurable),
        "valid_rows": len(df_valid),
        "annotation_column": annotation_type_col,
        "prediction_column": prediction_type_col,
        "positive_class": "condensation"
    }

    return metrics, df_valid


def dump_mismatch_images_condensation(df: pd.DataFrame, data_dir: Path, output_dir: Path, col_cfg: dict = None):
    """Copy focus images for condensation TP, TN, FP, FN cases and generate PDFs."""
    print("\n  DUMPING CONDENSATION CLASSIFICATION IMAGES & PDFs")
    print("  " + "-" * 40)

    if col_cfg is None:
        col_cfg = get_column_config(None)

    # Create directories
    tp_dir = output_dir / "tp_condensation"
    tn_dir = output_dir / "tn_condensation"
    fp_dir = output_dir / "fp_condensation"
    fn_dir = output_dir / "fn_condensation"
    tp_dir.mkdir(exist_ok=True)
    tn_dir.mkdir(exist_ok=True)
    fp_dir.mkdir(exist_ok=True)
    fn_dir.mkdir(exist_ok=True)

    # Find venue ID column
    venue_id_col = col_cfg["venue_id"]
    venue_col = None
    for col in df.columns:
        if col.lower() == venue_id_col.lower() or col == venue_id_col:
            venue_col = col
            break

    if venue_col is None:
        print("  Warning: Could not find venue ID column, skipping image dump")
        return

    # Filter TP, TN, FP and FN rows
    df_tp = df[(df["_annotation_condensation"] == "yes") & (df["_prediction_condensation"] == "yes")].copy()
    df_tn = df[(df["_annotation_condensation"] == "no") & (df["_prediction_condensation"] == "no")].copy()
    df_fp = df[(df["_annotation_condensation"] == "no") & (df["_prediction_condensation"] == "yes")].copy()
    df_fn = df[(df["_annotation_condensation"] == "yes") & (df["_prediction_condensation"] == "no")].copy()

    print(f"  True Positives  (pred=Condensation, actual=Condensation): {len(df_tp)}")
    print(f"  True Negatives  (pred=Other, actual=Other):               {len(df_tn)}")
    print(f"  False Positives (pred=Condensation, actual=Other):        {len(df_fp)}")
    print(f"  False Negatives (pred=Other, actual=Condensation):        {len(df_fn)}")

    # Copy images for each category
    for category, cat_df, cat_dir in [
        ("TP", df_tp, tp_dir),
        ("TN", df_tn, tn_dir),
        ("FP", df_fp, fp_dir),
        ("FN", df_fn, fn_dir),
    ]:
        copied = 0
        for _, row in cat_df.iterrows():
            venue_id = str(row[venue_col]).strip()
            if not venue_id or pd.isna(row[venue_col]):
                continue

            images = find_focus_images(data_dir, venue_id)
            for img in images:
                dest = cat_dir / f"{venue_id}_{img.name}"
                try:
                    shutil.copy2(img, dest)
                    copied += 1
                except Exception:
                    pass

        print(f"  Copied {copied} images to {cat_dir.name}/")

    # Generate PDFs
    if HAS_FPDF:
        print("\n  Generating condensation PDF reports...")

        if len(df_tp) > 0:
            tp_pdf = output_dir / "true_positives_condensation.pdf"
            generate_mismatch_pdf(df_tp, data_dir, tp_pdf,
                                  "True Positives Report (Condensation)",
                                  "Predicted condensation, Actual condensation",
                                  col_cfg)
            print(f"  Generated: {tp_pdf.name}")

        if len(df_tn) > 0:
            tn_pdf = output_dir / "true_negatives_condensation.pdf"
            generate_mismatch_pdf(df_tn, data_dir, tn_pdf,
                                  "True Negatives Report (Condensation)",
                                  "Predicted no condensation, Actual no condensation",
                                  col_cfg)
            print(f"  Generated: {tn_pdf.name}")

        if len(df_fp) > 0:
            fp_pdf = output_dir / "false_positives_condensation.pdf"
            generate_mismatch_pdf(df_fp, data_dir, fp_pdf,
                                  "False Positives Report (Condensation)",
                                  "Predicted condensation, Actual no condensation",
                                  col_cfg)
            print(f"  Generated: {fp_pdf.name}")

        if len(df_fn) > 0:
            fn_pdf = output_dir / "false_negatives_condensation.pdf"
            generate_mismatch_pdf(df_fn, data_dir, fn_pdf,
                                  "False Negatives Report (Condensation)",
                                  "Predicted no condensation, Actual condensation",
                                  col_cfg)
            print(f"  Generated: {fn_pdf.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Gemini quality_has_issue predictions vs human annotations'
    )
    parser.add_argument('output_dir', help='Directory containing concatenated_results.xlsx')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data directory with focus images (default: read from run_config.json)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f'Error: {args.output_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    # Load run config to get data_dir and column config
    run_config = load_run_config(output_dir)
    col_cfg = get_column_config(run_config)

    data_dir = None
    data_dir_source = None
    if args.data_dir:
        data_dir = Path(args.data_dir)
        data_dir_source = "command line"
    elif run_config and "data_dir" in run_config:
        data_dir = Path(run_config["data_dir"])
        data_dir_source = "run_config.json"

    # Print data_dir info at start
    if data_dir:
        print(f"  Data directory: {data_dir} (from {data_dir_source})")
        if not data_dir.exists():
            print(f"  WARNING: Data directory does not exist!")
    else:
        print("  Data directory: Not specified")

    metrics, df_valid, df_measurable = analyze_quality_issue(output_dir, col_cfg)

    if metrics:
        # Create subdirectory for results
        results_dir = output_dir / "analysis_quality_issue"
        results_dir.mkdir(exist_ok=True)
        print(f"\n  Results directory: {results_dir}")

        save_metrics(metrics, results_dir)

        # Dump mismatch images and generate PDFs
        if data_dir and data_dir.exists():
            dump_mismatch_images(df_valid, data_dir, results_dir, col_cfg)
        elif data_dir:
            print(f"\n  Warning: Data directory not found: {data_dir}")
            print("  Skipping image dump and PDF generation")
        else:
            print("\n  Warning: No data directory specified")
            print("  Use --data-dir or ensure run_config.json exists")

    # Condensation analysis
    if df_measurable is not None:
        condensation_metrics, df_valid_condensation = analyze_condensation(df_measurable, col_cfg)

        if condensation_metrics:
            condensation_dir = output_dir / "analysis_condensation"
            condensation_dir.mkdir(exist_ok=True)
            print(f"\n  Condensation results directory: {condensation_dir}")

            # Save condensation metrics
            condensation_metrics_path = condensation_dir / "condensation_metrics.json"
            with open(condensation_metrics_path, 'w') as f:
                json.dump(condensation_metrics, f, indent=2)
            print(f"  Metrics saved: {condensation_metrics_path}")

            # Dump condensation classification images and PDFs
            if data_dir and data_dir.exists():
                dump_mismatch_images_condensation(df_valid_condensation, data_dir, condensation_dir, col_cfg)
            elif data_dir:
                print(f"\n  Warning: Data directory not found: {data_dir}")
                print("  Skipping condensation image dump and PDF generation")
            else:
                print("\n  Warning: No data directory specified")
                print("  Use --data-dir or ensure run_config.json exists")

    if metrics or (df_measurable is not None):
        print("\n" + "=" * 60)
        print("  COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()
