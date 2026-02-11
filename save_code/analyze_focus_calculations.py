#!/usr/bin/env python3
"""
Analyze Focus Calculations: Compare focus severity predictions vs human quality annotations.

Compares "Has Quality Issue (Yes/No)" annotations vs "Focus severity" predictions.
Only analyzes rows where quality_is_measurable = Yes.

Generates two reports:
1. Error only: Focus_severity = Error means quality issue predicted
2. Error + Warning: Focus_severity = Error OR Warning means quality issue predicted

Usage:
    python analyze_focus_calculations.py output_2026-02-08_15-30/
    python analyze_focus_calculations.py /path/to/output_dir
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
PREDICTION_COL = "Focus severity"  # Error, Warning, Ok
MEASURABLE_COL = "quality_is_measurable"
DEFAULT_VENUE_ID_COL = "System ID"
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
    elif val_str in ("no", "n", "false", "0"):
        return "no"
    else:
        return ""


def normalize_severity_error_only(val) -> str:
    """Convert Focus severity to yes/no (Error only = yes)."""
    if pd.isna(val) or val == "":
        return ""

    val_str = str(val).strip().lower()

    if val_str == "error":
        return "yes"
    elif val_str in ("warning", "ok", "na"):
        return "no"
    else:
        return ""


def normalize_severity_error_warning(val) -> str:
    """Convert Focus severity to yes/no (Error or Warning = yes)."""
    if pd.isna(val) or val == "":
        return ""

    val_str = str(val).strip().lower()

    if val_str in ("error", "warning"):
        return "yes"
    elif val_str in ("ok", "na"):
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


def print_confusion_matrix(metrics: Dict, title: str):
    """Print a visual confusion matrix."""
    cm = metrics["confusion_matrix"]
    tp, tn, fp, fn = cm["true_positive"], cm["true_negative"], cm["false_positive"], cm["false_negative"]

    print(f"\n  CONFUSION MATRIX - {title}")
    print("  " + "-" * 40)
    print("                      Predicted")
    print("                    Yes      No")
    print("              +--------+--------+")
    print(f"  Actual Yes  |  {tp:4}   |  {fn:4}   |  (TP, FN)")
    print("              +--------+--------+")
    print(f"  Actual No   |  {fp:4}   |  {tn:4}   |  (FP, TN)")
    print("              +--------+--------+")


def print_metrics(metrics: Dict, title: str):
    """Print metrics in a formatted way."""
    print(f"\n  CLASSIFICATION METRICS - {title}")
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
                          title: str, mismatch_type: str, mode: str, col_cfg: dict = None):
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
    pdf.cell(0, 10, f"Mode: {mode}", ln=True, align='C')
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
            ("Prediction: Focus Severity", "Focus severity"),
            ("focus_right_mid", "focus_right_mid"),
            ("focus_left_mid", "focus_left_mid"),
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


def dump_mismatch_images(df_tp: pd.DataFrame, df_tn: pd.DataFrame,
                         df_fp: pd.DataFrame, df_fn: pd.DataFrame,
                         data_dir: Path, output_dir: Path, suffix: str, col_cfg: dict = None):
    """Copy focus images for TP, TN, FP and FN cases to separate directories and generate PDFs."""

    if col_cfg is None:
        col_cfg = get_column_config(None)

    # Create directories
    tp_dir = output_dir / f"focus_tp_{suffix}"
    tn_dir = output_dir / f"focus_tn_{suffix}"
    fp_dir = output_dir / f"focus_fp_{suffix}"
    fn_dir = output_dir / f"focus_fn_{suffix}"
    tp_dir.mkdir(exist_ok=True)
    tn_dir.mkdir(exist_ok=True)
    fp_dir.mkdir(exist_ok=True)
    fn_dir.mkdir(exist_ok=True)

    # Find venue ID column (use first non-empty df to find it)
    venue_id_col = col_cfg["venue_id"]
    venue_col = None
    for check_df in [df_tp, df_tn, df_fp, df_fn]:
        if len(check_df) > 0:
            for col in check_df.columns:
                if col.lower() == venue_id_col.lower() or col == venue_id_col:
                    venue_col = col
                    break
            if venue_col:
                break

    if venue_col is None:
        print("  Warning: Could not find venue ID column, skipping image dump")
        return

    print(f"\n  [{suffix.upper()}] TP: {len(df_tp)}, TN: {len(df_tn)}, FP: {len(df_fp)}, FN: {len(df_fn)}")

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
        if len(df_tp) > 0:
            tp_pdf = output_dir / f"focus_true_positives_{suffix}.pdf"
            generate_mismatch_pdf(df_tp, data_dir, tp_pdf,
                                  f"True Positives ({suffix})",
                                  "Predicted quality issue, Actual has issue",
                                  suffix, col_cfg)
            print(f"  Generated: {tp_pdf.name}")

        if len(df_tn) > 0:
            tn_pdf = output_dir / f"focus_true_negatives_{suffix}.pdf"
            generate_mismatch_pdf(df_tn, data_dir, tn_pdf,
                                  f"True Negatives ({suffix})",
                                  "Predicted no issue, Actual no issue",
                                  suffix, col_cfg)
            print(f"  Generated: {tn_pdf.name}")

        if len(df_fp) > 0:
            fp_pdf = output_dir / f"focus_false_positives_{suffix}.pdf"
            generate_mismatch_pdf(df_fp, data_dir, fp_pdf,
                                  f"False Positives ({suffix})",
                                  "Predicted quality issue, Actual no issue",
                                  suffix, col_cfg)
            print(f"  Generated: {fp_pdf.name}")

        if len(df_fn) > 0:
            fn_pdf = output_dir / f"focus_false_negatives_{suffix}.pdf"
            generate_mismatch_pdf(df_fn, data_dir, fn_pdf,
                                  f"False Negatives ({suffix})",
                                  "Predicted no issue, Actual has quality issue",
                                  suffix, col_cfg)
            print(f"  Generated: {fn_pdf.name}")


def analyze_focus_calculations(output_dir: Path, col_cfg: dict = None) -> Tuple[Optional[Dict], Optional[Dict], Optional[pd.DataFrame]]:
    """Analyze focus calculations from concatenated_results.xlsx."""
    xlsx_path = output_dir / "concatenated_results.xlsx"

    if col_cfg is None:
        col_cfg = get_column_config(None)

    if not xlsx_path.exists():
        print(f"Error: File not found: {xlsx_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  ANALYZE FOCUS CALCULATIONS")
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
        elif "focus severity" in col_clean or col.lower() == "focus_severity":
            prediction_col = col
        elif col == MEASURABLE_COL or col == "quality_is_measurable":
            measurable_col = col

    if annotation_col is None:
        print(f"Error: Annotation column 'Has Quality Issue' not found.", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    if prediction_col is None:
        print(f"Error: Prediction column 'Focus severity' not found.", file=sys.stderr)
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

    # Normalize annotation
    df_measurable["_annotation"] = df_measurable[annotation_col].apply(normalize_yes_no)

    # Create two prediction columns
    df_measurable["_pred_error_only"] = df_measurable[prediction_col].apply(normalize_severity_error_only)
    df_measurable["_pred_error_warning"] = df_measurable[prediction_col].apply(normalize_severity_error_warning)

    # Value distributions
    print("\n  VALUE DISTRIBUTION (measurable only):")
    print(f"  Annotations (Has Quality Issue): {df_measurable['_annotation'].value_counts().to_dict()}")
    print(f"  Focus Severity: {df_measurable[prediction_col].value_counts().to_dict()}")

    # Filter to valid rows for each mode
    df_valid_error = df_measurable[(df_measurable["_annotation"] != "") & (df_measurable["_pred_error_only"] != "")]
    df_valid_warning = df_measurable[(df_measurable["_annotation"] != "") & (df_measurable["_pred_error_warning"] != "")]

    print(f"\n  Valid rows (Error only): {len(df_valid_error)}")
    print(f"  Valid rows (Error+Warning): {len(df_valid_warning)}")

    # Calculate metrics for Error only
    metrics_error = None
    if len(df_valid_error) > 0:
        y_true = df_valid_error["_annotation"].tolist()
        y_pred = df_valid_error["_pred_error_only"].tolist()
        metrics_error = calculate_metrics(y_true, y_pred)
        print_confusion_matrix(metrics_error, "ERROR ONLY")
        print_metrics(metrics_error, "ERROR ONLY")
        metrics_error["mode"] = "error_only"
        metrics_error["metadata"] = {
            "annotation_column": annotation_col,
            "prediction_column": prediction_col,
            "valid_rows": len(df_valid_error)
        }

    # Calculate metrics for Error + Warning
    metrics_warning = None
    if len(df_valid_warning) > 0:
        y_true = df_valid_warning["_annotation"].tolist()
        y_pred = df_valid_warning["_pred_error_warning"].tolist()
        metrics_warning = calculate_metrics(y_true, y_pred)
        print_confusion_matrix(metrics_warning, "ERROR + WARNING")
        print_metrics(metrics_warning, "ERROR + WARNING")
        metrics_warning["mode"] = "error_warning"
        metrics_warning["metadata"] = {
            "annotation_column": annotation_col,
            "prediction_column": prediction_col,
            "valid_rows": len(df_valid_warning)
        }

    return metrics_error, metrics_warning, df_measurable


def save_metrics(metrics_error: Dict, metrics_warning: Dict, output_dir: Path):
    """Save metrics to JSON file."""
    output_path = output_dir / "focus_analysis_metrics.json"

    combined = {
        "error_only": metrics_error,
        "error_warning": metrics_warning
    }

    with open(output_path, 'w') as f:
        json.dump(combined, f, indent=2)

    print(f"\n  Metrics saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze focus severity predictions vs quality issue annotations'
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

    metrics_error, metrics_warning, df_measurable = analyze_focus_calculations(output_dir, col_cfg)

    if metrics_error or metrics_warning:
        # Create subdirectory for results
        results_dir = output_dir / "analysis_focus_calculations"
        results_dir.mkdir(exist_ok=True)
        print(f"\n  Results directory: {results_dir}")

        save_metrics(metrics_error, metrics_warning, results_dir)

        # Dump mismatch images and generate PDFs
        if data_dir and data_dir.exists() and df_measurable is not None:
            print("\n  DUMPING CLASSIFICATION IMAGES & PDFs")
            print("  " + "-" * 40)

            # Error only
            df_tp_error = df_measurable[(df_measurable["_annotation"] == "yes") & (df_measurable["_pred_error_only"] == "yes")]
            df_tn_error = df_measurable[(df_measurable["_annotation"] == "no") & (df_measurable["_pred_error_only"] == "no")]
            df_fp_error = df_measurable[(df_measurable["_annotation"] == "no") & (df_measurable["_pred_error_only"] == "yes")]
            df_fn_error = df_measurable[(df_measurable["_annotation"] == "yes") & (df_measurable["_pred_error_only"] == "no")]
            dump_mismatch_images(df_tp_error, df_tn_error, df_fp_error, df_fn_error, data_dir, results_dir, "error_only", col_cfg)

            # Error + Warning
            df_tp_warning = df_measurable[(df_measurable["_annotation"] == "yes") & (df_measurable["_pred_error_warning"] == "yes")]
            df_tn_warning = df_measurable[(df_measurable["_annotation"] == "no") & (df_measurable["_pred_error_warning"] == "no")]
            df_fp_warning = df_measurable[(df_measurable["_annotation"] == "no") & (df_measurable["_pred_error_warning"] == "yes")]
            df_fn_warning = df_measurable[(df_measurable["_annotation"] == "yes") & (df_measurable["_pred_error_warning"] == "no")]
            dump_mismatch_images(df_tp_warning, df_tn_warning, df_fp_warning, df_fn_warning, data_dir, results_dir, "error_warning", col_cfg)

        elif data_dir:
            print(f"\n  Warning: Data directory not found: {data_dir}")
        else:
            print("\n  Warning: No data directory specified")

        print("\n" + "=" * 60)
        print("  COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()
