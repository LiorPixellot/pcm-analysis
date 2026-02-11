#!/usr/bin/env python3
"""
Analyze Is Measurable: Calculate precision, recall, and confusion matrix for is_measurable predictions.

Compares human annotations vs Gemini predictions from concatenated_results.xlsx.

Usage:
    python analyze_is_measurable.py output_2026-02-08_15-30/
    python analyze_is_measurable.py /path/to/output_dir
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
DEFAULT_ANNOTATION_COL = "Is Measurable camera problems (Yes/No)"
DEFAULT_NOT_MEASURABLE_REASON_COL = "If Not Measurable Why? (fog\\lights of\\to dart\\to bright\\sun on camera )"
PREDICTION_COL = "quality_is_measurable"
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

    cols["annotation"] = ann_cols.get("is_measurable_annotation", DEFAULT_ANNOTATION_COL)
    cols["not_measurable_reason"] = ann_cols.get("not_measurable_reason_annotation", DEFAULT_NOT_MEASURABLE_REASON_COL)
    cols["venue_id"] = run_config.get("annotations", {}).get("join_key", DEFAULT_VENUE_ID_COL) if run_config else DEFAULT_VENUE_ID_COL
    cols["calibration"] = ann_cols.get("calibration", DEFAULT_CALIBRATION_COL)
    cols["indoor_outdoor"] = ann_cols.get("indoor_outdoor", DEFAULT_INDOOR_OUTDOOR_COL)
    cols["far_close"] = ann_cols.get("far_close", DEFAULT_FAR_CLOSE_COL)
    return cols


def get_display_columns(col_cfg):
    """Build DISPLAY_COLUMNS list from column config."""
    return [
        col_cfg["venue_id"],
        col_cfg["calibration"],
        col_cfg["indoor_outdoor"],
        col_cfg["far_close"],
        col_cfg["annotation"],
        col_cfg["not_measurable_reason"],
        "quality_is_measurable",
        "quality_not_measurable_reason",
    ]


def normalize_value(val) -> str:
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


def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """Calculate precision, recall, F1, and confusion matrix.

    Treats 'yes' (measurable) as positive class.
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

    print("\n  Class 'Yes' (Measurable):")
    pos = metrics["positive_class_yes"]
    print(f"    Precision: {pos['precision']:.4f}")
    print(f"    Recall:    {pos['recall']:.4f}")
    print(f"    F1-Score:  {pos['f1_score']:.4f}")
    print(f"    Support:   {pos['support']}")

    print("\n  Class 'No' (Not Measurable):")
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
    return clean.strip()[:50]  # Truncate long names


def sanitize_for_pdf(text: str) -> str:
    """Sanitize text for PDF output by removing non-Latin characters."""
    # Encode as latin-1, replacing unencodable chars with '?'
    return text.encode('latin-1', errors='replace').decode('latin-1')


def get_display_value(row: pd.Series, col_name: str) -> str:
    """Get display value for a column, handling missing columns."""
    # Try exact match first
    if col_name in row.index:
        val = row[col_name]
        if pd.isna(val):
            return "-"
        return sanitize_for_pdf(str(val)[:60])  # Truncate long values

    # Try cleaned column name match
    clean_target = clean_column_name(col_name).lower()
    for col in row.index:
        if clean_column_name(col).lower() == clean_target:
            val = row[col]
            if pd.isna(val):
                return "-"
            return sanitize_for_pdf(str(val)[:60])

    return "-"


def resize_image_for_pdf(img_path: Path, max_width: int = 250, max_height: int = 200) -> Optional[Path]:
    """Resize image for PDF and return path to temp file."""
    try:
        img = Image.open(img_path)
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        # Save to temp file
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

    # Use defaults if no col_cfg provided
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

    temp_files = []  # Track temp files for cleanup

    # Generate page for each case
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        pdf.add_page()

        venue_id = sanitize_for_pdf(str(row[venue_col]).strip())

        # Header
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, f"Case {idx}/{len(df)}: {venue_id}", ln=True)
        pdf.ln(5)

        # Data table
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(70, 8, "Field", border=1)
        pdf.cell(0, 8, "Value", border=1, ln=True)

        pdf.set_font('Helvetica', '', 9)

        # Display key columns
        display_cols = [
            ("System ID", venue_col),
            ("Calibration", col_cfg["calibration"]),
            ("Indoor/Outdoor", col_cfg["indoor_outdoor"]),
            ("Far/Close", col_cfg["far_close"]),
            ("Annotation (Human)", col_cfg["annotation"]),
            ("Annotation Reason", col_cfg["not_measurable_reason"]),
            ("Prediction (Gemini)", "quality_is_measurable"),
            ("Prediction Reason", "quality_not_measurable_reason"),
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

                for img_path in images[:2]:  # Max 2 images per row
                    resized = resize_image_for_pdf(img_path)
                    if resized:
                        temp_files.append(resized)
                        try:
                            pdf.image(str(resized), x=x_start + x_offset, y=y_start, w=90)
                            x_offset += 95
                        except Exception as e:
                            print(f"    Warning: Could not add image to PDF: {e}")

    # Save PDF
    pdf.output(output_path)

    # Cleanup temp files
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
                    print(f"  Warning: Could not copy {img}: {e}")

        print(f"  Copied {copied} images to {cat_dir}")

    # Generate PDFs
    if HAS_FPDF:
        print("\n  Generating PDF reports...")

        if len(df_tp) > 0:
            tp_pdf = output_dir / "true_positives.pdf"
            generate_mismatch_pdf(df_tp, data_dir, tp_pdf,
                                  "True Positives Report",
                                  "Predicted YES (measurable), Actual YES (measurable)",
                                  col_cfg)
            print(f"  Generated: {tp_pdf}")

        if len(df_tn) > 0:
            tn_pdf = output_dir / "true_negatives.pdf"
            generate_mismatch_pdf(df_tn, data_dir, tn_pdf,
                                  "True Negatives Report",
                                  "Predicted NO (not measurable), Actual NO (not measurable)",
                                  col_cfg)
            print(f"  Generated: {tn_pdf}")

        if len(df_fp) > 0:
            fp_pdf = output_dir / "false_positives.pdf"
            generate_mismatch_pdf(df_fp, data_dir, fp_pdf,
                                  "False Positives Report",
                                  "Predicted YES (measurable), Actual NO (not measurable)",
                                  col_cfg)
            print(f"  Generated: {fp_pdf}")

        if len(df_fn) > 0:
            fn_pdf = output_dir / "false_negatives.pdf"
            generate_mismatch_pdf(df_fn, data_dir, fn_pdf,
                                  "False Negatives Report",
                                  "Predicted NO (not measurable), Actual YES (measurable)",
                                  col_cfg)
            print(f"  Generated: {fn_pdf}")
    else:
        print("  Warning: fpdf2 not installed, skipping PDF generation")
        print("  Install with: pip install fpdf2")


def analyze_is_measurable(output_dir: Path, col_cfg: dict = None) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """Analyze results from concatenated_results.xlsx."""
    xlsx_path = output_dir / "concatenated_results.xlsx"

    if col_cfg is None:
        col_cfg = get_column_config(None)

    if not xlsx_path.exists():
        print(f"Error: File not found: {xlsx_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  ANALYZE RESULTS - IS MEASURABLE METRICS")
    print("=" * 60)
    print(f"  Input: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    print(f"  Total rows: {len(df)}")

    # Find columns using configured names
    annotation_col = None
    prediction_col = None
    target_annotation = col_cfg["annotation"].lower().replace("\n", " ")

    for col in df.columns:
        col_clean = str(col).lower().replace("\n", " ")
        if col_clean == target_annotation or ("is measurable" in col_clean and "camera problems" in col_clean):
            annotation_col = col
        elif col == PREDICTION_COL or col == "quality_is_measurable":
            prediction_col = col

    if annotation_col is None:
        print(f"Error: Annotation column not found.", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    if prediction_col is None:
        print(f"Error: Prediction column not found.", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    print(f"  Annotation column: {annotation_col}")
    print(f"  Prediction column: {prediction_col}")

    # Normalize values
    df["_annotation"] = df[annotation_col].apply(normalize_value)
    df["_prediction"] = df[prediction_col].apply(normalize_value)

    # Count value distributions
    print("\n  VALUE DISTRIBUTION:")
    print(f"  Annotations: {df['_annotation'].value_counts().to_dict()}")
    print(f"  Predictions: {df['_prediction'].value_counts().to_dict()}")

    # Filter to rows with both annotation and prediction
    df_valid = df[(df["_annotation"] != "") & (df["_prediction"] != "")]
    print(f"\n  Valid rows (both annotation and prediction): {len(df_valid)}")

    if len(df_valid) == 0:
        print("Error: No valid rows found", file=sys.stderr)
        return None, None

    # Calculate metrics
    y_true = df_valid["_annotation"].tolist()
    y_pred = df_valid["_prediction"].tolist()

    metrics = calculate_metrics(y_true, y_pred)

    print_confusion_matrix(metrics)
    print_metrics(metrics)

    metrics["metadata"] = {
        "input_file": str(xlsx_path),
        "total_rows": len(df),
        "valid_rows": len(df_valid),
        "annotation_column": annotation_col,
        "prediction_column": prediction_col
    }

    return metrics, df_valid


def save_metrics(metrics: Dict, output_dir: Path):
    """Save metrics to JSON file."""
    output_path = output_dir / "analysis_metrics.json"

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Metrics saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze is_measurable prediction results'
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

    metrics, df_valid = analyze_is_measurable(output_dir, col_cfg)

    if metrics:
        # Create subdirectory for results
        results_dir = output_dir / "analysis_is_measurable"
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

        print("\n" + "=" * 60)
        print("  COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()
