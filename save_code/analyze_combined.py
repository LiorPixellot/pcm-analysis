#!/usr/bin/env python3
"""
Analyze Combined: Compare Gemini vs Laplacian predictions for quality issues.

Combines both predictors, creates ensemble predictions, performs threshold sweeps,
and generates visualizations comparing all approaches.

Usage:
    python analyze_combined.py output_2026-02-10_18-18/
    python analyze_combined.py /path/to/output_dir
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# Default column names
DEFAULT_ANNOTATION_COL = "Has Quality Issue (Yes/No)"
GEMINI_PREDICTION_COL = "quality_has_issue"
FOCUS_SEVERITY_COL = "Focus severity"
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


def load_run_config(output_dir: Path) -> Optional[Dict]:
    """Load run configuration from run_config.json."""
    config_path = output_dir / "run_config.json"
    if not config_path.exists():
        return None
    with open(config_path, 'r') as f:
        return json.load(f)


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
    tp = tn = fp = fn = 0
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

    return {
        "total_samples": total,
        "confusion_matrix": {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn},
        "accuracy": accuracy,
        "positive_class_yes": {"precision": precision, "recall": recall, "f1_score": f1, "support": tp + fn},
        "negative_class_no": {
            "precision": tn / (tn + fn) if (tn + fn) > 0 else 0,
            "recall": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "f1_score": 2 * (tn / (tn + fn)) * (tn / (tn + fp)) / ((tn / (tn + fn)) + (tn / (tn + fp))) if (tn + fn) > 0 and (tn + fp) > 0 and ((tn / (tn + fn)) + (tn / (tn + fp))) > 0 else 0,
            "support": tn + fp,
        },
    }


def find_columns(df: pd.DataFrame, col_cfg: dict) -> Tuple[str, str, str, str]:
    """Find annotation, gemini prediction, focus severity, and measurable columns."""
    annotation_col = gemini_col = severity_col = measurable_col = None
    target_annotation = col_cfg["annotation"].lower().replace("\n", " ")

    for col in df.columns:
        col_clean = str(col).lower().replace("\n", " ")
        if col_clean == target_annotation or "has quality issue" in col_clean:
            annotation_col = col
        elif col == GEMINI_PREDICTION_COL or col == "quality_has_issue":
            gemini_col = col
        elif col == FOCUS_SEVERITY_COL or ("focus severity" in col_clean and col != "focus_severity"):
            severity_col = col
        elif col == MEASURABLE_COL or col == "quality_is_measurable":
            measurable_col = col

    # Fallback: if Focus severity (uppercase) not found, use focus_severity
    if severity_col is None:
        for col in df.columns:
            if col.lower() == "focus_severity":
                severity_col = col
                break

    missing = []
    if annotation_col is None:
        missing.append("Has Quality Issue annotation")
    if gemini_col is None:
        missing.append("quality_has_issue (Gemini)")
    if severity_col is None:
        missing.append("Focus severity (Laplacian)")
    if measurable_col is None:
        missing.append("quality_is_measurable")

    if missing:
        print(f"Error: Missing columns: {', '.join(missing)}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    return annotation_col, gemini_col, severity_col, measurable_col


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_confusion_matrices(all_metrics: Dict[str, Dict], output_path: Path):
    """Plot 2x2 grid of confusion matrices."""
    names = list(all_metrics.keys())
    n = len(names)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    if rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__len__') else [row])]

    for idx, name in enumerate(names):
        ax = axes_flat[idx]
        cm = all_metrics[name]["confusion_matrix"]
        matrix = np.array([[cm["true_positive"], cm["false_negative"]],
                           [cm["false_positive"], cm["true_negative"]]])
        im = ax.imshow(matrix, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Yes", "No"])
        ax.set_yticklabels(["Yes", "No"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontsize=11, fontweight="bold")
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                color = "white" if val > matrix.max() / 2 else "black"
                label_map = {(0, 0): "TP", (0, 1): "FN", (1, 0): "FP", (1, 1): "TN"}
                ax.text(j, i, f"{label_map[(i,j)]}\n{val}", ha="center", va="center",
                        color=color, fontsize=12, fontweight="bold")

    # Hide unused axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Confusion Matrices Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_comparison(all_metrics: Dict[str, Dict], output_path: Path):
    """Bar chart comparing precision, recall, F1 across predictors."""
    names = list(all_metrics.keys())
    precision_vals = [all_metrics[n]["positive_class_yes"]["precision"] for n in names]
    recall_vals = [all_metrics[n]["positive_class_yes"]["recall"] for n in names]
    f1_vals = [all_metrics[n]["positive_class_yes"]["f1_score"] for n in names]
    accuracy_vals = [all_metrics[n]["accuracy"] for n in names]

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - 1.5 * width, precision_vals, width, label="Precision", color="#2196F3")
    bars2 = ax.bar(x - 0.5 * width, recall_vals, width, label="Recall", color="#FF9800")
    bars3 = ax.bar(x + 0.5 * width, f1_vals, width, label="F1", color="#4CAF50")
    bars4 = ax.bar(x + 1.5 * width, accuracy_vals, width, label="Accuracy", color="#9C27B0")

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Metrics Comparison: Gemini vs Laplacian vs Ensembles", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_focus_distributions(df: pd.DataFrame, output_path: Path):
    """Box plots of focus_right_mid and focus_left_mid split by annotation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, col in enumerate(["focus_right_mid", "focus_left_mid"]):
        ax = axes[idx]
        if col not in df.columns:
            ax.set_title(f"{col} (not available)")
            continue

        yes_vals = df[df["_annotation"] == "yes"][col].dropna()
        no_vals = df[df["_annotation"] == "no"][col].dropna()

        bp = ax.boxplot([yes_vals, no_vals], tick_labels=["Has Issue (Yes)", "No Issue (No)"],
                        patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#EF5350")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("#66BB6A")
        bp["boxes"][1].set_alpha(0.6)

        ax.set_title(col, fontweight="bold")
        ax.set_ylabel("Laplacian Value")
        ax.grid(axis="y", alpha=0.3)

        # Add median labels
        for i, vals in enumerate([yes_vals, no_vals]):
            if len(vals) > 0:
                med = vals.median()
                ax.annotate(f"med={med:.1f}", xy=(i + 1, med), xytext=(10, 5),
                            textcoords="offset points", fontsize=9, color="blue")

    fig.suptitle("Focus Metric Distributions by Annotation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_focus_scatter(df: pd.DataFrame, gemini_col: str, output_path: Path):
    """Scatter plot of focus_right_mid vs focus_left_mid, colored by annotation."""
    if "focus_right_mid" not in df.columns or "focus_left_mid" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    for annotation, color, label in [("yes", "#EF5350", "Has Issue"), ("no", "#66BB6A", "No Issue")]:
        subset = df[df["_annotation"] == annotation]

        # Correct Gemini prediction
        correct = subset[subset["_gemini"] == subset["_annotation"]]
        wrong = subset[subset["_gemini"] != subset["_annotation"]]

        if len(correct) > 0:
            ax.scatter(correct["focus_right_mid"], correct["focus_left_mid"],
                       c=color, marker="o", alpha=0.7, s=60, edgecolors="black", linewidths=0.5,
                       label=f"{label} (Gemini correct)")
        if len(wrong) > 0:
            ax.scatter(wrong["focus_right_mid"], wrong["focus_left_mid"],
                       c=color, marker="X", alpha=0.9, s=100, edgecolors="black", linewidths=0.5,
                       label=f"{label} (Gemini wrong)")

    # Add threshold lines
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.4, label="Error threshold (10)")
    ax.axvline(x=10, color="red", linestyle="--", alpha=0.4)
    ax.axhline(y=20, color="orange", linestyle="--", alpha=0.4, label="Warning threshold (20)")
    ax.axvline(x=20, color="orange", linestyle="--", alpha=0.4)

    ax.set_xlabel("focus_right_mid")
    ax.set_ylabel("focus_left_mid")
    ax.set_title("Focus Right vs Left (colored by annotation, shaped by Gemini accuracy)", fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sweep(sweep_results: List[Dict], current_threshold: float,
                         optimal_f1_threshold: float, optimal_precision_threshold: Optional[float],
                         output_path: Path):
    """Precision-recall vs threshold curve."""
    thresholds = [r["threshold"] for r in sweep_results]
    precisions = [r["precision"] for r in sweep_results]
    recalls = [r["recall"] for r in sweep_results]
    f1s = [r["f1"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(thresholds, precisions, "b-", linewidth=2, label="Precision")
    ax.plot(thresholds, recalls, "r-", linewidth=2, label="Recall")
    ax.plot(thresholds, f1s, "g--", linewidth=1.5, label="F1")

    ax.axvline(x=current_threshold, color="gray", linestyle=":", linewidth=2, label=f"Current threshold ({current_threshold})")
    ax.axvline(x=optimal_f1_threshold, color="green", linestyle="-.", linewidth=2, label=f"Best F1 threshold ({optimal_f1_threshold})")
    if optimal_precision_threshold is not None:
        ax.axvline(x=optimal_precision_threshold, color="blue", linestyle="-.", linewidth=2,
                    label=f"Best precision@80%recall ({optimal_precision_threshold})")

    ax.set_xlabel("focus_mid threshold (min of right, left)")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep: Precision & Recall vs focus_mid Cutoff", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_segment_analysis(df: pd.DataFrame, col_cfg: dict, all_metrics_per_segment: Dict, output_path: Path):
    """Bar chart showing accuracy per segment for each predictor."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    segment_cols = [
        ("indoor_outdoor", "Indoor / Outdoor"),
        ("calibration", "Calibration"),
        ("far_close", "Far / Close"),
    ]

    for idx, (cfg_key, title) in enumerate(segment_cols):
        ax = axes[idx]
        seg_data = all_metrics_per_segment.get(cfg_key, {})
        if not seg_data:
            ax.set_title(f"{title} (no data)")
            continue

        segment_names = sorted(seg_data.keys())
        predictors = ["Gemini", "Laplacian(Error)", "AND Ensemble", "OR Ensemble"]
        x = np.arange(len(segment_names))
        width = 0.2

        for p_idx, predictor in enumerate(predictors):
            vals = []
            for seg in segment_names:
                m = seg_data[seg].get(predictor)
                vals.append(m["accuracy"] if m else 0)
            ax.bar(x + (p_idx - 1.5) * width, vals, width, label=predictor)

        ax.set_ylabel("Accuracy")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(segment_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Accuracy by Segment", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def threshold_sweep(df: pd.DataFrame, thresholds: range) -> List[Dict]:
    """Sweep focus_mid threshold and calculate metrics at each point."""
    results = []
    y_true = df["_annotation"].tolist()

    for th in thresholds:
        y_pred = []
        for _, row in df.iterrows():
            r = row.get("focus_right_mid", np.nan)
            l = row.get("focus_left_mid", np.nan)
            if pd.isna(r) or pd.isna(l):
                y_pred.append("no")
                continue
            min_focus = min(r, l)
            y_pred.append("yes" if min_focus <= th else "no")

        m = calculate_metrics(y_true, y_pred)
        pos = m["positive_class_yes"]
        results.append({
            "threshold": th,
            "precision": pos["precision"],
            "recall": pos["recall"],
            "f1": pos["f1_score"],
            "accuracy": m["accuracy"],
            "tp": m["confusion_matrix"]["true_positive"],
            "fp": m["confusion_matrix"]["false_positive"],
            "fn": m["confusion_matrix"]["false_negative"],
            "tn": m["confusion_matrix"]["true_negative"],
        })

    return results


def compute_segment_metrics(df: pd.DataFrame, col_cfg: dict) -> Dict:
    """Compute accuracy by segment (indoor/outdoor, calibration, far/close)."""
    segment_cols_map = {
        "indoor_outdoor": col_cfg["indoor_outdoor"],
        "calibration": col_cfg["calibration"],
        "far_close": col_cfg["far_close"],
    }

    all_seg = {}
    predictors = {
        "Gemini": "_gemini",
        "Laplacian(Error)": "_laplacian_error",
        "AND Ensemble": "_and_ensemble",
        "OR Ensemble": "_or_ensemble",
    }

    for seg_key, seg_col in segment_cols_map.items():
        if seg_col not in df.columns:
            continue

        seg_vals = df[seg_col].dropna().unique()
        seg_vals = [v for v in seg_vals if str(v).strip() and str(v).strip() != "-"]
        if len(seg_vals) < 2:
            continue

        seg_data = {}
        for val in sorted(seg_vals, key=str):
            subset = df[df[seg_col] == val]
            if len(subset) < 3:
                continue
            seg_data[str(val)] = {}
            y_true = subset["_annotation"].tolist()
            for pred_name, pred_col in predictors.items():
                if pred_col in subset.columns:
                    y_pred = subset[pred_col].tolist()
                    seg_data[str(val)][pred_name] = calculate_metrics(y_true, y_pred)

        if seg_data:
            all_seg[seg_key] = seg_data

    return all_seg


def main():
    parser = argparse.ArgumentParser(
        description="Combined analysis: Gemini vs Laplacian vs Ensemble predictions"
    )
    parser.add_argument("output_dir", help="Directory containing concatenated_results.xlsx")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"Error: {args.output_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    xlsx_path = output_dir / "concatenated_results.xlsx"
    if not xlsx_path.exists():
        print(f"Error: File not found: {xlsx_path}", file=sys.stderr)
        sys.exit(1)

    # Load config
    run_config = load_run_config(output_dir)
    col_cfg = get_column_config(run_config)

    print("=" * 70)
    print("  COMBINED ANALYSIS: Gemini vs Laplacian vs Ensemble")
    print("=" * 70)
    print(f"  Input: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    print(f"  Total rows: {len(df)}")

    # Find columns
    annotation_col, gemini_col, severity_col, measurable_col = find_columns(df, col_cfg)
    print(f"  Annotation column:     {annotation_col}")
    print(f"  Gemini prediction:     {gemini_col}")
    print(f"  Focus severity column: {severity_col}")
    print(f"  Measurable column:     {measurable_col}")

    # Filter to measurable
    df["_measurable"] = df[measurable_col].apply(normalize_yes_no)
    df_m = df[df["_measurable"] == "yes"].copy()
    print(f"\n  Measurable rows: {len(df_m)} (excluded {len(df) - len(df_m)})")

    if len(df_m) == 0:
        print("Error: No measurable rows found", file=sys.stderr)
        sys.exit(1)

    # Normalize columns
    df_m["_annotation"] = df_m[annotation_col].apply(normalize_yes_no)
    df_m["_gemini"] = df_m[gemini_col].apply(normalize_yes_no)
    df_m["_laplacian_error"] = df_m[severity_col].apply(normalize_severity_error_only)
    df_m["_laplacian_warn"] = df_m[severity_col].apply(normalize_severity_error_warning)

    # Filter to valid rows (all four columns non-empty)
    df_v = df_m[
        (df_m["_annotation"] != "") &
        (df_m["_gemini"] != "") &
        (df_m["_laplacian_error"] != "") &
        (df_m["_laplacian_warn"] != "")
    ].copy()
    print(f"  Valid rows (all columns present): {len(df_v)}")

    if len(df_v) == 0:
        print("Error: No valid rows with all required columns", file=sys.stderr)
        sys.exit(1)

    # Create ensemble predictions
    df_v["_and_ensemble"] = df_v.apply(
        lambda r: "yes" if r["_gemini"] == "yes" and r["_laplacian_error"] == "yes" else "no", axis=1
    )
    df_v["_or_ensemble"] = df_v.apply(
        lambda r: "yes" if r["_gemini"] == "yes" or r["_laplacian_error"] == "yes" else "no", axis=1
    )

    # -----------------------------------------------------------------------
    # 1. Three-way comparison table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  THREE-WAY COMPARISON TABLE")
    print("=" * 70)

    cross = df_v.groupby(["_annotation", "_gemini", "_laplacian_error", "_laplacian_warn"]).size().reset_index(name="count")
    cross = cross.sort_values("count", ascending=False)

    header = f"  {'Annotation':<12} {'Gemini':<8} {'Lap(Err)':<10} {'Lap(Err+Warn)':<15} {'Count':<6}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for _, row in cross.iterrows():
        note = ""
        if row["_annotation"] == "yes" and row["_gemini"] == "no" and row["_laplacian_error"] == "no":
            note = " <-- both miss"
        elif row["_annotation"] == "no" and row["_gemini"] == "yes" and row["_laplacian_error"] == "yes":
            note = " <-- both false alarm"
        print(f"  {row['_annotation']:<12} {row['_gemini']:<8} {row['_laplacian_error']:<10} {row['_laplacian_warn']:<15} {row['count']:<6}{note}")

    # -----------------------------------------------------------------------
    # 2. Metrics for all predictors
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  METRICS COMPARISON")
    print("=" * 70)

    y_true = df_v["_annotation"].tolist()
    all_metrics = {
        "Gemini": calculate_metrics(y_true, df_v["_gemini"].tolist()),
        "Laplacian(Error)": calculate_metrics(y_true, df_v["_laplacian_error"].tolist()),
        "AND Ensemble": calculate_metrics(y_true, df_v["_and_ensemble"].tolist()),
        "OR Ensemble": calculate_metrics(y_true, df_v["_or_ensemble"].tolist()),
    }

    # Print side-by-side table
    print(f"\n  {'Predictor':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    print("  " + "-" * 90)
    for name, m in all_metrics.items():
        pos = m["positive_class_yes"]
        cm = m["confusion_matrix"]
        print(f"  {name:<20} {pos['precision']:>10.4f} {pos['recall']:>10.4f} {pos['f1_score']:>10.4f} {m['accuracy']:>10.4f} {cm['true_positive']:>5} {cm['false_positive']:>5} {cm['false_negative']:>5} {cm['true_negative']:>5}")

    # -----------------------------------------------------------------------
    # 3. Threshold sweep
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  THRESHOLD SWEEP (focus_mid)")
    print("=" * 70)

    has_focus_cols = "focus_right_mid" in df_v.columns and "focus_left_mid" in df_v.columns
    sweep_results = []
    optimal_f1_th = 10
    optimal_precision_th = None

    if has_focus_cols:
        sweep_results = threshold_sweep(df_v, range(5, 61))

        # Best F1
        best_f1 = max(sweep_results, key=lambda r: r["f1"])
        optimal_f1_th = best_f1["threshold"]

        # Best precision at >= 80% recall
        candidates = [r for r in sweep_results if r["recall"] >= 0.8]
        if candidates:
            best_prec = max(candidates, key=lambda r: r["precision"])
            optimal_precision_th = best_prec["threshold"]

        # Current threshold (10)
        current = next((r for r in sweep_results if r["threshold"] == 10), None)

        print(f"  Current threshold (10):  precision={current['precision']:.4f}  recall={current['recall']:.4f}  F1={current['f1']:.4f}" if current else "  Current threshold (10): N/A")
        print(f"  Best F1 threshold ({optimal_f1_th}):  precision={best_f1['precision']:.4f}  recall={best_f1['recall']:.4f}  F1={best_f1['f1']:.4f}")
        if optimal_precision_th is not None:
            bp = next(r for r in sweep_results if r["threshold"] == optimal_precision_th)
            print(f"  Best precision@80%recall ({optimal_precision_th}):  precision={bp['precision']:.4f}  recall={bp['recall']:.4f}  F1={bp['f1']:.4f}")
        else:
            print("  Best precision@80%recall: No threshold achieves >=80% recall")
    else:
        print("  Skipped: focus_right_mid / focus_left_mid columns not found")

    # -----------------------------------------------------------------------
    # 4. Disagreement analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DISAGREEMENT ANALYSIS")
    print("=" * 70)

    disagree = df_v[df_v["_gemini"] != df_v["_laplacian_error"]].copy()
    agree = df_v[df_v["_gemini"] == df_v["_laplacian_error"]]

    print(f"  Total rows: {len(df_v)}")
    print(f"  Agree: {len(agree)} ({100 * len(agree) / len(df_v):.1f}%)")
    print(f"  Disagree: {len(disagree)} ({100 * len(disagree) / len(df_v):.1f}%)")

    if len(disagree) > 0:
        gemini_right = disagree[disagree["_gemini"] == disagree["_annotation"]]
        laplacian_right = disagree[disagree["_laplacian_error"] == disagree["_annotation"]]
        neither_right = disagree[(disagree["_gemini"] != disagree["_annotation"]) & (disagree["_laplacian_error"] != disagree["_annotation"])]

        print(f"\n  When they disagree:")
        print(f"    Gemini correct:    {len(gemini_right)} ({100 * len(gemini_right) / len(disagree):.1f}%)")
        print(f"    Laplacian correct: {len(laplacian_right)} ({100 * len(laplacian_right) / len(disagree):.1f}%)")
        print(f"    Neither correct:   {len(neither_right)} ({100 * len(neither_right) / len(disagree):.1f}%)")

    # -----------------------------------------------------------------------
    # 5. Segment analysis
    # -----------------------------------------------------------------------
    all_seg_metrics = compute_segment_metrics(df_v, col_cfg)

    if all_seg_metrics:
        print("\n" + "=" * 70)
        print("  SEGMENT ANALYSIS")
        print("=" * 70)

        for seg_key, seg_data in all_seg_metrics.items():
            print(f"\n  --- {seg_key} ---")
            for seg_val, pred_metrics in seg_data.items():
                print(f"  {seg_val}:")
                for pred_name, m in pred_metrics.items():
                    print(f"    {pred_name:<20} acc={m['accuracy']:.3f}  prec={m['positive_class_yes']['precision']:.3f}  rec={m['positive_class_yes']['recall']:.3f}  (n={m['total_samples']})")

    # -----------------------------------------------------------------------
    # 6. Key finding summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  KEY FINDINGS")
    print("=" * 70)

    best_predictor = max(all_metrics.items(), key=lambda x: x[1]["positive_class_yes"]["f1_score"])
    best_precision = max(all_metrics.items(), key=lambda x: x[1]["positive_class_yes"]["precision"])
    best_recall = max(all_metrics.items(), key=lambda x: x[1]["positive_class_yes"]["recall"])

    print(f"  Best F1:        {best_predictor[0]} (F1={best_predictor[1]['positive_class_yes']['f1_score']:.4f})")
    print(f"  Best Precision: {best_precision[0]} (precision={best_precision[1]['positive_class_yes']['precision']:.4f})")
    print(f"  Best Recall:    {best_recall[0]} (recall={best_recall[1]['positive_class_yes']['recall']:.4f})")

    if len(disagree) > 0 and len(gemini_right) > len(laplacian_right):
        print(f"  When predictors disagree, Gemini is correct more often ({len(gemini_right)} vs {len(laplacian_right)} cases)")
    elif len(disagree) > 0 and len(laplacian_right) > len(gemini_right):
        print(f"  When predictors disagree, Laplacian is correct more often ({len(laplacian_right)} vs {len(gemini_right)} cases)")

    # -----------------------------------------------------------------------
    # 7. Save outputs
    # -----------------------------------------------------------------------
    results_dir = output_dir / "analysis_combined"
    results_dir.mkdir(exist_ok=True)
    print(f"\n  Output directory: {results_dir}")

    # 7a. combined_metrics.json
    metrics_out = {}
    for name, m in all_metrics.items():
        metrics_out[name] = m
    metrics_out["threshold_sweep"] = {
        "current_threshold": 10,
        "optimal_f1_threshold": optimal_f1_th,
        "optimal_precision_at_80_recall_threshold": optimal_precision_th,
        "sweep_data": sweep_results,
    }
    metrics_out["disagreement"] = {
        "total": len(df_v),
        "agree": len(agree),
        "disagree": len(disagree),
        "gemini_right_when_disagree": len(gemini_right) if len(disagree) > 0 else 0,
        "laplacian_right_when_disagree": len(laplacian_right) if len(disagree) > 0 else 0,
    }
    if all_seg_metrics:
        seg_out = {}
        for seg_key, seg_data in all_seg_metrics.items():
            seg_out[seg_key] = {}
            for seg_val, pred_metrics in seg_data.items():
                seg_out[seg_key][seg_val] = {
                    pred_name: {"accuracy": m["accuracy"],
                                "precision": m["positive_class_yes"]["precision"],
                                "recall": m["positive_class_yes"]["recall"],
                                "f1": m["positive_class_yes"]["f1_score"],
                                "n": m["total_samples"]}
                    for pred_name, m in pred_metrics.items()
                }
        metrics_out["segments"] = seg_out

    with open(results_dir / "combined_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Saved: combined_metrics.json")

    # 7b. Visualizations
    plot_confusion_matrices(all_metrics, results_dir / "confusion_matrices.png")
    print(f"  Saved: confusion_matrices.png")

    plot_metrics_comparison(all_metrics, results_dir / "metrics_comparison.png")
    print(f"  Saved: metrics_comparison.png")

    plot_focus_distributions(df_v, results_dir / "focus_distributions.png")
    print(f"  Saved: focus_distributions.png")

    plot_focus_scatter(df_v, gemini_col, results_dir / "focus_scatter.png")
    print(f"  Saved: focus_scatter.png")

    if sweep_results:
        plot_threshold_sweep(sweep_results, 10, optimal_f1_th, optimal_precision_th,
                             results_dir / "threshold_sweep.png")
        print(f"  Saved: threshold_sweep.png")

    if all_seg_metrics:
        plot_segment_analysis(df_v, col_cfg, all_seg_metrics, results_dir / "segment_analysis.png")
        print(f"  Saved: segment_analysis.png")

    # 7c. Disagreements CSV
    if len(disagree) > 0:
        export_cols = [col_cfg["venue_id"], annotation_col, gemini_col, severity_col]
        if "focus_right_mid" in disagree.columns:
            export_cols.append("focus_right_mid")
        if "focus_left_mid" in disagree.columns:
            export_cols.append("focus_left_mid")
        for extra in [col_cfg["calibration"], col_cfg["indoor_outdoor"], col_cfg["far_close"]]:
            if extra in disagree.columns:
                export_cols.append(extra)
        # Add internal columns for clarity
        disagree_export = disagree[[c for c in export_cols if c in disagree.columns]].copy()
        disagree_export["gemini_correct"] = (disagree["_gemini"] == disagree["_annotation"]).map({True: "yes", False: "no"})
        disagree_export["laplacian_correct"] = (disagree["_laplacian_error"] == disagree["_annotation"]).map({True: "yes", False: "no"})
        disagree_export.to_csv(results_dir / "disagreements.csv", index=False)
        print(f"  Saved: disagreements.csv ({len(disagree_export)} rows)")

    # 7d. Summary text
    with open(results_dir / "summary.txt", "w") as f:
        f.write("COMBINED ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input: {xlsx_path}\n")
        f.write(f"Total rows: {len(df)}, Measurable: {len(df_m)}, Valid: {len(df_v)}\n\n")

        f.write("METRICS (positive class = 'has quality issue')\n")
        f.write(f"{'Predictor':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}\n")
        f.write("-" * 60 + "\n")
        for name, m in all_metrics.items():
            pos = m["positive_class_yes"]
            f.write(f"{name:<20} {pos['precision']:>10.4f} {pos['recall']:>10.4f} {pos['f1_score']:>10.4f} {m['accuracy']:>10.4f}\n")

        f.write(f"\nBest F1:        {best_predictor[0]} ({best_predictor[1]['positive_class_yes']['f1_score']:.4f})\n")
        f.write(f"Best Precision: {best_precision[0]} ({best_precision[1]['positive_class_yes']['precision']:.4f})\n")
        f.write(f"Best Recall:    {best_recall[0]} ({best_recall[1]['positive_class_yes']['recall']:.4f})\n")

        if has_focus_cols:
            f.write(f"\nTHRESHOLD SWEEP\n")
            f.write(f"Current threshold (10): F1={next((r for r in sweep_results if r['threshold'] == 10), {}).get('f1', 0):.4f}\n")
            f.write(f"Optimal F1 threshold: {optimal_f1_th}\n")
            if optimal_precision_th:
                f.write(f"Optimal precision@80%recall threshold: {optimal_precision_th}\n")

        f.write(f"\nDISAGREEMENT\n")
        f.write(f"Agree: {len(agree)}, Disagree: {len(disagree)}\n")
        if len(disagree) > 0:
            f.write(f"Gemini correct when disagree: {len(gemini_right)}\n")
            f.write(f"Laplacian correct when disagree: {len(laplacian_right)}\n")

    print(f"  Saved: summary.txt")

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
