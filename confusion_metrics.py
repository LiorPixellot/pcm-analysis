#!/usr/bin/env python3
"""
Confusion Metrics for Annotation Pipeline

Reads the predictions CSV from run_annotation_pipeline.py, computes precision/recall/F1,
generates image dumps (symlinked) and confusion matrix plots.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def norm_pred(val: str) -> str:
    """Normalize a prediction value to lowercase, treating blanks/dashes as empty."""
    v = str(val).strip().lower()
    if v in ("-", ""):
        return ""
    return v


def norm_severity(val: str) -> str:
    """Normalize severity to uppercase."""
    v = str(val).strip().upper()
    if v in ("-", ""):
        return ""
    return v


def severity_to_binary(val: str) -> str | None:
    """Map severity to binary issue prediction.

    OK → "no", WARNING/ERROR → "yes", NA → None (excluded).
    """
    v = norm_severity(val)
    if v in ("WARNING", "ERROR"):
        return "yes"
    if v == "OK":
        return "no"
    return None  # NA or unrecognized → excluded


def severity_error_only_to_binary(val: str) -> str | None:
    """Map severity to binary: only ERROR → yes, OK/WARNING → no, NA → excluded."""
    v = norm_severity(val)
    if v == "ERROR":
        return "yes"
    if v in ("OK", "WARNING"):
        return "no"
    return None  # NA or unrecognized → excluded


# ---------------------------------------------------------------------------
# Binary metrics
# ---------------------------------------------------------------------------

def binary_confusion(gt_list, pred_list, positive="yes"):
    """Compute TP, TN, FP, FN for binary classification.

    Both lists should contain lowercase strings ('yes'/'no').
    Returns dict with tp, tn, fp, fn.
    """
    tp = tn = fp = fn = 0
    for g, p in zip(gt_list, pred_list):
        if g == positive and p == positive:
            tp += 1
        elif g != positive and p != positive:
            tn += 1
        elif g != positive and p == positive:
            fp += 1
        else:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def prf(cm: dict) -> dict:
    """Precision, recall, F1, accuracy from confusion dict."""
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# Multi-class metrics
# ---------------------------------------------------------------------------

def multiclass_confusion(gt_list, pred_list, classes):
    """Build NxN confusion matrix (rows=gt, cols=pred).

    Returns 2D numpy array.
    """
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    mat = np.zeros((n, n), dtype=int)
    for g, p in zip(gt_list, pred_list):
        gi = idx.get(g)
        pi = idx.get(p)
        if gi is not None and pi is not None:
            mat[gi, pi] += 1
    return mat


def per_class_metrics(mat, classes):
    """Per-class precision/recall/F1 from confusion matrix."""
    results = {}
    for i, c in enumerate(classes):
        tp = mat[i, i]
        fp = mat[:, i].sum() - tp
        fn = mat[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[c] = {"precision": prec, "recall": rec, "f1": f1, "support": int(mat[i, :].sum())}
    return results


# ---------------------------------------------------------------------------
# Image dumps (symlinks)
# ---------------------------------------------------------------------------

def create_image_dumps(rows, dimension, gt_key, pred_key, output_dir,
                       positive="yes", is_binary=True):
    """Create symlinked image dumps for TP/TN/FP/FN (binary) or correct/mismatched (multiclass)."""
    dump_dir = output_dir / "image_dumps" / dimension

    if is_binary:
        for cat in ("TP", "TN", "FP", "FN"):
            (dump_dir / cat).mkdir(parents=True, exist_ok=True)

        for row in rows:
            g = norm_pred(row[gt_key])
            p = norm_pred(row[pred_key])
            if g == positive and p == positive:
                cat = "TP"
            elif g != positive and p != positive:
                cat = "TN"
            elif g != positive and p == positive:
                cat = "FP"
            else:
                cat = "FN"
            _symlink_images(row, dump_dir / cat)
    else:
        (dump_dir / "correct").mkdir(parents=True, exist_ok=True)
        for row in rows:
            g = norm_severity(row[gt_key])
            p = norm_severity(row[pred_key])
            if g == p:
                _symlink_images(row, dump_dir / "correct")
            else:
                mismatch_dir = dump_dir / "mismatched" / f"{g}_vs_{p}"
                mismatch_dir.mkdir(parents=True, exist_ok=True)
                _symlink_images(row, mismatch_dir)


def _symlink_images(row, dest_dir):
    """Create symlinks for cam0, cam1, joined images in dest_dir."""
    venue_id = row.get("System ID", "unknown")
    date_str = row.get("Checkup date", "").strip().replace(" ", "_").replace(":", "-")
    for col, suffix in [("cam0_image", "CAM0"), ("cam1_image", "CAM1"), ("joined_image", "joined")]:
        src = row.get(col, "").strip()
        if not src or not Path(src).exists():
            continue
        link_name = dest_dir / f"{venue_id}_{date_str}_{suffix}.jpg"
        try:
            link_name.symlink_to(src)
        except OSError:
            pass  # skip if can't create symlink


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(mat, classes, title, output_path):
    """Save annotated confusion matrix heatmap as PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Annotate cells
    thresh = mat.max() / 2.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]),
                    ha="center", va="center",
                    color="white" if mat[i, j] > thresh else "black",
                    fontsize=12)

    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_binary_confusion(cm, title, output_path):
    """Plot 2x2 confusion matrix from TP/TN/FP/FN dict."""
    classes = ["Positive", "Negative"]
    mat = np.array([[cm["tp"], cm["fn"]], [cm["fp"], cm["tn"]]])
    plot_confusion_matrix(mat, classes, title, output_path)


def plot_prf_bars(metrics_dict, title, output_path):
    """Plot grouped bar chart of precision/recall/F1 per class."""
    classes = list(metrics_dict.keys())
    precision = [metrics_dict[c]["precision"] for c in classes]
    recall = [metrics_dict[c]["recall"] for c in classes]
    f1 = [metrics_dict[c]["f1"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#4c72b0")
    ax.bar(x, recall, width, label="Recall", color="#55a868")
    ax.bar(x + width, f1, width, label="F1", color="#c44e52")

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_distribution(gt_counts, pred_counts, classes, title, output_path):
    """Side-by-side bar chart comparing GT vs predicted distributions."""
    x = np.arange(len(classes))
    width = 0.35

    gt_vals = [gt_counts.get(c, 0) for c in classes]
    pred_vals = [pred_counts.get(c, 0) for c in classes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, gt_vals, width, label="Ground Truth", color="#4c72b0")
    ax.bar(x + width/2, pred_vals, width, label="Predicted", color="#c44e52")

    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute confusion metrics from annotation predictions")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions CSV from run_annotation_pipeline.py")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to write metrics, plots, image dumps "
                             "(default: metrics/ next to predictions CSV)")
    parser.add_argument("--skip-image-dumps", action="store_true",
                        help="Compute metrics and plots only, no image symlinks")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = predictions_path.parent / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Read predictions
    with open(args.predictions, 'r') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    print(f"Loaded {len(all_rows)} rows from {args.predictions}")

    # ======================================================================
    # 1. MEASURABILITY (binary: yes/no)
    # ======================================================================
    meas_rows = [r for r in all_rows
                 if norm_pred(r["pred_measurable"]) in ("yes", "no")
                 and norm_pred(r["gt_measurable"]) in ("yes", "no")]
    meas_excluded = len(all_rows) - len(meas_rows)

    gt_meas = [norm_pred(r["gt_measurable"]) for r in meas_rows]
    pred_meas = [norm_pred(r["pred_measurable"]) for r in meas_rows]

    cm_meas = binary_confusion(gt_meas, pred_meas, positive="yes")
    prf_meas = prf(cm_meas)

    # By venue type
    meas_by_type = {}
    for vtype in ("indoor", "outdoor"):
        subset = [(g, p) for r, g, p in zip(meas_rows, gt_meas, pred_meas)
                  if str(r.get("Indoor / Outdoor", "")).strip().lower() == vtype]
        if subset:
            gt_sub, pred_sub = zip(*subset)
            cm_sub = binary_confusion(gt_sub, pred_sub, positive="yes")
            prf_sub = prf(cm_sub)
            prf_sub["n"] = len(subset)
            meas_by_type[vtype] = prf_sub

    # By source_sheet
    meas_by_sheet = {}
    sheets = set(r.get("source_sheet", "") for r in meas_rows)
    for sheet in sorted(sheets):
        if not sheet:
            continue
        subset = [(g, p) for r, g, p in zip(meas_rows, gt_meas, pred_meas)
                  if r.get("source_sheet", "") == sheet]
        if subset:
            gt_sub, pred_sub = zip(*subset)
            cm_sub = binary_confusion(gt_sub, pred_sub, positive="yes")
            prf_sub = prf(cm_sub)
            prf_sub["n"] = len(subset)
            meas_by_sheet[sheet] = prf_sub

    # ======================================================================
    # 2. QUALITY ISSUE (binary: yes/no) — using detect_issues predictions
    # ======================================================================
    issue_rows = [r for r in all_rows
                  if norm_pred(r["pred_has_issue"]) in ("yes", "no")
                  and norm_pred(r["gt_has_issue"]) in ("yes", "no")]
    issue_excluded = len(all_rows) - len(issue_rows)

    gt_issue = [norm_pred(r["gt_has_issue"]) for r in issue_rows]
    pred_issue = [norm_pred(r["pred_has_issue"]) for r in issue_rows]

    cm_issue = binary_confusion(gt_issue, pred_issue, positive="yes")
    prf_issue = prf(cm_issue)

    # By venue type
    issue_by_type = {}
    for vtype in ("indoor", "outdoor"):
        subset = [(g, p) for r, g, p in zip(issue_rows, gt_issue, pred_issue)
                  if str(r.get("Indoor / Outdoor", "")).strip().lower() == vtype]
        if subset:
            gt_sub, pred_sub = zip(*subset)
            cm_sub = binary_confusion(gt_sub, pred_sub, positive="yes")
            prf_sub = prf(cm_sub)
            prf_sub["n"] = len(subset)
            issue_by_type[vtype] = prf_sub

    # By source_sheet
    issue_by_sheet = {}
    sheets = set(r.get("source_sheet", "") for r in issue_rows)
    for sheet in sorted(sheets):
        if not sheet:
            continue
        subset = [(g, p) for r, g, p in zip(issue_rows, gt_issue, pred_issue)
                  if r.get("source_sheet", "") == sheet]
        if subset:
            gt_sub, pred_sub = zip(*subset)
            cm_sub = binary_confusion(gt_sub, pred_sub, positive="yes")
            prf_sub = prf(cm_sub)
            prf_sub["n"] = len(subset)
            issue_by_sheet[sheet] = prf_sub

    # ======================================================================
    # 3. QUALITY ISSUE — LAPLACIAN (binary: severity mapped to yes/no, NA excluded)
    # ======================================================================
    # Map pred_severity to binary: OK→no, WARNING/ERROR→yes, NA→excluded
    sev_binary_rows = []  # rows where severity maps to yes/no
    sev_na_rows = []      # rows where pred_severity is NA
    for r in all_rows:
        mapped = severity_to_binary(r.get("pred_severity", ""))
        gt = norm_pred(r.get("gt_has_issue", ""))
        if gt not in ("yes", "no"):
            continue  # skip rows without valid GT
        if mapped is None:
            sev_na_rows.append(r)
        else:
            r["_sev_binary_pred"] = mapped
            sev_binary_rows.append(r)

    gt_sev_bin = [norm_pred(r["gt_has_issue"]) for r in sev_binary_rows]
    pred_sev_bin = [r["_sev_binary_pred"] for r in sev_binary_rows]

    cm_sev = binary_confusion(gt_sev_bin, pred_sev_bin, positive="yes")
    prf_sev = prf(cm_sev)

    # NA breakdown
    sev_na_gt_yes = sum(1 for r in sev_na_rows if norm_pred(r.get("gt_has_issue", "")) == "yes")
    sev_na_gt_no = len(sev_na_rows) - sev_na_gt_yes

    # By venue type
    sev_by_type = {}
    for vtype in ("indoor", "outdoor"):
        subset = [(g, p) for r, g, p in zip(sev_binary_rows, gt_sev_bin, pred_sev_bin)
                  if str(r.get("Indoor / Outdoor", "")).strip().lower() == vtype]
        if subset:
            gt_sub, pred_sub = zip(*subset)
            cm_sub = binary_confusion(gt_sub, pred_sub, positive="yes")
            prf_sub = prf(cm_sub)
            prf_sub["n"] = len(subset)
            sev_by_type[vtype] = prf_sub

    # By source_sheet
    sev_by_sheet = {}
    sheets = set(r.get("source_sheet", "") for r in sev_binary_rows)
    for sheet in sorted(sheets):
        if not sheet:
            continue
        subset = [(g, p) for r, g, p in zip(sev_binary_rows, gt_sev_bin, pred_sev_bin)
                  if r.get("source_sheet", "") == sheet]
        if subset:
            gt_sub, pred_sub = zip(*subset)
            cm_sub = binary_confusion(gt_sub, pred_sub, positive="yes")
            prf_sub = prf(cm_sub)
            prf_sub["n"] = len(subset)
            sev_by_sheet[sheet] = prf_sub

    # ======================================================================
    # 4. QUALITY ISSUE — LAPLACIAN ERROR ONLY (binary: only ERROR→yes, OK/WARNING→no)
    # ======================================================================
    sev_eo_binary_rows = []  # rows where severity maps to yes/no
    sev_eo_na_rows = []      # rows where pred_severity is NA
    for r in all_rows:
        mapped = severity_error_only_to_binary(r.get("pred_severity", ""))
        gt = norm_pred(r.get("gt_has_issue", ""))
        if gt not in ("yes", "no"):
            continue  # skip rows without valid GT
        if mapped is None:
            sev_eo_na_rows.append(r)
        else:
            r["_sev_eo_binary_pred"] = mapped
            sev_eo_binary_rows.append(r)

    gt_sev_eo_bin = [norm_pred(r["gt_has_issue"]) for r in sev_eo_binary_rows]
    pred_sev_eo_bin = [r["_sev_eo_binary_pred"] for r in sev_eo_binary_rows]

    cm_sev_eo = binary_confusion(gt_sev_eo_bin, pred_sev_eo_bin, positive="yes")
    prf_sev_eo = prf(cm_sev_eo)

    # NA breakdown
    sev_eo_na_gt_yes = sum(1 for r in sev_eo_na_rows if norm_pred(r.get("gt_has_issue", "")) == "yes")
    sev_eo_na_gt_no = len(sev_eo_na_rows) - sev_eo_na_gt_yes

    # By venue type
    sev_eo_by_type = {}
    for vtype in ("indoor", "outdoor"):
        subset = [(g, p) for r, g, p in zip(sev_eo_binary_rows, gt_sev_eo_bin, pred_sev_eo_bin)
                  if str(r.get("Indoor / Outdoor", "")).strip().lower() == vtype]
        if subset:
            gt_sub, pred_sub = zip(*subset)
            cm_sub = binary_confusion(gt_sub, pred_sub, positive="yes")
            prf_sub = prf(cm_sub)
            prf_sub["n"] = len(subset)
            sev_eo_by_type[vtype] = prf_sub

    # By source_sheet
    sev_eo_by_sheet = {}
    sheets = set(r.get("source_sheet", "") for r in sev_eo_binary_rows)
    for sheet in sorted(sheets):
        if not sheet:
            continue
        subset = [(g, p) for r, g, p in zip(sev_eo_binary_rows, gt_sev_eo_bin, pred_sev_eo_bin)
                  if r.get("source_sheet", "") == sheet]
        if subset:
            gt_sub, pred_sub = zip(*subset)
            cm_sub = binary_confusion(gt_sub, pred_sub, positive="yes")
            prf_sub = prf(cm_sub)
            prf_sub["n"] = len(subset)
            sev_eo_by_sheet[sheet] = prf_sub

    # ======================================================================
    # Text report
    # ======================================================================
    report_lines = []

    def rprint(line=""):
        report_lines.append(line)

    rprint("=" * 60)
    rprint("  CONFUSION METRICS REPORT")
    rprint("=" * 60)

    rprint("\n=== MEASURABILITY ===")
    rprint(f"Evaluated: {len(meas_rows)} rows (excluded {meas_excluded} with N/A/Error predictions)")
    rprint(f"TP: {cm_meas['tp']}  FP: {cm_meas['fp']}  FN: {cm_meas['fn']}  TN: {cm_meas['tn']}")
    rprint(f"Precision: {prf_meas['precision']:.4f}  Recall: {prf_meas['recall']:.4f}"
           f"  F1: {prf_meas['f1']:.4f}  Accuracy: {prf_meas['accuracy']:.4f}")
    rprint("\nBy venue type:")
    for vtype, m in sorted(meas_by_type.items()):
        rprint(f"  {vtype.capitalize():8s}: P={m['precision']:.4f} R={m['recall']:.4f}"
               f" F1={m['f1']:.4f} (N={m['n']})")
    rprint("\nBy source_sheet:")
    for sheet, m in sorted(meas_by_sheet.items()):
        rprint(f"  {sheet:20s}: P={m['precision']:.4f} R={m['recall']:.4f}"
               f" F1={m['f1']:.4f} (N={m['n']})")

    rprint("\n=== QUALITY ISSUE (detect_issues) ===")
    rprint(f"Evaluated: {len(issue_rows)} rows (excluded {issue_excluded} with N/A/Error predictions)")
    rprint(f"TP: {cm_issue['tp']}  FP: {cm_issue['fp']}  FN: {cm_issue['fn']}  TN: {cm_issue['tn']}")
    rprint(f"Precision: {prf_issue['precision']:.4f}  Recall: {prf_issue['recall']:.4f}"
           f"  F1: {prf_issue['f1']:.4f}  Accuracy: {prf_issue['accuracy']:.4f}")
    rprint("\nBy venue type:")
    for vtype, m in sorted(issue_by_type.items()):
        rprint(f"  {vtype.capitalize():8s}: P={m['precision']:.4f} R={m['recall']:.4f}"
               f" F1={m['f1']:.4f} (N={m['n']})")
    rprint("\nBy source_sheet:")
    for sheet, m in sorted(issue_by_sheet.items()):
        rprint(f"  {sheet:20s}: P={m['precision']:.4f} R={m['recall']:.4f}"
               f" F1={m['f1']:.4f} (N={m['n']})")

    rprint("\n=== QUALITY ISSUE — LAPLACIAN (severity → binary) ===")
    rprint(f"Mapping: OK→no, WARNING/ERROR→yes, NA→excluded")
    rprint(f"Evaluated: {len(sev_binary_rows)} rows  |  NA excluded: {len(sev_na_rows)}")
    rprint(f"TP: {cm_sev['tp']}  FP: {cm_sev['fp']}  FN: {cm_sev['fn']}  TN: {cm_sev['tn']}")
    rprint(f"Precision: {prf_sev['precision']:.4f}  Recall: {prf_sev['recall']:.4f}"
           f"  F1: {prf_sev['f1']:.4f}  Accuracy: {prf_sev['accuracy']:.4f}")
    rprint("\nBy venue type:")
    for vtype, m in sorted(sev_by_type.items()):
        rprint(f"  {vtype.capitalize():8s}: P={m['precision']:.4f} R={m['recall']:.4f}"
               f" F1={m['f1']:.4f} (N={m['n']})")
    rprint("\nBy source_sheet:")
    for sheet, m in sorted(sev_by_sheet.items()):
        rprint(f"  {sheet:20s}: P={m['precision']:.4f} R={m['recall']:.4f}"
               f" F1={m['f1']:.4f} (N={m['n']})")
    rprint(f"\n  NA (excluded — not measurable):")
    rprint(f"    Total: {len(sev_na_rows)} rows")
    rprint(f"    GT has_issue=yes: {sev_na_gt_yes}  (venues annotated as having issues but too dark to detect)")
    rprint(f"    GT has_issue=no:  {sev_na_gt_no}  (correctly unmeasurable)")

    rprint("\n=== QUALITY ISSUE — LAPLACIAN ERROR ONLY ===")
    rprint(f"Mapping: ERROR→yes, OK/WARNING→no, NA→excluded")
    rprint(f"Evaluated: {len(sev_eo_binary_rows)} rows  |  NA excluded: {len(sev_eo_na_rows)}")
    rprint(f"TP: {cm_sev_eo['tp']}  FP: {cm_sev_eo['fp']}  FN: {cm_sev_eo['fn']}  TN: {cm_sev_eo['tn']}")
    rprint(f"Precision: {prf_sev_eo['precision']:.4f}  Recall: {prf_sev_eo['recall']:.4f}"
           f"  F1: {prf_sev_eo['f1']:.4f}  Accuracy: {prf_sev_eo['accuracy']:.4f}")
    rprint("\nBy venue type:")
    for vtype, m in sorted(sev_eo_by_type.items()):
        rprint(f"  {vtype.capitalize():8s}: P={m['precision']:.4f} R={m['recall']:.4f}"
               f" F1={m['f1']:.4f} (N={m['n']})")
    rprint("\nBy source_sheet:")
    for sheet, m in sorted(sev_eo_by_sheet.items()):
        rprint(f"  {sheet:20s}: P={m['precision']:.4f} R={m['recall']:.4f}"
               f" F1={m['f1']:.4f} (N={m['n']})")
    rprint(f"\n  NA (excluded — not measurable):")
    rprint(f"    Total: {len(sev_eo_na_rows)} rows")
    rprint(f"    GT has_issue=yes: {sev_eo_na_gt_yes}  (venues annotated as having issues but too dark to detect)")
    rprint(f"    GT has_issue=no:  {sev_eo_na_gt_no}  (correctly unmeasurable)")

    rprint("\n" + "=" * 60)

    report_text = "\n".join(report_lines)
    print(report_text)

    # Write report file
    report_path = output_dir / "metrics_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text + "\n")
    print(f"\nReport written to: {report_path}")

    # ======================================================================
    # Plots
    # ======================================================================

    # Measurability confusion matrix
    plot_binary_confusion(cm_meas, "Measurability Confusion Matrix",
                          plots_dir / "measurability_confusion.png")

    # Quality issue confusion matrix
    plot_binary_confusion(cm_issue, "Quality Issue Confusion Matrix",
                          plots_dir / "quality_issue_confusion.png")

    # Quality Issue — Laplacian confusion matrix (2x2)
    plot_binary_confusion(cm_sev, "Quality Issue (Laplacian) Confusion Matrix",
                          plots_dir / "severity_confusion.png")

    # PRF bar charts
    meas_prf_dict = {
        "Yes (measurable)": {"precision": prf_meas["precision"],
                             "recall": prf_meas["recall"],
                             "f1": prf_meas["f1"]},
    }
    # Add negative class metrics
    tn, fp, fn, tp = cm_meas["tn"], cm_meas["fp"], cm_meas["fn"], cm_meas["tp"]
    neg_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    neg_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    neg_f1 = 2 * neg_prec * neg_rec / (neg_prec + neg_rec) if (neg_prec + neg_rec) > 0 else 0.0
    meas_prf_dict["No (not measurable)"] = {"precision": neg_prec, "recall": neg_rec, "f1": neg_f1}
    plot_prf_bars(meas_prf_dict, "Measurability — Precision / Recall / F1",
                  plots_dir / "measurability_prf.png")

    issue_prf_dict = {
        "Yes (has issue)": {"precision": prf_issue["precision"],
                            "recall": prf_issue["recall"],
                            "f1": prf_issue["f1"]},
    }
    tn, fp, fn, tp = cm_issue["tn"], cm_issue["fp"], cm_issue["fn"], cm_issue["tp"]
    neg_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    neg_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    neg_f1 = 2 * neg_prec * neg_rec / (neg_prec + neg_rec) if (neg_prec + neg_rec) > 0 else 0.0
    issue_prf_dict["No (no issue)"] = {"precision": neg_prec, "recall": neg_rec, "f1": neg_f1}
    plot_prf_bars(issue_prf_dict, "Quality Issue — Precision / Recall / F1",
                  plots_dir / "quality_issue_prf.png")

    sev_prf_dict = {
        "Yes (has issue)": {"precision": prf_sev["precision"],
                            "recall": prf_sev["recall"],
                            "f1": prf_sev["f1"]},
    }
    tn, fp, fn, tp = cm_sev["tn"], cm_sev["fp"], cm_sev["fn"], cm_sev["tp"]
    neg_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    neg_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    neg_f1 = 2 * neg_prec * neg_rec / (neg_prec + neg_rec) if (neg_prec + neg_rec) > 0 else 0.0
    sev_prf_dict["No (no issue)"] = {"precision": neg_prec, "recall": neg_rec, "f1": neg_f1}
    plot_prf_bars(sev_prf_dict, "Quality Issue (Laplacian) — Precision / Recall / F1",
                  plots_dir / "severity_prf.png")

    # Distribution comparisons
    gt_meas_counts = defaultdict(int)
    pred_meas_counts = defaultdict(int)
    for g, p in zip(gt_meas, pred_meas):
        gt_meas_counts[g] += 1
        pred_meas_counts[p] += 1
    plot_distribution(gt_meas_counts, pred_meas_counts, ["yes", "no"],
                      "Measurability — GT vs Predicted Distribution",
                      plots_dir / "measurability_distribution.png")

    gt_issue_counts = defaultdict(int)
    pred_issue_counts = defaultdict(int)
    for g, p in zip(gt_issue, pred_issue):
        gt_issue_counts[g] += 1
        pred_issue_counts[p] += 1
    plot_distribution(gt_issue_counts, pred_issue_counts, ["yes", "no"],
                      "Quality Issue — GT vs Predicted Distribution",
                      plots_dir / "quality_issue_distribution.png")

    gt_sev_bin_counts = defaultdict(int)
    pred_sev_bin_counts = defaultdict(int)
    for g, p in zip(gt_sev_bin, pred_sev_bin):
        gt_sev_bin_counts[g] += 1
        pred_sev_bin_counts[p] += 1
    plot_distribution(gt_sev_bin_counts, pred_sev_bin_counts, ["yes", "no"],
                      "Quality Issue (Laplacian) — GT vs Predicted Distribution",
                      plots_dir / "severity_distribution.png")

    # Quality Issue — Laplacian Error Only confusion matrix (2x2)
    plot_binary_confusion(cm_sev_eo, "Quality Issue (Laplacian Error Only) Confusion Matrix",
                          plots_dir / "severity_error_only_confusion.png")

    sev_eo_prf_dict = {
        "Yes (has issue)": {"precision": prf_sev_eo["precision"],
                            "recall": prf_sev_eo["recall"],
                            "f1": prf_sev_eo["f1"]},
    }
    tn, fp, fn, tp = cm_sev_eo["tn"], cm_sev_eo["fp"], cm_sev_eo["fn"], cm_sev_eo["tp"]
    neg_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    neg_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    neg_f1 = 2 * neg_prec * neg_rec / (neg_prec + neg_rec) if (neg_prec + neg_rec) > 0 else 0.0
    sev_eo_prf_dict["No (no issue)"] = {"precision": neg_prec, "recall": neg_rec, "f1": neg_f1}
    plot_prf_bars(sev_eo_prf_dict, "Quality Issue (Laplacian Error Only) — Precision / Recall / F1",
                  plots_dir / "severity_error_only_prf.png")

    gt_sev_eo_bin_counts = defaultdict(int)
    pred_sev_eo_bin_counts = defaultdict(int)
    for g, p in zip(gt_sev_eo_bin, pred_sev_eo_bin):
        gt_sev_eo_bin_counts[g] += 1
        pred_sev_eo_bin_counts[p] += 1
    plot_distribution(gt_sev_eo_bin_counts, pred_sev_eo_bin_counts, ["yes", "no"],
                      "Quality Issue (Laplacian Error Only) — GT vs Predicted Distribution",
                      plots_dir / "severity_error_only_distribution.png")

    print(f"Plots written to: {plots_dir}")

    # ======================================================================
    # Image dumps
    # ======================================================================
    if not args.skip_image_dumps:
        print("Creating image dumps (symlinks)...")

        create_image_dumps(meas_rows, "measurability", "gt_measurable", "pred_measurable",
                           output_dir, positive="yes", is_binary=True)

        create_image_dumps(issue_rows, "quality_issue", "gt_has_issue", "pred_has_issue",
                           output_dir, positive="yes", is_binary=True)

        create_image_dumps(sev_binary_rows, "severity", "gt_has_issue", "_sev_binary_pred",
                           output_dir, positive="yes", is_binary=True)
        # NA dump: pred_severity=NA, can't evaluate
        na_dump_dir = output_dir / "image_dumps" / "severity" / "NA"
        na_dump_dir.mkdir(parents=True, exist_ok=True)
        for row in sev_na_rows:
            _symlink_images(row, na_dump_dir)

        create_image_dumps(sev_eo_binary_rows, "severity_error_only", "gt_has_issue",
                           "_sev_eo_binary_pred", output_dir, positive="yes", is_binary=True)
        # NA dump for error-only
        eo_na_dump_dir = output_dir / "image_dumps" / "severity_error_only" / "NA"
        eo_na_dump_dir.mkdir(parents=True, exist_ok=True)
        for row in sev_eo_na_rows:
            _symlink_images(row, eo_na_dump_dir)

        # Skipped (not measurable) dump: quality issue was not evaluated
        not_meas_rows = [r for r in all_rows
                         if r.get("pred_observation", "").strip() == "Skipped — not measurable"]
        not_meas_dir = output_dir / "image_dumps" / "quality_issue" / "skipped_not_measurable"
        not_meas_dir.mkdir(parents=True, exist_ok=True)
        for row in not_meas_rows:
            _symlink_images(row, not_meas_dir)
        print(f"  Skipped (not measurable): {len(not_meas_rows)} venues dumped")

        print(f"Image dumps written to: {output_dir / 'image_dumps'}")
    else:
        print("Skipping image dumps (--skip-image-dumps)")


if __name__ == "__main__":
    main()
