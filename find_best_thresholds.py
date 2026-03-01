"""
Find optimal severity thresholds via Decision Tree analysis.

Goal: find rules from focus metrics that catch ALL cases where
AVERAGE_BLUR_AVG < 400 (Error), using symmetric left/right features.

Sweeps class_weight to find the precision/recall trade-off at depth=2.

Usage:
    python find_best_thresholds.py
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

DATA_PATH = (
    "output_dir/more_examples_play_is_messure/concat_data/"
    "laplacian_th_with_blur_measurable_issues.xlsx"
)


def load_data():
    df = pd.read_excel(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    # Encode categoricals
    df["is_measurable_enc"] = (
        df["is_measurable"].map({"Yes": 1, "No": 0, "Unknown": 0}).fillna(0).astype(int)
    )
    df["has_issue_enc"] = (
        df["has_issue"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    )

    # Symmetric features — order-independent of left/right
    df["mean_min"] = df[["focus_right_mean", "focus_left_mean"]].min(axis=1)
    df["mean_max"] = df[["focus_right_mean", "focus_left_mean"]].max(axis=1)
    df["mid_min"] = df[["focus_right_mid", "focus_left_mid"]].min(axis=1)
    df["mid_max"] = df[["focus_right_mid", "focus_left_mid"]].max(axis=1)

    feature_cols = [
        "mean_min", "mean_max", "mid_min", "mid_max",
        "focus_abs_dif_rel", "is_measurable_enc", "has_issue_enc",
    ]
    df_valid = df.dropna(subset=["AVERAGE_BLUR_AVG"] + feature_cols).copy()
    df_valid["is_error"] = (df_valid["AVERAGE_BLUR_AVG"] < 400).astype(int)

    n_err = df_valid["is_error"].sum()
    n_ok = len(df_valid) - n_err
    print(f"Rows: {len(df_valid)}  Error(blur<400): {n_err}  Not-Error: {n_ok}")

    return df_valid, feature_cols


def fit_tree(X, y, feature_names, max_depth, error_weight):
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight={0: 1, 1: error_weight},
        random_state=42,
    )
    tree.fit(X, y)
    y_pred = tree.predict(X)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return tree, y_pred, recall, precision, tp, fp, fn, tn


def sweep_weights(X, y, feature_names, max_depth, label):
    print(f"\n{'#' * 70}")
    print(f"# {label} — depth={max_depth}")
    print(f"# Sweeping Error class weight to maximize recall")
    print(f"{'#' * 70}")

    header = f"{'weight':>7} {'recall':>7} {'prec':>7} {'f1':>7} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    weights = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]
    best_tree = None
    best_weight = None
    best_recall = 0

    for w in weights:
        tree, y_pred, recall, precision, tp, fp, fn, tn = fit_tree(
            X, y, feature_names, max_depth, w
        )
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{w:>7} {recall:>7.1%} {precision:>7.1%} {f1:>7.3f} {tp:>6} {fp:>6} {fn:>6} {tn:>6}")

        if recall > best_recall or (recall == best_recall and precision > 0):
            best_recall = recall
            best_tree = tree
            best_weight = w

    # Show the best tree that achieves max recall
    print(f"\n--- Best tree: weight={best_weight}, recall={best_recall:.1%} ---")
    print(f"\nFeature importances:")
    for name, imp in sorted(
        zip(feature_names, best_tree.feature_importances_), key=lambda x: -x[1]
    ):
        if imp > 0.001:
            print(f"  {name:25s} {imp:.4f}")

    print(f"\nTree rules:")
    print(export_text(best_tree, feature_names=feature_names, decimals=2))

    # Show what blur values fall in each leaf
    y_pred = best_tree.predict(X)
    print("Blur distribution when tree says Error:")
    blur = X_df["AVERAGE_BLUR_AVG"]  # will be set in main
    pred_error = y_pred == 1
    actual_error = y == 1
    print(f"  Flagged as Error: {pred_error.sum()}")
    print(f"    truly Error (blur<400): {(pred_error & actual_error).sum()}")
    print(f"    false alarm (blur>=400): {(pred_error & ~actual_error).sum()}")
    missed = (~pred_error) & actual_error
    print(f"  Missed Errors: {missed.sum()}")
    if missed.sum() > 0:
        missed_blur = blur[missed]
        print(f"    blur range of missed: {missed_blur.min():.0f} - {missed_blur.max():.0f}")

    return best_tree, best_weight


# Global ref for blur values (used in sweep_weights reporting)
X_df = None


def main():
    global X_df
    df, all_features = load_data()
    X_df = df

    focus_only = ["mean_min", "mean_max", "mid_min", "mid_max", "focus_abs_dif_rel"]
    y = df["is_error"].values

    for features, label in [
        (focus_only, "FOCUS-ONLY"),
        (all_features, "ALL FEATURES"),
    ]:
        X = df[features].values
        for depth in [2, 3]:
            sweep_weights(X, y, features, depth, label)


if __name__ == "__main__":
    main()
