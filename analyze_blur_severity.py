"""Analyze correlation between AVERAGE_BLUR_AVG and Focus_severity."""

import sys
import os
from datetime import datetime
from itertools import combinations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


SEVERITY_ORDER = ["Ok", "Warning", "Error", "NA"]
SEVERITY_NUMERIC = {"Ok": 0, "Warning": 1, "Error": 2}


def run_analysis(df, label, out_dir):
    """Run full blur-severity analysis on df, saving outputs with label prefix.

    Returns dict with key metrics for comparison (or None if insufficient data).
    """
    prefix = label + "_"
    print("=" * 60)
    print(f"[{label}] SUMMARY STATISTICS: AVERAGE_BLUR_AVG by Focus_severity")
    print(f"  Rows: {len(df)}")
    print("=" * 60)

    if len(df) == 0:
        print(f"  No data for [{label}], skipping.\n")
        return None

    # --- 1. Summary Statistics ---
    groups = df.groupby("Focus_severity")["AVERAGE_BLUR_AVG"]
    summary = groups.agg(["count", "mean", "median", "std", "min", "max"])
    summary = summary.reindex([s for s in SEVERITY_ORDER if s in summary.index])
    summary.to_csv(os.path.join(out_dir, f"{prefix}summary_stats.csv"))
    print(summary.to_string())
    print()

    # --- 2. Box Plot (interactive) ---
    present = [s for s in SEVERITY_ORDER if s in df["Focus_severity"].values]
    colors = {"Ok": "#4CAF50", "Warning": "#FFC107", "Error": "#F44336", "NA": "#9E9E9E"}

    fig = go.Figure()
    for sev in present:
        vals = df.loc[df["Focus_severity"] == sev, "AVERAGE_BLUR_AVG"]
        fig.add_trace(go.Box(
            y=vals, name=f"{sev} (n={len(vals)})",
            marker_color=colors.get(sev, "#BBDEFB"),
            boxpoints="outliers",
            hoverinfo="y+name",
        ))
    fig.update_layout(
        title=f"[{label}] AVERAGE_BLUR_AVG Distribution by Focus Severity",
        yaxis_title="AVERAGE_BLUR_AVG",
        xaxis_title="Focus Severity",
        showlegend=False,
    )
    fig.write_html(os.path.join(out_dir, f"{prefix}boxplot.html"))
    print(f"Saved {prefix}boxplot.html")

    # --- 3. Histogram / Overlapping Distributions (interactive) ---
    fig = go.Figure()
    for sev in present:
        vals = df.loc[df["Focus_severity"] == sev, "AVERAGE_BLUR_AVG"]
        fig.add_trace(go.Histogram(
            x=vals, name=f"{sev} (n={len(vals)})",
            marker_color=colors.get(sev, "#BBDEFB"),
            opacity=0.6,
            nbinsx=50,
        ))
    fig.update_layout(
        barmode="overlay",
        title=f"[{label}] AVERAGE_BLUR_AVG Distribution by Focus Severity",
        xaxis_title="AVERAGE_BLUR_AVG",
        yaxis_title="Count",
    )
    fig.write_html(os.path.join(out_dir, f"{prefix}histogram.html"))
    print(f"Saved {prefix}histogram.html")

    # --- 4. Statistical Tests ---
    print("\n" + "=" * 60)
    print(f"[{label}] STATISTICAL TESTS")
    print("=" * 60)

    group_arrays = [df.loc[df["Focus_severity"] == s, "AVERAGE_BLUR_AVG"].values
                    for s in present if len(df[df["Focus_severity"] == s]) > 0]

    metrics = {}

    if len(group_arrays) >= 2:
        kw_stat, kw_p = stats.kruskal(*group_arrays)
        print(f"\nKruskal-Wallis H-test (all groups): H={kw_stat:.4f}, p={kw_p:.2e}")
        metrics["kruskal_wallis_p"] = kw_p

        test_rows = [{"test": "Kruskal-Wallis (all groups)", "statistic": kw_stat, "p_value": kw_p}]
        print("\nPairwise Mann-Whitney U tests:")
        for a, b in combinations(present, 2):
            va = df.loc[df["Focus_severity"] == a, "AVERAGE_BLUR_AVG"].values
            vb = df.loc[df["Focus_severity"] == b, "AVERAGE_BLUR_AVG"].values
            u_stat, u_p = stats.mannwhitneyu(va, vb, alternative="two-sided")
            sig = "***" if u_p < 0.001 else "**" if u_p < 0.01 else "*" if u_p < 0.05 else "ns"
            print(f"  {a} vs {b}: U={u_stat:.1f}, p={u_p:.2e} {sig}")
            test_rows.append({"test": f"{a} vs {b}", "statistic": u_stat, "p_value": u_p})

        pd.DataFrame(test_rows).to_csv(os.path.join(out_dir, f"{prefix}statistical_tests.csv"), index=False)
        print(f"\nSaved {prefix}statistical_tests.csv")
    else:
        print("\n  Not enough groups for statistical tests.")
        metrics["kruskal_wallis_p"] = None

    # --- 5. Spearman Correlation (excluding NA) ---
    print("\n" + "=" * 60)
    print(f"[{label}] SPEARMAN RANK CORRELATION (excluding NA)")
    print("=" * 60)

    df_ranked = df[df["Focus_severity"].isin(SEVERITY_NUMERIC)].copy()
    if len(df_ranked) >= 3:
        df_ranked["severity_rank"] = df_ranked["Focus_severity"].map(SEVERITY_NUMERIC)
        rho, sp_p = stats.spearmanr(df_ranked["AVERAGE_BLUR_AVG"], df_ranked["severity_rank"])
        print(f"  rho={rho:.4f}, p={sp_p:.2e}")
        print(f"  Interpretation: {'positive' if rho > 0 else 'negative'} correlation — "
              f"higher blur {'associates with worse' if rho > 0 else 'associates with better'} focus severity")
        metrics["spearman_rho"] = rho
    else:
        print("  Not enough non-NA rows for correlation.")
        metrics["spearman_rho"] = None

    # Collect per-severity means for comparison
    for sev in SEVERITY_ORDER:
        mask = df["Focus_severity"] == sev
        metrics[f"mean_{sev}"] = df.loc[mask, "AVERAGE_BLUR_AVG"].mean() if mask.any() else None
        metrics[f"count_{sev}"] = int(mask.sum())

    metrics["total_rows"] = len(df)
    print()
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze correlation between AVERAGE_BLUR_AVG and Focus_severity"
    )
    parser.add_argument("input_csv", nargs="?", default="laplacian_th_with_blur_and_measurable.csv",
                        help="Input CSV file (default: laplacian_th_with_blur_and_measurable.csv)")
    parser.add_argument("--output-dir", default=None,
                        help="Parent output directory (creates analyze_blur/ subdir). If omitted, creates timestamped dir.")
    args = parser.parse_args()

    csv_path = args.input_csv

    if not os.path.isfile(csv_path):
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    if csv_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # pandas reads the string "NA" as NaN — restore it so groupby includes NA severity
    df["Focus_severity"] = df["Focus_severity"].fillna("NA")

    # Filter to rows with blur data
    df_blur = df.dropna(subset=["AVERAGE_BLUR_AVG"]).copy()
    print(f"Rows with AVERAGE_BLUR_AVG data: {len(df_blur)}")
    print(f"Rows without blur data: {len(df) - len(df_blur)}")

    # Check if is_measurable column exists
    has_measurable = "is_measurable" in df.columns
    if has_measurable:
        df_measurable = df_blur[df_blur["is_measurable"] == "Yes"].copy()
        print(f"Rows with is_measurable == 'Yes' AND blur data: {len(df_measurable)}")
    else:
        print("Column 'is_measurable' not found — running all-data analysis only.")

    # Create output directory
    if args.output_dir:
        out_dir = os.path.join(args.output_dir, "analyze_blur")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        out_dir = os.path.join("output_dir", f"analyze_blur_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    # --- Run A: All Data ---
    metrics_all = run_analysis(df_blur, "all", out_dir)

    # --- Run B: Measurable Only ---
    metrics_meas = None
    if has_measurable:
        metrics_meas = run_analysis(df_measurable, "measurable", out_dir)

    # --- Comparison Summary ---
    if metrics_all and metrics_meas:
        print("=" * 60)
        print("COMPARISON SUMMARY: All Data vs Measurable Only")
        print("=" * 60)

        rows = []
        rows.append({"metric": "total_rows", "all": metrics_all["total_rows"], "measurable": metrics_meas["total_rows"]})
        for sev in SEVERITY_ORDER:
            rows.append({
                "metric": f"count_{sev}",
                "all": metrics_all[f"count_{sev}"],
                "measurable": metrics_meas[f"count_{sev}"],
            })
            rows.append({
                "metric": f"mean_{sev}",
                "all": metrics_all.get(f"mean_{sev}"),
                "measurable": metrics_meas.get(f"mean_{sev}"),
            })
        rows.append({"metric": "spearman_rho", "all": metrics_all.get("spearman_rho"), "measurable": metrics_meas.get("spearman_rho")})
        rows.append({"metric": "kruskal_wallis_p", "all": metrics_all.get("kruskal_wallis_p"), "measurable": metrics_meas.get("kruskal_wallis_p")})

        comp_df = pd.DataFrame(rows)
        comp_df.to_csv(os.path.join(out_dir, "comparison_summary.csv"), index=False)
        print(comp_df.to_string(index=False))
        print(f"\nSaved comparison_summary.csv")

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
