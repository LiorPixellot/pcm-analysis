#!/usr/bin/env python3
"""
Read laplacian_calculations.csv and add Focus_severity column based on threshold logic.
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path


def worst_severity(a, b):
    """Return the worse severity (higher rank = worse)."""
    rank = {"Ok": 0, "Warning": 1, "Error": 2}
    if rank[a] >= rank[b]:
        return a
    return b


def mid_severity(mid):
    """
    Calculate severity from mid focus value.
    Error: 0 <= mid <= 10
    Warning: 10 < mid <= 20
    Ok: mid > 20
    """
    if mid < 0:
        return "Error"
    if mid <= 10:
        return "Error"
    if mid <= 20:
        return "Warning"
    return "Ok"


def calc_focus_severity(focus_right_mean, focus_left_mean,
                        focus_abs_dif_rel,
                        focus_right_mid, focus_left_mid):
    """
    Calculate focus severity based on thresholds.

    Returns one of: "NA", "Ok", "Warning", "Error"
    """
    # 1) NA gate: either mean < 70
    if focus_right_mean < 70 or focus_left_mean < 70:
        return "NA"

    # 2) Severity from abs_dif_rel
    if focus_abs_dif_rel > 1.25:
        sev_diff = "Error"
    elif focus_abs_dif_rel >= 0.7:
        sev_diff = "Warning"
    else:
        sev_diff = "Ok"

    # 3) Severity from mid values (per-side, then worst)
    sev_mid_right = mid_severity(focus_right_mid)
    sev_mid_left = mid_severity(focus_left_mid)
    sev_mid = worst_severity(sev_mid_right, sev_mid_left)

    # 4) Final = worst of diff and mid severities
    return worst_severity(sev_diff, sev_mid)


def main():
    parser = argparse.ArgumentParser(
        description='Add Focus_severity column to laplacian calculations CSV'
    )
    parser.add_argument('input_csv', nargs='?', default='laplacian_calculations.csv',
                        help='Input CSV file (default: laplacian_calculations.csv)')
    parser.add_argument('-o', '--output', default='laplacian_th_calculations.csv',
                        help='Output CSV file (default: laplacian_th_calculations.csv)')
    parser.add_argument('--output-dir', default=None,
                        help='Parent output directory (creates laplacian_th/ subdir). If omitted, creates timestamped dir.')
    args = parser.parse_args()

    # Create output directory
    script_dir = Path(__file__).parent.resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir) / "laplacian_th"
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_dir = script_dir / "output_dir" / f"laplacian_th_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output CSV path: use user-provided path or default inside output_dir
    if args.output != 'laplacian_th_calculations.csv':
        output_path = args.output
    else:
        output_path = str(output_dir / "laplacian_th_calculations.csv")

    try:
        with open(args.input_csv, 'r', newline='') as infile:
            reader = csv.DictReader(infile)

            # Add Focus_severity to fieldnames
            fieldnames = reader.fieldnames + ['Focus_severity']

            with open(output_path, 'w', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()

                counts = {"NA": 0, "Ok": 0, "Warning": 0, "Error": 0}
                total = 0

                for row in reader:
                    # Parse values
                    focus_right_mean = float(row['focus_right_mean'])
                    focus_left_mean = float(row['focus_left_mean'])
                    focus_abs_dif_rel = float(row['focus_abs_dif_rel'])
                    focus_right_mid = float(row['focus_right_mid'])
                    focus_left_mid = float(row['focus_left_mid'])

                    # Calculate severity
                    severity = calc_focus_severity(
                        focus_right_mean, focus_left_mean,
                        focus_abs_dif_rel,
                        focus_right_mid, focus_left_mid
                    )

                    row['Focus_severity'] = severity
                    writer.writerow(row)

                    counts[severity] += 1
                    total += 1

    except FileNotFoundError:
        print(f'Error: {args.input_csv} not found', file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f'Error: Missing column {e} in input CSV', file=sys.stderr)
        sys.exit(1)

    print(f'Processed: {total}')
    print(f'  NA: {counts["NA"]}')
    print(f'  Ok: {counts["Ok"]}')
    print(f'  Warning: {counts["Warning"]}')
    print(f'  Error: {counts["Error"]}')
    print(f'Output dir: {output_dir}')
    print(f'Output written to: {output_path}')


if __name__ == '__main__':
    main()
