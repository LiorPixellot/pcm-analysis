#!/usr/bin/env python3
"""
Read laplacian_calculations.csv and add Focus_severity column based on threshold logic.
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path


def calc_focus_severity(focus_right_mean, focus_left_mean,
                        focus_abs_dif_rel,
                        focus_right_mid, focus_left_mid,
                        focus_center_mean=None, focus_center_mid=None,
                        focus_abs_dif_rel_12=None):
    """
    Calculate focus severity based on thresholds.

    For 3-camera setups, also considers center camera metrics and pair (1,2) dif_rel.
    Empty/None values for 3-cam fields → original 2-cam logic.

    Returns one of: "NA", "Ok", "Warning", "Error"
    """
    # Collect all camera means and mids
    all_means = [focus_right_mean, focus_left_mean]
    all_mids = [focus_right_mid, focus_left_mid]
    all_dif_rels = [focus_abs_dif_rel]

    if focus_center_mean is not None and focus_center_mean != '':
        all_means.append(float(focus_center_mean))
    if focus_center_mid is not None and focus_center_mid != '':
        all_mids.append(float(focus_center_mid))
    if focus_abs_dif_rel_12 is not None and focus_abs_dif_rel_12 != '':
        all_dif_rels.append(float(focus_abs_dif_rel_12))

    # NA: too dark to trust Laplacian values
    if min(all_means) < 50:
        return "NA"

    avg_mean = sum(all_means) / len(all_means)
    min_mid = min(all_mids)
    max_mid = max(all_mids)
    max_dif_rel = max(all_dif_rels)

    # Error (any triggers):
    if max_dif_rel > 1.0:               # large diff between any camera pair
        return "Error"
    if min_mid <= 15:                     # one camera very blurry
        return "Error"
    if avg_mean < 95 and min_mid <= 25:   # dark + bad mid
        return "Error"
    if max_mid <= 50 and avg_mean < 100:  # all cameras bad
        return "Error"

    # Warning:
    if max_dif_rel >= 0.5 or min_mid <= 25:
        return "Warning"

    return "Ok"


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

                    # Parse optional 3-cam columns
                    focus_center_mean = row.get('focus_center_mean', '')
                    focus_center_mid = row.get('focus_center_mid', '')
                    focus_abs_dif_rel_12 = row.get('focus_abs_dif_rel_12', '')

                    # Calculate severity
                    severity = calc_focus_severity(
                        focus_right_mean, focus_left_mean,
                        focus_abs_dif_rel,
                        focus_right_mid, focus_left_mid,
                        focus_center_mean, focus_center_mid,
                        focus_abs_dif_rel_12,
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
