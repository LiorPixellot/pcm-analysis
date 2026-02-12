#!/usr/bin/env python3
"""
Run the full Laplacian focus pipeline:
  1. laplacian_calculations.py — compute focus metrics
  2. laplacian_th_calculations.py — add severity classification
  3. concat_with_is_measurable.py — join blur + is_measurable data
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Run Laplacian focus calculation + threshold classification pipeline'
    )
    parser.add_argument('data_dir', help='Path to data directory (e.g., all_data_02_09)')
    parser.add_argument('--blur-csv', default='PQS_blur_by_venue.xlsx',
                        help='Blur by venue Excel file to join (default: PQS_blur_by_venue.xlsx)')
    parser.add_argument('--is-measurable-csv', default='is_measurable.csv',
                        help='is_measurable CSV to join (default: is_measurable.csv)')
    parser.add_argument('--no-concat', action='store_true',
                        help='Skip the concat/join step (Step 3)')
    args = parser.parse_args()

    python = sys.executable
    script_dir = Path(__file__).parent.resolve()

    # Step 1: Laplacian calculations
    print("=" * 60)
    print("Step 1: Running laplacian_calculations.py")
    print("=" * 60)
    step1 = subprocess.run(
        [python, '-u', str(script_dir / 'laplacian_calculations.py'), args.data_dir]
    )
    if step1.returncode != 0:
        print("Step 1 failed, aborting.", file=sys.stderr)
        sys.exit(1)

    # Find the CSV output from step 1
    output_base = script_dir / 'output_dir'
    laplacian_dirs = sorted(output_base.glob('laplacian_[0-9]*'), key=lambda p: p.name)
    if not laplacian_dirs:
        print("Error: No laplacian output directory found.", file=sys.stderr)
        sys.exit(1)

    step1_csv = laplacian_dirs[-1] / 'laplacian_calculations.csv'
    if not step1_csv.exists():
        print(f"Error: {step1_csv} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Step 1 output: {step1_csv}")
    print()

    # Step 2: Threshold calculations
    print("=" * 60)
    print("Step 2: Running laplacian_th_calculations.py")
    print("=" * 60)
    step2 = subprocess.run(
        [python, '-u', str(script_dir / 'laplacian_th_calculations.py'), str(step1_csv)]
    )
    if step2.returncode != 0:
        print("Step 2 failed.", file=sys.stderr)
        sys.exit(1)

    # Find the CSV output from step 2
    th_dirs = sorted(output_base.glob('laplacian_th_[0-9]*'), key=lambda p: p.name)
    if not th_dirs:
        print("Error: No laplacian_th output directory found.", file=sys.stderr)
        sys.exit(1)

    step2_csv = th_dirs[-1] / 'laplacian_th_calculations.csv'

    # Step 3: Concat with blur + is_measurable (on by default, skip with --no-concat)
    blur_file = script_dir / args.blur_csv
    ism_file = script_dir / args.is_measurable_csv
    if not args.no_concat:
        if not blur_file.exists():
            print(f"Warning: {blur_file} not found, skipping Step 3.", file=sys.stderr)
        elif not ism_file.exists():
            print(f"Warning: {ism_file} not found, skipping Step 3.", file=sys.stderr)
        else:
            print()
            print("=" * 60)
            print("Step 3: Running concat_with_is_measurable.py")
            print("=" * 60)
            output_csv = str(th_dirs[-1] / 'laplacian_th_with_blur_and_measurable.csv')
            step3 = subprocess.run(
                [python, '-u', str(script_dir / 'concat_with_is_measurable.py'),
                 str(step2_csv), str(blur_file), str(ism_file),
                 '-o', output_csv]
            )
            if step3.returncode != 0:
                print("Step 3 failed.", file=sys.stderr)
                sys.exit(1)

    print("\nPipeline complete.")


if __name__ == '__main__':
    main()
