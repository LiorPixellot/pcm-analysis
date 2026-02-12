#!/usr/bin/env python3
"""
Run the full analysis pipeline:
  1. run_laplacian_pipeline.py       — focus metrics + severity classification
  2. concat_blur.py                  — join blur data
  3. run_is_measurable.py            — AI measurability analysis (or use pre-computed CSV)
  4. concat_with_is_measurable.py    — join is_measurable data
  5. detect_issues.py                — AI issue detection (focus/object/condensation)
  6. concat_with_detect_issues.py    — join detect_issues data
  7. analyze_blur_severity.py        — plots + stats

Usage:
    python run_full_pipeline.py <data_dir> <blur_xlsx>
    python run_full_pipeline.py <data_dir> <blur_xlsx> --is-measurable-csv <path>
    python run_full_pipeline.py <data_dir> <blur_xlsx> --examples-dir examples/
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Run full focus analysis pipeline'
    )
    parser.add_argument('data_dir', help='Path to data directory (e.g., all_data_02_09)')
    parser.add_argument('blur_xlsx', help='PQS blur by venue file (e.g., PQS_blur_by_venue.xlsx)')
    parser.add_argument('--is-measurable-csv', default=None,
                        help='Pre-computed measurable CSV (skip run_is_measurable.py)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of venues for run_is_measurable.py')
    parser.add_argument('--output-dir', default=None,
                        help='Master output directory. If omitted, creates output_dir/YYYY-MM-DD_HH-MM/')
    parser.add_argument('--examples-dir', default=None,
                        help='Examples directory for detect_issues.py (passed as --use-examples)')
    args = parser.parse_args()

    python = sys.executable
    script_dir = Path(__file__).parent.resolve()

    # Validate inputs before running expensive computations
    for path, label in [(args.data_dir, 'data_dir'), (args.blur_xlsx, 'blur_xlsx')]:
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    if args.is_measurable_csv and not Path(args.is_measurable_csv).exists():
        print(f"Error: is_measurable_csv not found: {args.is_measurable_csv}", file=sys.stderr)
        sys.exit(1)

    # Create master output directory
    if args.output_dir:
        master_output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        master_output_dir = script_dir / 'output_dir' / timestamp
    master_output_dir.mkdir(parents=True, exist_ok=True)
    concat_data_dir = master_output_dir / 'concat_data'
    concat_data_dir.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(master_output_dir)

    print(f"Master output directory: {master_output_dir}\n")

    # Copy source files for reproducibility
    source_files_dir = master_output_dir / 'source_files'
    source_files_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for entry in script_dir.iterdir():
        if entry.is_file():
            shutil.copy2(entry, source_files_dir / entry.name)
            copied += 1
    print(f"Copied {copied} source files to {source_files_dir}\n")

    # Step 1: Laplacian pipeline (calculations + threshold)
    print("=" * 60)
    print("Step 1: Running run_laplacian_pipeline.py")
    print("=" * 60)
    step1 = subprocess.run(
        [python, '-u', str(script_dir / 'run_laplacian_pipeline.py'), args.data_dir,
         '--output-dir', output_dir_str]
    )
    if step1.returncode != 0:
        print("Step 1 failed, aborting.", file=sys.stderr)
        sys.exit(1)

    step1_csv = master_output_dir / 'laplacian_th' / 'laplacian_th_calculations.csv'
    if not step1_csv.exists():
        print(f"Error: {step1_csv} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Step 1 output: {step1_csv}\n")

    # Step 2: Concat blur data
    print("=" * 60)
    print("Step 2: Running concat_blur.py")
    print("=" * 60)
    step2_csv = concat_data_dir / 'laplacian_th_with_blur.xlsx'
    step2 = subprocess.run(
        [python, '-u', str(script_dir / 'concat_blur.py'), str(step1_csv), args.blur_xlsx,
         '-o', str(step2_csv)]
    )
    if step2.returncode != 0:
        print("Step 2 failed, aborting.", file=sys.stderr)
        sys.exit(1)
    if not step2_csv.exists():
        print(f"Error: {step2_csv} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Step 2 output: {step2_csv}\n")

    # Step 3: Run or use pre-computed is_measurable
    if args.is_measurable_csv:
        measurable_csv = args.is_measurable_csv
        print(f"Using pre-computed measurable CSV: {measurable_csv}\n")
    else:
        print("=" * 60)
        print("Step 3: Running run_is_measurable.py")
        print("=" * 60)
        cmd = [python, '-u', str(script_dir / 'run_is_measurable.py'),
               '--dataset', args.data_dir, '--blur-xlsx', args.blur_xlsx,
               '--output-dir', output_dir_str]
        if args.limit:
            cmd += ['--limit', str(args.limit)]
        step3 = subprocess.run(cmd)
        if step3.returncode != 0:
            print("Step 3 failed, aborting.", file=sys.stderr)
            sys.exit(1)

        measurable_csv = str(master_output_dir / 'run_is_measurable' / 'run_is_measurable.csv')
        if not Path(measurable_csv).exists():
            print(f"Error: {measurable_csv} not found.", file=sys.stderr)
            sys.exit(1)

        print(f"Step 3 output: {measurable_csv}\n")

    # Step 4: Concat is_measurable data
    print("=" * 60)
    print("Step 4: Running concat_with_is_measurable.py")
    print("=" * 60)
    step4_csv = concat_data_dir / 'laplacian_th_with_blur_and_measurable.xlsx'
    step4 = subprocess.run(
        [python, '-u', str(script_dir / 'concat_with_is_measurable.py'),
         str(step2_csv), measurable_csv, '-o', str(step4_csv)]
    )
    if step4.returncode != 0:
        print("Step 4 failed, aborting.", file=sys.stderr)
        sys.exit(1)
    if not step4_csv.exists():
        print(f"Error: {step4_csv} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Step 4 output: {step4_csv}\n")

    # Step 5: Detect issues
    print("=" * 60)
    print("Step 5: Running detect_issues.py")
    print("=" * 60)
    cmd = [python, '-u', str(script_dir / 'detect_issues.py'),
           '--dataset', args.data_dir, '--blur-xlsx', args.blur_xlsx,
           '--output-dir', output_dir_str]
    if args.limit:
        cmd += ['--limit', str(args.limit)]
    if args.examples_dir:
        cmd += ['--use-examples', args.examples_dir]
    step5 = subprocess.run(cmd)
    if step5.returncode != 0:
        print("Step 5 failed, aborting.", file=sys.stderr)
        sys.exit(1)

    detect_issues_csv = str(master_output_dir / 'detect_issues' / 'detect_issues.csv')
    if not Path(detect_issues_csv).exists():
        print(f"Error: {detect_issues_csv} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Step 5 output: {detect_issues_csv}\n")

    # Step 6: Concat detect_issues data
    print("=" * 60)
    print("Step 6: Running concat_with_detect_issues.py")
    print("=" * 60)
    step6_xlsx = concat_data_dir / 'laplacian_th_with_blur_measurable_issues.xlsx'
    step6 = subprocess.run(
        [python, '-u', str(script_dir / 'concat_with_detect_issues.py'),
         str(step4_csv), detect_issues_csv, '-o', str(step6_xlsx)]
    )
    if step6.returncode != 0:
        print("Step 6 failed, aborting.", file=sys.stderr)
        sys.exit(1)
    if not step6_xlsx.exists():
        print(f"Error: {step6_xlsx} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Step 6 output: {step6_xlsx}\n")

    # Step 7: Analyze blur severity
    print("=" * 60)
    print("Step 7: Running analyze_blur_severity.py")
    print("=" * 60)
    step7 = subprocess.run(
        [python, '-u', str(script_dir / 'analyze_blur_severity.py'), str(step6_xlsx),
         '--output-dir', output_dir_str]
    )
    if step7.returncode != 0:
        print("Step 7 failed.", file=sys.stderr)
        sys.exit(1)

    print(f"\nFull pipeline complete. All outputs in: {master_output_dir}")


if __name__ == '__main__':
    main()
