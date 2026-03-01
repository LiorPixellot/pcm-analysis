#!/usr/bin/env python3
"""
Run the full Laplacian focus pipeline:
  1. laplacian_calculations.py — compute focus metrics
  2. laplacian_th_calculations.py — add severity classification
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
    parser.add_argument('--output-dir', default=None,
                        help='Parent output directory (passed to child scripts). If omitted, each step creates its own timestamped dir.')
    args = parser.parse_args()

    python = sys.executable
    script_dir = Path(__file__).parent.resolve()

    # Step 1: Laplacian calculations
    print("=" * 60)
    print("Step 1: Running laplacian_calculations.py")
    print("=" * 60)
    cmd1 = [python, '-u', str(script_dir / 'laplacian_calculations.py'), args.data_dir]
    if args.output_dir:
        cmd1 += ['--output-dir', args.output_dir]
    step1 = subprocess.run(cmd1)
    if step1.returncode != 0:
        print("Step 1 failed, aborting.", file=sys.stderr)
        sys.exit(1)

    # Find the CSV output from step 1
    if args.output_dir:
        step1_csv = Path(args.output_dir) / 'laplacian' / 'laplacian_calculations.csv'
    else:
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
    cmd2 = [python, '-u', str(script_dir / 'laplacian_th_calculations.py'), str(step1_csv)]
    if args.output_dir:
        cmd2 += ['--output-dir', args.output_dir]
    step2 = subprocess.run(cmd2)
    if step2.returncode != 0:
        print("Step 2 failed.", file=sys.stderr)
        sys.exit(1)

    # Find the CSV output from step 2
    if args.output_dir:
        step2_csv = Path(args.output_dir) / 'laplacian_th' / 'laplacian_th_calculations.csv'
    else:
        output_base = script_dir / 'output_dir'
        th_dirs = sorted(output_base.glob('laplacian_th_[0-9]*'), key=lambda p: p.name)
        if not th_dirs:
            print("Error: No laplacian_th output directory found.", file=sys.stderr)
            sys.exit(1)
        step2_csv = th_dirs[-1] / 'laplacian_th_calculations.csv'

    print(f"\nStep 2 output: {step2_csv}")
    print("\nPipeline complete.")


if __name__ == '__main__':
    main()
