#!/usr/bin/env python3
"""
Join laplacian threshold CSV with PQS blur-by-venue CSV on venue ID.

Usage:
    python concat_blur.py <laplacian_th.csv> <blur_by_venue.csv> [-o output.csv]
"""

import argparse
import csv
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Join laplacian threshold CSV with blur-by-venue CSV on venue ID'
    )
    parser.add_argument('laplacian_csv', help='Laplacian threshold calculations CSV (has venue_id column)')
    parser.add_argument('blur_csv', help='Blur by venue CSV (has PIXELLOT_VENUE_ID column)')
    parser.add_argument('-o', '--output', default='laplacian_th_with_blur.csv',
                        help='Output CSV file (default: laplacian_th_with_blur.csv)')
    args = parser.parse_args()

    # Load blur lookup: PIXELLOT_VENUE_ID -> row
    blur_lookup = {}
    blur_fields = []
    try:
        with open(args.blur_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            blur_fields = [col for col in reader.fieldnames if col != 'PIXELLOT_VENUE_ID']
            for row in reader:
                vid = row['PIXELLOT_VENUE_ID']
                if vid:
                    blur_lookup[vid] = {k: row[k] for k in blur_fields}
    except FileNotFoundError:
        print(f'Error: {args.blur_csv} not found', file=sys.stderr)
        sys.exit(1)
    except KeyError:
        print(f'Error: {args.blur_csv} missing PIXELLOT_VENUE_ID column', file=sys.stderr)
        sys.exit(1)

    print(f'Loaded {len(blur_lookup)} venues from {args.blur_csv}')

    # Join and write output
    matched = 0
    unmatched = 0

    try:
        with open(args.laplacian_csv, 'r', newline='') as infile:
            reader = csv.DictReader(infile)
            out_fields = reader.fieldnames + blur_fields

            with open(args.output, 'w', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=out_fields)
                writer.writeheader()

                for row in reader:
                    vid = row['venue_id']
                    if vid in blur_lookup:
                        row.update(blur_lookup[vid])
                        matched += 1
                    else:
                        for col in blur_fields:
                            row[col] = ''
                        unmatched += 1
                    writer.writerow(row)

    except FileNotFoundError:
        print(f'Error: {args.laplacian_csv} not found', file=sys.stderr)
        sys.exit(1)
    except KeyError:
        print(f'Error: {args.laplacian_csv} missing venue_id column', file=sys.stderr)
        sys.exit(1)

    print(f'Matched: {matched}')
    print(f'Unmatched (no blur data): {unmatched}')
    print(f'Output written to: {args.output}')


if __name__ == '__main__':
    main()
