#!/usr/bin/env python3
"""
Join laplacian threshold CSV with PQS blur-by-venue Excel and is_measurable CSV on venue ID.

Usage:
    python concat_with_is_measurable.py <laplacian_th.csv> <blur_by_venue.xlsx> <is_measurable.csv> [-o output.csv]
"""

import argparse
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Join laplacian threshold CSV with blur Excel and is_measurable CSV on venue ID'
    )
    parser.add_argument('laplacian_csv', help='Laplacian threshold calculations CSV (has venue_id column)')
    parser.add_argument('blur_xlsx', help='PQS blur by venue Excel (has PIXELLOT_VENUE_ID column)')
    parser.add_argument('is_measurable_csv', help='is_measurable CSV (has venue_id, is_measurable, has_quality_issue columns)')
    parser.add_argument('-o', '--output', default='laplacian_th_with_blur_and_measurable.csv',
                        help='Output CSV file (default: laplacian_th_with_blur_and_measurable.csv)')
    args = parser.parse_args()

    # Load laplacian data
    lap = pd.read_csv(args.laplacian_csv)
    print(f'Laplacian rows: {len(lap)}')

    # Load blur data
    blur = pd.read_excel(args.blur_xlsx)
    blur = blur.rename(columns={'PIXELLOT_VENUE_ID': 'venue_id'})
    print(f'Blur venues: {len(blur)}')

    # Load is_measurable data - keep only needed columns, dedupe by venue_id (first row)
    ism = pd.read_csv(args.is_measurable_csv)
    ism = ism[['venue_id', 'is_measurable', 'has_quality_issue']].drop_duplicates(subset='venue_id', keep='first')
    print(f'is_measurable unique venues: {len(ism)}')

    # Join laplacian with blur on venue_id (left join)
    merged = lap.merge(blur, on='venue_id', how='left')
    blur_matched = merged['AVERAGE_BLUR_AVG'].notna().sum()
    print(f'Blur matched: {blur_matched}, unmatched: {len(merged) - blur_matched}')

    # Join with is_measurable on venue_id (left join)
    merged = merged.merge(ism, on='venue_id', how='left')
    ism_matched = merged['is_measurable'].notna().sum()
    print(f'is_measurable matched: {ism_matched}, unmatched: {len(merged) - ism_matched}')

    # Write output
    merged.to_csv(args.output, index=False)
    print(f'Total rows: {len(merged)}')
    print(f'Output written to: {args.output}')


if __name__ == '__main__':
    main()
