#!/usr/bin/env python3
"""Download S3 annotation data for venues listed in complete_annotation.csv."""

import csv
import subprocess
import sys
from pathlib import Path

CSV_PATH = Path("complete_annotation.csv")
OUTPUT_DIR = Path("annotation_data")
S3_BUCKET = "s3://venue-logs/autofixes-reports/camera-degradation"


def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Deduplicate by (date, venue_id) in case of duplicate rows
    seen = set()
    unique_rows = []
    for row in rows:
        date_str = row["Checkup date"].split(" ")[0]  # "2026-01-12 00:00:00" -> "2026-01-12"
        venue_id = row["System ID"].strip()
        key = (date_str, venue_id)
        if key not in seen:
            seen.add(key)
            unique_rows.append((date_str, venue_id))

    total = len(unique_rows)
    print(f"Found {total} unique (date, venue) pairs to download")

    downloaded = 0
    skipped = 0
    failed = 0

    for i, (date_str, venue_id) in enumerate(unique_rows, 1):
        s3_path = f"{S3_BUCKET}/{date_str}/{venue_id}/"
        local_path = OUTPUT_DIR / date_str / venue_id

        print(f"[{i}/{total}] {venue_id} ({date_str})... ", end="", flush=True)

        result = subprocess.run(
            ["aws", "s3", "cp", "--recursive", s3_path, str(local_path) + "/"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "NoSuchKey" in stderr or "does not exist" in stderr:
                print("SKIPPED (not found on S3)")
                skipped += 1
            else:
                print(f"FAILED: {stderr}")
                failed += 1
        elif not result.stdout.strip():
            print("SKIPPED (empty)")
            skipped += 1
        else:
            file_count = result.stdout.strip().count("\n") + 1
            print(f"OK ({file_count} files)")
            downloaded += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
