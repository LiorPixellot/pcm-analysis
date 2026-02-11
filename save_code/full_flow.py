#!/usr/bin/env python3
"""
Full Flow: Combined pipeline for camera focus analysis.

Steps:
1. Calculate Laplacian focus metrics from dataset directory
2. Add Focus_severity thresholds (pass/fail criteria)
3. Check if frames are measurable using Gemini AI
4. Concatenate results with annotations from Excel

Usage:
    python full_flow.py /path/to/data_dir --limit 5        # Test with 5 events
    python full_flow.py /path/to/data_dir                   # Full run
    python full_flow.py /path/to/data_dir --venues venues.yaml  # Process only venues from YAML
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import cv2

# Try to import pandas for concatenation step
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()

# Defaults for annotations Excel (overridden by config.yaml)
DEFAULT_EXCEL_FILE = "Annotators - Working Table 2025.xlsx"
DEFAULT_SHEET_NAME = "POC"
DEFAULT_JOIN_KEY = "System ID"

# Output directory and files (set at runtime by init_output_dir)
OUTPUT_DIR: Path = None
LAPLACIAN_CSV: Path = None
THRESHOLD_CSV: Path = None
MEASURABLE_CSV: Path = None
CONCATENATED_XLSX: Path = None
COST_FILE: Path = None
CONFIG_FILE: Path = None


def init_output_dir(existing_dir: Optional[Path] = None) -> Path:
    """Create timestamped output directory and set output file paths.

    Args:
        existing_dir: If provided, use this directory instead of creating new one.
    """
    global OUTPUT_DIR, LAPLACIAN_CSV, THRESHOLD_CSV, MEASURABLE_CSV, CONCATENATED_XLSX, COST_FILE, CONFIG_FILE

    if existing_dir:
        OUTPUT_DIR = existing_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        OUTPUT_DIR = SCRIPT_DIR / "output_dir" / f"full_flow_{timestamp}"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LAPLACIAN_CSV = OUTPUT_DIR / "laplacian_calculations.csv"
    THRESHOLD_CSV = OUTPUT_DIR / "laplacian_th.csv"
    MEASURABLE_CSV = OUTPUT_DIR / "is_measurable.csv"
    CONCATENATED_XLSX = OUTPUT_DIR / "concatenated_results.xlsx"
    COST_FILE = OUTPUT_DIR / "cost.txt"
    CONFIG_FILE = OUTPUT_DIR / "run_config.json"

    return OUTPUT_DIR


def copy_source_files(output_dir: Path):
    """Copy Python scripts and config files to output directory for reproducibility."""
    source_dir = output_dir / "source"
    source_dir.mkdir(exist_ok=True)

    # Copy .py files
    for py_file in SCRIPT_DIR.glob("*.py"):
        shutil.copy2(py_file, source_dir / py_file.name)

    # Copy .yaml and .cfg files
    for pattern in ["*.yaml", "*.cfg"]:
        for cfg_file in SCRIPT_DIR.glob(pattern):
            shutil.copy2(cfg_file, source_dir / cfg_file.name)

    print(f"  Source files copied to: {source_dir}")


def save_run_config(data_dir: Path, limit: Optional[int], venues_filter: Optional[Set[str]],
                    flow_config: Dict[str, bool], venues_path: Optional[Path] = None,
                    annotations_config: Optional[Dict] = None):
    """Save run configuration to JSON file."""
    config = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir.resolve()),
        "limit": limit,
        "venues_filter": list(venues_filter) if venues_filter else None,
        "venues_file": str(venues_path) if venues_path and venues_path.exists() else None,
        "flow_config": flow_config,
        "annotations": annotations_config or {},
        "output_dir": str(OUTPUT_DIR),
        "output_files": {
            "laplacian_csv": str(LAPLACIAN_CSV),
            "threshold_csv": str(THRESHOLD_CSV),
            "measurable_csv": str(MEASURABLE_CSV),
            "concatenated_xlsx": str(CONCATENATED_XLSX),
            "cost_file": str(COST_FILE)
        }
    }

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  Run config saved: {CONFIG_FILE}")



# ============================================================================
# VENUE FILTERING
# ============================================================================

def load_venues_from_yaml(yaml_path: Path) -> Optional[Set[str]]:
    """Load venue IDs from a YAML file.

    Returns None if 'full: true' is set (process all venues).

    Expected YAML format:
        venues:
          - 5bed37433e663c077bc36cef
          - 5bed375c3e663c077bc36cf1
          - 5bed84843e663c077bc36daf

    Or simple list format:
        - 5bed37433e663c077bc36cef
        - 5bed375c3e663c077bc36cf1

    Or full mode:
        full: true
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if data is None:
        return set()

    # Check for "full: true" option - return None to process all venues
    if isinstance(data, dict) and data.get('full', False):
        return None

    # Handle both formats
    if isinstance(data, list):
        venues = data
    elif isinstance(data, dict) and 'venues' in data:
        venues = data['venues']
    else:
        raise ValueError(f"Invalid YAML format. Expected list or dict with 'venues' key.")

    return set(str(v).strip() for v in venues if v)


def load_flow_config(flow_path: Path) -> Dict[str, bool]:
    """Load flow configuration from YAML file.

    Returns dict with step names and whether they should run.
    Default: all steps enabled.
    """
    default_config = {
        "laplacian": True,
        "threshold": True,
        "is_measurable": True,
        "concatenate": True,
    }

    if not flow_path.exists():
        return default_config

    with open(flow_path, 'r') as f:
        data = yaml.safe_load(f) or {}

    steps = data.get("steps", {})

    # Merge with defaults (unknown keys ignored)
    for key in default_config:
        if key in steps:
            default_config[key] = bool(steps[key])

    return default_config


# ============================================================================
# STEP 1: LAPLACIAN CALCULATIONS
# ============================================================================

def image_focus(image_view):
    """Calculate focus metrics for an image region."""
    laplacian_image = cv2.Laplacian(image_view, cv2.CV_64F)
    mean_l, stddev = cv2.meanStdDev(laplacian_image)
    variance = stddev[0].mean() * stddev[0].mean()
    mean_i = image_view.mean()
    return mean_i, variance


def extract_region(image, side, joined_frame_size=800):
    """Extract 800x800 region from center of image."""
    center_y, size_x, _ = image.shape
    center_y = center_y // 2

    if side == 'right':
        region = image[center_y - joined_frame_size // 2:center_y + joined_frame_size // 2,
                       0:joined_frame_size]
    else:
        region = image[center_y - joined_frame_size // 2:center_y + joined_frame_size // 2,
                       size_x - joined_frame_size:size_x]
    return region


def process_focus_folder(focus_folder: Path) -> Optional[Dict]:
    """Process a single focus folder and return calculated metrics."""
    cam0_path = focus_folder / 'CAM0_1.jpg'
    cam1_path = focus_folder / 'CAM1_1.jpg'

    if not cam0_path.exists() or not cam1_path.exists():
        return None

    img_right = cv2.imread(str(cam0_path))
    img_left = cv2.imread(str(cam1_path))

    if img_right is None or img_left is None:
        return None

    region_right = extract_region(img_right, 'right')
    region_left = extract_region(img_left, 'left')

    right_mean, right_mid_focus = image_focus(region_right)
    left_mean, left_mid_focus = image_focus(region_left)

    focus_abs_dif = abs(right_mid_focus - left_mid_focus)
    focus_abs_dif_rel = focus_abs_dif / (right_mid_focus + left_mid_focus) * 2

    return {
        'focus_right_mean': right_mean,
        'focus_left_mean': left_mean,
        'focus_right_mid': right_mid_focus,
        'focus_left_mid': left_mid_focus,
        'focus_abs_dif_rel': focus_abs_dif_rel
    }


def find_focus_folders(data_dir: Path):
    """Find all focus folders under data_dir."""
    for venue_dir in sorted(data_dir.iterdir()):
        if not venue_dir.is_dir():
            continue
        for event_dir in sorted(venue_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            focus_folder = event_dir / 'focus'
            if focus_folder.exists() and focus_folder.is_dir():
                yield venue_dir.name, event_dir.name, focus_folder


def validate_against_json(calculated, focus_json_path, tolerance=0.001):
    """Validate calculated values against focus.json."""
    if not focus_json_path.exists():
        return 'NO_JSON', 'focus.json not found'

    try:
        with open(focus_json_path) as f:
            expected = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return 'JSON_ERROR', str(e)

    mismatches = []
    fields = ['focus_right_mean', 'focus_left_mean', 'focus_right_mid', 'focus_left_mid', 'focus_abs_dif_rel']

    for field in fields:
        if field not in expected:
            mismatches.append(f'{field}: missing in JSON')
            continue

        calc_val = calculated[field]
        exp_val = expected[field]

        if exp_val == 0:
            if abs(calc_val) >= tolerance:
                mismatches.append(f'{field}: calc={calc_val:.6f} exp={exp_val:.6f}')
        elif abs(calc_val - exp_val) / abs(exp_val) >= tolerance:
            diff_pct = abs(calc_val - exp_val) / abs(exp_val) * 100
            mismatches.append(f'{field}: calc={calc_val:.6f} exp={exp_val:.6f} ({diff_pct:.2f}% diff)')

    if mismatches:
        return 'FAIL', '; '.join(mismatches)
    return 'PASS', ''


def step1_laplacian_calculations(data_dir: Path, limit: Optional[int] = None,
                                  venues_filter: Optional[Set[str]] = None) -> int:
    """Step 1: Calculate Laplacian focus metrics."""
    print("\n" + "=" * 60)
    print("  STEP 1: LAPLACIAN CALCULATIONS")
    print("=" * 60)

    if venues_filter:
        print(f"  Filtering to {len(venues_filter)} venues")

    fieldnames = [
        'venue_id', 'event_id', 'focus_right_mean', 'focus_left_mean',
        'focus_right_mid', 'focus_left_mid', 'focus_abs_dif_rel',
        'validation_status', 'validation_details'
    ]

    processed = 0
    skipped = 0

    with open(LAPLACIAN_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for venue_id, event_id, focus_folder in find_focus_folders(data_dir):
            # Filter by venue if specified
            if venues_filter and venue_id not in venues_filter:
                continue

            if limit and processed >= limit:
                break

            metrics = process_focus_folder(focus_folder)

            if metrics is None:
                skipped += 1
                continue

            focus_json_path = focus_folder / 'focus.json'
            status, details = validate_against_json(metrics, focus_json_path)

            row = {
                'venue_id': venue_id,
                'event_id': event_id,
                'focus_right_mean': metrics['focus_right_mean'],
                'focus_left_mean': metrics['focus_left_mean'],
                'focus_right_mid': metrics['focus_right_mid'],
                'focus_left_mid': metrics['focus_left_mid'],
                'focus_abs_dif_rel': metrics['focus_abs_dif_rel'],
                'validation_status': status,
                'validation_details': details
            }
            writer.writerow(row)
            processed += 1

            if processed % 50 == 0:
                print(f"  Processed: {processed}")

    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Output: {LAPLACIAN_CSV}")

    return processed


# ============================================================================
# STEP 2: THRESHOLD CALCULATIONS
# ============================================================================

def worst_severity(a, b):
    """Return the worse severity."""
    rank = {"Ok": 0, "Warning": 1, "Error": 2}
    return a if rank[a] >= rank[b] else b


def mid_severity(mid):
    """Calculate severity from mid focus value."""
    if mid <= 10:
        return "Error"
    if mid <= 20:
        return "Warning"
    return "Ok"


def calc_focus_severity(focus_right_mean, focus_left_mean, focus_abs_dif_rel,
                        focus_right_mid, focus_left_mid):
    """Calculate focus severity based on thresholds."""
    if focus_right_mean < 70 or focus_left_mean < 70:
        return "NA"

    if focus_abs_dif_rel > 1.25:
        sev_diff = "Error"
    elif focus_abs_dif_rel >= 0.70:
        sev_diff = "Warning"
    else:
        sev_diff = "Ok"

    sev_mid_right = mid_severity(focus_right_mid)
    sev_mid_left = mid_severity(focus_left_mid)
    sev_mid = worst_severity(sev_mid_right, sev_mid_left)

    return worst_severity(sev_diff, sev_mid)


def step2_threshold_calculations() -> int:
    """Step 2: Add Focus_severity column based on thresholds."""
    print("\n" + "=" * 60)
    print("  STEP 2: THRESHOLD CALCULATIONS")
    print("=" * 60)

    with open(LAPLACIAN_CSV, 'r', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['Focus_severity']

        rows = []
        counts = {"NA": 0, "Ok": 0, "Warning": 0, "Error": 0}

        for row in reader:
            severity = calc_focus_severity(
                float(row['focus_right_mean']),
                float(row['focus_left_mean']),
                float(row['focus_abs_dif_rel']),
                float(row['focus_right_mid']),
                float(row['focus_left_mid'])
            )
            row['Focus_severity'] = severity
            rows.append(row)
            counts[severity] += 1

    with open(THRESHOLD_CSV, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  NA: {counts['NA']}")
    print(f"  Ok: {counts['Ok']}")
    print(f"  Warning: {counts['Warning']}")
    print(f"  Error: {counts['Error']}")
    print(f"  Output: {THRESHOLD_CSV}")

    return len(rows)


# ============================================================================
# STEP 3: IS MEASURABLE (GEMINI) - calls is_measurable.py subprocess
# ============================================================================

def parse_cost_file() -> Dict[str, float]:
    """Parse cost.txt to extract pricing information."""
    if not COST_FILE.exists():
        return {"input_price": 0.0, "output_price": 0.0, "total_price": 0.0}

    pricing = {"input_price": 0.0, "output_price": 0.0, "total_price": 0.0}
    try:
        with open(COST_FILE, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'Input cost:' in line:
                    pricing["input_price"] = float(line.split('$')[1].strip())
                elif 'Output cost:' in line:
                    pricing["output_price"] = float(line.split('$')[1].strip())
                elif 'Total cost:' in line:
                    pricing["total_price"] = float(line.split('$')[1].strip())
    except (ValueError, IndexError):
        pass
    return pricing


def step3_is_measurable(data_dir: Path) -> Tuple[int, Dict[str, float]]:
    """Step 3: Run is_measurable.py to check if frames are measurable."""
    print("\n" + "=" * 60)
    print("  STEP 3: IS MEASURABLE (GEMINI)")
    print("=" * 60)

    # Call is_measurable.py as subprocess
    is_measurable_script = SCRIPT_DIR / "is_measurable.py"
    cmd = [
        sys.executable,
        str(is_measurable_script),
        "--input", str(THRESHOLD_CSV),
        "--output", str(MEASURABLE_CSV),
        "--data-dir", str(data_dir),
        "--cost-file", str(COST_FILE)
    ]

    print(f"  Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))

    if result.returncode != 0:
        print(f"  ERROR: is_measurable.py exited with code {result.returncode}")
        return 0, {"total_price": 0.0}

    # Parse cost from cost.txt written by is_measurable.py
    pricing = parse_cost_file()

    # Count processed rows from output CSV
    processed_count = 0
    if MEASURABLE_CSV.exists():
        with open(MEASURABLE_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("is_measurable") not in ("N/A", "Error", ""):
                    processed_count += 1

    return processed_count, pricing


# ============================================================================
# STEP 4: CONCATENATE WITH ANNOTATIONS
# ============================================================================

# Focus metrics columns (from Laplacian calculations)
FOCUS_COLUMNS = {
    "focus_right_mean": "focus_right_mean",
    "focus_left_mean": "focus_left_mean",
    "focus_right_mid": "focus_right_mid",
    "focus_left_mid": "focus_left_mid",
    "focus_abs_dif_rel": "focus_abs_dif_rel",
    "Focus_severity": "focus_severity",
}

# Quality metrics columns (from Gemini analysis)
QUALITY_COLUMNS = {
    "is_measurable": "quality_is_measurable",
    "not_measurable_reason": "quality_not_measurable_reason",
    "has_quality_issue": "quality_has_issue",
    "quality_issue_type": "quality_issue_type",
    "which_camera": "quality_which_camera",
}



def clean_column_name(col_name):
    """Clean column name by replacing newlines with spaces."""
    if col_name is None:
        return col_name
    clean = str(col_name).replace('\n', ' ').replace('\r', ' ')
    while '  ' in clean:
        clean = clean.replace('  ', ' ')
    return clean.strip()


def step4_concatenate(annotations_config: Optional[Dict] = None) -> int:
    """Step 4: Concatenate results with annotations from Excel.

    Reads from two separate files:
    - laplacian_th.csv: Focus metrics (focus_* columns)
    - is_measurable.csv: Gemini quality results (quality_* columns)
    """
    print("\n" + "=" * 60)
    print("  STEP 4: CONCATENATE WITH ANNOTATIONS")
    print("=" * 60)

    if not HAS_PANDAS:
        print("  SKIPPED: pandas not installed")
        return 0

    # Get annotations settings from config with defaults
    ann = annotations_config or {}
    excel_file = SCRIPT_DIR / ann.get("excel_file", DEFAULT_EXCEL_FILE)
    sheet_name = ann.get("sheet_name", DEFAULT_SHEET_NAME)
    join_key = ann.get("join_key", DEFAULT_JOIN_KEY)

    if not excel_file.exists():
        print(f"  SKIPPED: Excel file not found: {excel_file}")
        return 0

    # Read Excel
    df_excel = pd.read_excel(excel_file, sheet_name=sheet_name)
    print(f"  Excel rows: {len(df_excel)}")

    # Read laplacian_th.csv for focus metrics
    df_laplacian = pd.read_csv(THRESHOLD_CSV)
    print(f"  Laplacian rows: {len(df_laplacian)}")

    # Read is_measurable.csv for Gemini quality results
    # keep_default_na=False prevents "None" strings from becoming NaN
    df_measurable = pd.read_csv(MEASURABLE_CSV, keep_default_na=False)
    print(f"  Measurable rows: {len(df_measurable)}")

    # Create lookup for focus metrics from laplacian_th.csv
    focus_lookup = {}
    for _, row in df_laplacian.iterrows():
        venue_id = row["venue_id"]
        if venue_id not in focus_lookup:
            data = {}
            for src_col, dst_col in FOCUS_COLUMNS.items():
                if src_col in row:
                    data[dst_col] = row[src_col]
            focus_lookup[venue_id] = data

    # Create lookup for quality metrics from is_measurable.csv
    quality_lookup = {}
    for _, row in df_measurable.iterrows():
        venue_id = row["venue_id"]
        if venue_id not in quality_lookup:
            data = {}
            for src_col, dst_col in QUALITY_COLUMNS.items():
                if src_col in row:
                    data[dst_col] = row[src_col]
            quality_lookup[venue_id] = data

    # Add new columns to Excel dataframe
    all_new_columns = list(FOCUS_COLUMNS.values()) + list(QUALITY_COLUMNS.values())
    for col in all_new_columns:
        if col not in df_excel.columns:
            df_excel[col] = None

    # Match and fill
    focus_matched = 0
    quality_matched = 0
    for idx, row in df_excel.iterrows():
        system_id = row[join_key]
        if pd.isna(system_id):
            continue

        system_id = str(system_id).strip()

        # Fill focus metrics from laplacian_th.csv
        if system_id in focus_lookup:
            for col_name, value in focus_lookup[system_id].items():
                df_excel.at[idx, col_name] = value
            focus_matched += 1

        # Fill quality metrics from is_measurable.csv (Gemini predictions)
        # Note: Original Excel columns are preserved as human annotations
        if system_id in quality_lookup:
            result = quality_lookup[system_id]
            for col_name, value in result.items():
                df_excel.at[idx, col_name] = value
            quality_matched += 1

    # Clean column names (remove newlines) before saving
    df_excel.columns = [clean_column_name(col) for col in df_excel.columns]

    # Save as Excel
    df_excel.to_excel(CONCATENATED_XLSX, index=False)

    print(f"  Matched rows (focus): {focus_matched}")
    print(f"  Matched rows (quality): {quality_matched}")
    print(f"  New columns: focus_* ({len(FOCUS_COLUMNS)}), quality_* ({len(QUALITY_COLUMNS)})")
    print(f"  Output: {CONCATENATED_XLSX}")

    return max(focus_matched, quality_matched)


# ============================================================================
# MAIN
# ============================================================================

def load_config() -> Dict:
    """Load configuration from config.yaml."""
    config_path = SCRIPT_DIR / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(
        description='Full flow: Laplacian -> Thresholds -> Measurable -> Concatenate'
    )
    parser.add_argument('data_dir', nargs='?', default=None,
                        help='Path to data directory (default: from config.yaml)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of events to process')
    parser.add_argument('--flow', type=str, default=None,
                        help='YAML file with flow config (default: flow.yaml if exists)')
    parser.add_argument('--venues', type=str, default=None,
                        help='YAML file with list of venue IDs (default: venues.yaml if exists)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Use existing output directory (default: create new timestamped dir)')
    args = parser.parse_args()

    # Load config
    config = load_config()

    # Get data_dir from args or config
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir_str = config.get("data_dir", "data_13")
        data_dir = SCRIPT_DIR / data_dir_str

    if not data_dir.is_dir():
        print(f'Error: {data_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    # Load venues filter (default: venues.yaml if exists)
    venues_filter = None
    venues_path = Path(args.venues) if args.venues else SCRIPT_DIR / "venues.yaml"

    if venues_path.exists():
        venues_filter = load_venues_from_yaml(venues_path)
        if venues_filter is None:
            print(f"Venues: ALL (full mode)")
        else:
            print(f"Loaded {len(venues_filter)} venues from {venues_path}")
    elif args.venues:
        # User specified a file but it doesn't exist
        print(f'Error: Venues file not found: {args.venues}', file=sys.stderr)
        sys.exit(1)

    # Load flow configuration
    flow_path = Path(args.flow) if args.flow else SCRIPT_DIR / "flow.yaml"
    flow_config = load_flow_config(flow_path)

    # Initialize output directory (use existing or create new timestamped)
    existing_dir = Path(args.output_dir) if args.output_dir else None
    output_dir = init_output_dir(existing_dir)

    print("=" * 60)
    print("  FULL FLOW - CAMERA FOCUS ANALYSIS")
    print("=" * 60)
    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {output_dir}")
    if args.limit:
        print(f"  Limit: {args.limit} events")
    if venues_filter:
        print(f"  Venues filter: {len(venues_filter)} venues")
    print(f"  Steps: laplacian={flow_config['laplacian']}, threshold={flow_config['threshold']}, "
          f"is_measurable={flow_config['is_measurable']}, concatenate={flow_config['concatenate']}")
    print("=" * 60)

    # Get annotations config
    annotations_config = config.get("annotations", {})

    # Save run configuration
    save_run_config(data_dir, args.limit, venues_filter, flow_config, venues_path, annotations_config)
    copy_source_files(OUTPUT_DIR)

    # Step 1: Laplacian calculations
    if flow_config["laplacian"]:
        count1 = step1_laplacian_calculations(data_dir, args.limit, venues_filter)
        if count1 == 0:
            print("\nNo events processed. Exiting.")
            sys.exit(1)
    else:
        print("\n  STEP 1: SKIPPED (flow.yaml)")
        count1 = 0

    # Step 2: Threshold calculations
    if flow_config["threshold"]:
        count2 = step2_threshold_calculations()
    else:
        print("\n  STEP 2: SKIPPED (flow.yaml)")

    # Step 3: Is Measurable (Gemini)
    if flow_config["is_measurable"]:
        count3, total_cost = step3_is_measurable(data_dir)
    else:
        print("\n  STEP 3: SKIPPED (flow.yaml)")
        total_cost = {"total_price": 0.0}

    # Step 4: Concatenate
    if flow_config["concatenate"]:
        step4_concatenate(annotations_config)
    else:
        print("\n  STEP 4: SKIPPED (flow.yaml)")

    # Final summary
    print("\n" + "=" * 60)
    print("  COMPLETE")
    print("=" * 60)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Output files:")
    print(f"    - {CONFIG_FILE.name}")
    print(f"    - {LAPLACIAN_CSV.name}")
    print(f"    - {THRESHOLD_CSV.name}")
    if flow_config["is_measurable"]:
        print(f"    - {MEASURABLE_CSV.name}")
        print(f"    - {COST_FILE.name}")
    if flow_config["concatenate"]:
        print(f"    - {CONCATENATED_XLSX.name}")
    print(f"\n  Total Gemini cost: ${total_cost['total_price']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
