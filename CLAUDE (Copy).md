

  Two new files, no existing files modified:

  1. run_annotation_pipeline.py — Iterates complete_annotation.csv, runs Laplacian severity + Gemini measurability + Gemini issue detection per row. Reuses existing
  analyze_indoor()/analyze_outdoor()/analyze_venue() functions. Outputs a predictions CSV with ground truth + predictions side by side. Supports --limit, --skip-gemini, --max-workers.
  2. confusion_metrics.py — Reads predictions CSV, computes confusion metrics for 3 dimensions (measurability, quality issue via detect_issues, severity). Generates TP/TN/FP/FN image dump
  directories with symlinks, confusion matrix heatmaps (matplotlib PNG), precision/recall/F1 charts, and a text report with indoor/outdoor breakdowns.

  Key design decisions:
  - All string comparisons normalized to lowercase, - and empty treated as "no value"
  - Quality issue GT compared against detect_issues predictions (not measurability side-output)
  - Image dumps use symlinks to save disk space
  - Rows with N/A/Error predictions excluded from confusion matrix, counted separately

  3 tasks (1 done, 1 in progress, 1 open)
  ◼ Write confusion_metrics.py
  ◻ Test both scripts with --limit and --skip-gemini
  ✔ Write run_annotation_pipeline.py

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Ready to code?

 Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Confusion Metrics System for Annotation Pipeline

 Context

 We have 335 manually annotated venue samples in complete_annotation.csv (across 5 dates: 2026-01-12 through 2026-02-09) with ground truth for:
 - Measurability: Is Measurable Focus Image (Yes/No) — 211 yes, 123 no
 - Quality issues: Has Quality Issue (Yes/No) + Quality Issue Type — focus (109), focus-with-trailing-space (4), condensation (1), none (9), dash (211)
 - Focus severity: Focus severity — OK, WARNING, ERROR, NA

 Images are at paths stored in columns cam0_image, cam1_image, joined_image (full absolute paths to annotation_data/<date>/<venue_id>/<event_id>/focus/).

 Goal: Run the pipeline (measurability + issue detection) against each annotated row, then compute precision/recall/F1 with image dumps and confusion matrix plots.

 Data Normalization Rules

 The annotation CSV has inconsistent casing/values that must be normalized before comparison:

 ┌────────────────────────────────────┬─────────────────────────────────────────────┬──────────────────────────────────────────────┐
 │               Column               │                 Raw values                  │                Normalization                 │
 ├────────────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────────┤
 │ Is Measurable Focus Image (Yes/No) │ yes, no (lowercase)                         │ .strip().lower() → compare as yes/no         │
 ├────────────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────────┤
 │ Has Quality Issue (Yes/No)         │ yes, no (lowercase)                         │ .strip().lower() → compare as yes/no         │
 ├────────────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────────┤
 │ Quality Issue Type                 │ focus, focus , condensation, none, -, empty │ .strip().lower(), treat -/empty/none as none │
 ├────────────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────────┤
 │ Indoor / Outdoor                   │ Indoor, Outdoor, indoor, outdoor            │ .strip().lower()                             │
 ├────────────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────────┤
 │ Focus severity                     │ OK, WARNING, ERROR, NA                      │ .strip().upper()                             │
 ├────────────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────────┤
 │ Pipeline predictions               │ Yes/No, Focus/None                          │ .strip().lower() before comparison           │
 └────────────────────────────────────┴─────────────────────────────────────────────┴──────────────────────────────────────────────┘

 All comparisons done in lowercase. Dashes (-) and empty strings treated as "no value".

 ---
 Script 1: run_annotation_pipeline.py

 Iterates complete_annotation.csv, runs the Gemini AI analysis per row, writes results CSV.

 Approach

 Import pipeline functions directly (no subprocess calls):

 ┌──────────────────────────────────────────────────────────────┬──────────────────────────────┬────────────────────────────────┐
 │                            Import                            │             From             │            Purpose             │
 ├──────────────────────────────────────────────────────────────┼──────────────────────────────┼────────────────────────────────┤
 │ load_config()                                                │ is_measurable.py             │ Load config.yaml               │
 ├──────────────────────────────────────────────────────────────┼──────────────────────────────┼────────────────────────────────┤
 │ load_images(focus_dir)                                       │ is_measurable.py             │ Load CAM0/CAM1/joined as bytes │
 ├──────────────────────────────────────────────────────────────┼──────────────────────────────┼────────────────────────────────┤
 │ load_model_pricing(), calculate_pricing(), get_token_usage() │ is_measurable.py             │ Cost tracking                  │
 ├──────────────────────────────────────────────────────────────┼──────────────────────────────┼────────────────────────────────┤
 │ analyze_indoor(), load_examples()                            │ is_measurable_indoor.py      │ Indoor measurability           │
 ├──────────────────────────────────────────────────────────────┼──────────────────────────────┼────────────────────────────────┤
 │ analyze_outdoor(), load_examples()                           │ is_measurable_outdoor.py     │ Outdoor measurability          │
 ├──────────────────────────────────────────────────────────────┼──────────────────────────────┼────────────────────────────────┤
 │ analyze_venue(), load_examples()                             │ detect_issues.py             │ Issue detection                │
 ├──────────────────────────────────────────────────────────────┼──────────────────────────────┼────────────────────────────────┤
 │ process_focus_folder()                                       │ laplacian_calculations.py    │ Laplacian focus metrics        │
 ├──────────────────────────────────────────────────────────────┼──────────────────────────────┼────────────────────────────────┤
 │ calc_focus_severity()                                        │ laplacian_th_calculations.py │ Severity classification        │
 └──────────────────────────────────────────────────────────────┴──────────────────────────────┴────────────────────────────────┘

 Processing Per Row

 1. Extract focus_dir from cam0_image column: Path(row['cam0_image']).parent
 2. Determine indoor/outdoor from Indoor / Outdoor column (normalized lowercase)
 3. Laplacian step: process_focus_folder(focus_dir) → metrics, then calc_focus_severity(...) → predicted severity
 4. Measurability step (Gemini): load_images(focus_dir) → get cam0/cam1 bytes (no joined — same as pipeline). Call analyze_indoor()/analyze_outdoor() based on venue type. Output:
 is_measurable, reason, has_quality_issue, quality_issue_type, which_camera
 5. Issue detection step (Gemini): same images + joined. Call analyze_venue(). Output: has_issue, issue_type, observation

 Output CSV Columns

 # Identity
 System ID, Checkup date, Calibration, Indoor / Outdoor, source_sheet

 # Ground truth (normalized from annotation)
 gt_measurable, gt_has_issue, gt_issue_type, gt_severity, gt_which_camera

 # Predictions - measurability (from analyze_indoor/analyze_outdoor)
 pred_measurable, pred_measurable_reason, pred_has_quality_issue, pred_quality_issue_type, pred_which_camera

 # Predictions - issue detection (from analyze_venue)
 pred_has_issue, pred_issue_type, pred_observation

 # Predictions - severity (from Laplacian)
 pred_severity

 # Laplacian metrics
 focus_right_mean, focus_left_mean, focus_right_mid, focus_left_mid, focus_abs_dif_rel

 # Image paths (for confusion_metrics.py to use)
 cam0_image, cam1_image, joined_image

 # Cost tracking
 measurable_tokens, issues_tokens

 CLI

 # Full run
 python run_annotation_pipeline.py --annotations complete_annotation.csv --output-dir output_dir/eval

 # Test with 5 rows
 python run_annotation_pipeline.py --annotations complete_annotation.csv --output-dir output_dir/eval --limit 5

 # Skip Gemini (Laplacian severity only)
 python run_annotation_pipeline.py --annotations complete_annotation.csv --output-dir output_dir/eval --skip-gemini

 # Limit workers
 python run_annotation_pipeline.py --annotations complete_annotation.csv --output-dir output_dir/eval --max-workers 5

 Flags

 - --annotations — path to CSV (default: complete_annotation.csv)
 - --output-dir — output directory (creates timestamped subdir if not specified)
 - --limit N — process first N rows only
 - --skip-gemini — skip Gemini calls, run only Laplacian severity prediction
 - --max-workers N — parallel Gemini workers (default from config.yaml)
 - --examples-dir — examples directory for detect_issues (default: examples/)

 Parallelization

 Use concurrent.futures.ThreadPoolExecutor (same pattern as run_is_measurable.py and detect_issues.py). Each worker processes one row: loads images, calls both Gemini APIs, returns results.

 Error Handling

 - If images don't exist: mark predictions as N/A
 - If Gemini fails: mark predictions as Error, log traceback, continue
 - Cost report written at end (same format as existing scripts)

 ---
 Script 2: confusion_metrics.py

 Reads the predictions CSV from Script 1, computes metrics, generates image dumps and plots.

 Three Evaluation Dimensions

 1. Measurability (binary Yes/No):
   - GT: gt_measurable
   - Pred: pred_measurable
   - Positive class = yes (measurable)
 2. Has Quality Issue (binary Yes/No):
   - GT: gt_has_issue
   - Pred: pred_has_issue (from detect_issues / analyze_venue())
   - Positive class = yes (has issue)
   - Note: we use detect_issues predictions, NOT the measurability step's has_quality_issue side-output
 3. Focus Severity (4-class: OK/Warning/Error/NA):
   - GT: gt_severity
   - Pred: pred_severity (from Laplacian)
   - Multi-class confusion matrix

 Metrics Computed

 For binary dimensions (measurability, quality issue):
 - TP, TN, FP, FN counts
 - Precision, Recall, F1-score, Accuracy
 - Breakdown by indoor/outdoor
 - Breakdown by source_sheet (POC version)

 For severity (multi-class):
 - 4x4 confusion matrix
 - Per-class precision/recall/F1
 - Overall accuracy

 Image Dumps

 <output_dir>/image_dumps/
 ├── measurability/
 │   ├── TP/   # pred=yes, gt=yes (correct: measurable)
 │   │   ├── <venue_id>_CAM0.jpg
 │   │   ├── <venue_id>_CAM1.jpg
 │   │   └── <venue_id>_joined.jpg
 │   ├── TN/   # pred=no, gt=no (correct: not measurable)
 │   ├── FP/   # pred=yes, gt=no (wrong: said measurable but isn't)
 │   └── FN/   # pred=no, gt=yes (wrong: said not measurable but is)
 ├── quality_issue/
 │   ├── TP/   # pred=yes, gt=yes
 │   ├── TN/   # pred=no, gt=no
 │   ├── FP/   # pred=yes, gt=no
 │   └── FN/   # pred=no, gt=no
 └── severity/
     ├── correct/    # pred_severity == gt_severity
     └── mismatched/ # subdirs like Error_vs_OK/ for each mismatch pair

 Images are symlinked (saves disk space). Filename format: <venue_id>_<image_type>.jpg (e.g., 5bebd54a4d2a230799c92df7_CAM0.jpg).

 Plots (matplotlib, saved as PNG)

 1. Confusion matrix heatmaps — one per dimension (2x2 for binary, 4x4 for severity), annotated with counts
 2. Precision/Recall/F1 bar chart — grouped bars per class
 3. Distribution comparison — predicted vs ground truth value counts side-by-side

 Text Report (metrics_report.txt)

 === MEASURABILITY ===
 TP: X  FP: X  FN: X  TN: X
 Precision: X.XX  Recall: X.XX  F1: X.XX  Accuracy: X.XX

 By venue type:
   Indoor:  P=X.XX R=X.XX F1=X.XX (N=X)
   Outdoor: P=X.XX R=X.XX F1=X.XX (N=X)

 === QUALITY ISSUE ===
 ...same format...

 === SEVERITY ===
 ...per-class metrics...

 CLI

 python confusion_metrics.py --predictions output_dir/eval/annotation_predictions.csv --output-dir output_dir/eval/metrics
 python confusion_metrics.py --predictions output_dir/eval/annotation_predictions.csv --output-dir output_dir/eval/metrics --skip-image-dumps

 Flags

 - --predictions — path to predictions CSV from Script 1
 - --output-dir — where to write metrics, plots, image dumps
 - --skip-image-dumps — compute metrics and plots only, no image copying

 Rows Excluded from Metrics

 - Rows where prediction is N/A or Error → excluded from confusion matrix, counted separately in report
 - For severity: rows with gt_severity = ABORTED → excluded

 ---
 Key Files to Create/Modify

 ┌────────────────────────────┬────────────────────────────────────────────────────┐
 │            File            │                       Action                       │
 ├────────────────────────────┼────────────────────────────────────────────────────┤
 │ run_annotation_pipeline.py │ CREATE — pipeline runner on annotation data        │
 ├────────────────────────────┼────────────────────────────────────────────────────┤
 │ confusion_metrics.py       │ CREATE — metrics computation + image dumps + plots │
 └────────────────────────────┴────────────────────────────────────────────────────┘

 No existing files modified.

 Verification

 1. python run_annotation_pipeline.py --annotations complete_annotation.csv --output-dir output_dir/eval_test --limit 5 --skip-gemini — test Laplacian-only
 2. python run_annotation_pipeline.py --annotations complete_annotation.csv --output-dir output_dir/eval_test --limit 5 — test with Gemini (5 rows)
 3. python confusion_metrics.py --predictions output_dir/eval_test/annotation_predictions.csv --output-dir output_dir/eval_test/metrics — verify metrics + image dumps + plots
 4. Check image dumps have correct images in TP/TN/FP/FN directories
 5. Open PNG plots to verify confusion matrices render correctly
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

 Claude has written up a plan and is ready to execute. Would you like to proceed?

 ❯ 1. Yes, clear context and auto-accept edits (shift+tab)
   2. Yes, auto-accept edits
   3. Yes, manually approve edits
   4. Type here to tell Claude what to change

 ctrl-g to edit in VS Code · ~/.claude/plans/joyful-beaming-toast.md



