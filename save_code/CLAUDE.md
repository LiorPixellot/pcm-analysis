# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PCM (Proactive Camera Measurement) is a system for analyzing camera setup quality, focus metrics, and image degradation at sports venues. It combines legacy C++ analysis, Python data collection, OpenCV-based metrics, and Google Gemini AI for quality assessment.

**Primary Technologies:** Python 3, C++ (legacy binary), Google Gemini AI, AWS S3, OpenCV

## Development Setup

### Install Dependencies

```bash
pip install -r scrapanyzer/requirements.txt
```

### Environment Configuration

- **GCP credentials:** Place `service-account-key.json` in the main directory or set `GOOGLE_APPLICATION_CREDENTIALS`
- **Gemini config:** Edit `config.yaml` for model settings (default: `gemini-3-flash-preview`)
- **AWS credentials:** Use environment variables or IAM roles (avoid hardcoding)

## Running Analysis

### Main Data Collection (Windows)
```bash
python scrapanyzer/camera_degradation_collect.py
```
This orchestrates the full pipeline: captures images from cameras, runs setup/focus/movement analysis, uploads to S3.

### Focus Analysis Pipeline
```bash
python full_flow.py /path/to/data_dir
```
Runs the complete focus analysis: Laplacian calculations, threshold classification, Gemini measurability analysis, and result concatenation.

### Setup Analysis (calls C++ binary)
```bash
python scrapanyzer/run_all_setup.py
```

## Architecture

### Data Flow

1. **Collection** (`camera_degradation_collect.py`): Captures from RTSP/HTTP camera streams (CAM0, CAM1, OCR)
2. **Analysis** runs in parallel:
   - Focus: `run_all_focus.py` - Laplacian variance on 800x800 regions
   - Setup: `run_all_setup.py` - calls `ProactiveCameraSetupMeasure.exe`
   - Movement: `camera_degradation_analyze.py` - optical flow (Lucas-Kanade)
3. **Upload**: Results go to S3 bucket `venue-logs` at `autofixes-reports/camera-degradation/...`

### Key Modules

| Module | Purpose |
|--------|---------|
| `scrapanyzer/pxl_utils.py` | Utilities: archive extraction, config loading, Papertrail |
| `scrapanyzer/aws_utils.py` | S3 upload functions for reports |
| `scrapanyzer/image_movement_calc.py` | Optical flow movement detection |
| `is_measurable.py` | Gemini-based measurability and focus analysis |
| `full_flow.py` | Complete focus analysis pipeline |

### Data Directory Structure

```
data_13/{venue_id}/{event_id}/
├── focus/          # Focus images (CAM0_1.jpg, CAM1_1.jpg) & focus.json
├── setup/          # Setup analysis results
├── movement/       # Movement analysis results
└── log.txt         # Execution logs (JSON format)
```

### Output Formats

- **CSV:** Focus results with headers (venue_id, event_id, cam0_focus, cam1_focus, severity, etc.)
- **JSON:** focus.json with metrics, config, token usage
- **Images:** side-by-side comparisons, blended images

## Cursor Rules

This project follows rules in `.cursor/rules/remote/`:
- `baseline.mdc`: Core coding guidelines (concise communication, no hardcoded secrets, minimal edits)
- `pr-guidelines.mdc`: PR format `[<Jira>-<Issue>] <summary>`, one logical change per PR
- `docker.mdc`: Use minimal base images, multi-stage builds, non-root user
