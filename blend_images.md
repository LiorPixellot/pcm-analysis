# Blend Images for Movement Analysis

## What was done

Two scripts were created to generate visual movement analysis images and add clickable links to the movement xlsx.

### `batch_blend.py`

Creates three panoramic images per venue/calibration in the movement directory:

1. **Current pano** (`_current.jpg`) — side-by-side of current camera view (CAM1 left, CAM0 right)
2. **Reference pano** (`_reference.jpg`) — side-by-side of calibration reference images (1.jpg left, 0.jpg right)
3. **Blend** (`_blend.jpg`) — 50/50 alpha overlay of current vs reference. Camera movement shows as ghosting/misalignment.

The blend technique is the same used in `proactive-camera-monitoring/camera_degradation_analyze.py`.

```bash
python3 batch_blend.py data/16_2_linux_s2
```

Output files are saved under each venue's movement directory:
```
data/16_2_linux_s2/<venue_id>/<ts>/movement/
    movement.json                                         (already existed)
    movement_<venue_id>_pano_<sport>_current.jpg         (new)
    movement_<venue_id>_pano_<sport>_reference.jpg       (new)
    movement_<venue_id>_pano_<sport>_blend.jpg           (new)
```

Idempotent — skips venues where all three images already exist.

### `add_blend_links_to_xlsx.py`

Adds three clickable `file:///` hyperlink columns to the movement xlsx:

- `current_image` — link to the current camera pano
- `reference_image` — link to the calibration reference pano
- `blend_image` — link to the blend overlay

```bash
python3 add_blend_links_to_xlsx.py data/16_2_linux_s2
```

## Results on `16_2_linux_s2`

| Metric | Count |
|--------|-------|
| Venues processed | 1,521 |
| Image sets created | 3,744 |
| Xlsx rows with hyperlinks | 2,839 |
| Rows without calibration images (no 0.jpg/1.jpg) | 3,527 |
| Rows without calibration data | 11,804 |
| Broken images (skipped) | 5 |

Only calibrations that have reference images (0.jpg + 1.jpg in `multisportcalibration/<sport>/`) get blend images. Many calibrations only have XML config files without reference images.
