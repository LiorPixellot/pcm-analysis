#!/usr/bin/env python3
"""
Create blend images for each venue × calibration in a data directory.

For each venue with a movement.json, blends the current camera panorama
(CAM1 left, CAM0 right) with the calibration reference panorama (1.jpg left, 0.jpg right)
at 50% alpha. This makes camera movement visually obvious as ghosting/misalignment.

Output: <data_dir>/<venue>/<ts>/movement/movement_<venue>_pano_<sport>_blend.jpg

Usage:
    python batch_blend.py <data_dir>
    python batch_blend.py data/16_2_linux_s2
"""

import json
import os
import sys
from pathlib import Path

from PIL import Image


def side_by_side(images: list[Image.Image]) -> Image.Image:
    """Concatenate images horizontally, resizing all to the same height."""
    if not images:
        raise ValueError("No images")
    if len(images) == 1:
        return images[0]
    target_h = min(img.size[1] for img in images)
    resized = []
    for img in images:
        if img.size[1] != target_h:
            img = img.resize((int(img.size[0] * target_h / img.size[1]), target_h))
        resized.append(img)
    total_width = sum(img.size[0] for img in resized)
    new_im = Image.new("RGB", (total_width, target_h))
    x_offset = 0
    for img in resized:
        new_im.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    return new_im


def find_event_with_movement(venue_path: Path):
    """Find the event directory that contains movement/movement.json."""
    for entry in sorted(venue_path.iterdir()):
        if entry.is_dir():
            mov_json = entry / "movement" / "movement.json"
            if mov_json.exists():
                return entry, mov_json
    return None, None


def find_cam_image(directory: Path, cam: str) -> Path | None:
    """Find CAM0_1.jpg or CAM1_1.jpg with _0.jpg and _rot.jpg fallbacks."""
    for suffix in ["_1.jpg", "_1_rot.jpg", "_0.jpg", "_0_rot.jpg"]:
        path = directory / f"{cam}{suffix}"
        if path.exists():
            return path
    return None


def process_venue(venue_path: Path):
    """Create blend images for all calibrations in a venue."""
    venue_id = venue_path.name
    event_dir, mov_json = find_event_with_movement(venue_path)
    if event_dir is None:
        return 0, 0

    collected = event_dir / "collected"
    cam0_path = find_cam_image(collected, "CAM0")
    cam1_path = find_cam_image(collected, "CAM1")
    if cam0_path is None or cam1_path is None:
        print(f"  {venue_id}: missing camera images in {collected}")
        return 0, 0
    cam2_path = find_cam_image(collected, "CAM2")  # optional 3rd camera

    # Load movement.json to get calibration list
    try:
        with open(mov_json) as f:
            mov_data = json.load(f)
    except Exception as e:
        print(f"  {venue_id}: error reading {mov_json}: {e}")
        return 0, 0

    # Get unique calibrations
    calibrations = set()
    for m in mov_data.get("movements", []):
        calib = m.get("calibration", "").lower()
        if calib:
            calibrations.add(calib)

    if not calibrations:
        return 0, 0

    movement_dir = event_dir / "movement"
    msc_dir = collected / "multisportcalibration"
    created = 0
    skipped = 0

    for sport in sorted(calibrations):
        blend_name = f"movement_{venue_id}_pano_{sport}_blend.jpg"
        blend_path = movement_dir / blend_name

        current_name = f"movement_{venue_id}_pano_{sport}_current.jpg"
        reference_name = f"movement_{venue_id}_pano_{sport}_reference.jpg"
        current_path = movement_dir / current_name
        reference_path = movement_dir / reference_name

        if blend_path.exists() and current_path.exists() and reference_path.exists():
            skipped += 1
            continue

        # Check calibration reference images
        ref_0 = msc_dir / sport / "0.jpg"
        ref_1 = msc_dir / sport / "1.jpg"
        if not ref_0.exists() or not ref_1.exists():
            continue

        # Determine if CAM2 is usable for this calibration
        has_cam2 = cam2_path is not None and (msc_dir / sport / "2.jpg").exists()

        try:
            # Current panorama: right-to-left numbering (CAM2, CAM1, CAM0)
            cam0_img = Image.open(cam0_path)
            cam1_img = Image.open(cam1_path)
            cam_images = [cam1_img, cam0_img]
            if has_cam2:
                cam_images.insert(0, Image.open(cam2_path))
            pano = side_by_side(cam_images)

            # Reference panorama: same order (2.jpg, 1.jpg, 0.jpg)
            ref0_img = Image.open(ref_0)
            ref1_img = Image.open(ref_1)
            ref_images = [ref1_img, ref0_img]
            if has_cam2:
                ref_images.insert(0, Image.open(msc_dir / sport / "2.jpg"))
            ref_pano = side_by_side(ref_images)

            # Resize ref_pano to match pano dimensions for blend
            if ref_pano.size != pano.size:
                ref_pano = ref_pano.resize(pano.size)

            # Save current and reference panos
            if not current_path.exists():
                pano.save(current_path, quality=85)
            if not reference_path.exists():
                ref_pano.save(reference_path, quality=85)

            blend = Image.blend(pano, ref_pano, 0.5)
            if not blend_path.exists():
                blend.save(blend_path, quality=85)
            created += 1
        except Exception as e:
            print(f"  {venue_id}/{sport}: error creating blend: {e}")

    return created, skipped


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <data_dir>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory")
        sys.exit(1)

    venues = sorted(d for d in data_dir.iterdir() if d.is_dir())
    print(f"Processing {len(venues)} venues in {data_dir}...")

    total_created = 0
    total_skipped = 0
    venues_with_blends = 0

    for i, venue_path in enumerate(venues, 1):
        created, skipped = process_venue(venue_path)
        total_created += created
        total_skipped += skipped
        if created > 0:
            venues_with_blends += 1
        if i % 100 == 0:
            print(f"  [{i}/{len(venues)}] created={total_created} skipped={total_skipped}")

    print(f"\nDone:")
    print(f"  Venues processed:      {len(venues)}")
    print(f"  Venues with new blends: {venues_with_blends}")
    print(f"  Blends created:        {total_created}")
    print(f"  Blends skipped (exist): {total_skipped}")


if __name__ == "__main__":
    main()
