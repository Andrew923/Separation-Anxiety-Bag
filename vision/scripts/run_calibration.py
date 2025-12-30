#!/usr/bin/env python3
"""
Process captured calibration images and generate calibration data.
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration import (
    CheckerboardConfig,
    StereoCalibrator,
    save_calibration
)
from src.utils import draw_epipolar_lines, compute_calibration_quality


def main():
    parser = argparse.ArgumentParser(
        description='Process calibration images and generate stereo calibration'
    )
    parser.add_argument(
        '--input-dir', type=str, default='data/calibration_images',
        help='Directory containing left/ and right/ image subdirectories'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data/calibration_data',
        help='Output directory for calibration files'
    )
    parser.add_argument(
        '--rows', type=int, default=7,
        help='Checkerboard internal corners (rows)'
    )
    parser.add_argument(
        '--cols', type=int, default=10,
        help='Checkerboard internal corners (cols)'
    )
    parser.add_argument(
        '--square-size', type=float, default=25.0,
        help='Checkerboard square size in mm'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.0,
        help='Rectification alpha (0=crop to valid region, 1=keep all pixels)'
    )
    parser.add_argument(
        '--show-results', action='store_true',
        help='Display calibration verification images'
    )
    parser.add_argument(
        '--min-pairs', type=int, default=10,
        help='Minimum number of valid pairs required'
    )
    args = parser.parse_args()

    # Setup paths
    base_path = Path(__file__).parent.parent / args.input_dir
    left_dir = base_path / 'left'
    right_dir = base_path / 'right'
    output_path = Path(__file__).parent.parent / args.output_dir

    if not left_dir.exists() or not right_dir.exists():
        print(f"Error: Image directories not found")
        print(f"  Expected: {left_dir}")
        print(f"  Expected: {right_dir}")
        return 1

    # Find image pairs
    left_images = sorted(left_dir.glob('*.png'))
    right_images = sorted(right_dir.glob('*.png'))

    if len(left_images) == 0:
        print("Error: No images found in left directory")
        return 1

    print(f"Found {len(left_images)} left images, {len(right_images)} right images")

    # Match pairs by filename
    pairs = []
    for left_path in left_images:
        right_path = right_dir / left_path.name
        if right_path.exists():
            pairs.append((left_path, right_path))

    print(f"Matched {len(pairs)} image pairs")

    if len(pairs) < args.min_pairs:
        print(f"Error: Need at least {args.min_pairs} pairs, found {len(pairs)}")
        return 1

    # Load first image to get size
    sample = cv2.imread(str(pairs[0][0]))
    image_size = (sample.shape[1], sample.shape[0])
    print(f"Image size: {image_size}")

    # Setup calibrator
    checkerboard = CheckerboardConfig(
        rows=args.rows,
        cols=args.cols,
        square_size_mm=args.square_size
    )
    calibrator = StereoCalibrator(checkerboard, image_size)

    print(f"\nProcessing calibration images...")
    print(f"  Checkerboard: {args.cols}x{args.rows} internal corners")
    print(f"  Square size: {args.square_size} mm")
    print()

    # Process each pair
    valid_count = 0
    skipped_count = 0

    for i, (left_path, right_path) in enumerate(pairs):
        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))

        # Find corners
        found_left, corners_left, _ = calibrator.find_corners(left_img)
        found_right, corners_right, _ = calibrator.find_corners(right_img)

        if found_left and found_right:
            calibrator.add_calibration_pair(corners_left, corners_right)
            valid_count += 1
            status = "OK"
        else:
            skipped_count += 1
            missing = []
            if not found_left:
                missing.append("left")
            if not found_right:
                missing.append("right")
            status = f"SKIP (no corners in {', '.join(missing)})"

        print(f"  Pair {i+1:02d}/{len(pairs)}: {status}")

    print(f"\nValid pairs: {valid_count}/{len(pairs)}")

    if valid_count < args.min_pairs:
        print(f"Error: Not enough valid pairs (need {args.min_pairs})")
        return 1

    # Run calibration
    print()
    calibration_data = calibrator.run_full_calibration(alpha=args.alpha)

    # Compute quality metrics
    quality = compute_calibration_quality(calibration_data)
    print(f"\nCalibration Quality:")
    print(f"  Individual cameras: {quality['individual_quality']}")
    print(f"  Stereo calibration: {quality['stereo_quality']}")

    # Save calibration
    print()
    save_calibration(calibration_data, str(output_path))

    print(f"\nCalibration complete!")
    print(f"  Output: {output_path}")

    # Show verification if requested
    if args.show_results:
        print("\nShowing rectification verification...")
        print("Press any key to close")

        # Load and rectify a sample pair
        left_img = cv2.imread(str(pairs[0][0]))
        right_img = cv2.imread(str(pairs[0][1]))

        # Apply rectification
        rect = calibration_data['rectification']
        left_rect = cv2.remap(left_img, rect['map1_left'], rect['map2_left'], cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, rect['map1_right'], rect['map2_right'], cv2.INTER_LINEAR)

        # Draw epipolar lines
        combined = draw_epipolar_lines(left_rect, right_rect, num_lines=20)

        # Resize for display
        max_width = 1400
        if combined.shape[1] > max_width:
            scale = max_width / combined.shape[1]
            combined = cv2.resize(combined, None, fx=scale, fy=scale)

        cv2.imshow('Rectification Verification (epipolar lines should be horizontal)', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    sys.exit(main())
