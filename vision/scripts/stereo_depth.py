#!/usr/bin/env python3
"""
Real-time stereo depth estimation using StereoSGBM.

Uses robot_config.yaml as the single source of truth for stereo/WLS configuration.
"""

import sys
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.camera import StereoCamera
from src.calibration import load_calibration
from src.stereo_matcher import StereoMatcher, SGBMParams, WLSParams, create_parameter_trackbars
from src.utils import draw_epipolar_lines, resize_for_display
from robot.src.gpio_config import load_robot_config


def main():
    parser = argparse.ArgumentParser(
        description='Real-time stereo depth estimation'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to robot_config.yaml (default: robot/config/robot_config.yaml)'
    )
    parser.add_argument(
        '--calibration', type=str, default=None,
        help='Path to calibration directory (overrides config)'
    )
    parser.add_argument(
        '--device', type=int, default=None,
        help='Camera device ID (overrides config)'
    )
    parser.add_argument(
        '--resolution', choices=['high', 'medium', 'low'], default=None,
        help='Resolution: high=2560x960, medium=1280x480, low=640x240 (overrides config)'
    )
    parser.add_argument(
        '--show-params', action='store_true',
        help='Show parameter adjustment trackbars'
    )
    parser.add_argument(
        '--colormap', type=str, default='JET',
        choices=['JET', 'TURBO', 'MAGMA', 'INFERNO', 'PLASMA', 'VIRIDIS'],
        help='Colormap for disparity visualization'
    )
    parser.add_argument(
        '--no-wls', action='store_true',
        help='Disable WLS filtering (overrides config)'
    )
    args = parser.parse_args()

    # Load robot config (source of truth)
    print("Loading robot configuration...")
    robot_config = load_robot_config(args.config)
    stereo_cfg = robot_config.get('stereo_camera', {})

    # Resolution mapping
    resolutions = {
        'high': (2560, 960),
        'medium': (1280, 480),
        'low': (640, 240)
    }

    # Use CLI args if provided, otherwise fall back to config
    resolution_key = args.resolution or stereo_cfg.get('resolution', 'low')
    resolution = resolutions[resolution_key]
    device_id = args.device if args.device is not None else stereo_cfg.get('device_id', 0)

    # Load calibration
    if args.calibration:
        calib_path = Path(args.calibration)
    else:
        # Use config path (relative to project root)
        config_calib_path = stereo_cfg.get('calibration_path', 'vision/data/calibration_data')
        project_root = Path(__file__).parent.parent.parent
        calib_path = project_root / config_calib_path

    print(f"Loading calibration from {calib_path}...")

    try:
        calibration_data = load_calibration(str(calib_path))
    except FileNotFoundError as e:
        print(f"Error: Calibration files not found at {calib_path}")
        print("Run capture_calibration.py and run_calibration.py first")
        return 1

    # Check resolution compatibility
    calib_size = tuple(calibration_data['image_size'])
    single_res = (resolution[0] // 2, resolution[1])

    if single_res != calib_size:
        print(f"Warning: Requested resolution {single_res} differs from calibration {calib_size}")
        print("Rectification maps may not work correctly")
        print("For best results, use the same resolution as calibration")

    # Setup SGBM parameters
    params = SGBMParams()

    # Setup WLS parameters from config
    wls_params = None
    wls_cfg = stereo_cfg.get('wls_filter', {})
    if wls_cfg.get('enabled', False) and not args.no_wls:
        wls_params = WLSParams(
            enabled=True,
            lambda_=wls_cfg.get('lambda', 8000.0),
            sigma_color=wls_cfg.get('sigma_color', 1.5),
            confidence_threshold=wls_cfg.get('confidence_threshold', 0.0)
        )
        print(f"WLS filtering enabled: lambda={wls_params.lambda_}, sigma={wls_params.sigma_color}, conf_thresh={wls_params.confidence_threshold}")
    else:
        print("WLS filtering disabled")

    # Create stereo matcher
    matcher = StereoMatcher(calibration_data, params, wls_params)
    matcher.set_colormap(args.colormap)

    print(f"Starting stereo depth estimation")
    print(f"  Camera: /dev/video{device_id}")
    print(f"  Resolution: {resolution[0]}x{resolution[1]} ({single_res[0]}x{single_res[1]} per camera)")
    print(f"  WLS Filter: {'enabled' if wls_params else 'disabled'}")
    print()
    print("Controls:")
    print("  Q - Quit")
    print("  E - Toggle epipolar lines")
    print("  C - Cycle colormap")
    if args.show_params:
        print("  Trackbars - Adjust SGBM parameters")
    print()

    # Open camera
    camera = StereoCamera(device_id, resolution, fps=30)
    if not camera.open():
        print("Error: Could not open camera")
        return 1

    # Create windows
    cv2.namedWindow('Stereo Depth', cv2.WINDOW_NORMAL)

    if args.show_params:
        cv2.namedWindow('Parameters', cv2.WINDOW_NORMAL)
        create_parameter_trackbars('Parameters', matcher)

    # State
    show_epipolar = False
    colormap_idx = 0
    colormaps = ['JET', 'TURBO', 'MAGMA', 'INFERNO', 'PLASMA', 'VIRIDIS']

    # FPS tracking
    frame_times = []
    fps = 0

    print("Running... Press Q to quit")

    while True:
        start_time = time.time()

        success, left, right = camera.read()
        if not success:
            continue

        # Process frame
        left_rect, right_rect, disparity = matcher.process_frame(left, right)

        # Get colorized disparity
        disp_color = matcher.get_colorized_disparity(disparity)

        # Create display
        if show_epipolar:
            stereo_view = draw_epipolar_lines(left_rect, right_rect, num_lines=15)
        else:
            stereo_view = np.hstack([left_rect, right_rect])

        # Resize disparity to match stereo view width
        disp_resized = cv2.resize(disp_color, (stereo_view.shape[1], disp_color.shape[0]))

        # Stack vertically
        display = np.vstack([stereo_view, disp_resized])

        # Add info overlay
        cv2.rectangle(display, (0, 0), (400, 30), (40, 40, 40), -1)
        info_text = f"FPS: {fps:.1f} | Disp: {matcher.params.num_disparities} | Block: {matcher.params.block_size}"
        cv2.putText(display, info_text, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Resize for display
        display = resize_for_display(display, max_width=1400, max_height=900)

        cv2.imshow('Stereo Depth', display)

        # Handle input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('e'):
            show_epipolar = not show_epipolar
            print(f"Epipolar lines: {'ON' if show_epipolar else 'OFF'}")
        elif key == ord('c'):
            colormap_idx = (colormap_idx + 1) % len(colormaps)
            matcher.set_colormap(colormaps[colormap_idx])
            print(f"Colormap: {colormaps[colormap_idx]}")

        # Update FPS
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times))

    camera.release()
    cv2.destroyAllWindows()

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
