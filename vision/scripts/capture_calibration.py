#!/usr/bin/env python3
"""
Calibration image capture tool.
Captures stereo image pairs for camera calibration.
Supports both GUI and headless modes.
"""

import sys
import argparse
import time
import select
import termios
import tty
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera import StereoCamera
from src.calibration import CheckerboardConfig


def check_display_available():
    """Check if a display is available for GUI mode."""
    import os
    display = os.environ.get('DISPLAY')
    if not display:
        return False
    try:
        # Try to create a small test window
        cv2.namedWindow('_test', cv2.WINDOW_NORMAL)
        cv2.destroyWindow('_test')
        return True
    except:
        return False


def get_key_nonblocking():
    """Non-blocking key read from terminal."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def is_interactive():
    """Check if stdin is an interactive terminal."""
    return sys.stdin.isatty()


def run_headless(camera, args, pattern_size, find_flags, left_dir, right_dir):
    """Run capture in headless mode (terminal only)."""
    interactive = is_interactive()

    print("\n=== HEADLESS MODE ===")
    if interactive:
        print("Controls:")
        print("  ENTER - Capture when checkerboard detected")
        print("  q     - Quit")
    else:
        print("Running non-interactively (auto-capture mode)")
        print("  Will auto-capture when checkerboard is detected")
        print(f"  Delay between captures: {args.delay}ms")
    print()
    print("Tip: Move checkerboard to different positions/angles between captures")
    print()

    old_settings = None
    if interactive:
        # Set terminal to raw mode for non-blocking input
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    captured = 0
    last_capture_time = 0
    frames_since_status = 0
    last_status = ""

    try:
        while captured < args.target:
            success, left, right = camera.read()
            if not success:
                continue

            # Convert to grayscale for corner detection
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

            # Find corners
            found_left, corners_left = cv2.findChessboardCorners(
                left_gray, pattern_size, find_flags
            )
            found_right, corners_right = cv2.findChessboardCorners(
                right_gray, pattern_size, find_flags
            )

            both_found = found_left and found_right
            current_time = time.time() * 1000

            # Build status
            if both_found:
                if interactive:
                    status = f"\r[READY] Checkerboard detected! Press ENTER to capture | {captured}/{args.target}"
                else:
                    status = f"\r[READY] Checkerboard detected! | {captured}/{args.target}"
            else:
                missing = []
                if not found_left:
                    missing.append("L")
                if not found_right:
                    missing.append("R")
                status = f"\r[SEARCHING] No corners in: {','.join(missing)} | {captured}/{args.target}    "

            # Update status line (avoid flickering)
            frames_since_status += 1
            if status != last_status or frames_since_status > 10:
                sys.stdout.write(status)
                sys.stdout.flush()
                last_status = status
                frames_since_status = 0

            # Handle capture
            should_capture = False

            if interactive:
                # Check for keypress
                key = get_key_nonblocking()
                if key == '\n' or key == '\r':  # Enter
                    should_capture = both_found
                elif key == 'q':
                    print("\n\nCapture cancelled by user")
                    break
            else:
                # Auto-capture when checkerboard detected
                should_capture = both_found

            if should_capture and (current_time - last_capture_time >= args.delay):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                left_path = left_dir / f'{timestamp}.png'
                right_path = right_dir / f'{timestamp}.png'

                cv2.imwrite(str(left_path), left)
                cv2.imwrite(str(right_path), right)

                captured += 1
                last_capture_time = current_time
                print(f"\n>>> Captured pair {captured}/{args.target}")

    finally:
        # Restore terminal settings
        if old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    return captured


def run_gui(camera, args, pattern_size, find_flags, left_dir, right_dir):
    """Run capture in GUI mode with display."""
    print("\n=== GUI MODE ===")
    print("Controls:")
    print("  SPACE - Capture image pair (when corners detected)")
    print("  Q     - Quit")
    print()

    cv2.namedWindow('Calibration Capture', cv2.WINDOW_NORMAL)

    captured = 0
    last_capture_time = 0

    while captured < args.target:
        success, left, right = camera.read()
        if not success:
            print("Error: Could not read frame")
            continue

        # Convert to grayscale for corner detection
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Find corners
        found_left, corners_left = cv2.findChessboardCorners(
            left_gray, pattern_size, find_flags
        )
        found_right, corners_right = cv2.findChessboardCorners(
            right_gray, pattern_size, find_flags
        )

        # Create display copies
        display_left = left.copy()
        display_right = right.copy()

        # Draw corners if found
        if found_left:
            cv2.drawChessboardCorners(display_left, pattern_size, corners_left, found_left)
        if found_right:
            cv2.drawChessboardCorners(display_right, pattern_size, corners_right, found_right)

        # Combine for display
        display = np.hstack([display_left, display_right])

        # Resize for display if too large
        max_width = 1600
        if display.shape[1] > max_width:
            scale = max_width / display.shape[1]
            display = cv2.resize(display, None, fx=scale, fy=scale)

        # Add status overlay
        both_found = found_left and found_right
        if both_found:
            status = "READY - Press SPACE to capture"
            color = (0, 255, 0)  # Green
        else:
            missing = []
            if not found_left:
                missing.append("left")
            if not found_right:
                missing.append("right")
            status = f"Searching... (no corners in {', '.join(missing)})"
            color = (0, 165, 255)  # Orange

        # Draw status bar
        cv2.rectangle(display, (0, 0), (display.shape[1], 40), (40, 40, 40), -1)
        cv2.putText(
            display, f"{status} | Captured: {captured}/{args.target}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

        cv2.imshow('Calibration Capture', display)

        key = cv2.waitKey(1) & 0xFF
        current_time = cv2.getTickCount() / cv2.getTickFrequency() * 1000

        if key == ord(' ') and both_found:
            # Check delay since last capture
            if current_time - last_capture_time >= args.delay:
                # Save image pair
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                left_path = left_dir / f'{timestamp}.png'
                right_path = right_dir / f'{timestamp}.png'

                cv2.imwrite(str(left_path), left)
                cv2.imwrite(str(right_path), right)

                captured += 1
                last_capture_time = current_time
                print(f"Captured pair {captured}/{args.target}: {timestamp}")

                # Flash effect
                flash = np.ones_like(display) * 255
                cv2.imshow('Calibration Capture', flash)
                cv2.waitKey(100)

        elif key == ord('q'):
            print("\nCapture cancelled by user")
            break

    cv2.destroyAllWindows()
    return captured


def main():
    parser = argparse.ArgumentParser(
        description='Capture calibration images for stereo camera'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--resolution', choices=['high', 'medium', 'low'], default='high',
        help='Resolution: high=2560x960, medium=1280x480, low=640x240'
    )
    parser.add_argument(
        '--target', type=int, default=20,
        help='Target number of image pairs to capture (default: 20)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data/calibration_images',
        help='Output directory for captured images'
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
        '--delay', type=int, default=2000,
        help='Minimum delay between captures in ms (default: 2000)'
    )
    parser.add_argument(
        '--headless', action='store_true',
        help='Force headless mode (no GUI)'
    )
    args = parser.parse_args()

    # Resolution mapping
    resolutions = {
        'high': (2560, 960),
        'medium': (1280, 480),
        'low': (640, 240)
    }
    resolution = resolutions[args.resolution]

    # Create output directories
    base_path = Path(__file__).parent.parent / args.output_dir
    left_dir = base_path / 'left'
    right_dir = base_path / 'right'
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    # Setup checkerboard config
    checkerboard = CheckerboardConfig(rows=args.rows, cols=args.cols)
    pattern_size = checkerboard.pattern_size

    # Corner detection flags
    find_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH |
        cv2.CALIB_CB_FAST_CHECK |
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    print(f"Starting calibration capture")
    print(f"  Camera: /dev/video{args.device}")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")
    print(f"  Checkerboard: {args.cols}x{args.rows} internal corners")
    print(f"  Target: {args.target} image pairs")
    print(f"  Output: {base_path}")

    # Open camera
    camera = StereoCamera(args.device, resolution, fps=30)
    if not camera.open():
        print("Error: Could not open camera")
        return 1

    print(f"Camera opened. Single camera resolution: {camera.get_single_resolution()}")

    # Determine mode
    use_gui = not args.headless and check_display_available()

    try:
        if use_gui:
            captured = run_gui(camera, args, pattern_size, find_flags, left_dir, right_dir)
        else:
            captured = run_headless(camera, args, pattern_size, find_flags, left_dir, right_dir)
    finally:
        camera.release()

    print(f"\nCapture complete!")
    print(f"  Total captured: {captured} pairs")
    print(f"  Left images: {left_dir}")
    print(f"  Right images: {right_dir}")

    if captured < args.target:
        print(f"\nWarning: Only captured {captured}/{args.target} pairs")

    return 0


if __name__ == '__main__':
    sys.exit(main())
