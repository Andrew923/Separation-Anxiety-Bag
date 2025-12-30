#!/usr/bin/env python3
"""
UWB front/back calibration routine.

Calibrates the UWB triangulation to distinguish front from back.
"""

import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from robot.src.gpio_config import load_gpio_config, load_robot_config
from robot.src.uwb_tracker import UWBModuleConfig, DualUWBAnchors
from robot.src.uwb_triangulation import (
    UWBTriangulator, UWBCalibrator, TriangulationConfig
)
from robot.src.utils import get_default_config_path


def collect_samples(
    anchors: DualUWBAnchors,
    calibrator: UWBCalibrator,
    is_front: bool,
    num_samples: int = 10
) -> int:
    """
    Collect calibration samples.

    Args:
        anchors: UWB anchors
        calibrator: Calibrator instance
        is_front: True if collecting front samples
        num_samples: Number of samples to collect

    Returns:
        Number of successful samples
    """
    position = "FRONT" if is_front else "BACK"
    print(f"\nCollecting {num_samples} {position} samples...")
    print("Hold position steady.")

    successful = 0
    attempts = 0
    max_attempts = num_samples * 3

    while successful < num_samples and attempts < max_attempts:
        attempts += 1

        range1, range2 = anchors.poll_ranges()

        if range1 is not None and range2 is not None:
            if is_front:
                angle = calibrator.record_front_sample(range1, range2)
            else:
                angle = calibrator.record_back_sample(range1, range2)

            if angle is not None:
                successful += 1
                print(f"  Sample {successful}/{num_samples}: "
                      f"R1={range1:.0f}mm, R2={range2:.0f}mm, "
                      f"Angle={angle:.1f}deg")
            else:
                print(f"  Failed to triangulate (R1={range1:.0f}, R2={range2:.0f})")
        else:
            print(f"  No range data")

        time.sleep(0.3)

    return successful


def main():
    parser = argparse.ArgumentParser(description="Calibrate UWB front/back")
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples per position')
    parser.add_argument('--config', type=str, help='Path to robot config')
    parser.add_argument('--skip-back', action='store_true',
                        help='Skip back samples (front only)')
    args = parser.parse_args()

    # Load configuration
    gpio_config = load_gpio_config()
    robot_config = load_robot_config(args.config)
    uwb_config = robot_config.get('uwb', {})

    config_path = args.config or get_default_config_path()

    print("=" * 60)
    print("UWB Front/Back Calibration")
    print("=" * 60)
    print("\nThis routine calibrates the UWB triangulation to correctly")
    print("identify when the person is in front vs behind the robot.")
    print("\nInstructions:")
    print("1. Stand directly in FRONT of the robot (~1.5m away)")
    print("2. Hold the UWB tag steady")
    print("3. Press Enter when ready to collect front samples")
    if not args.skip_back:
        print("4. Then move behind the robot and collect back samples")
    print("=" * 60)

    # Setup UWB
    anchor1_cfg = UWBModuleConfig(
        uart_port=gpio_config.uwb_anchor1.uart_port,
        baud_rate=uwb_config.get('baud_rate', 115200),
        network_id=uwb_config.get('network_id', 0x1234)
    )
    anchor2_cfg = UWBModuleConfig(
        uart_port=gpio_config.uwb_anchor2.uart_port,
        baud_rate=uwb_config.get('baud_rate', 115200),
        network_id=uwb_config.get('network_id', 0x1234)
    )

    anchors = DualUWBAnchors(anchor1_cfg, anchor2_cfg)

    if not anchors.connect():
        print("\nError: Failed to connect to UWB anchors")
        return

    print("\nUWB anchors connected.")

    # Setup triangulator and calibrator
    anchor1_offset = uwb_config.get('anchor1_offset_mm', [100, 50, 0])
    anchor2_offset = uwb_config.get('anchor2_offset_mm', [-100, 50, 0])

    tri_config = TriangulationConfig(
        anchor1_position=(anchor1_offset[0], anchor1_offset[1]),
        anchor2_position=(anchor2_offset[0], anchor2_offset[1])
    )
    triangulator = UWBTriangulator(tri_config)
    calibrator = UWBCalibrator(triangulator)

    try:
        # Collect front samples
        input("\nPress Enter when standing in FRONT of the robot...")
        front_count = collect_samples(
            anchors, calibrator, is_front=True, num_samples=args.samples
        )
        print(f"\nCollected {front_count} front samples.")

        # Optionally collect back samples
        if not args.skip_back:
            input("\nPress Enter when standing BEHIND the robot...")
            back_count = collect_samples(
                anchors, calibrator, is_front=False, num_samples=args.samples
            )
            print(f"\nCollected {back_count} back samples.")

        # Compute calibration
        print("\n" + "=" * 60)
        print("Computing calibration...")

        front_offset = calibrator.compute_calibration()

        if front_offset is not None:
            print(f"\nCalibration successful!")
            print(f"Front offset angle: {front_offset:.1f} degrees")

            # Save to config
            save = input("\nSave calibration to config? (y/n): ")
            if save.lower() == 'y':
                if calibrator.save_calibration(config_path):
                    print(f"Calibration saved to {config_path}")
                else:
                    print("Failed to save calibration")
        else:
            print("\nCalibration failed - not enough valid samples")

    except KeyboardInterrupt:
        print("\nCalibration interrupted.")

    finally:
        anchors.close()

    print("\nCalibration complete.")


if __name__ == '__main__':
    main()
