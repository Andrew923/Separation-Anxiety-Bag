#!/usr/bin/env python3
"""
UWB range testing utility.

Tests UWB module communication and ranging.
"""

import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from robot.src.gpio_config import load_gpio_config, load_robot_config
from robot.src.uwb_tracker import RYUW122, UWBModuleConfig, DualUWBAnchors
from robot.src.uwb_triangulation import (
    UWBTriangulator, UWBCalibrator, TriangulationConfig
)


def test_single_anchor(config: UWBModuleConfig, name: str):
    """Test a single UWB anchor."""
    print(f"\nTesting {name} on {config.uart_port}...")

    uwb = RYUW122(config)

    if not uwb.connect():
        print(f"  Failed to connect to {name}")
        return

    print(f"  Connected to {name}")

    if not uwb.configure_as_anchor():
        print(f"  Failed to configure {name} as anchor")
        uwb.close()
        return

    print(f"  Configured as ANCHOR")

    # Read ranges for a few seconds
    print(f"  Reading ranges for 10 seconds...")
    print("-" * 40)

    start_time = time.time()
    while time.time() - start_time < 10:
        range_mm = uwb.get_range()
        if range_mm is not None:
            print(f"  Range: {range_mm:.0f} mm ({range_mm/1000:.2f} m)")
        time.sleep(0.2)

    uwb.close()
    print(f"\n{name} test complete.")


def test_dual_anchors(robot_config: dict, gpio_config):
    """Test both UWB anchors and triangulation."""
    print("\nTesting dual anchors with triangulation...")

    uwb_config = robot_config.get('uwb', {})

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
        print("  Failed to connect to one or both anchors")
        return

    print("  Both anchors connected")

    # Setup triangulator
    anchor1_offset = uwb_config.get('anchor1_offset_mm', [100, 50, 0])
    anchor2_offset = uwb_config.get('anchor2_offset_mm', [-100, 50, 0])

    tri_config = TriangulationConfig(
        anchor1_position=(anchor1_offset[0], anchor1_offset[1]),
        anchor2_position=(anchor2_offset[0], anchor2_offset[1])
    )
    triangulator = UWBTriangulator(tri_config)

    # Read and triangulate
    print("\n  Reading ranges and computing angles...")
    print("-" * 60)

    start_time = time.time()
    while time.time() - start_time < 15:
        range1, range2 = anchors.get_ranges()

        if range1 is not None and range2 is not None:
            result = triangulator.triangulate(range1, range2)

            if result is not None:
                print(f"  R1: {range1:5.0f}mm  R2: {range2:5.0f}mm  "
                      f"Angle: {result.angle_deg:+6.1f}deg  "
                      f"Dist: {result.estimated_distance_mm:.0f}mm  "
                      f"Conf: {result.confidence:.2f}")
            else:
                print(f"  R1: {range1:5.0f}mm  R2: {range2:5.0f}mm  "
                      f"(triangulation failed)")
        else:
            print(f"  R1: {'None':>5}  R2: {'None':>5}")

        time.sleep(0.3)

    anchors.close()
    print("\nDual anchor test complete.")


def main():
    parser = argparse.ArgumentParser(description="Test UWB modules")
    parser.add_argument('--test', choices=['anchor1', 'anchor2', 'both', 'all'],
                        default='all', help='What to test')
    parser.add_argument('--config', type=str, help='Path to GPIO config')
    args = parser.parse_args()

    # Load configuration
    gpio_config = load_gpio_config(args.config)
    robot_config = load_robot_config()
    uwb_config = robot_config.get('uwb', {})

    print("=" * 50)
    print("UWB Test Utility")
    print("=" * 50)
    print(f"Anchor 1: {gpio_config.uwb_anchor1.uart_port}")
    print(f"Anchor 2: {gpio_config.uwb_anchor2.uart_port}")
    print(f"Baud rate: {uwb_config.get('baud_rate', 115200)}")
    print(f"Network ID: 0x{uwb_config.get('network_id', 0x1234):04X}")
    print("=" * 50)

    try:
        if args.test in ['anchor1', 'all']:
            anchor1_cfg = UWBModuleConfig(
                uart_port=gpio_config.uwb_anchor1.uart_port,
                baud_rate=uwb_config.get('baud_rate', 115200),
                network_id=uwb_config.get('network_id', 0x1234)
            )
            test_single_anchor(anchor1_cfg, "Anchor 1")

        if args.test in ['anchor2', 'all']:
            anchor2_cfg = UWBModuleConfig(
                uart_port=gpio_config.uwb_anchor2.uart_port,
                baud_rate=uwb_config.get('baud_rate', 115200),
                network_id=uwb_config.get('network_id', 0x1234)
            )
            test_single_anchor(anchor2_cfg, "Anchor 2")

        if args.test in ['both', 'all']:
            test_dual_anchors(robot_config, gpio_config)

    except KeyboardInterrupt:
        print("\nTest interrupted.")

    print("\nTest complete.")


if __name__ == '__main__':
    main()
