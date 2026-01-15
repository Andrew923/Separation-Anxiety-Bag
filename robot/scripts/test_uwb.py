#!/usr/bin/env python3
"""
UWB range testing utility.

Tests UWB module communication and ranging.

To reset UWB modules via GPIO before running (requires sudo):
    sudo python3 -c "
import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in [18, 7]:  # Anchor 1 and 2 reset pins
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)
    time.sleep(0.1)
    GPIO.output(pin, GPIO.HIGH)
GPIO.cleanup()
print('Both UWB modules reset')
" && sleep 0.5
"""

import sys
import time
import argparse
import select
import threading
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

from robot.src.gpio_config import load_gpio_config, load_robot_config
from robot.src.uwb_tracker import (
    RYUW122, UWBModuleConfig, DualUWBAnchors, RangeFilterConfig
)
from robot.src.uwb_triangulation import (
    UWBTriangulator, UWBCalibrator, TriangulationConfig
)


def test_single_anchor(config: UWBModuleConfig, name: str):
    """Test a single UWB anchor."""
    print(f"\nTesting {name} on {config.uart_port}...")
    print(f"  Network ID: {config.network_id}")
    print(f"  Address: {config.address}")
    print(f"  Target TAG: {config.target_address}")

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

    # Read ranges for a few seconds using active polling
    print(f"  Polling range to {config.target_address} for 10 seconds...")
    print("-" * 40)

    start_time = time.time()
    while time.time() - start_time < 10:
        # Active poll - sends data to TAG to trigger TWR
        range_mm = uwb.poll_range()
        if range_mm is not None:
            print(f"  Range: {range_mm:.0f} mm ({range_mm/1000:.2f} m)")
        else:
            print(f"  No response from TAG")
        time.sleep(0.3)

    uwb.close()
    print(f"\n{name} test complete.")


def passthrough_mode(uart_port: str, baud_rate: int, reset_pin: Optional[int] = None):
    """
    Interactive passthrough mode for direct AT command communication.

    Type AT commands directly and see responses in real-time.

    Args:
        uart_port: Serial port path
        baud_rate: Baud rate
        reset_pin: Optional GPIO pin for hardware reset (BCM numbering)
    """
    if not SERIAL_AVAILABLE:
        print("Error: pyserial not available")
        return

    print(f"\n{'=' * 50}")
    print("UWB Passthrough Mode")
    print(f"{'=' * 50}")
    print(f"Port: {uart_port}")
    print(f"Baud: {baud_rate}")
    print(f"Reset Pin: {reset_pin if reset_pin else 'None'}")
    print("-" * 50)

    # Perform hardware reset if pin is configured
    if reset_pin is not None and GPIO_AVAILABLE:
        print("Performing hardware reset...")
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(reset_pin, GPIO.OUT)
            GPIO.output(reset_pin, GPIO.LOW)
            time.sleep(0.1)
            GPIO.output(reset_pin, GPIO.HIGH)
            time.sleep(0.5)  # Wait for module to boot
            print("  Reset complete.")
        except Exception as e:
            print(f"  Reset failed: {e}")
    elif reset_pin is not None:
        print("Warning: Reset pin specified but RPi.GPIO not available")

    print("-" * 50)
    print("Type AT commands and press Enter to send.")
    print("Responses will be displayed automatically.")
    print("Type 'quit' or 'exit' to exit passthrough mode.")
    print("Type 'help' for common commands.")
    print("Type 'reset' to perform hardware reset.")
    print(f"{'=' * 50}\n")

    try:
        ser = serial.Serial(
            port=uart_port,
            baudrate=baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1
        )
    except Exception as e:
        print(f"Failed to open serial port: {e}")
        return

    # Clear buffers
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # Check for boot message
    time.sleep(0.2)
    if ser.in_waiting > 0:
        boot_msg = ser.read(ser.in_waiting).decode('utf-8', errors='replace')
        print(f"  << (boot) {boot_msg.strip()}")

    # Flag to stop reader thread
    running = True

    def reader_thread():
        """Background thread to read and display serial responses."""
        while running:
            try:
                if ser.in_waiting > 0:
                    data = ser.read(ser.in_waiting)
                    text = data.decode('utf-8', errors='replace')
                    # Print response with visual distinction
                    for line in text.split('\n'):
                        line = line.strip()
                        if line:
                            print(f"  << {line}")
                time.sleep(0.05)
            except Exception:
                if running:
                    pass  # Ignore errors during shutdown

    # Start reader thread
    reader = threading.Thread(target=reader_thread, daemon=True)
    reader.start()

    # Common commands help
    help_text = """
Common AT Commands:
  AT              - Test module (expect +OK)
  AT+RESET        - Software reset
  AT+MODE=0       - Set TAG mode
  AT+MODE=1       - Set ANCHOR mode
  AT+NETWORKID=X  - Set network ID (8 bytes ASCII)
  AT+ADDRESS=X    - Set device address (8 bytes ASCII)
  AT+RSSI=1       - Enable RSSI in responses
  AT+VER?         - Query firmware version
  AT+UID?         - Query hardware UID
  AT+ANCHOR_SEND=<Addr>,<Len>,<Data>  - Send to TAG (triggers ranging)
    Example: AT+ANCHOR_SEND=TAG001,4,PING

Special commands:
  reset           - Perform hardware reset (if reset pin configured)
  quit/exit       - Exit passthrough mode
"""

    try:
        while True:
            try:
                # Read user input
                cmd = input(">> ").strip()

                if not cmd:
                    continue

                if cmd.lower() in ['quit', 'exit']:
                    print("Exiting passthrough mode...")
                    break

                if cmd.lower() == 'help':
                    print(help_text)
                    continue

                if cmd.lower() == 'reset':
                    if reset_pin is not None and GPIO_AVAILABLE:
                        print("Performing hardware reset...")
                        try:
                            GPIO.output(reset_pin, GPIO.LOW)
                            time.sleep(0.1)
                            GPIO.output(reset_pin, GPIO.HIGH)
                            time.sleep(0.5)
                            ser.reset_input_buffer()
                            print("  Reset complete.")
                        except Exception as e:
                            print(f"  Reset failed: {e}")
                    else:
                        print("  No reset pin configured or GPIO not available")
                    continue

                # Send command with CR+LF
                ser.write(f"{cmd}\r\n".encode('utf-8'))

                # Small delay to allow response
                time.sleep(0.1)

            except EOFError:
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        running = False
        time.sleep(0.1)  # Let reader thread finish
        ser.close()
        if reset_pin is not None and GPIO_AVAILABLE:
            try:
                GPIO.cleanup(reset_pin)
            except Exception:
                pass
        print("Serial port closed.")


def test_dual_anchors(robot_config: dict, gpio_config, use_filter: bool = False, filter_alpha: float = 0.3):
    """Test both UWB anchors and triangulation.
    
    Args:
        robot_config: Robot configuration dict
        gpio_config: GPIO configuration
        use_filter: Enable EMA filtering on ranges
        filter_alpha: EMA alpha value (0.1-0.5, lower = more smoothing)
    """
    print("\nTesting dual anchors with triangulation...")

    uwb_config = robot_config.get('uwb', {})

    anchor1_cfg = UWBModuleConfig(
        uart_port=gpio_config.uwb_anchor1.uart_port,
        baud_rate=uwb_config.get('baud_rate', 115200),
        network_id=str(uwb_config.get('network_id', '0x1234')),
        address=uwb_config.get('anchor1_address', 'ANCHOR01'),
        target_address=uwb_config.get('target_address', 'TAG001')
    )
    anchor2_cfg = UWBModuleConfig(
        uart_port=gpio_config.uwb_anchor2.uart_port,
        baud_rate=uwb_config.get('baud_rate', 115200),
        network_id=str(uwb_config.get('network_id', '0x1234')),
        address=uwb_config.get('anchor2_address', 'ANCHOR02'),
        target_address=uwb_config.get('target_address', 'TAG001')
    )

    # Setup filter config if enabled
    filter_config = None
    if use_filter:
        filter_config = RangeFilterConfig(
            enabled=True,
            ema_alpha=filter_alpha,
            outlier_threshold_mm=200.0,
            min_samples=3
        )
        print(f"  Filtering: ENABLED (alpha={filter_alpha})")
    else:
        print(f"  Filtering: DISABLED")

    anchors = DualUWBAnchors(anchor1_cfg, anchor2_cfg, filter_config)

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
    print("\n  Polling ranges and computing angles...")
    print("-" * 70)
    
    if use_filter:
        print("  (First few readings may show None while filter initializes)")

    start_time = time.time()
    while time.time() - start_time < 15:
        # Get both raw and filtered for comparison if filtering
        if use_filter:
            raw1, raw2 = anchors.poll_ranges_raw()
            # Re-poll would double the time, so we'll just show filtered
            range1, range2 = anchors._apply_filters(raw1, raw2)
        else:
            range1, range2 = anchors.poll_ranges()
            raw1, raw2 = range1, range2

        if range1 is not None and range2 is not None:
            result = triangulator.triangulate(range1, range2)

            if result is not None:
                if use_filter and raw1 is not None and raw2 is not None:
                    print(f"  Raw: {raw1:4.0f}/{raw2:4.0f}  "
                          f"Filt: {range1:4.0f}/{range2:4.0f}  "
                          f"Angle: {result.angle_deg:+6.1f}°  "
                          f"Dist: {result.estimated_distance_mm:4.0f}mm")
                else:
                    print(f"  R1: {range1:5.0f}mm  R2: {range2:5.0f}mm  "
                          f"Angle: {result.angle_deg:+6.1f}deg  "
                          f"Dist: {result.estimated_distance_mm:.0f}mm  "
                          f"Conf: {result.confidence:.2f}")
            else:
                print(f"  R1: {range1:5.0f}mm  R2: {range2:5.0f}mm  "
                      f"(triangulation failed)")
        else:
            if use_filter:
                raw_str = f"Raw: {raw1 or 'None':>4}/{raw2 or 'None':>4}"
                print(f"  {raw_str}  (filter initializing...)")
            else:
                print(f"  R1: {'None':>5}  R2: {'None':>5}")

        time.sleep(0.3)

    anchors.close()
    print("\nDual anchor test complete.")


def main():
    parser = argparse.ArgumentParser(description="Test UWB modules")
    parser.add_argument('--test', choices=['anchor1', 'anchor2', 'both', 'all', 'passthrough'],
                        default='all', help='What to test (passthrough for interactive mode)')
    parser.add_argument('--config', type=str, help='Path to GPIO config')
    parser.add_argument('--port', type=str, help='Serial port override for passthrough mode')
    parser.add_argument('--anchor', choices=['1', '2'], default='1',
                        help='Which anchor to use for passthrough mode (1 or 2, default: 1)')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('--filter', action='store_true',
                        help='Enable EMA filtering on range measurements')
    parser.add_argument('--filter-alpha', type=float, default=0.3,
                        help='EMA filter alpha (0.1-0.5, lower=smoother, default: 0.3)')
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
    print(f"Network ID: {uwb_config.get('network_id', '0x1234')}")
    print(f"Target TAG: {uwb_config.get('target_address', 'TAG001')}")
    print("=" * 50)

    try:
        if args.test == 'passthrough':
            # Use specified port, or select based on --anchor flag
            if args.port:
                port = args.port
                reset_pin = None  # Unknown reset pin for custom port
            elif args.anchor == '2':
                port = gpio_config.uwb_anchor2.uart_port
                reset_pin = gpio_config.uwb_anchor2.reset_pin
            else:
                port = gpio_config.uwb_anchor1.uart_port
                reset_pin = gpio_config.uwb_anchor1.reset_pin
            baud = args.baud or uwb_config.get('baud_rate', 115200)
            passthrough_mode(port, baud, reset_pin)

        elif args.test in ['anchor1', 'all']:
            anchor1_cfg = UWBModuleConfig(
                uart_port=gpio_config.uwb_anchor1.uart_port,
                baud_rate=uwb_config.get('baud_rate', 115200),
                network_id=str(uwb_config.get('network_id', '0x1234')),
                address=uwb_config.get('anchor1_address', 'ANCHOR01'),
                target_address=uwb_config.get('target_address', 'TAG001')
            )
            test_single_anchor(anchor1_cfg, "Anchor 1")

        if args.test in ['anchor2', 'all']:
            anchor2_cfg = UWBModuleConfig(
                uart_port=gpio_config.uwb_anchor2.uart_port,
                baud_rate=uwb_config.get('baud_rate', 115200),
                network_id=str(uwb_config.get('network_id', '0x1234')),
                address=uwb_config.get('anchor2_address', 'ANCHOR02'),
                target_address=uwb_config.get('target_address', 'TAG001')
            )
            test_single_anchor(anchor2_cfg, "Anchor 2")

        if args.test in ['both', 'all']:
            test_dual_anchors(
                robot_config, gpio_config,
                use_filter=args.filter,
                filter_alpha=args.filter_alpha
            )

    except KeyboardInterrupt:
        print("\nTest interrupted.")

    print("\nTest complete.")


if __name__ == '__main__':
    main()
