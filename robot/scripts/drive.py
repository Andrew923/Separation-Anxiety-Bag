#!/usr/bin/env python3
"""
Manual robot control using ShanWan gamepad.

Open-loop PWM control without obstacle avoidance.
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from robot.src.gpio_config import load_gpio_config
from robot.src.motor_driver import MotorDriverConfig, DualMotorDriver
from robot.src.gamepad import Gamepad, ControllerConfig


def print_status(
    throttle: float,
    turn: float,
    left_pwm: float,
    right_pwm: float,
    speed_limit: int,
    deadzone: float,
    safety_status: str = "OK"
) -> None:
    """Print single-line status."""
    status = (
        f"Input: Throt={throttle:+5.2f} Turn={turn:+5.2f} | "
        f"Motor: L={left_pwm:+4.0f}% R={right_pwm:+4.0f}% | "
        f"Limit: {speed_limit:2d}% | Deadzone: {deadzone*100:3.0f}% | "
        f"Safety: {safety_status}"
    )
    print(f"\r{status}    ", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Manual robot control using PS2 gamepad (open-loop PWM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  Left Stick Y   - Throttle (forward/back)
  Left Stick X   - Turn (left/right)
  Cross (X)      - Coast/stop
  Circle (O)      - Brake (active)
  D-pad Up/Down  - Speed limit +/- 5%%
  Start           - Quit

Examples:
  %(prog)s                        # Default settings
  %(prog)s --max-speed 30         # Limit to 30%% speed
  %(prog)s --device /dev/input/js1  # Specify controller device
        """
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default=None,
        help='Controller device path (auto-detect if not specified)'
    )
    parser.add_argument(
        '--max-speed', '-s',
        type=int,
        default=5,
        help='Maximum speed percentage (default: 5%%)'
    )
    parser.add_argument(
        '--deadzone',
        type=float,
        default=0.9,
        help='Axis deadzone 0-1 (default: 0.9)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to GPIO config file'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Manual Robot Control - Open-Loop PWM")
    print("=" * 60)

    # Initialize gamepad
    print("Initializing gamepad...")
    gamepad_config = ControllerConfig(
        deadzone=args.deadzone,
        device_path=args.device,
        invert_throttle=True
    )
    gamepad = Gamepad(gamepad_config)
    gamepad.start()

    print(f"  Controller: {gamepad.name} ({gamepad.path})")

    # Load motor configuration
    print("Loading motor configuration...")
    gpio_config = load_gpio_config(args.config)

    left_motor_cfg = MotorDriverConfig(
        in1_pin=gpio_config.left_motor.in1_pin,
        in2_pin=gpio_config.left_motor.in2_pin,
        ena_pin=gpio_config.left_motor.ena_pin,
        pwm_frequency=gpio_config.pwm_frequency
    )
    right_motor_cfg = MotorDriverConfig(
        in1_pin=gpio_config.right_motor.in1_pin,
        in2_pin=gpio_config.right_motor.in2_pin,
        ena_pin=gpio_config.right_motor.ena_pin,
        pwm_frequency=gpio_config.pwm_frequency
    )
    motors = DualMotorDriver(left_motor_cfg, right_motor_cfg)

    print("=" * 60)
    print("Controls (ShanWan mapping):")
    print("  Left Stick Y   - Throttle (forward/back)")
    print("  Left Stick X   - Turn (left/right)")
    print("  Cross (button 2) - Coast/stop")
    print("  Circle (button 1) - Brake (active)")
    print("  D-pad Up/Down  - Speed limit +/- 5%")
    print("  Start (button 9) - Quit")
    print("=" * 60)
    print(f"Speed Limit: {args.max_speed}% | Deadzone: {args.deadzone*100:.0f}%")
    print("-" * 60)

    # Control state
    speed_limit = args.max_speed
    last_dpad_y = 0  # For edge detection
    running = True

    try:
        while running:
            # Get controller input
            throttle, turn = gamepad.get_state()

            # D-pad speed limit adjustment (edge detection)
            dpad_x, dpad_y = gamepad.get_dpad()

            if dpad_y != last_dpad_y:  # State changed
                if dpad_y == -1:  # Up pressed
                    speed_limit = min(100, speed_limit + 5)
                elif dpad_y == 1:  # Down pressed
                    speed_limit = max(0, speed_limit - 5)
                last_dpad_y = dpad_y

            # Button handling (ShanWan mapping)
            if gamepad.is_button_pressed(2):  # Cross = stop
                motors.stop()
                print_status(throttle, turn, 0, 0, speed_limit, args.deadzone, "STOPPED")
                time.sleep(0.1)
                continue

            if gamepad.is_button_pressed(1):  # Circle = brake
                motors.brake()
                print_status(throttle, turn, 0, 0, speed_limit, args.deadzone, "BRAKING")
                time.sleep(0.1)
                continue

            if gamepad.is_button_pressed(9):  # Start = quit
                running = False
                break

            # Apply deadzone - use fixed speed when pushed past deadzone
            if abs(throttle) >= args.deadzone:
                drive_throttle = 1.0 if throttle > 0 else -1.0
            else:
                drive_throttle = 0.0

            if abs(turn) >= args.deadzone:
                drive_turn = 1.0 if turn > 0 else -1.0
            else:
                drive_turn = 0.0

            motors.arcade_drive(
                drive_throttle * speed_limit,
                drive_turn * speed_limit
            )

            # Get actual PWM values for display
            left_pwm, right_pwm = motors.get_speeds()

            # Display status
            print_status(
                drive_throttle,
                drive_turn,
                left_pwm,
                right_pwm,
                speed_limit,
                args.deadzone,
                "OK"
            )

            time.sleep(0.02)  # 50Hz control loop

    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    finally:
        motors.stop()
        motors.cleanup()
        gamepad.cleanup()
        print("\nCleanup complete.")


if __name__ == '__main__':
    main()
