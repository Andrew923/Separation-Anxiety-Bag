#!/usr/bin/env python3
"""
Motor and encoder test utility.

Interactive manual control for testing motors with real-time encoder feedback.
"""

import sys
import time
import argparse
import termios
import tty
import select
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from robot.src.gpio_config import load_gpio_config, load_robot_config
from robot.src.motor_driver import MotorDriver, MotorDriverConfig, DualMotorDriver
from robot.src.encoder import QuadratureEncoder, EncoderConfig, DualEncoders


def get_key(timeout: float = 0.1) -> str:
    """
    Get a single keypress without blocking.

    Args:
        timeout: How long to wait for a key (seconds)

    Returns:
        The key pressed, or empty string if no key
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
            return key
        return ''
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def test_manual(
    motors: DualMotorDriver,
    encoders: Optional[DualEncoders] = None,
    step: int = 10
):
    """
    Manual motor control with WASD arcade drive.

    Controls:
        W/S - Throttle (Forward/Reverse)
        A/D - Turn (Left/Right)
        Space - Stop (Coast)
        B - Brake
        R - Reset encoder counts
        Q - Quit

    Args:
        motors: DualMotorDriver instance
        encoders: Optional DualEncoders instance for feedback
        step: Speed increment per keypress (default 10%)
    """
    throttle = 0.0
    turn = 0.0

    print("\n" + "=" * 60)
    print("Manual Arcade Control")
    print("=" * 60)
    print("Controls:")
    print("  W/S   - Throttle (Forward/Reverse)")
    print("  A/D   - Turn (Left/Right)")
    print("  Space - Coast (stop both motors)")
    print("  B     - Brake (active braking)")
    if encoders:
        print("  R     - Reset encoder counts")
    print("  Q     - Quit")
    print("=" * 60)
    print(f"Speed step: {step}%")
    print("-" * 60)

    if encoders:
        encoders.reset()

    try:
        while True:
            # Build status line
            left_speed, right_speed = motors.get_speeds()
            status = f"Input: Throt={throttle:+4.0f}% Turn={turn:+4.0f}% | Motor: L={left_speed:+4.0f}% R={right_speed:+4.0f}%"

            if encoders:
                left_count, right_count = encoders.get_counts()
                left_rpm, right_rpm = encoders.get_rpms()
                status += f" | Enc: L={left_count:+6d} R={right_count:+6d}"

            # Display status
            print(f"\r{status}    ", end="")
            sys.stdout.flush()

            key = get_key(0.05)

            if key == '':
                continue

            key_lower = key.lower()

            if key_lower == 'q':
                print("\n\nQuitting...")
                break

            elif key_lower == 'w':
                throttle = min(100, throttle + step)
            elif key_lower == 's':
                throttle = max(-100, throttle - step)
            elif key_lower == 'a':
                turn = max(-100, turn - step)  # Left is negative turn
            elif key_lower == 'd':
                turn = min(100, turn + step)   # Right is positive turn
            elif key == ' ':
                throttle = 0
                turn = 0
                motors.stop()
                continue
            elif key_lower == 'b':
                throttle = 0
                turn = 0
                motors.brake()
                continue
            elif key_lower == 'r' and encoders:
                encoders.reset()
                continue

            # Apply mixing
            motors.arcade_drive(throttle, turn)

    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    finally:
        motors.stop()
        print("Motors stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive motor and encoder test utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Test motors and encoders together
  %(prog)s --no-encoders      # Test motors only (no encoder feedback)
  %(prog)s --step 5           # Use 5%% speed increments
"""
    )
    parser.add_argument('--no-encoders', action='store_true',
                        help='Disable encoder feedback')
    parser.add_argument('--step', type=int, default=10,
                        help='Speed step per keypress (default: 10%%)')
    parser.add_argument('--config', type=str,
                        help='Path to GPIO config file')
    args = parser.parse_args()

    # Load configuration
    gpio_config = load_gpio_config(args.config)
    robot_config = load_robot_config()

    print("=" * 60)
    print("Motor and Encoder Test Utility")
    print("=" * 60)
    print(f"Left motor:  IN1={gpio_config.left_motor.in1_pin}, "
          f"IN2={gpio_config.left_motor.in2_pin}, "
          f"ENA={gpio_config.left_motor.ena_pin}")
    print(f"Right motor: IN1={gpio_config.right_motor.in1_pin}, "
          f"IN2={gpio_config.right_motor.in2_pin}, "
          f"ENA={gpio_config.right_motor.ena_pin}")
    if not args.no_encoders:
        print(f"Left encoder:  A={gpio_config.left_encoder.channel_a}, "
              f"B={gpio_config.left_encoder.channel_b}")
        print(f"Right encoder: A={gpio_config.right_encoder.channel_a}, "
              f"B={gpio_config.right_encoder.channel_b}")
    print("=" * 60)

    # Create motor drivers
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

    # Create encoders unless disabled
    encoders = None
    if not args.no_encoders:
        left_enc_cfg = EncoderConfig(
            channel_a_pin=gpio_config.left_encoder.channel_a,
            channel_b_pin=gpio_config.left_encoder.channel_b,
            counts_per_revolution=robot_config['robot']['encoder_cpr']
        )
        right_enc_cfg = EncoderConfig(
            channel_a_pin=gpio_config.right_encoder.channel_a,
            channel_b_pin=gpio_config.right_encoder.channel_b,
            counts_per_revolution=robot_config['robot']['encoder_cpr']
        )
        encoders = DualEncoders(left_enc_cfg, right_enc_cfg)
        if encoders.is_initialized:
            print("Encoders initialized successfully.")
        else:
            print("Warning: Encoders failed to initialize (callbacks may not work).")

    try:
        test_manual(motors, encoders, step=args.step)

    except KeyboardInterrupt:
        print("\nTest interrupted.")

    finally:
        motors.stop()
        motors.cleanup()
        if encoders is not None:
            encoders.cleanup()
        print("\nCleanup complete.")


if __name__ == '__main__':
    main()
