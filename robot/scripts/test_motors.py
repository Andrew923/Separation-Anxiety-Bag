#!/usr/bin/env python3
"""
Motor and encoder test utility.

Tests motor control and encoder feedback.
"""

import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from robot.src.gpio_config import load_gpio_config, load_robot_config
from robot.src.motor_driver import MotorDriver, MotorDriverConfig, DualMotorDriver
from robot.src.encoder import QuadratureEncoder, EncoderConfig, DualEncoders


def test_single_motor(motor: MotorDriver, name: str):
    """Test a single motor with ramp up/down."""
    print(f"\nTesting {name} motor...")

    # Ramp up
    print("  Ramping up...")
    for speed in range(0, 51, 10):
        motor.set_speed(speed)
        print(f"    Speed: {speed}%")
        time.sleep(0.3)

    # Hold
    print("  Holding at 50%...")
    time.sleep(1.0)

    # Ramp down
    print("  Ramping down...")
    for speed in range(50, -1, -10):
        motor.set_speed(speed)
        print(f"    Speed: {speed}%")
        time.sleep(0.3)

    # Test reverse
    print("  Testing reverse...")
    for speed in range(0, -51, -10):
        motor.set_speed(speed)
        print(f"    Speed: {speed}%")
        time.sleep(0.3)

    motor.stop()
    print(f"  {name} motor test complete.")


def test_encoders(encoders: DualEncoders, duration: float = 5.0):
    """Test encoder reading."""
    print(f"\nTesting encoders for {duration} seconds...")
    print("Rotate wheels manually to see counts.")
    print("-" * 50)

    encoders.reset()
    start_time = time.time()

    while time.time() - start_time < duration:
        left_count, right_count = encoders.get_counts()
        left_rpm, right_rpm = encoders.get_rpms()

        print(f"\rLeft: {left_count:6d} counts, {left_rpm:6.1f} RPM | "
              f"Right: {right_count:6d} counts, {right_rpm:6.1f} RPM", end="")

        time.sleep(0.1)

    print("\n" + "-" * 50)
    print("Encoder test complete.")


def test_drive(motors: DualMotorDriver, encoders: DualEncoders):
    """Test closed-loop drive briefly."""
    print("\nTesting drive (forward for 2 seconds)...")

    encoders.reset()
    motors.set_speeds(30, 30)

    for i in range(20):
        time.sleep(0.1)
        left_rpm, right_rpm = encoders.get_rpms()
        print(f"\rLeft: {left_rpm:6.1f} RPM | Right: {right_rpm:6.1f} RPM", end="")

    motors.stop()
    print("\nDrive test complete.")


def main():
    parser = argparse.ArgumentParser(description="Test motors and encoders")
    parser.add_argument('--test', choices=['motors', 'encoders', 'drive', 'all'],
                        default='all', help='What to test')
    parser.add_argument('--config', type=str, help='Path to GPIO config')
    args = parser.parse_args()

    # Load configuration
    gpio_config = load_gpio_config(args.config)
    robot_config = load_robot_config()

    print("=" * 50)
    print("Motor and Encoder Test Utility")
    print("=" * 50)
    print(f"Left motor: PWM={gpio_config.left_motor.pwm_pin}, "
          f"DIR={gpio_config.left_motor.dir_pin}")
    print(f"Right motor: PWM={gpio_config.right_motor.pwm_pin}, "
          f"DIR={gpio_config.right_motor.dir_pin}")
    print(f"Left encoder: A={gpio_config.left_encoder.channel_a}, "
          f"B={gpio_config.left_encoder.channel_b}")
    print(f"Right encoder: A={gpio_config.right_encoder.channel_a}, "
          f"B={gpio_config.right_encoder.channel_b}")
    print("=" * 50)

    # Create motor drivers
    left_motor_cfg = MotorDriverConfig(
        pwm_pin=gpio_config.left_motor.pwm_pin,
        dir_pin=gpio_config.left_motor.dir_pin,
        pwm_frequency=gpio_config.pwm_frequency
    )
    right_motor_cfg = MotorDriverConfig(
        pwm_pin=gpio_config.right_motor.pwm_pin,
        dir_pin=gpio_config.right_motor.dir_pin,
        pwm_frequency=gpio_config.pwm_frequency
    )
    motors = DualMotorDriver(left_motor_cfg, right_motor_cfg)

    # Create encoders
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

    try:
        if args.test in ['motors', 'all']:
            test_single_motor(motors.left, "Left")
            test_single_motor(motors.right, "Right")

        if args.test in ['encoders', 'all']:
            test_encoders(encoders)

        if args.test in ['drive', 'all']:
            test_drive(motors, encoders)

    except KeyboardInterrupt:
        print("\nTest interrupted.")

    finally:
        motors.stop()
        motors.cleanup()
        encoders.cleanup()
        print("\nCleanup complete.")


if __name__ == '__main__':
    main()
