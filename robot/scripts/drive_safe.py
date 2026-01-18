#!/usr/bin/env python3
"""
Manual robot control using PS2 gamepad with obstacle avoidance.

Closed-loop PID control with stereo vision safety checks.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from robot.src.gpio_config import load_gpio_config, load_robot_config
from robot.src.motor_driver import MotorDriverConfig, DualMotorDriver
from robot.src.encoder import EncoderConfig, DualEncoders
from robot.src.differential_drive import DriveConfig, PIDConfig, DifferentialDriveController
from robot.src.depth_preprocessor import DepthPreprocessor, DepthPreprocessorConfig
from robot.src.path_planner_factory import create_path_planner
from robot.src.gamepad import Gamepad, ControllerConfig

from vision.src.camera import StereoCamera
from vision.src.stereo_matcher import StereoMatcher, SGBMParams
from vision.src.calibration import load_calibration


def print_status(
    throttle: float,
    turn: float,
    linear_mm_s: float,
    angular_deg_s: float,
    speed_limit: int,
    safety_status: str,
    obstacles_blocked: int = 0
) -> None:
    """Print single-line status."""
    status = (
        f"Input: Throt={throttle:+5.2f} Turn={turn:+5.2f} | "
        f"Motor: Lin={linear_mm_s:+4.0f}mm/s Ang={angular_deg_s:+3.0f}d/s | "
        f"Obstacles: {obstacles_blocked} | Safety: {safety_status} | "
        f"Limit: {speed_limit:2d}%"
    )
    print(f"\r{status}    ", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Manual robot control with obstacle avoidance (closed-loop PID)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls (ShanWan mapping):
  Left Stick Y   - Throttle (forward/back)
  Left Stick X   - Turn (left/right)
  Cross (button 2) - Coast/stop
  Circle (button 1) - Brake (active)
  D-pad Up/Down  - Speed limit +/- 5%%
  Start (button 9)  - Quit

Safety:
  Uses stereo camera and path planner to detect obstacles.
  Full stop when obstacles within safety distance (emergency_stop_distance from config).
  Robot can still rotate in place when blocked.

Examples:
  %(prog)s                        # Default settings
  %(prog)s --no-vision            # Disable obstacle avoidance
  %(prog)s --algorithm apf         # Use APF instead of follow-gap
  %(prog)s --max-speed 30         # Limit to 30%% speed
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
        '--no-vision',
        action='store_true',
        help='Disable obstacle avoidance (fallback to open-loop)'
    )
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        choices=['follow_gap', 'apf'],
        default=None,
        help='Path planning algorithm (uses config if not specified)'
    )
    parser.add_argument(
        '--calibration',
        type=str,
        default='vision/data/calibration_data',
        help='Path to calibration data directory'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to robot config file'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Manual Robot Control - Closed-Loop + Safety")
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

    # Load configurations
    print("Loading configurations...")
    gpio_config = load_gpio_config(args.config)
    robot_config = load_robot_config(args.config)

    # Override algorithm if specified
    if args.algorithm:
        if 'path_planning' not in robot_config:
            robot_config['path_planning'] = {}
        robot_config['path_planning']['algorithm'] = args.algorithm

    # Get safety distance from config
    apf_cfg = robot_config.get('path_planning', {}).get('apf', {})
    safety_distance_mm = apf_cfg.get('emergency_stop_distance_mm', 150.0)
    print(f"  Safety distance: {safety_distance_mm}mm")

    # Initialize motors
    print("Initializing motors...")
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

    # Initialize encoders
    print("Initializing encoders...")
    enc_cfg = robot_config.get('robot', {})
    left_enc_cfg = EncoderConfig(
        channel_a_pin=gpio_config.left_encoder.channel_a,
        channel_b_pin=gpio_config.left_encoder.channel_b,
        counts_per_revolution=enc_cfg.get('encoder_cpr', 233)
    )
    right_enc_cfg = EncoderConfig(
        channel_a_pin=gpio_config.right_encoder.channel_a,
        channel_b_pin=gpio_config.right_encoder.channel_b,
        counts_per_revolution=enc_cfg.get('encoder_cpr', 233)
    )
    encoders = DualEncoders(left_enc_cfg, right_enc_cfg)

    # Initialize differential drive controller
    print("Initializing differential drive controller...")
    pid_cfg = robot_config.get('pid', {})
    nav_cfg = robot_config.get('navigation', {})
    drive_config = DriveConfig(
        wheel_diameter_mm=enc_cfg.get('wheel_diameter_mm', 52.0),
        wheel_base_mm=enc_cfg.get('wheel_base_mm', 215.9),
        encoder_cpr=enc_cfg.get('encoder_cpr', 233),
        max_rpm=enc_cfg.get('max_rpm', 753.0),
        deadband=robot_config.get('motors', {}).get('deadband', 2.0),
        left_pid=PIDConfig(
            kp=pid_cfg.get('left', {}).get('kp', 1.5),
            ki=pid_cfg.get('left', {}).get('ki', 0.5),
            kd=pid_cfg.get('left', {}).get('kd', 0.05),
            integral_limit=pid_cfg.get('left', {}).get('integral_limit', 100.0),
            output_limit=pid_cfg.get('left', {}).get('output_limit', 100.0)
        ),
        right_pid=PIDConfig(
            kp=pid_cfg.get('right', {}).get('kp', 1.5),
            ki=pid_cfg.get('right', {}).get('ki', 0.5),
            kd=pid_cfg.get('right', {}).get('kd', 0.05),
            integral_limit=pid_cfg.get('right', {}).get('integral_limit', 100.0),
            output_limit=pid_cfg.get('right', {}).get('output_limit', 100.0)
        ),
        control_rate_hz=robot_config.get('control', {}).get('motor_loop_hz', 100.0),
        heading_correction_enabled=False  # Disable for manual control
    )
    drive = DifferentialDriveController(drive_config, motors, encoders)
    drive.start()

    # Vision components (if enabled)
    camera = None
    matcher = None
    depth_preprocessor = None
    path_planner = None

    if not args.no_vision:
        try:
            # Initialize stereo camera
            print("Initializing stereo camera...")
            resolution = StereoCamera.RESOLUTIONS['low']
            stereo_cfg = robot_config.get('stereo_camera', {})
            flip_180 = stereo_cfg.get('flip_180', False)

            camera = StereoCamera(
                device_id=0,
                resolution=resolution,
                fps=30,
                flip_180=flip_180
            )

            if not camera.open():
                raise RuntimeError("Failed to open camera")

            print(f"  Camera opened: {camera.resolution[0]}x{camera.resolution[1]}")

            # Load calibration
            print("Loading calibration...")
            calib_path = Path(args.calibration)
            if not calib_path.exists():
                raise RuntimeError(f"Calibration not found: {calib_path}")

            calibration_data = load_calibration(str(calib_path))

            # Initialize stereo matcher
            params = SGBMParams(
                num_disparities=64,
                block_size=5,
                uniqueness_ratio=10,
                speckle_window_size=100,
                speckle_range=32
            )
            matcher = StereoMatcher(calibration_data, params, None)

            # Initialize depth preprocessor
            print("Initializing depth preprocessor...")
            camera_cfg = robot_config.get('camera', {})
            depth_cfg = robot_config.get('depth_preprocessing', {})

            preproc_config = DepthPreprocessorConfig(
                horizontal_fov_deg=camera_cfg.get('horizontal_fov_deg', 90.0),
                vertical_fov_deg=camera_cfg.get('vertical_fov_deg', 73.7),
                image_width=resolution[0],
                image_height=resolution[1],
                camera_height_mm=camera_cfg.get('height_mm', 139.7),
                camera_tilt_deg=camera_cfg.get('tilt_deg', -8.0),
                min_range_mm=depth_cfg.get('min_range_mm', 200.0),
                max_range_mm=depth_cfg.get('max_range_mm', 2000.0),
                num_sectors=depth_cfg.get('num_sectors', 72),
                floor_detection_method=depth_cfg.get('floor_detection_method', 'height'),
                floor_threshold_mm=depth_cfg.get('height_floor', {}).get('floor_threshold_mm', 75.0),
                robot_height_mm=depth_cfg.get('height_floor', {}).get('robot_height_mm', 533.0)
            )
            depth_preprocessor = DepthPreprocessor(preproc_config)

            # Initialize path planner
            path_planner = create_path_planner(robot_config)
            print(f"  Path planner: {path_planner.name}")

        except Exception as e:
            print(f"  Vision init failed: {e}")
            print("  Continuing without obstacle avoidance...")
            camera = None

    print("=" * 60)
    print("Controls:")
    print("  Left Stick Y   - Throttle (forward/back)")
    print("  Left Stick X   - Turn (left/right)")
    print("  Cross (X)      - Coast/stop")
    print("  Circle (O)      - Brake (active)")
    print("  D-pad Up/Down  - Speed limit +/- 5%")
    print("  Start           - Quit")
    print("=" * 60)
    print(f"Mode: {'Closed-Loop + Safety' if camera else 'Closed-Loop (No Safety)'}")
    print(f"Speed Limit: {args.max_speed}% | Deadzone: {args.deadzone*100:.0f}%")
    print("-" * 60)

    # Control state
    speed_limit = args.max_speed
    last_dpad_y = 0  # For edge detection
    running = True
    max_linear_speed = robot_config.get('navigation', {}).get('max_linear_speed_mm_s', 500.0)
    max_angular_speed = robot_config.get('navigation', {}).get('max_angular_speed_deg_s', 90.0)

    try:
        while running:
            # Get controller input
            cmd_throttle, cmd_turn = gamepad.get_state()

            # Apply deadzone - use fixed speed when pushed past deadzone
            if abs(cmd_throttle) >= args.deadzone:
                cmd_throttle = 1.0 if cmd_throttle > 0 else -1.0
            else:
                cmd_throttle = 0.0

            if abs(cmd_turn) >= args.deadzone:
                cmd_turn = 1.0 if cmd_turn > 0 else -1.0
            else:
                cmd_turn = 0.0

            # D-pad speed limit adjustment (edge detection)
            dpad_x, dpad_y = gamepad.get_dpad()
            if dpad_y != last_dpad_y:  # State changed
                if dpad_y == -1:  # Up pressed
                    speed_limit = min(100, speed_limit + 5)
                elif dpad_y == 1:  # Down pressed
                    speed_limit = max(0, speed_limit - 5)
                last_dpad_y = dpad_y

            # Button handling (ShanWan mapping)
            if gamepad.is_button_pressed(2):  # Cross = coast (free-wheel)
                drive.stop()  # Stop PID control
                motors.stop()  # Coast motors (no active braking)
                print_status(
                    cmd_throttle, cmd_turn, 0.0, 0.0,
                    speed_limit, "COAST"
                )
                time.sleep(0.1)
                drive.start()  # Restart control loop
                continue

            if gamepad.is_button_pressed(1):  # Circle = brake
                drive.stop()  # Use hardware brake, not PID
                motors.brake()
                print_status(
                    cmd_throttle, cmd_turn, 0.0, 0.0,
                    speed_limit, "BRAKING"
                )
                time.sleep(0.1)
                drive.start()  # Restart control loop
                continue

            if gamepad.is_button_pressed(9):  # Start = quit
                running = False
                break

            # Obstacle avoidance
            safety_status = "OK"
            obstacles_blocked = 0

            if camera is not None and matcher is not None:
                # Capture frame
                ret, left, right = camera.read()
                if ret and left is not None and right is not None:
                    # Compute depth
                    _, _, disparity = matcher.process_frame(left, right)
                    depth = matcher.disparity_to_depth(disparity)

                    # Preprocess depth
                    preproc_result = depth_preprocessor.process(depth)

                    # Check for obstacles
                    planner_result = path_planner.compute(
                        preproc_result.distances,
                        preproc_result.sector_angles,
                        target_heading_deg=0.0
                    )

                    # Count blocked sectors (closer than safety distance)
                    if len(preproc_result.distances) > 0:
                        obstacles_blocked = int(
                            (preproc_result.distances < safety_distance_mm).sum()
                        )

                    # Safety: full stop on throttle if blocked
                    if not planner_result.can_proceed:
                        cmd_throttle = 0.0
                        safety_status = "BLOCKED"
                else:
                    # Camera frame failed
                    safety_status = "NO_DATA"

            # Apply speed limit
            target_linear = cmd_throttle * max_linear_speed * (speed_limit / 100.0)
            target_angular = cmd_turn * max_angular_speed * (speed_limit / 100.0)

            # Send command to differential drive controller
            drive.set_velocity(target_linear, target_angular)

            # Get actual velocities for display
            actual_linear, actual_angular = drive.get_actual_velocity()

            # Display status
            print_status(
                cmd_throttle,
                cmd_turn,
                actual_linear,
                actual_angular,
                speed_limit,
                safety_status,
                obstacles_blocked
            )

            time.sleep(0.02)  # 50Hz control loop

    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    finally:
        drive.stop()
        if camera is not None:
            camera.release()
        gamepad.cleanup()
        print("\nCleanup complete.")


if __name__ == '__main__':
    main()
