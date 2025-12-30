#!/usr/bin/env python3
"""
Main robot control loop for person following.

Integrates:
- Stereo vision for depth/obstacle detection
- UWB for person tracking
- VFH for obstacle avoidance
- Differential drive with PID control
"""

import sys
import signal
import time
import threading
import argparse
from pathlib import Path
from typing import Optional
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from robot.src.gpio_config import load_gpio_config, load_robot_config
from robot.src.motor_driver import MotorDriverConfig, DualMotorDriver
from robot.src.encoder import EncoderConfig, DualEncoders
from robot.src.differential_drive import DriveConfig, PIDConfig, DifferentialDriveController
from robot.src.odometry import OdometryConfig, WheelOdometry
from robot.src.uwb_tracker import UWBModuleConfig, DualUWBAnchors
from robot.src.uwb_triangulation import UWBTriangulator, TriangulationConfig
from robot.src.depth_to_polar import DepthToPolar, DepthToPolarConfig
from robot.src.vfh import VectorFieldHistogram, VFHConfig
from robot.src.navigation import NavigationController, NavigationConfig, NavigationState


class PersonFollowingRobot:
    """
    Main robot controller integrating all subsystems.

    Architecture:
    - Main thread: Coordination and navigation
    - Motor thread: PID control loop (via DifferentialDriveController)
    - Vision: Depth maps from stereo camera (optional external process)
    """

    def __init__(self, config_path: Optional[str] = None, use_vision: bool = True):
        """
        Initialize all robot subsystems.

        Args:
            config_path: Path to robot config (uses default if None)
            use_vision: Whether to use stereo vision for obstacles
        """
        # Load configuration
        self._gpio_config = load_gpio_config()
        self._robot_config = load_robot_config(config_path)
        self._use_vision = use_vision

        # State
        self._running = False
        self._shutdown_event = threading.Event()

        # Components (initialized in _init_* methods)
        self._motors: Optional[DualMotorDriver] = None
        self._encoders: Optional[DualEncoders] = None
        self._drive: Optional[DifferentialDriveController] = None
        self._odometry: Optional[WheelOdometry] = None
        self._uwb_anchors: Optional[DualUWBAnchors] = None
        self._triangulator: Optional[UWBTriangulator] = None
        self._depth_converter: Optional[DepthToPolar] = None
        self._vfh: Optional[VectorFieldHistogram] = None
        self._navigation: Optional[NavigationController] = None

        # Latest sensor data
        self._latest_depth: Optional[np.ndarray] = None
        self._depth_lock = threading.Lock()

        # Statistics
        self._loop_count = 0
        self._start_time = 0.0

    def initialize(self) -> bool:
        """
        Initialize all subsystems.

        Returns:
            True if initialization successful
        """
        print("Initializing robot subsystems...")

        try:
            self._init_motors()
            self._init_encoders()
            self._init_drive()
            self._init_odometry()
            self._init_uwb()
            self._init_navigation()

            if self._use_vision:
                self._init_vision()

            print("All subsystems initialized.")
            return True

        except Exception as e:
            print(f"Initialization failed: {e}")
            self.shutdown()
            return False

    def _init_motors(self) -> None:
        """Initialize motor drivers."""
        print("  Initializing motors...")
        gpio = self._gpio_config

        left_cfg = MotorDriverConfig(
            pwm_pin=gpio.left_motor.pwm_pin,
            dir_pin=gpio.left_motor.dir_pin,
            pwm_frequency=gpio.pwm_frequency
        )
        right_cfg = MotorDriverConfig(
            pwm_pin=gpio.right_motor.pwm_pin,
            dir_pin=gpio.right_motor.dir_pin,
            pwm_frequency=gpio.pwm_frequency
        )
        self._motors = DualMotorDriver(left_cfg, right_cfg)

    def _init_encoders(self) -> None:
        """Initialize encoders."""
        print("  Initializing encoders...")
        gpio = self._gpio_config
        robot = self._robot_config['robot']

        left_cfg = EncoderConfig(
            channel_a_pin=gpio.left_encoder.channel_a,
            channel_b_pin=gpio.left_encoder.channel_b,
            counts_per_revolution=robot['encoder_cpr']
        )
        right_cfg = EncoderConfig(
            channel_a_pin=gpio.right_encoder.channel_a,
            channel_b_pin=gpio.right_encoder.channel_b,
            counts_per_revolution=robot['encoder_cpr']
        )
        self._encoders = DualEncoders(left_cfg, right_cfg)

    def _init_drive(self) -> None:
        """Initialize differential drive controller."""
        print("  Initializing drive controller...")
        robot = self._robot_config['robot']
        pid = self._robot_config['pid']
        control = self._robot_config['control']

        left_pid = PIDConfig(
            kp=pid['left']['kp'],
            ki=pid['left']['ki'],
            kd=pid['left']['kd'],
            integral_limit=pid['left'].get('integral_limit', 100),
            output_limit=pid['left'].get('output_limit', 100)
        )
        right_pid = PIDConfig(
            kp=pid['right']['kp'],
            ki=pid['right']['ki'],
            kd=pid['right']['kd'],
            integral_limit=pid['right'].get('integral_limit', 100),
            output_limit=pid['right'].get('output_limit', 100)
        )

        drive_cfg = DriveConfig(
            wheel_diameter_mm=robot['wheel_diameter_mm'],
            wheel_base_mm=robot['wheel_base_mm'],
            encoder_cpr=robot['encoder_cpr'],
            max_rpm=robot['max_rpm'],
            left_pid=left_pid,
            right_pid=right_pid,
            control_rate_hz=control['motor_loop_hz']
        )

        self._drive = DifferentialDriveController(
            drive_cfg, self._motors, self._encoders
        )

    def _init_odometry(self) -> None:
        """Initialize odometry."""
        print("  Initializing odometry...")
        robot = self._robot_config['robot']

        odo_cfg = OdometryConfig(
            wheel_diameter_mm=robot['wheel_diameter_mm'],
            wheel_base_mm=robot['wheel_base_mm'],
            encoder_cpr=robot['encoder_cpr']
        )
        self._odometry = WheelOdometry(odo_cfg)

    def _init_uwb(self) -> None:
        """Initialize UWB tracking."""
        print("  Initializing UWB...")
        gpio = self._gpio_config
        uwb = self._robot_config.get('uwb', {})

        anchor1_cfg = UWBModuleConfig(
            uart_port=gpio.uwb_anchor1.uart_port,
            baud_rate=uwb.get('baud_rate', 115200),
            network_id=uwb.get('network_id', 0x1234),
            timeout_ms=uwb.get('timeout_ms', 100)
        )
        anchor2_cfg = UWBModuleConfig(
            uart_port=gpio.uwb_anchor2.uart_port,
            baud_rate=uwb.get('baud_rate', 115200),
            network_id=uwb.get('network_id', 0x1234),
            timeout_ms=uwb.get('timeout_ms', 100)
        )

        self._uwb_anchors = DualUWBAnchors(anchor1_cfg, anchor2_cfg)

        if not self._uwb_anchors.connect():
            print("    Warning: UWB connection failed")

        # Triangulator
        anchor1_offset = uwb.get('anchor1_offset_mm', [100, 50, 0])
        anchor2_offset = uwb.get('anchor2_offset_mm', [-100, 50, 0])
        calibration = uwb.get('calibration', {})

        tri_cfg = TriangulationConfig(
            anchor1_position=(anchor1_offset[0], anchor1_offset[1]),
            anchor2_position=(anchor2_offset[0], anchor2_offset[1]),
            front_offset_deg=calibration.get('front_offset_deg', 0.0),
            calibration_valid=calibration.get('valid', False)
        )
        self._triangulator = UWBTriangulator(tri_cfg)

    def _init_vision(self) -> None:
        """Initialize vision-based obstacle detection."""
        print("  Initializing vision integration...")
        camera = self._robot_config.get('camera', {})
        vfh_cfg = self._robot_config.get('vfh', {})

        # Depth converter
        depth_cfg = DepthToPolarConfig(
            horizontal_fov_deg=camera.get('horizontal_fov_deg', 60.0),
            vertical_fov_deg=camera.get('vertical_fov_deg', 45.0),
            camera_height_mm=camera.get('height_mm', 200.0),
            camera_tilt_deg=camera.get('tilt_deg', 0.0)
        )
        self._depth_converter = DepthToPolar(depth_cfg)

        # VFH
        vfh_config = VFHConfig(
            num_sectors=vfh_cfg.get('num_sectors', 72),
            min_range_mm=vfh_cfg.get('min_range_mm', 200),
            max_range_mm=vfh_cfg.get('max_range_mm', 3000),
            min_height_mm=vfh_cfg.get('min_height_mm', 50),
            max_height_mm=vfh_cfg.get('max_height_mm', 500),
            obstacle_threshold=vfh_cfg.get('obstacle_threshold', 0.3),
            safety_margin_mm=vfh_cfg.get('safety_margin_mm', 150),
            wide_valley_threshold=vfh_cfg.get('wide_valley_threshold', 3),
            narrow_valley_threshold=vfh_cfg.get('narrow_valley_threshold', 1)
        )
        self._vfh = VectorFieldHistogram(vfh_config, self._depth_converter)

    def _init_navigation(self) -> None:
        """Initialize navigation controller."""
        print("  Initializing navigation...")
        nav = self._robot_config.get('navigation', {})

        nav_cfg = NavigationConfig(
            target_follow_distance_mm=nav.get('target_follow_distance_mm', 1500),
            follow_distance_tolerance_mm=nav.get('follow_distance_tolerance_mm', 200),
            angular_tolerance_deg=nav.get('angular_tolerance_deg', 10),
            spin_threshold_deg=nav.get('spin_threshold_deg', 45),
            stop_if_blocked_timeout_s=nav.get('stop_if_blocked_timeout_s', 3),
            max_linear_speed_mm_s=nav.get('max_linear_speed_mm_s', 500),
            max_angular_speed_deg_s=nav.get('max_angular_speed_deg_s', 90),
            approach_speed_factor=nav.get('approach_speed_factor', 0.5)
        )
        self._navigation = NavigationController(nav_cfg)

    def set_depth_map(self, depth_map: np.ndarray) -> None:
        """
        Update depth map from external vision process.

        Args:
            depth_map: Depth values in mm
        """
        with self._depth_lock:
            self._latest_depth = depth_map.copy()

    def _get_latest_depth(self) -> Optional[np.ndarray]:
        """Get latest depth map."""
        with self._depth_lock:
            if self._latest_depth is not None:
                return self._latest_depth.copy()
            return None

    def run(self) -> None:
        """Main control loop."""
        print("\nStarting main control loop...")
        print("Press Ctrl+C to stop.\n")

        self._running = True
        self._start_time = time.time()

        # Start drive controller
        self._drive.start()

        # Control loop timing
        control = self._robot_config.get('control', {})
        loop_hz = control.get('main_loop_hz', 20)
        loop_period = 1.0 / loop_hz

        camera = self._robot_config.get('camera', {})
        camera_fov = camera.get('horizontal_fov_deg', 60.0)

        while self._running and not self._shutdown_event.is_set():
            loop_start = time.time()
            self._loop_count += 1

            try:
                # 1. Get UWB readings
                range1, range2 = self._uwb_anchors.get_ranges()

                # 2. Triangulate person position
                target_angle = None
                target_range = None

                if range1 is not None and range2 is not None:
                    result = self._triangulator.triangulate(range1, range2)
                    if result is not None:
                        target_angle = result.angle_deg
                        target_range = result.estimated_distance_mm

                # 3. Update VFH from depth map
                if self._use_vision and self._vfh is not None:
                    depth_map = self._get_latest_depth()
                    if depth_map is not None:
                        self._vfh.update_from_depth(depth_map, camera_fov)

                # 4. Get safe direction from VFH
                vfh_result = self._vfh.find_safe_direction(
                    target_angle if target_angle is not None else 0.0
                )

                # 5. Compute navigation command
                command = self._navigation.update(
                    target_angle,
                    target_range,
                    vfh_result
                )

                # 6. Send to drive controller
                self._drive.set_velocity(
                    command.linear_velocity_mm_s,
                    command.angular_velocity_deg_s
                )

                # 7. Update odometry
                left_count, right_count = self._encoders.get_counts()
                pose = self._odometry.update(left_count, right_count)

                # 8. Print status periodically
                if self._loop_count % 20 == 0:
                    self._print_status(command, target_angle, target_range, pose)

            except Exception as e:
                print(f"Control loop error: {e}")
                self._drive.set_velocity(0, 0)

            # Maintain loop timing
            elapsed = time.time() - loop_start
            sleep_time = loop_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _print_status(self, command, target_angle, target_range, pose):
        """Print status line."""
        state = command.state.name[:8]
        angle_str = f"{target_angle:+6.1f}" if target_angle else "  None"
        range_str = f"{target_range:5.0f}" if target_range else " None"

        print(f"State: {state:8s} | "
              f"Target: {angle_str}deg {range_str}mm | "
              f"Cmd: {command.linear_velocity_mm_s:+6.0f}mm/s "
              f"{command.angular_velocity_deg_s:+5.0f}deg/s | "
              f"Pose: ({pose.x/1000:.1f}, {pose.y/1000:.1f})m "
              f"{pose.theta_deg():+.0f}deg")

    def shutdown(self) -> None:
        """Gracefully shutdown all subsystems."""
        print("\nShutting down...")
        self._running = False
        self._shutdown_event.set()

        # Stop drive controller
        if self._drive is not None:
            self._drive.stop()

        # Cleanup hardware
        if self._motors is not None:
            self._motors.cleanup()

        if self._encoders is not None:
            self._encoders.cleanup()

        if self._uwb_anchors is not None:
            self._uwb_anchors.close()

        # Print statistics
        elapsed = time.time() - self._start_time if self._start_time else 0
        if elapsed > 0:
            avg_hz = self._loop_count / elapsed
            print(f"Ran {self._loop_count} loops in {elapsed:.1f}s ({avg_hz:.1f} Hz)")

        print("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Person-following robot")
    parser.add_argument('--config', type=str, help='Path to robot config')
    parser.add_argument('--no-vision', action='store_true',
                        help='Disable vision-based obstacles')
    args = parser.parse_args()

    robot = PersonFollowingRobot(
        config_path=args.config,
        use_vision=not args.no_vision
    )

    # Signal handlers
    def signal_handler(sig, frame):
        print("\nShutdown requested...")
        robot.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize and run
    if robot.initialize():
        robot.run()
    else:
        print("Failed to initialize robot.")
        sys.exit(1)


if __name__ == '__main__':
    main()
