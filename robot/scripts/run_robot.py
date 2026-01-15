#!/usr/bin/env python3
"""
Main robot control loop for person following.

Integrates:
- Stereo vision for depth/obstacle detection
- Tracking camera for brightness-based target detection
- UWB for person tracking and range estimation
- Sensor fusion for robust target localization
- Path planning for obstacle avoidance
- Differential drive with PID control
"""

import sys
import signal
import time
import threading
import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from robot.src.gpio_config import load_gpio_config, load_robot_config
from robot.src.motor_driver import MotorDriverConfig, DualMotorDriver
from robot.src.encoder import EncoderConfig, DualEncoders
from robot.src.differential_drive import DriveConfig, PIDConfig, DifferentialDriveController
from robot.src.odometry import OdometryConfig, WheelOdometry
from robot.src.uwb_tracker import UWBModuleConfig, DualUWBAnchors, RangeFilterConfig
from robot.src.uwb_triangulation import UWBTriangulator, TriangulationConfig
from robot.src.depth_preprocessor import DepthPreprocessor, DepthPreprocessorConfig
from robot.src.path_planner import PathPlanner, PlannerResult
from robot.src.path_planner_factory import create_path_planner
from robot.src.navigation import NavigationController, NavigationConfig, NavigationState
from robot.src.target_detector import TargetDetector, TargetDetectorConfig, TargetDetection
from robot.src.target_tracker import TargetTracker, TargetTrackerConfig, TargetState
from robot.src.tracking_camera import TrackingCamera, TrackingCameraConfig

from vision.src.camera import StereoCamera
from vision.src.stereo_matcher import StereoMatcher, SGBMParams, WLSParams
from vision.src.calibration import load_calibration


class PersonFollowingRobot:
    """
    Main robot controller integrating all subsystems.

    Architecture:
    - Main thread: Coordination and navigation (20 Hz)
    - Motor thread: PID control loop (100 Hz, via DifferentialDriveController)
    - Vision thread: Camera capture and processing (runs continuously)
      - Stereo camera (device 0): Depth estimation for obstacle avoidance
      - Tracking camera (device 2): Brightness detection for target tracking
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        use_vision: bool = True,
        filter_enabled: bool = False,
        filter_alpha: Optional[float] = None
    ):
        """
        Initialize all robot subsystems.

        Args:
            config_path: Path to robot config (uses default if None)
            use_vision: Whether to use stereo vision for obstacles
            filter_enabled: Force enable UWB range filtering (overrides config)
            filter_alpha: EMA alpha for filtering (overrides config if set)
        """
        # Load configuration
        self._gpio_config = load_gpio_config()
        self._robot_config = load_robot_config(config_path)
        self._use_vision = use_vision
        self._filter_enabled = filter_enabled
        self._filter_alpha = filter_alpha

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
        self._depth_preprocessor: Optional[DepthPreprocessor] = None
        self._path_planner: Optional[PathPlanner] = None
        self._navigation: Optional[NavigationController] = None
        self._target_detector: Optional[TargetDetector] = None
        self._target_tracker: Optional[TargetTracker] = None

        # Vision components
        self._stereo_camera: Optional[StereoCamera] = None
        self._stereo_matcher: Optional[StereoMatcher] = None
        self._tracking_camera: Optional[TrackingCamera] = None
        self._vision_thread: Optional[threading.Thread] = None

        # Latest sensor data (shared between vision thread and main loop)
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_target_detection: Optional[TargetDetection] = None
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
            self._init_target_tracker()

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
        """Initialize UWB tracking with optional filtering."""
        print("  Initializing UWB...")
        gpio = self._gpio_config
        uwb = self._robot_config.get('uwb', {})

        # Build anchor configs
        anchor1_cfg = UWBModuleConfig(
            uart_port=gpio.uwb_anchor1.uart_port,
            baud_rate=uwb.get('baud_rate', 115200),
            network_id=uwb.get('network_id', '0x1234'),
            address=uwb.get('anchor1_address', 'ANCHOR01'),
            target_address=uwb.get('target_address', 'TAG001'),
            timeout_ms=uwb.get('timeout_ms', 100),
            reset_pin=gpio.uwb_anchor1.reset_pin
        )
        anchor2_cfg = UWBModuleConfig(
            uart_port=gpio.uwb_anchor2.uart_port,
            baud_rate=uwb.get('baud_rate', 115200),
            network_id=uwb.get('network_id', '0x1234'),
            address=uwb.get('anchor2_address', 'ANCHOR02'),
            target_address=uwb.get('target_address', 'TAG001'),
            timeout_ms=uwb.get('timeout_ms', 100),
            reset_pin=gpio.uwb_anchor2.reset_pin
        )

        # Build filter config from YAML (can be overridden by CLI)
        filter_cfg = None
        filter_yaml = uwb.get('filter', {})
        if filter_yaml.get('enabled', False) or self._filter_enabled:
            # CLI override takes precedence
            alpha = self._filter_alpha if self._filter_alpha else filter_yaml.get('ema_alpha', 0.25)
            filter_cfg = RangeFilterConfig(
                enabled=True,
                ema_alpha=alpha,
                outlier_threshold_mm=filter_yaml.get('outlier_threshold_mm', 200.0),
                min_samples=filter_yaml.get('min_samples', 3)
            )
            print(f"    Range filtering enabled (alpha={filter_cfg.ema_alpha})")

        self._uwb_anchors = DualUWBAnchors(anchor1_cfg, anchor2_cfg, filter_cfg)

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
        """Initialize vision-based obstacle detection and target tracking."""
        print("  Initializing vision integration...")
        camera_cfg = self._robot_config.get('camera', {})
        depth_cfg = self._robot_config.get('depth_preprocessing', {})
        stereo_cfg = self._robot_config.get('stereo_camera', {})
        tracking_cfg = self._robot_config.get('tracking_camera', {})
        target_cfg = self._robot_config.get('target_detection', {})

        # Get FOV for angle calculations
        horizontal_fov = camera_cfg.get('horizontal_fov_deg', 60.0)

        # Initialize stereo camera for depth estimation
        self._init_stereo_camera(stereo_cfg)

        # Initialize tracking camera for brightness detection (if enabled)
        pattern_type = target_cfg.get('pattern_type', 'checkerboard')
        if pattern_type == 'brightness':
            self._init_tracking_camera(tracking_cfg)

        # Depth preprocessor
        preprocessor_config = DepthPreprocessorConfig(
            horizontal_fov_deg=horizontal_fov,
            vertical_fov_deg=camera_cfg.get('vertical_fov_deg', 45.0),
            camera_height_mm=camera_cfg.get('height_mm', 200.0),
            camera_tilt_deg=camera_cfg.get('tilt_deg', 0.0),
            min_range_mm=depth_cfg.get('min_range_mm', 200.0),
            max_range_mm=depth_cfg.get('max_range_mm', 3000.0),
            num_sectors=depth_cfg.get('num_sectors', 72),
            floor_detection_method=depth_cfg.get('floor_detection_method', 'height'),
            floor_threshold_mm=depth_cfg.get('height_floor', {}).get('floor_threshold_mm', 50.0),
            robot_height_mm=depth_cfg.get('height_floor', {}).get('robot_height_mm', 500.0)
        )
        self._depth_preprocessor = DepthPreprocessor(preprocessor_config)

        # Path planner (VFH, Follow-the-Gap, or APF based on config)
        self._path_planner = create_path_planner(self._robot_config)
        print(f"    Using path planning algorithm: {self._path_planner.name}")

        # Target detector for visual target tracking
        # For brightness mode, use tracking camera FOV; otherwise use stereo camera FOV
        if pattern_type == 'brightness':
            detector_fov = tracking_cfg.get('horizontal_fov_deg', 60.0)
        else:
            detector_fov = horizontal_fov

        self._init_target_detector(target_cfg, detector_fov)

    def _init_stereo_camera(self, stereo_cfg: dict) -> None:
        """Initialize stereo camera and stereo matcher for depth estimation."""
        print("    Initializing stereo camera...")

        device_id = stereo_cfg.get('device_id', 0)
        resolution_name = stereo_cfg.get('resolution', 'low')
        fps = stereo_cfg.get('fps', 30)
        calibration_path = stereo_cfg.get('calibration_path', 'vision/data/calibration_data')

        # Map resolution name to actual resolution
        resolution_map = {
            'low': StereoCamera.RESOLUTIONS['low'],
            'medium': StereoCamera.RESOLUTIONS['medium'],
            'high': StereoCamera.RESOLUTIONS['high'],
        }
        resolution = resolution_map.get(resolution_name, StereoCamera.RESOLUTIONS['low'])

        self._stereo_camera = StereoCamera(
            device_id=device_id,
            resolution=resolution,
            fps=fps
        )

        if not self._stereo_camera.open():
            raise RuntimeError(f"Failed to open stereo camera (device {device_id})")

        actual_res = self._stereo_camera.resolution
        print(f"      Stereo camera opened: {actual_res[0]}x{actual_res[1]}")

        # Load calibration and create stereo matcher
        calib_path = Path(calibration_path)
        if not calib_path.exists():
            raise RuntimeError(f"Calibration path not found: {calib_path}")

        calibration_data = load_calibration(str(calib_path))
        print(f"      Calibration loaded: {calibration_data['calibration_date']}")
        print(f"      Baseline: {calibration_data['baseline_mm']:.1f}mm")

        # Create stereo matcher with default SGBM parameters
        params = SGBMParams(
            num_disparities=64,
            block_size=5,
            uniqueness_ratio=10,
            speckle_window_size=100,
            speckle_range=32
        )

        # Create WLS filter parameters if configured
        wls_cfg = stereo_cfg.get('wls_filter', {})
        wls_params = None
        if wls_cfg.get('enabled', False):
            wls_params = WLSParams(
                enabled=True,
                lambda_=wls_cfg.get('lambda', 8000.0),
                sigma_color=wls_cfg.get('sigma_color', 1.5),
                confidence_threshold=wls_cfg.get('confidence_threshold', 0.0)
            )
            print(f"      WLS filtering enabled (lambda={wls_params.lambda_}, "
                  f"sigma={wls_params.sigma_color}, conf_thresh={wls_params.confidence_threshold})")

        self._stereo_matcher = StereoMatcher(calibration_data, params, wls_params)

    def _init_tracking_camera(self, tracking_cfg: dict) -> None:
        """Initialize dedicated tracking camera for brightness detection."""
        print("    Initializing tracking camera...")

        resolution = tracking_cfg.get('resolution', [640, 480])
        if isinstance(resolution, list):
            resolution = tuple(resolution)

        config = TrackingCameraConfig(
            device_id=tracking_cfg.get('device_id', 2),
            resolution=resolution,
            fps=tracking_cfg.get('fps', 30),
            exposure=tracking_cfg.get('exposure', 20),
            auto_exposure=tracking_cfg.get('auto_exposure', False),
            horizontal_fov_deg=tracking_cfg.get('horizontal_fov_deg', 60.0)
        )

        self._tracking_camera = TrackingCamera(config)

        if not self._tracking_camera.open():
            raise RuntimeError(f"Failed to open tracking camera (device {config.device_id})")

        actual_res = self._tracking_camera.get_resolution()
        print(f"      Tracking camera opened: {actual_res[0]}x{actual_res[1]}")
        print(f"      Exposure: {config.exposure} (auto={config.auto_exposure})")

    def _init_target_detector(self, target_cfg: dict, horizontal_fov: float) -> None:
        """Initialize target detector for visual target tracking."""
        bullseye_cfg = target_cfg.get('bullseye', {})
        checker_cfg = target_cfg.get('checkerboard', {})
        brightness_cfg = target_cfg.get('brightness', {})

        detector_config = TargetDetectorConfig(
            pattern_type=target_cfg.get('pattern_type', 'checkerboard'),
            depth_sample_radius=target_cfg.get('depth_sample_radius', 3),
            # Bullseye settings
            bullseye_min_rings=bullseye_cfg.get('min_rings', 1),
            bullseye_circularity_threshold=bullseye_cfg.get('circularity_threshold', 0.3),
            bullseye_concentricity_threshold_px=bullseye_cfg.get('concentricity_threshold_px', 5),
            bullseye_min_radius_px=bullseye_cfg.get('min_radius_px', 10),
            bullseye_max_radius_px=bullseye_cfg.get('max_radius_px', 150),
            bullseye_blur_kernel_size=bullseye_cfg.get('blur_kernel_size', 1),
            bullseye_canny_low=bullseye_cfg.get('canny_low', 50),
            bullseye_canny_high=bullseye_cfg.get('canny_high', 150),
            # Checkerboard settings
            checker_sample_size=checker_cfg.get('sample_size', 8),
            checker_contrast_threshold=checker_cfg.get('contrast_threshold', 0.3),
            checker_corner_quality=checker_cfg.get('corner_quality', 0.1),
            checker_min_distance=checker_cfg.get('min_distance', 20),
            checker_max_corners=checker_cfg.get('max_corners', 50),
            checker_block_size=checker_cfg.get('block_size', 7),
            checker_diagonal_tolerance=checker_cfg.get('diagonal_tolerance', 0.2),
            # Brightness settings
            brightness_threshold=brightness_cfg.get('threshold', 230),
            brightness_gain=brightness_cfg.get('gain', 1.6),
            brightness_blur_kernel_size=brightness_cfg.get('blur_kernel_size', 1),
            brightness_min_area_px=brightness_cfg.get('min_area_px', 20),
            brightness_max_area_px=brightness_cfg.get('max_area_px', 200),
        )
        self._target_detector = TargetDetector(detector_config, horizontal_fov)
        print(f"    Target detection enabled: {detector_config.pattern_type.upper()}")

    def _init_target_tracker(self) -> None:
        """Initialize target tracker for sensor fusion."""
        print("  Initializing target tracker...")
        tracker_cfg = self._robot_config.get('target_tracking', {})

        tracker_config = TargetTrackerConfig(
            ema_alpha=tracker_cfg.get('ema_alpha', 0.25),
            outlier_threshold_mm=tracker_cfg.get('outlier_threshold_mm', 200.0),
            angle_ema_alpha=tracker_cfg.get('angle_ema_alpha', 0.3),
            visual_range_threshold_mm=tracker_cfg.get('visual_range_threshold_mm', 3000.0),
            visual_confidence_threshold=tracker_cfg.get('visual_confidence_threshold', 0.5),
            visual_timeout_ms=tracker_cfg.get('visual_timeout_ms', 500.0),
            uwb_timeout_ms=tracker_cfg.get('uwb_timeout_ms', 500.0),
            min_samples=tracker_cfg.get('min_samples', 3)
        )
        self._target_tracker = TargetTracker(tracker_config)
        print("    Sensor fusion enabled (UWB + Visual)")

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

    def _start_vision_thread(self) -> None:
        """Start the vision processing thread."""
        if not self._use_vision:
            return

        if self._stereo_camera is None:
            print("Warning: Vision thread not started - no stereo camera")
            return

        self._vision_thread = threading.Thread(
            target=self._vision_loop,
            name="VisionThread",
            daemon=True
        )
        self._vision_thread.start()
        print("  Vision thread started")

    def _vision_loop(self) -> None:
        """
        Vision processing loop running in separate thread.

        Continuously captures from:
        - Stereo camera: Computes depth map for obstacle avoidance
        - Tracking camera: Detects brightness target (if enabled)

        Results are stored in shared variables protected by _depth_lock.
        """
        print("    Vision loop running...")

        while self._running and not self._shutdown_event.is_set():
            try:
                # Capture and process stereo frame for depth
                stereo_frame = self._stereo_camera.read()
                depth_map = None
                left_frame = None

                if stereo_frame is not None and self._stereo_matcher is not None:
                    left, right = self._stereo_camera.split_frames(stereo_frame)
                    left_frame = left
                    depth_map = self._stereo_matcher.compute_depth(left, right)

                # Capture from tracking camera and detect brightness target
                target_detection = None
                if self._tracking_camera is not None and self._target_detector is not None:
                    ret, tracking_frame = self._tracking_camera.read()
                    if ret and tracking_frame is not None:
                        # Brightness detection uses angle only (UWB provides range)
                        target_detection = self._target_detector.detect(tracking_frame, None)

                # Store results for main loop
                with self._depth_lock:
                    self._latest_depth = depth_map
                    self._latest_frame = left_frame
                    self._latest_target_detection = target_detection

            except Exception as e:
                print(f"Vision loop error: {e}")
                time.sleep(0.1)  # Avoid tight loop on persistent errors

        print("    Vision loop stopped")

    def set_depth_map(self, depth_map: np.ndarray) -> None:
        """
        Update depth map from external vision process.

        Args:
            depth_map: Depth values in mm
        """
        with self._depth_lock:
            self._latest_depth = depth_map.copy()

    def set_frame_and_depth(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray
    ) -> None:
        """
        Update both camera frame and depth map from external vision process.

        The frame is used for bullseye detection, and depth map for obstacle
        detection. Both should come from the same camera capture for consistency.

        Args:
            frame: BGR or grayscale camera frame (for bullseye detection)
            depth_map: Depth values in mm (for obstacle detection)
        """
        with self._depth_lock:
            self._latest_frame = frame.copy()
            self._latest_depth = depth_map.copy()

    def _get_latest_vision_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get latest camera frame and depth map.

        Returns:
            Tuple of (frame, depth_map) - either may be None
        """
        with self._depth_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
            depth = self._latest_depth.copy() if self._latest_depth is not None else None
            return frame, depth

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

        # Start vision thread (captures from stereo + tracking cameras)
        self._start_vision_thread()

        # Control loop timing
        control = self._robot_config.get('control', {})
        loop_hz = control.get('main_loop_hz', 20)
        loop_period = 1.0 / loop_hz

        while self._running and not self._shutdown_event.is_set():
            loop_start = time.time()
            self._loop_count += 1

            try:
                # 1. Get UWB readings (raw, unfiltered - tracker handles filtering)
                range1, range2 = self._uwb_anchors.poll_ranges_raw()

                # 2. Triangulate UWB position
                uwb_angle = None
                uwb_range = None

                if range1 is not None and range2 is not None:
                    result = self._triangulator.triangulate(range1, range2)
                    if result is not None:
                        uwb_angle = result.angle_deg
                        uwb_range = result.estimated_distance_mm

                # 3. Get latest vision data from vision thread
                with self._depth_lock:
                    depth_map = self._latest_depth.copy() if self._latest_depth is not None else None
                    visual_detection = self._latest_target_detection

                # 4. Fuse UWB and visual with target tracker
                target_state = self._target_tracker.update(
                    uwb_angle, uwb_range, visual_detection
                )
                target_angle = target_state.angle_deg
                target_range = target_state.range_mm

                # 5. Process depth map and compute path
                planner_result = None
                if self._use_vision and self._path_planner is not None and depth_map is not None:
                    # Preprocess depth to 1D distances
                    preproc_result = self._depth_preprocessor.process(depth_map)

                    # Compute path using selected algorithm
                    planner_result = self._path_planner.compute(
                        preproc_result.distances,
                        preproc_result.sector_angles,
                        target_angle if target_angle is not None else 0.0
                    )

                # 6. Fallback if no vision or no planner result
                if planner_result is None:
                    # No obstacles detected - proceed toward target
                    planner_result = PlannerResult(
                        best_heading_deg=target_angle if target_angle else 0.0,
                        can_proceed=True
                    )

                # 7. Compute navigation command
                command = self._navigation.update(
                    target_angle,
                    target_range,
                    planner_result
                )

                # 8. Send to drive controller
                self._drive.set_velocity(
                    command.linear_velocity_mm_s,
                    command.angular_velocity_deg_s
                )

                # 9. Update odometry
                left_count, right_count = self._encoders.get_counts()
                pose = self._odometry.update(left_count, right_count)

                # 10. Print status periodically
                if self._loop_count % 20 == 0:
                    self._print_status(command, target_state, pose)

            except Exception as e:
                print(f"Control loop error: {e}")
                self._drive.set_velocity(0, 0)

            # Maintain loop timing
            elapsed = time.time() - loop_start
            sleep_time = loop_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _print_status(self, command, target_state: TargetState, pose):
        """Print status line."""
        state = command.state.name[:8]
        angle_str = f"{target_state.angle_deg:+6.1f}" if target_state.angle_deg else "  None"
        range_str = f"{target_state.range_mm:5.0f}" if target_state.range_mm else " None"
        source = target_state.source[:6]

        print(f"State: {state:8s} | "
              f"Target: {angle_str}deg {range_str}mm ({source}) | "
              f"Cmd: {command.linear_velocity_mm_s:+6.0f}mm/s "
              f"{command.angular_velocity_deg_s:+5.0f}deg/s | "
              f"Pose: ({pose.x/1000:.1f}, {pose.y/1000:.1f})m "
              f"{pose.theta_deg():+.0f}deg")

    def shutdown(self) -> None:
        """Gracefully shutdown all subsystems."""
        print("\nShutting down...")
        self._running = False
        self._shutdown_event.set()

        # Wait for vision thread to stop
        if self._vision_thread is not None and self._vision_thread.is_alive():
            print("  Stopping vision thread...")
            self._vision_thread.join(timeout=2.0)

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

        # Release cameras
        if self._stereo_camera is not None:
            print("  Releasing stereo camera...")
            self._stereo_camera.release()

        if self._tracking_camera is not None:
            print("  Releasing tracking camera...")
            self._tracking_camera.release()

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
    parser.add_argument('--filter', action='store_true',
                        help='Force enable UWB range filtering (overrides config)')
    parser.add_argument('--filter-alpha', type=float, default=None,
                        help='EMA alpha for UWB filtering (0.1=smooth, 0.5=responsive)')
    args = parser.parse_args()

    robot = PersonFollowingRobot(
        config_path=args.config,
        use_vision=not args.no_vision,
        filter_enabled=args.filter,
        filter_alpha=args.filter_alpha
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
