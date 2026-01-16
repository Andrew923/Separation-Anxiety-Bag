#!/usr/bin/env python3
"""
Integration test GUI for person-following robot.

Fuses stereo vision, UWB tracking, and visual target detection with visualization.
Does NOT require motors - for sensor integration testing only.

Usage:
    # Default (follow_gap algorithm)
    python robot/scripts/integration_test.py

    # Use APF algorithm
    python robot/scripts/integration_test.py --algorithm apf
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

import cv2
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from robot.src.gpio_config import load_gpio_config, load_robot_config
from robot.src.uwb_tracker import UWBModuleConfig, DualUWBAnchors, RangeFilterConfig
from robot.src.uwb_triangulation import UWBTriangulator, TriangulationConfig, TriangulationResult
from robot.src.depth_preprocessor import DepthPreprocessor, DepthPreprocessorConfig
from robot.src.path_planner import PathPlanner, PlannerResult
from robot.src.path_planner_factory import create_path_planner, get_available_algorithms
from robot.src.target_detector import TargetDetector, TargetDetectorConfig, TargetDetection
from robot.src.target_tracker import TargetTracker, TargetTrackerConfig, TargetState
from robot.src.tracking_camera import TrackingCamera, TrackingCameraConfig
from robot.src.navigation import (
    NavigationController, NavigationConfig, NavigationState, NavigationCommand
)
from robot.src.depth_preprocessor import PreprocessorResult
from robot.src.visualizers.depth_preprocessor_viz import (
    DepthPreprocessorVisualizer, DepthVisualizerConfig
)

from vision.src.camera import StereoCamera
from vision.src.stereo_matcher import StereoMatcher, SGBMParams, WLSParams
from vision.src.calibration import load_calibration


# Available colormaps for depth visualization
COLORMAPS = ['JET', 'TURBO', 'MAGMA', 'INFERNO', 'PLASMA', 'VIRIDIS']


@dataclass
class IntegrationConfig:
    """Configuration for integration test."""
    calibration_path: str = "vision/data/calibration_data"
    device_id: int = 0
    config_path: Optional[str] = None
    algorithm: str = "follow_gap"  # "follow_gap" or "apf"
    target_fps: float = 20.0
    debug_view: bool = False  # Show debug visualization instead of camera feed


class IntegrationTester:
    """
    Integration test controller for sensor fusion.

    Manages stereo camera, UWB modules, visual target detection,
    and visualization without requiring motor hardware.
    """

    def __init__(self, config: IntegrationConfig):
        """
        Initialize integration tester.

        Args:
            config: Integration test configuration
        """
        self._config = config
        self._running = False

        # Load robot configuration
        self._gpio_config = load_gpio_config()
        self._robot_config = load_robot_config(config.config_path)

        # Override algorithm in config if specified
        if config.algorithm:
            if 'path_planning' not in self._robot_config:
                self._robot_config['path_planning'] = {}
            self._robot_config['path_planning']['algorithm'] = config.algorithm

        # Components (initialized in initialize())
        self._camera: Optional[StereoCamera] = None
        self._matcher: Optional[StereoMatcher] = None
        self._tracking_camera: Optional[TrackingCamera] = None
        self._uwb_anchors: Optional[DualUWBAnchors] = None
        self._triangulator: Optional[UWBTriangulator] = None
        self._depth_preprocessor: Optional[DepthPreprocessor] = None
        self._path_planner: Optional[PathPlanner] = None
        self._target_detector: Optional[TargetDetector] = None
        self._target_tracker: Optional[TargetTracker] = None
        self._navigation: Optional[NavigationController] = None
        self._horizontal_fov: float = 60.0  # Default, updated in _init_path_planning

        # State
        self._colormap_index = 0
        self._show_epipolar = False
        self._debug_view = config.debug_view  # Toggle for debug visualization
        self._fps = 0.0
        self._frame_times: List[float] = []

        # Visualizers
        self._depth_viz: Optional[DepthPreprocessorVisualizer] = None

        # Latest data
        self._latest_ranges: Tuple[Optional[float], Optional[float]] = (None, None)
        self._latest_triangulation: Optional[TriangulationResult] = None
        self._latest_planner_result: Optional[PlannerResult] = None
        self._latest_preproc_result: Optional[PreprocessorResult] = None
        self._latest_nav_cmd: Optional[NavigationCommand] = None
        self._latest_target_detection: Optional[TargetDetection] = None
        self._latest_target_state: Optional[TargetState] = None
        self._latest_left_frame: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None

    def initialize(self) -> bool:
        """
        Initialize all subsystems.

        Returns:
            True if initialization successful
        """
        print("Initializing integration test...")
        print(f"  Algorithm: {self._config.algorithm}")

        try:
            # Vision (always enabled)
            self._init_camera()
            self._init_stereo()

            # UWB (always enabled)
            if not self._init_uwb():
                print("Warning: UWB initialization failed")

            # Path planning and obstacle detection
            self._init_path_planning()

            # Target tracking (bullseye + sensor fusion)
            self._init_target_tracking()

            # Navigation
            self._init_navigation()

            print("Initialization complete.")
            return True

        except Exception as e:
            print(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.cleanup()
            return False

    def _init_camera(self) -> None:
        """Initialize stereo camera."""
        print("  Initializing camera...")

        # Use low resolution (640x240 full frame = 320x240 per camera)
        resolution = StereoCamera.RESOLUTIONS['low']
        self._camera = StereoCamera(
            device_id=self._config.device_id,
            resolution=resolution,
            fps=30
        )

        if not self._camera.open():
            raise RuntimeError(f"Failed to open camera device {self._config.device_id}")

        actual_res = self._camera.resolution
        print(f"    Camera opened: {actual_res[0]}x{actual_res[1]}")

    def _init_stereo(self) -> None:
        """Initialize stereo matcher with calibration."""
        print("  Loading calibration...")

        calib_path = Path(self._config.calibration_path)
        if not calib_path.exists():
            raise RuntimeError(f"Calibration path not found: {calib_path}")

        calibration_data = load_calibration(str(calib_path))
        print(f"    Calibration loaded: {calibration_data['calibration_date']}")
        print(f"    Image size: {calibration_data['image_size']}")
        print(f"    Baseline: {calibration_data['baseline_mm']:.1f}mm")

        # Create stereo matcher
        params = SGBMParams(
            num_disparities=64,
            block_size=5,
            uniqueness_ratio=10,
            speckle_window_size=100,
            speckle_range=32
        )

        # Create WLS filter parameters from config
        stereo_cfg = self._robot_config.get('stereo_camera', {})
        wls_cfg = stereo_cfg.get('wls_filter', {})
        wls_params = None
        if wls_cfg.get('enabled', False):
            wls_params = WLSParams(
                enabled=True,
                lambda_=wls_cfg.get('lambda', 8000.0),
                sigma_color=wls_cfg.get('sigma_color', 1.5),
                confidence_threshold=wls_cfg.get('confidence_threshold', 0.0)
            )
            print(f"    WLS filtering enabled (lambda={wls_params.lambda_}, "
                  f"sigma={wls_params.sigma_color}, conf_thresh={wls_params.confidence_threshold})")

        self._matcher = StereoMatcher(calibration_data, params, wls_params)
        self._matcher.set_colormap(COLORMAPS[self._colormap_index])

    def _init_uwb(self) -> bool:
        """
        Initialize UWB tracking with retry on failure.

        Note: No filtering here - TargetTracker handles EMA filtering.

        Returns:
            True if UWB initialized successfully
        """
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

        # No filter config - TargetTracker handles filtering
        self._uwb_anchors = DualUWBAnchors(anchor1_cfg, anchor2_cfg, None)

        # First attempt: connect without reset
        print("    Connecting to UWB modules...")
        if self._uwb_anchors.connect():
            print("    UWB connected successfully")
            self._init_triangulator(uwb)
            return True

        # Second attempt: hardware reset and retry
        print("    Initial connection failed, attempting hardware reset...")
        self._uwb_anchors.anchor1.hardware_reset()
        self._uwb_anchors.anchor2.hardware_reset()

        if self._uwb_anchors.connect():
            print("    UWB connected after reset")
            self._init_triangulator(uwb)
            return True

        print("    UWB connection failed after reset")
        return False

    def _init_triangulator(self, uwb_config: dict) -> None:
        """Initialize UWB triangulator."""
        anchor1_offset = uwb_config.get('anchor1_offset_mm', [100, 50, 0])
        anchor2_offset = uwb_config.get('anchor2_offset_mm', [-100, 50, 0])
        calibration = uwb_config.get('calibration', {})

        tri_cfg = TriangulationConfig(
            anchor1_position=(anchor1_offset[0], anchor1_offset[1]),
            anchor2_position=(anchor2_offset[0], anchor2_offset[1]),
            front_offset_deg=calibration.get('front_offset_deg', 0.0),
            calibration_valid=calibration.get('valid', False)
        )
        self._triangulator = UWBTriangulator(tri_cfg)

    def _init_path_planning(self) -> None:
        """Initialize depth preprocessor and path planner."""
        print("  Initializing path planning...")
        camera_cfg = self._robot_config.get('camera', {})
        depth_cfg = self._robot_config.get('depth_preprocessing', {})

        self._horizontal_fov = camera_cfg.get('horizontal_fov_deg', 60.0)

        # Depth preprocessor
        preprocessor_config = DepthPreprocessorConfig(
            horizontal_fov_deg=self._horizontal_fov,
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

        # Path planner (Follow-the-Gap or APF based on config)
        self._path_planner = create_path_planner(self._robot_config)
        print(f"    Using algorithm: {self._path_planner.name}")

    def _init_tracking_camera(self) -> None:
        """Initialize dedicated tracking camera for brightness detection."""
        print("    Initializing tracking camera...")
        tracking_cfg = self._robot_config.get('tracking_camera', {})

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

        # Update horizontal FOV for target detector
        self._horizontal_fov = config.horizontal_fov_deg

    def _init_target_tracking(self) -> None:
        """Initialize target detector and target tracker."""
        print("  Initializing target tracking...")

        # Target detector (bullseye, checkerboard, or brightness)
        target_cfg = self._robot_config.get('target_detection', {})
        pattern_type = target_cfg.get('pattern_type', 'checkerboard')

        # Initialize tracking camera for brightness mode (before creating detector)
        if pattern_type == 'brightness':
            self._init_tracking_camera()

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
            brightness_gain=brightness_cfg.get('gain', 2.0),
            brightness_blur_kernel_size=brightness_cfg.get('blur_kernel_size', 1),
            brightness_min_area_px=brightness_cfg.get('min_area_px', 1),
            brightness_max_area_px=brightness_cfg.get('max_area_px', 100),
            brightness_use_low_exposure=brightness_cfg.get('use_low_exposure', True),
            brightness_low_exposure=brightness_cfg.get('low_exposure', 5.0),
            brightness_settle_frames=brightness_cfg.get('settle_frames', 2),
        )
        self._target_detector = TargetDetector(detector_config, self._horizontal_fov)
        print(f"    Target detection enabled: {detector_config.pattern_type.upper()}")

        # Target tracker (sensor fusion)
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

    def run(self) -> None:
        """Main integration test loop."""
        print("\nStarting integration test...")
        print("Controls: Q=quit, E=epipolar, C=colormap, D=debug view")
        print(f"Debug view: {'ON' if self._debug_view else 'OFF'}")
        print("")

        self._running = True
        loop_period = 1.0 / self._config.target_fps

        # Initialize depth visualizer for debug view
        self._depth_viz = DepthPreprocessorVisualizer(DepthVisualizerConfig(
            polar_size=240,  # Match panel size
            max_range_mm=3000.0
        ))

        cv2.namedWindow("Integration Test", cv2.WINDOW_AUTOSIZE)

        while self._running:
            loop_start = time.time()

            # 1. Process vision (capture frame, compute depth)
            left_rect, depth_color = self._process_vision()

            # 2. Process UWB (get raw ranges, triangulate)
            self._process_uwb()

            # 3. Process visual target detection
            self._process_target_detection()

            # 4. Fuse sensors with target tracker
            self._process_target_tracking()

            # 5. Process path planning and navigation
            self._process_navigation()

            # 6. Render combined view
            combined = self._render_combined_view(left_rect, depth_color)
            cv2.imshow("Integration Test", combined)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key):
                break

            # Update FPS
            self._update_fps(time.time() - loop_start)

            # Maintain loop timing
            elapsed = time.time() - loop_start
            sleep_time = loop_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cv2.destroyAllWindows()

    def _process_vision(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process stereo vision frame.

        Returns:
            Tuple of (left_rectified, depth_colorized) images
        """
        if self._camera is None:
            return None, None

        # Capture frame
        ret, left, right = self._camera.read()
        if not ret or left is None or right is None:
            return None, None

        # Process stereo
        left_rect, right_rect, disparity = self._matcher.process_frame(left, right)

        # Compute depth
        depth = self._matcher.disparity_to_depth(disparity)

        # Store for bullseye detection and path planning
        self._latest_left_frame = left_rect.copy()
        self._latest_depth = depth.copy()

        # Get colorized depth
        depth_color = self._matcher.get_colorized_disparity(disparity)

        # Draw epipolar lines if enabled
        if self._show_epipolar:
            left_rect = self._draw_epipolar_lines(left_rect)

        return left_rect, depth_color

    def _draw_epipolar_lines(self, img: np.ndarray) -> np.ndarray:
        """Draw horizontal epipolar lines on image."""
        result = img.copy()
        h = result.shape[0]
        for y in range(0, h, 20):
            cv2.line(result, (0, y), (result.shape[1], y), (0, 255, 255), 1)
        return result

    def _process_uwb(self) -> None:
        """Process UWB tracking (raw, unfiltered - tracker handles EMA)."""
        if self._uwb_anchors is None:
            self._latest_ranges = (None, None)
            self._latest_triangulation = None
            return

        # Poll raw ranges (no filtering - target tracker handles that)
        range1, range2 = self._uwb_anchors.poll_ranges_raw()
        self._latest_ranges = (range1, range2)

        # Triangulate if both ranges available
        if range1 is not None and range2 is not None and self._triangulator is not None:
            self._latest_triangulation = self._triangulator.triangulate(range1, range2)
        else:
            self._latest_triangulation = None

    def _process_target_detection(self) -> None:
        """Process visual target detection on current frame."""
        if self._target_detector is None:
            self._latest_target_detection = None
            return

        # For brightness mode, use tracking camera; otherwise use stereo left frame
        if self._tracking_camera is not None:
            # Brightness mode: capture from dedicated tracking camera (low exposure)
            ret, tracking_frame = self._tracking_camera.read()
            if not ret or tracking_frame is None:
                self._latest_target_detection = None
                return

            # Brightness detection uses angle only (UWB provides range)
            self._latest_target_detection = self._target_detector.detect(
                tracking_frame, None
            )
        else:
            # Bullseye/checkerboard mode: use stereo left frame with depth
            if self._latest_left_frame is None:
                self._latest_target_detection = None
                return

            self._latest_target_detection = self._target_detector.detect(
                self._latest_left_frame,
                self._latest_depth
            )

    def _process_target_tracking(self) -> None:
        """Fuse UWB and visual detection with target tracker."""
        if self._target_tracker is None:
            self._latest_target_state = None
            return

        # Get UWB data
        uwb_angle = None
        uwb_range = None
        if self._latest_triangulation is not None:
            uwb_angle = self._latest_triangulation.angle_deg
            uwb_range = self._latest_triangulation.estimated_distance_mm

        # Fuse with target tracker
        self._latest_target_state = self._target_tracker.update(
            uwb_angle, uwb_range, self._latest_target_detection
        )

    def _process_navigation(self) -> None:
        """Process path planning and navigation command."""
        # Get fused target info from tracker
        target_angle: Optional[float] = None
        target_range: Optional[float] = None

        if self._latest_target_state is not None:
            target_angle = self._latest_target_state.angle_deg
            target_range = self._latest_target_state.range_mm

        # Compute path using selected algorithm
        if self._depth_preprocessor is not None and self._path_planner is not None and self._latest_depth is not None:
            # Preprocess depth to 1D distances
            preproc_result = self._depth_preprocessor.process(self._latest_depth)
            self._latest_preproc_result = preproc_result  # Store for debug visualization

            # Compute path
            self._latest_planner_result = self._path_planner.compute(
                preproc_result.distances,
                preproc_result.sector_angles,
                target_angle if target_angle is not None else 0.0
            )
        else:
            self._latest_preproc_result = None
            # Fallback - proceed toward target
            self._latest_planner_result = PlannerResult(
                best_heading_deg=target_angle if target_angle else 0.0,
                can_proceed=True
            )

        # Get navigation command
        self._latest_nav_cmd = self._navigation.update(
            target_angle,
            target_range,
            self._latest_planner_result
        )

    def _render_combined_view(
        self,
        left_rect: Optional[np.ndarray],
        depth_color: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Render combined visualization window.

        Layout:
        ┌─────────────┬─────────────┬─────────────────────┐
        │   Left      │   Depth     │                     │
        │  (bullseye) │  Colorized  │    Info Panel       │
        │  (320x240)  │  (320x240)  │    (320x240)        │
        ├─────────────┴─────────────┴─────────────────────┤
        │                Status Bar (960x40)              │
        └─────────────────────────────────────────────────┘

        Returns:
            Combined BGR image
        """
        # Create placeholder if images missing
        if left_rect is None:
            left_rect = np.zeros((240, 320, 3), dtype=np.uint8)
        if depth_color is None:
            depth_color = np.zeros((240, 320, 3), dtype=np.uint8)

        # Ensure correct size
        left_rect = cv2.resize(left_rect, (320, 240))
        depth_color = cv2.resize(depth_color, (320, 240))

        if self._debug_view:
            # Debug view: Virtual LIDAR + Depth with masks + Debug info panel
            top_row = self._render_debug_view(depth_color)
        else:
            # Normal view: Left camera + Depth colorized + Info panel
            left_with_target = self._draw_target_overlay(left_rect.copy())
            info_panel = self._render_info_panel()
            top_row = np.hstack([left_with_target, depth_color, info_panel])  # 960x240

        # Create status bar
        status_bar = self._render_status_bar()  # 960x40

        # Combine vertically
        combined = np.vstack([top_row, status_bar])  # 960x280

        return combined

    def _render_debug_view(self, depth_color: np.ndarray) -> np.ndarray:
        """
        Render debug view with 1D depth visualization and planner info.

        Layout:
        ┌─────────────────┬─────────────────┬─────────────────────┐
        │  Virtual LIDAR  │ Depth+Masks     │   Planner Debug     │
        │   (polar plot)  │                 │   (reason, etc.)    │
        │    (320x240)    │   (320x240)     │     (320x240)       │
        └─────────────────┴─────────────────┴─────────────────────┘

        Args:
            depth_color: Colorized depth image (fallback if no masks)

        Returns:
            Combined top row image (960x240)
        """
        # Panel 1: Virtual LIDAR with heading overlay
        lidar_panel = self._render_lidar_panel()

        # Panel 2: Depth with floor/obstacle masks
        depth_panel = self._render_depth_masks_panel(depth_color)

        # Panel 3: Planner debug info
        debug_panel = self._render_planner_debug_panel()

        return np.hstack([lidar_panel, depth_panel, debug_panel])

    def _render_lidar_panel(self) -> np.ndarray:
        """Render virtual LIDAR polar plot with heading overlays."""
        if self._depth_viz is None or self._latest_preproc_result is None:
            # Fallback placeholder
            panel = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(panel, "No depth data", (100, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
            return panel

        preproc = self._latest_preproc_result
        planner = self._latest_planner_result

        # Get heading info
        selected_heading = planner.best_heading_deg if planner else None
        can_proceed = planner.can_proceed if planner else True

        # Get target heading
        target_heading = None
        if self._latest_target_state is not None:
            target_heading = self._latest_target_state.angle_deg

        # Get safety distance from config
        safety_distance = None
        path_cfg = self._robot_config.get('path_planning', {})
        if self._config.algorithm == 'apf':
            apf_cfg = path_cfg.get('apf', {})
            safety_distance = apf_cfg.get('emergency_stop_distance_mm', 150)
        else:
            fg_cfg = path_cfg.get('follow_gap', {})
            safety_distance = fg_cfg.get('min_range_mm', 200)

        # Draw virtual LIDAR with overlays
        lidar_img = self._depth_viz.draw_virtual_lidar_with_heading(
            distances=preproc.distances,
            sector_angles=preproc.sector_angles,
            valid_sectors=preproc.valid_sectors,
            selected_heading_deg=selected_heading,
            target_heading_deg=target_heading,
            safety_distance_mm=safety_distance,
            can_proceed=can_proceed
        )

        # Resize to fit panel (240x240 polar -> pad to 320x240)
        h, w = lidar_img.shape[:2]
        panel = np.zeros((240, 320, 3), dtype=np.uint8)
        x_offset = (320 - w) // 2
        y_offset = (240 - h) // 2
        panel[y_offset:y_offset+h, x_offset:x_offset+w] = lidar_img

        return panel

    def _render_depth_masks_panel(self, depth_color: np.ndarray) -> np.ndarray:
        """Render depth map with floor/obstacle mask overlays."""
        if self._depth_viz is None or self._latest_preproc_result is None or self._latest_depth is None:
            return depth_color

        preproc = self._latest_preproc_result

        # Draw depth with mask overlays
        depth_with_masks = self._depth_viz.draw_depth_with_masks(
            depth_map=self._latest_depth,
            floor_mask=preproc.floor_mask,
            obstacle_mask=preproc.obstacle_mask
        )

        # Resize to panel size
        return cv2.resize(depth_with_masks, (320, 240))

    def _render_planner_debug_panel(self) -> np.ndarray:
        """Render planner debug info panel."""
        if self._depth_viz is None:
            return np.zeros((240, 320, 3), dtype=np.uint8)

        # Get target info
        target_angle = None
        target_range = None
        if self._latest_target_state is not None:
            target_angle = self._latest_target_state.angle_deg
            target_range = self._latest_target_state.range_mm

        return self._depth_viz.draw_planner_debug_panel(
            planner_result=self._latest_planner_result,
            algorithm=self._config.algorithm,
            target_angle_deg=target_angle,
            target_range_mm=target_range
        )

    def _draw_target_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw target detection overlay on frame."""
        if self._latest_target_detection is None:
            # No detection - show "No target" text
            cv2.putText(
                frame, "No target", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA
            )
            return frame

        detection = self._latest_target_detection
        cx, cy = detection.center_x, detection.center_y

        # Scale coordinates if frame was resized
        scale_x = frame.shape[1] / (self._latest_left_frame.shape[1] if self._latest_left_frame is not None else frame.shape[1])
        scale_y = frame.shape[0] / (self._latest_left_frame.shape[0] if self._latest_left_frame is not None else frame.shape[0])
        cx = int(cx * scale_x)
        cy = int(cy * scale_y)

        # Draw crosshair
        color = (0, 255, 0)  # Green
        cv2.circle(frame, (cx, cy), 15, color, 2)
        cv2.line(frame, (cx - 25, cy), (cx + 25, cy), color, 2)
        cv2.line(frame, (cx, cy - 25), (cx, cy + 25), color, 2)

        # Draw detection info (pattern type)
        pattern_label = detection.pattern_type.upper()[:6]
        cv2.putText(frame, pattern_label, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        conf_text = f"Conf: {detection.confidence:.2f}"
        cv2.putText(frame, conf_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if detection.range_mm is not None:
            range_text = f"Range: {detection.range_mm:.0f}mm"
            cv2.putText(frame, range_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return frame

    def _render_info_panel(self) -> np.ndarray:
        """Render info panel with sensor and navigation data."""
        panel = np.zeros((240, 320, 3), dtype=np.uint8)
        y = 20
        line_height = 22

        # Title
        cv2.putText(panel, f"Algorithm: {self._config.algorithm.upper()}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_height

        # UWB ranges
        range1, range2 = self._latest_ranges
        r1_str = f"{range1:.0f}" if range1 else "---"
        r2_str = f"{range2:.0f}" if range2 else "---"
        cv2.putText(panel, f"UWB: R1={r1_str}mm R2={r2_str}mm", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 255), 1, cv2.LINE_AA)
        y += line_height

        # Target state (fused)
        if self._latest_target_state:
            ts = self._latest_target_state
            angle_str = f"{ts.angle_deg:+.1f}" if ts.angle_deg else "---"
            range_str = f"{ts.range_mm:.0f}" if ts.range_mm else "---"
            source_color = (100, 255, 100) if ts.source == "visual" else (255, 200, 100)
            cv2.putText(panel, f"Target: {angle_str}deg {range_str}mm", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            y += line_height
            cv2.putText(panel, f"Source: {ts.source.upper()}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, source_color, 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "Target: ---", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)
        y += line_height

        # Navigation state
        if self._latest_nav_cmd:
            nav = self._latest_nav_cmd
            state_color = (100, 255, 100) if nav.state == NavigationState.FOLLOWING else (255, 255, 100)
            cv2.putText(panel, f"Nav: {nav.state.name}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_color, 1, cv2.LINE_AA)
            y += line_height
            cv2.putText(panel, f"Cmd: {nav.linear_velocity_mm_s:+.0f}mm/s {nav.angular_velocity_deg_s:+.0f}d/s", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += line_height

        # Path planner result
        if self._latest_planner_result:
            pr = self._latest_planner_result
            heading_str = f"{pr.best_heading_deg:.1f}" if pr.best_heading_deg else "---"
            proceed_color = (100, 255, 100) if pr.can_proceed else (100, 100, 255)
            cv2.putText(panel, f"Heading: {heading_str}deg", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            y += line_height
            cv2.putText(panel, f"Can proceed: {pr.can_proceed}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, proceed_color, 1, cv2.LINE_AA)
        y += line_height

        # FPS
        cv2.putText(panel, f"FPS: {self._fps:.1f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

        # Colormap indicator
        cv2.putText(panel, f"Colormap: {COLORMAPS[self._colormap_index]}", (10, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

        return panel

    def _render_status_bar(self) -> np.ndarray:
        """Render status bar at bottom."""
        bar = np.zeros((40, 960, 3), dtype=np.uint8)

        # Controls help
        cv2.putText(bar, "Q=Quit  E=Epipolar  C=Colormap  D=Debug", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        # Debug view indicator
        if self._debug_view:
            cv2.putText(bar, "DEBUG", (350, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Sensor status indicators
        uwb_color = (100, 255, 100) if self._uwb_anchors and self._uwb_anchors.is_connected else (100, 100, 255)
        cv2.putText(bar, "UWB", (700, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, uwb_color, 1, cv2.LINE_AA)

        cam_color = (100, 255, 100) if self._camera else (100, 100, 255)
        cv2.putText(bar, "CAM", (760, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, cam_color, 1, cv2.LINE_AA)

        # Show pattern type and detection status
        target_color = (100, 255, 100) if self._latest_target_detection else (100, 100, 100)
        pattern_label = self._target_detector.pattern_type.upper()[:6] if self._target_detector else "TARGET"
        cv2.putText(bar, pattern_label, (820, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, target_color, 1, cv2.LINE_AA)

        return bar

    def _handle_key(self, key: int) -> bool:
        """
        Handle keyboard input.

        Args:
            key: Key code from cv2.waitKey()

        Returns:
            False if should quit, True otherwise
        """
        if key == ord('q') or key == ord('Q'):
            print("\nQuit requested.")
            return False

        if key == ord('e') or key == ord('E'):
            self._show_epipolar = not self._show_epipolar
            print(f"Epipolar lines: {'ON' if self._show_epipolar else 'OFF'}")

        if key == ord('c') or key == ord('C'):
            self._colormap_index = (self._colormap_index + 1) % len(COLORMAPS)
            if self._matcher is not None:
                self._matcher.set_colormap(COLORMAPS[self._colormap_index])
            print(f"Colormap: {COLORMAPS[self._colormap_index]}")

        if key == ord('d') or key == ord('D'):
            self._debug_view = not self._debug_view
            print(f"Debug view: {'ON' if self._debug_view else 'OFF'}")

        return True

    def _update_fps(self, frame_time: float) -> None:
        """Update FPS calculation."""
        self._frame_times.append(frame_time)

        # Keep last 30 frames for averaging
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)

        if self._frame_times:
            avg_time = sum(self._frame_times) / len(self._frame_times)
            if avg_time > 0:
                self._fps = 1.0 / avg_time

    def cleanup(self) -> None:
        """Clean up resources."""
        print("\nCleaning up...")

        if self._camera is not None:
            self._camera.release()

        if self._tracking_camera is not None:
            self._tracking_camera.release()

        if self._uwb_anchors is not None:
            self._uwb_anchors.close()

        cv2.destroyAllWindows()
        print("Cleanup complete.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Integration test GUI for person-following robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python robot/scripts/integration_test.py                    # Default (follow_gap)
  python robot/scripts/integration_test.py --algorithm apf    # Use APF
        """
    )
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        choices=['follow_gap', 'apf'],
        default='follow_gap',
        help='Path planning algorithm (default: follow_gap)'
    )
    parser.add_argument(
        '--calibration',
        type=str,
        default='vision/data/calibration_data',
        help='Path to calibration data directory'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='Camera device ID'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to robot config YAML'
    )
    parser.add_argument(
        '--debug-view', '-d',
        action='store_true',
        help='Start with debug view enabled (shows 1D depth data and planner info)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = IntegrationConfig(
        calibration_path=args.calibration,
        device_id=args.device,
        config_path=args.config,
        algorithm=args.algorithm,
        debug_view=args.debug_view
    )

    tester = IntegrationTester(config)

    try:
        if tester.initialize():
            tester.run()
        else:
            print("Failed to initialize integration test.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        tester.cleanup()


if __name__ == '__main__':
    main()
