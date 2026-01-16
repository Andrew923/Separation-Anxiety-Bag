#!/usr/bin/env python3
"""
Live GUI for tuning depth preprocessing floor detection parameters.

Provides real-time visualization and parameter adjustment using OpenCV trackbars.
Uses robot_config.yaml as the single source of truth for stereo/WLS configuration.

=== PARAMETER REFERENCE ===

RANGE FILTERING:
    Min Range (cm)      Minimum distance to consider valid. Anything closer is
                        ignored (helps filter noise near the camera). Typical: 20-30cm

    Max Range (cm)      Maximum distance to consider. Anything farther is ignored
                        (out of reliable stereo range). Typical: 200-400cm

HEIGHT-BASED FLOOR DETECTION (default method):
    Works by computing the height of each pixel relative to the floor plane
    using camera geometry. Valid obstacles = pixels between floor and robot height.

    Floor Threshold (mm)    Pixels with height BELOW this value are considered floor
                            and ignored. Relative to computed floor plane.
                            Example: 50mm = anything within 5cm of floor is filtered.

    Robot Height (mm)       Pixels with height ABOVE this value are considered
                            ceiling/overhead and ignored.
                            Example: 500mm = anything 50cm+ above floor is filtered.

    Visual representation:
        Ceiling (ignored)     | above robot_height_mm
        ----------------------+
        Obstacles (detected)  | valid range
        ----------------------+
        Floor (ignored)       | below floor_threshold_mm

ADAPTIVE FLOOR DETECTION (press 'M' to switch):
    Alternative method where floor threshold varies based on distance from camera.
    Helps account for stereo depth noise increasing with distance.

    Base Threshold (mm)     Floor threshold at zero distance

    Depth Scaling x100      How much threshold increases per mm of depth.
                            Value of 2 = 0.02, meaning threshold grows 20mm per 1000mm.
                            Formula: threshold = base + (depth * scaling_factor)

    Robot Height (mm)       Same as above - ceiling cutoff

OUTPUT CONFIGURATION:
    Num Sectors             Number of angular sectors in virtual LIDAR output.
                            More sectors = finer angular resolution but more processing.
                            Typical: 36-72

CAMERA GEOMETRY (CLI arguments, not trackbars):
    --camera-height         Height of camera above floor in mm. Critical for height
                            calculations - measure this accurately!

    --camera-tilt           Downward tilt angle in degrees. Positive = looking down.
                            Use 0 if camera is level.

=== TUNING TIPS ===

1. Start with camera geometry - measure actual --camera-height before running
2. Adjust Floor Threshold - increase until floor pixels turn blue in "Depth + Masks"
3. Check raw camera view (View 3) - red outlines should only be on real obstacles
4. Verify Virtual LIDAR - no false close-range points from floor
5. Press 'S' to save when happy - saves to robot/config/floor_detection_params.yaml
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from robot.src.depth_preprocessor import DepthPreprocessor, DepthPreprocessorConfig
from robot.src.floor_detection import (
    HeightBasedFloorDetector,
    HeightBasedFloorConfig,
    AdaptiveFloorDetector,
    AdaptiveFloorConfig,
)
from robot.src.visualizers.depth_preprocessor_viz import DepthPreprocessorVisualizer
from robot.src.gpio_config import load_robot_config

# Import vision modules
from vision.src.camera import StereoCamera
from vision.src.calibration import load_calibration
from vision.src.stereo_matcher import StereoMatcher, SGBMParams, WLSParams


class FloorDetectionTuner:
    """
    Interactive GUI for tuning floor detection parameters.

    Windows:
    - "Depth + Masks": Raw depth with floor (blue) and obstacles (red) overlay
    - "Virtual LIDAR": Polar plot of 1D distance output
    - "View 3": Raw camera with obstacle outlines (default) or height histogram (press H)
    - "Parameters": Trackbars for parameter adjustment

    Keys:
    - 's': Save current params to YAML
    - 'l': Load params from YAML
    - 'm': Switch floor detection method (height/adaptive)
    - 'h': Toggle between raw camera and height histogram
    - 'q': Quit
    """

    def __init__(
        self,
        camera: StereoCamera,
        matcher: StereoMatcher,
        config: DepthPreprocessorConfig,
        output_dir: Path
    ):
        """
        Initialize tuner.

        Args:
            camera: Stereo camera instance
            matcher: Stereo matcher for depth computation
            config: Initial preprocessor configuration
            output_dir: Directory for saving/loading configs
        """
        self._camera = camera
        self._matcher = matcher
        self._config = config
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Create preprocessor with height-based detector by default
        self._current_method = "height"
        self._floor_detector = HeightBasedFloorDetector(HeightBasedFloorConfig(
            floor_threshold_mm=config.floor_threshold_mm,
            robot_height_mm=config.robot_height_mm,
        ))
        self._preprocessor = DepthPreprocessor(config, self._floor_detector)

        # Visualizer
        self._visualizer = DepthPreprocessorVisualizer()

        # State
        self._running = False
        self._show_histogram = False  # Toggle for View 3: camera (default) vs histogram
        self._filtering_enabled = True  # Temporal persistence filter toggle

        # Temporal filtering state (N-frame persistence filter on raw depth)
        self._persistence_frames = 2  # Number of frames required for persistence
        self._depth_history: list = []  # List of recent depth maps

        # Trackbar values (scaled for integer trackbars)
        self._trackbar_values = {
            'min_range_cm': int(config.min_range_mm / 10),
            'max_range_cm': int(config.max_range_mm / 10),
            'floor_threshold_mm': int(config.floor_threshold_mm),
            'robot_height_mm': int(config.robot_height_mm),
            'num_sectors': config.num_sectors,
            'base_threshold_mm': 50,  # For adaptive
            'depth_scaling_x100': 2,  # 0.02 * 100
        }

    def _create_trackbars(self) -> None:
        """Create all parameter trackbars."""
        cv2.namedWindow('Parameters', cv2.WINDOW_NORMAL)

        # --- Range filtering ---
        # Min/max distance to consider valid depth readings

        cv2.createTrackbar(
            'Min Range (cm)', 'Parameters',
            self._trackbar_values['min_range_cm'], 50,
            lambda v: self._on_trackbar_change('min_range_cm', v)
        )
        cv2.createTrackbar(
            'Max Range (cm)', 'Parameters',
            self._trackbar_values['max_range_cm'], 500,
            lambda v: self._on_trackbar_change('max_range_cm', v)
        )

        # --- Height-based floor detection ---
        # Floor Threshold: heights BELOW this are floor (ignored)
        # Robot Height: heights ABOVE this are ceiling (ignored)
        # Valid obstacles are between these two thresholds

        cv2.createTrackbar(
            'Floor Threshold (mm)', 'Parameters',
            self._trackbar_values['floor_threshold_mm'], 200,
            lambda v: self._on_trackbar_change('floor_threshold_mm', v)
        )
        cv2.createTrackbar(
            'Robot Height (mm)', 'Parameters',
            self._trackbar_values['robot_height_mm'], 1000,
            lambda v: self._on_trackbar_change('robot_height_mm', v)
        )

        # --- Output configuration ---
        # Number of angular sectors in 1D virtual LIDAR output

        cv2.createTrackbar(
            'Num Sectors', 'Parameters',
            self._trackbar_values['num_sectors'], 180,
            lambda v: self._on_trackbar_change('num_sectors', max(18, v))
        )

        # --- Adaptive floor detection (press 'M' to enable) ---
        # threshold = base_threshold + (depth * depth_scaling_factor)
        # Accounts for stereo noise increasing with distance

        cv2.createTrackbar(
            'Base Threshold (mm)', 'Parameters',
            self._trackbar_values['base_threshold_mm'], 200,
            lambda v: self._on_trackbar_change('base_threshold_mm', v)
        )
        cv2.createTrackbar(
            'Depth Scaling x100', 'Parameters',
            self._trackbar_values['depth_scaling_x100'], 10,
            lambda v: self._on_trackbar_change('depth_scaling_x100', v)
        )

    def _on_trackbar_change(self, name: str, value: int) -> None:
        """Handle trackbar value change."""
        self._trackbar_values[name] = value
        self._update_preprocessor()

    def _update_preprocessor(self) -> None:
        """Update preprocessor with current trackbar values."""
        v = self._trackbar_values

        # Update config
        self._config.min_range_mm = v['min_range_cm'] * 10
        self._config.max_range_mm = v['max_range_cm'] * 10
        self._config.floor_threshold_mm = v['floor_threshold_mm']
        self._config.robot_height_mm = v['robot_height_mm']
        self._config.num_sectors = max(18, v['num_sectors'])

        # Update floor detector
        if self._current_method == "height":
            self._floor_detector.floor_threshold_mm = v['floor_threshold_mm']
            self._floor_detector.robot_height_mm = v['robot_height_mm']
        else:  # adaptive
            self._floor_detector.base_threshold_mm = v['base_threshold_mm']
            self._floor_detector.depth_scaling_factor = v['depth_scaling_x100'] / 100.0
            self._floor_detector.robot_height_mm = v['robot_height_mm']

        # Recreate preprocessor with new config
        self._preprocessor = DepthPreprocessor(self._config, self._floor_detector)

    def _switch_method(self) -> None:
        """Switch between height-based and adaptive floor detection."""
        v = self._trackbar_values

        if self._current_method == "height":
            self._current_method = "adaptive"
            self._floor_detector = AdaptiveFloorDetector(AdaptiveFloorConfig(
                base_threshold_mm=v['base_threshold_mm'],
                depth_scaling_factor=v['depth_scaling_x100'] / 100.0,
                robot_height_mm=v['robot_height_mm'],
            ))
            print("Switched to ADAPTIVE floor detection")
        else:
            self._current_method = "height"
            self._floor_detector = HeightBasedFloorDetector(HeightBasedFloorConfig(
                floor_threshold_mm=v['floor_threshold_mm'],
                robot_height_mm=v['robot_height_mm'],
            ))
            print("Switched to HEIGHT-BASED floor detection")

        self._preprocessor.set_floor_detector(self._floor_detector)

    def _save_config(self, filename: str = "floor_detection_params.yaml") -> None:
        """Save current parameters to YAML."""
        filepath = self._output_dir / filename
        v = self._trackbar_values

        config = {
            'depth_preprocessing': {
                'min_range_mm': v['min_range_cm'] * 10,
                'max_range_mm': v['max_range_cm'] * 10,
                'num_sectors': v['num_sectors'],
                'floor_detection_method': self._current_method,
                'height_floor': {
                    'floor_threshold_mm': v['floor_threshold_mm'],
                    'robot_height_mm': v['robot_height_mm'],
                },
                'adaptive_floor': {
                    'base_threshold_mm': v['base_threshold_mm'],
                    'depth_scaling_factor': v['depth_scaling_x100'] / 100.0,
                    'robot_height_mm': v['robot_height_mm'],
                }
            }
        }

        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Saved config to {filepath}")

    def _load_config(self, filename: str = "floor_detection_params.yaml") -> None:
        """Load parameters from YAML."""
        filepath = self._output_dir / filename

        if not filepath.exists():
            print(f"Config file not found: {filepath}")
            return

        with open(filepath) as f:
            config = yaml.safe_load(f)

        dp = config.get('depth_preprocessing', {})

        # Update trackbar values
        self._trackbar_values['min_range_cm'] = int(dp.get('min_range_mm', 200) / 10)
        self._trackbar_values['max_range_cm'] = int(dp.get('max_range_mm', 3000) / 10)
        self._trackbar_values['num_sectors'] = dp.get('num_sectors', 72)

        height_cfg = dp.get('height_floor', {})
        self._trackbar_values['floor_threshold_mm'] = int(height_cfg.get('floor_threshold_mm', 50))
        self._trackbar_values['robot_height_mm'] = int(height_cfg.get('robot_height_mm', 500))

        adaptive_cfg = dp.get('adaptive_floor', {})
        self._trackbar_values['base_threshold_mm'] = int(adaptive_cfg.get('base_threshold_mm', 50))
        self._trackbar_values['depth_scaling_x100'] = int(adaptive_cfg.get('depth_scaling_factor', 0.02) * 100)

        # Update trackbar positions
        cv2.setTrackbarPos('Min Range (cm)', 'Parameters', self._trackbar_values['min_range_cm'])
        cv2.setTrackbarPos('Max Range (cm)', 'Parameters', self._trackbar_values['max_range_cm'])
        cv2.setTrackbarPos('Floor Threshold (mm)', 'Parameters', self._trackbar_values['floor_threshold_mm'])
        cv2.setTrackbarPos('Robot Height (mm)', 'Parameters', self._trackbar_values['robot_height_mm'])
        cv2.setTrackbarPos('Num Sectors', 'Parameters', self._trackbar_values['num_sectors'])
        cv2.setTrackbarPos('Base Threshold (mm)', 'Parameters', self._trackbar_values['base_threshold_mm'])
        cv2.setTrackbarPos('Depth Scaling x100', 'Parameters', self._trackbar_values['depth_scaling_x100'])

        # Set method
        method = dp.get('floor_detection_method', 'height')
        if method != self._current_method:
            self._switch_method()

        self._update_preprocessor()
        print(f"Loaded config from {filepath}")

    def _render_camera_with_obstacles(
        self,
        frame: np.ndarray,
        obstacle_mask: np.ndarray
    ) -> np.ndarray:
        """
        Render raw camera view with red obstacle outlines.

        Args:
            frame: Left rectified camera frame (BGR)
            obstacle_mask: Boolean mask where True = obstacle

        Returns:
            BGR image with red obstacle outlines overlaid
        """
        result = frame.copy()

        # Resize mask if dimensions don't match
        if obstacle_mask.shape[:2] != frame.shape[:2]:
            obstacle_mask = cv2.resize(
                obstacle_mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        # Get obstacle edges (outline only, not filled)
        mask_uint8 = obstacle_mask.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
        eroded = cv2.erode(mask_uint8, kernel, iterations=1)
        edges = (dilated - eroded) > 0

        # Draw red outlines
        result[edges] = (0, 0, 255)  # BGR red

        return result

    def _apply_depth_filter(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Apply N-frame persistence filter to raw depth map.

        For each pixel, outputs the maximum (farthest) depth seen
        in the last N frames. This rejects transient "close" noise spikes
        while being conservative (farther = safer for navigation).

        Args:
            depth_map: Current frame 2D depth map in mm

        Returns:
            Filtered depth map with max depth per pixel over N frames
        """
        self._depth_history.append(depth_map.copy())
        if len(self._depth_history) > self._persistence_frames:
            self._depth_history.pop(0)

        # Stack and take max across frames (safest/farthest reading)
        stacked = np.stack(self._depth_history, axis=0)
        return np.max(stacked, axis=0)

    def _reset_filter_state(self) -> None:
        """Reset the temporal filter history."""
        self._depth_history = []

    def run(self) -> None:
        """Main loop with live preview."""
        # Create windows
        cv2.namedWindow('Depth + Masks', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Virtual LIDAR', cv2.WINDOW_NORMAL)
        cv2.namedWindow('View 3', cv2.WINDOW_NORMAL)

        self._create_trackbars()

        print("\n=== Floor Detection Tuner ===")
        print("Controls:")
        print("  s - Save parameters to YAML")
        print("  l - Load parameters from YAML")
        print("  m - Switch floor detection method (height/adaptive)")
        print("  h - Toggle View 3: raw camera / height histogram")
        print("  f - Toggle temporal persistence filter")
        print("  [ / ] - Decrease / increase persistence frames (1-10)")
        print("  q - Quit")
        print()
        print(f"Temporal filter: {'ON' if self._filtering_enabled else 'OFF'} ({self._persistence_frames} frames)")
        print()

        # FPS tracking
        frame_times = []
        fps = 0

        self._running = True
        while self._running:
            start_time = time.time()

            # Capture frame
            success, left, right = self._camera.read()
            if not success:
                continue

            # Compute depth (also get left rectified for camera view)
            left_rect, _, disparity = self._matcher.process_frame(left, right)
            depth_map = self._matcher.disparity_to_depth(disparity)

            # Handle NaN/inf values
            depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply temporal persistence filter on raw depth if enabled
            if self._filtering_enabled:
                depth_map = self._apply_depth_filter(depth_map)
            else:
                self._reset_filter_state()

            # Process with current settings
            result = self._preprocessor.process(depth_map)

            # Visualize
            depth_viz = self._visualizer.draw_depth_with_masks(
                depth_map,
                result.floor_mask,
                result.obstacle_mask,
                max_range_mm=self._config.max_range_mm
            )

            lidar_viz = self._visualizer.draw_virtual_lidar(
                result.distances,
                result.sector_angles,
                result.valid_sectors,
                max_range_mm=self._config.max_range_mm
            )

            # View 3: Raw camera with obstacles (default) or height histogram
            if self._show_histogram:
                heights = self._preprocessor.compute_heights(depth_map)
                view3_viz = self._visualizer.draw_height_histogram(
                    heights,
                    self._trackbar_values['floor_threshold_mm'],
                    self._trackbar_values['robot_height_mm']
                )
            else:
                view3_viz = self._render_camera_with_obstacles(left_rect, result.obstacle_mask)

            # Add info overlay to depth viz
            filter_status = f"FILT({self._persistence_frames})" if self._filtering_enabled else "RAW"
            info_text = f"FPS: {fps:.1f} | Method: {self._current_method.upper()} | {filter_status}"
            cv2.rectangle(depth_viz, (0, 0), (400, 25), (40, 40, 40), -1)
            cv2.putText(depth_viz, info_text, (10, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Show windows
            cv2.imshow('Depth + Masks', depth_viz)
            cv2.imshow('Virtual LIDAR', lidar_viz)
            cv2.imshow('View 3', view3_viz)

            # Handle input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self._running = False
            elif key == ord('s'):
                self._save_config()
            elif key == ord('l'):
                self._load_config()
            elif key == ord('m'):
                self._switch_method()
            elif key == ord('h'):
                self._show_histogram = not self._show_histogram
                view_name = "Height Histogram" if self._show_histogram else "Raw Camera"
                print(f"View 3: {view_name}")
            elif key == ord('f'):
                self._filtering_enabled = not self._filtering_enabled
                self._reset_filter_state()
                status = f"ON ({self._persistence_frames} frames)" if self._filtering_enabled else "OFF"
                print(f"Temporal filter: {status}")
            elif key == ord('['):
                if self._persistence_frames > 1:
                    self._persistence_frames -= 1
                    self._reset_filter_state()
                    print(f"Persistence frames: {self._persistence_frames}")
            elif key == ord(']'):
                if self._persistence_frames < 10:
                    self._persistence_frames += 1
                    self._reset_filter_state()
                    print(f"Persistence frames: {self._persistence_frames}")

            # Update FPS
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Tune floor detection parameters for depth preprocessing'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to robot_config.yaml (default: robot/config/robot_config.yaml)'
    )
    parser.add_argument(
        '--calibration', type=str, default=None,
        help='Path to calibration directory (overrides config)'
    )
    parser.add_argument(
        '--device', type=int, default=None,
        help='Camera device ID (overrides config)'
    )
    parser.add_argument(
        '--resolution', choices=['high', 'medium', 'low'], default=None,
        help='Resolution: high=2560x960, medium=1280x480, low=640x240 (overrides config)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='config',
        help='Directory for saving/loading configs (relative to robot/)'
    )
    parser.add_argument(
        '--camera-height', type=float, default=200.0,
        help='Camera height above ground in mm'
    )
    parser.add_argument(
        '--camera-tilt', type=float, default=0.0,
        help='Camera tilt angle in degrees (positive = looking down)'
    )
    parser.add_argument(
        '--no-wls', action='store_true',
        help='Disable WLS filtering (overrides config)'
    )
    args = parser.parse_args()

    # Load robot config (source of truth)
    print("Loading robot configuration...")
    robot_config = load_robot_config(args.config)
    stereo_cfg = robot_config.get('stereo_camera', {})

    # Resolution mapping
    resolutions = {
        'high': (2560, 960),
        'medium': (1280, 480),
        'low': (640, 240)
    }

    # Use CLI args if provided, otherwise fall back to config
    resolution_key = args.resolution or stereo_cfg.get('resolution', 'medium')
    resolution = resolutions[resolution_key]
    single_res = (resolution[0] // 2, resolution[1])
    device_id = args.device if args.device is not None else stereo_cfg.get('device_id', 0)

    # Paths
    project_root = Path(__file__).parent.parent.parent
    robot_dir = Path(__file__).parent.parent
    output_dir = robot_dir / args.output_dir

    # Load calibration
    if args.calibration:
        calib_path = Path(args.calibration)
    else:
        # Use config path (relative to project root)
        config_calib_path = stereo_cfg.get('calibration_path', 'vision/data/calibration_data')
        calib_path = project_root / config_calib_path

    print(f"Loading calibration from {calib_path}...")
    try:
        calibration_data = load_calibration(str(calib_path))
    except FileNotFoundError:
        print(f"Error: Calibration files not found at {calib_path}")
        print("Run capture_calibration.py and run_calibration.py first")
        return 1

    # Check resolution compatibility
    calib_size = tuple(calibration_data['image_size'])
    if single_res != calib_size:
        print(f"Warning: Resolution {single_res} differs from calibration {calib_size}")

    # Setup SGBM parameters
    params = SGBMParams(num_disparities=64, block_size=5)

    # Setup WLS parameters from config
    wls_params = None
    wls_cfg = stereo_cfg.get('wls_filter', {})
    if wls_cfg.get('enabled', False) and not args.no_wls:
        wls_params = WLSParams(
            enabled=True,
            lambda_=wls_cfg.get('lambda', 8000.0),
            sigma_color=wls_cfg.get('sigma_color', 1.5),
            confidence_threshold=wls_cfg.get('confidence_threshold', 0.0)
        )
        print(f"WLS filtering enabled: lambda={wls_params.lambda_}, sigma={wls_params.sigma_color}, conf_thresh={wls_params.confidence_threshold}")
    else:
        print("WLS filtering disabled")

    # Create stereo matcher
    matcher = StereoMatcher(calibration_data, params, wls_params)

    # Create preprocessor config
    config = DepthPreprocessorConfig(
        horizontal_fov_deg=60.0,
        vertical_fov_deg=45.0,
        image_width=single_res[0],
        image_height=single_res[1],
        camera_height_mm=args.camera_height,
        camera_tilt_deg=args.camera_tilt,
        min_range_mm=200.0,
        max_range_mm=3000.0,
        floor_threshold_mm=50.0,
        robot_height_mm=500.0,
        num_sectors=72,
    )

    print(f"Starting floor detection tuner")
    print(f"  Camera: /dev/video{device_id}")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")
    print(f"  Camera height: {args.camera_height}mm")
    print(f"  Camera tilt: {args.camera_tilt}deg")
    print(f"  WLS Filter: {'enabled' if wls_params else 'disabled'}")
    print()

    # Open camera
    camera = StereoCamera(device_id, resolution, fps=30)
    if not camera.open():
        print("Error: Could not open camera")
        return 1

    # Create and run tuner
    tuner = FloorDetectionTuner(camera, matcher, config, output_dir)

    try:
        tuner.run()
    finally:
        camera.release()

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
