"""
Robot control module for person-following robot.

Components:
- gpio_config: GPIO pin configuration management
- motor_driver: 3-pin H-bridge motor driver interface (lgpio)
- encoder: Quadrature encoder reading (libgpiod)
- differential_drive: PID-controlled differential drive
- odometry: Wheel odometry for pose tracking
- uwb_tracker: RYUW122 UWB module communication
- uwb_triangulation: UWB triangulation and calibration
- depth_to_polar: Depth map to polar coordinate conversion (deprecated)
- depth_preprocessor: Unified depth preprocessing with 1D distance output
- floor_detection: Pluggable floor detection strategies
- path_planner: Abstract base for path planning algorithms
- follow_gap: Follow-the-Gap path planning
- apf: Artificial Potential Fields path planning
- navigation: High-level navigation controller
- target_detector: Visual target detection (bullseye, checkerboard, or brightness)
- target_tracker: Sensor fusion for UWB and visual tracking
- tracking_camera: Dedicated camera for brightness-based target detection
"""

from .gpio_config import GPIOConfig, load_gpio_config
from .gpio_manager import GPIOManager
from .motor_driver import MotorDriver, DualMotorDriver
from .encoder import QuadratureEncoder, DualEncoders, EncoderConfig
from .differential_drive import PIDController, DifferentialDriveController
from .odometry import Pose2D, WheelOdometry
from .uwb_tracker import (
    RYUW122, DualUWBAnchors, UWBModuleConfig,
    RangeFilterConfig, EMAFilter, AngleFilter
)
from .uwb_triangulation import UWBTriangulator, UWBCalibrator
from .depth_to_polar import DepthToPolar  # Deprecated, use DepthPreprocessor
from .depth_preprocessor import DepthPreprocessor, DepthPreprocessorConfig, PreprocessorResult
from .floor_detection import (
    FloorDetector,
    HeightBasedFloorDetector, HeightBasedFloorConfig,
    AdaptiveFloorDetector, AdaptiveFloorConfig,
)
from .path_planner import PathPlanner, PlannerResult
from .follow_gap import FollowTheGap, FollowGapConfig, Gap
from .apf import ArtificialPotentialFields, APFConfig
from .path_planner_factory import create_path_planner, get_available_algorithms
from .navigation import NavigationController, NavigationState, NavigationCommand
from .target_detector import (
    TargetDetector, TargetDetectorConfig, TargetDetection,
    BullseyePattern, CheckerboardPattern, BrightnessPattern
)
from .target_tracker import TargetTracker, TargetTrackerConfig, TargetState
from .tracking_camera import TrackingCamera, TrackingCameraConfig

__all__ = [
    'GPIOConfig', 'load_gpio_config',
    'GPIOManager',
    'MotorDriver', 'DualMotorDriver',
    'QuadratureEncoder', 'DualEncoders', 'EncoderConfig',
    'PIDController', 'DifferentialDriveController',
    'Pose2D', 'WheelOdometry',
    'RYUW122', 'DualUWBAnchors', 'UWBModuleConfig',
    'RangeFilterConfig', 'EMAFilter', 'AngleFilter',
    'UWBTriangulator', 'UWBCalibrator',
    'DepthToPolar',  # Deprecated
    'DepthPreprocessor', 'DepthPreprocessorConfig', 'PreprocessorResult',
    'FloorDetector',
    'HeightBasedFloorDetector', 'HeightBasedFloorConfig',
    'AdaptiveFloorDetector', 'AdaptiveFloorConfig',
    'PathPlanner', 'PlannerResult',
    'FollowTheGap', 'FollowGapConfig', 'Gap',
    'ArtificialPotentialFields', 'APFConfig',
    'create_path_planner', 'get_available_algorithms',
    'NavigationController', 'NavigationState', 'NavigationCommand',
    'TargetDetector', 'TargetDetectorConfig', 'TargetDetection',
    'BullseyePattern', 'CheckerboardPattern', 'BrightnessPattern',
    'TargetTracker', 'TargetTrackerConfig', 'TargetState',
    'TrackingCamera', 'TrackingCameraConfig',
]
