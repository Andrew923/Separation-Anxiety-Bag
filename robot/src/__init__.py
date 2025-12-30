"""
Robot control module for person-following robot.

Components:
- gpio_config: GPIO pin configuration management
- motor_driver: Cytron MDD10A motor driver interface
- encoder: Quadrature encoder reading
- differential_drive: PID-controlled differential drive
- odometry: Wheel odometry for pose tracking
- uwb_tracker: RYUW122 UWB module communication
- uwb_triangulation: UWB triangulation and calibration
- depth_to_polar: Depth map to polar coordinate conversion
- vfh: Vector Field Histogram obstacle avoidance
- navigation: High-level navigation controller
"""

from .gpio_config import GPIOConfig, load_gpio_config
from .motor_driver import MotorDriver, DualMotorDriver
from .encoder import QuadratureEncoder, DualEncoders
from .differential_drive import PIDController, DifferentialDriveController
from .odometry import Pose2D, WheelOdometry
from .uwb_tracker import RYUW122, DualUWBAnchors
from .uwb_triangulation import UWBTriangulator, UWBCalibrator
from .depth_to_polar import DepthToPolar
from .vfh import VectorFieldHistogram, VFHResult
from .navigation import NavigationController, NavigationState, NavigationCommand

__all__ = [
    'GPIOConfig', 'load_gpio_config',
    'MotorDriver', 'DualMotorDriver',
    'QuadratureEncoder', 'DualEncoders',
    'PIDController', 'DifferentialDriveController',
    'Pose2D', 'WheelOdometry',
    'RYUW122', 'DualUWBAnchors',
    'UWBTriangulator', 'UWBCalibrator',
    'DepthToPolar',
    'VectorFieldHistogram', 'VFHResult',
    'NavigationController', 'NavigationState', 'NavigationCommand',
]
