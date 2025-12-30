"""
Wheel odometry for robot position tracking.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import threading


@dataclass
class OdometryConfig:
    """Odometry configuration."""
    wheel_diameter_mm: float = 52.0
    wheel_base_mm: float = 200.0
    encoder_cpr: int = 700


@dataclass
class Pose2D:
    """2D pose representation."""
    x: float = 0.0       # mm (positive = right)
    y: float = 0.0       # mm (positive = forward)
    theta: float = 0.0   # radians (positive = counter-clockwise)

    def to_tuple(self) -> Tuple[float, float, float]:
        """Return (x, y, theta) tuple."""
        return (self.x, self.y, self.theta)

    def theta_deg(self) -> float:
        """Get heading in degrees."""
        return math.degrees(self.theta)

    def copy(self) -> 'Pose2D':
        """Create a copy of this pose."""
        return Pose2D(self.x, self.y, self.theta)


class WheelOdometry:
    """
    Computes robot pose from wheel encoder counts.

    Uses differential drive kinematics to integrate
    wheel movements into position and heading.
    """

    def __init__(self, config: OdometryConfig):
        """
        Initialize odometry tracker.

        Args:
            config: Odometry configuration
        """
        self._config = config
        self._pose = Pose2D()
        self._last_left_count: int = 0
        self._last_right_count: int = 0
        self._total_distance: float = 0.0
        self._lock = threading.Lock()

        # Precompute constants
        self._mm_per_count = (
            math.pi * config.wheel_diameter_mm / config.encoder_cpr
        )

    def update(self, left_count: int, right_count: int) -> Pose2D:
        """
        Update pose from new encoder counts.

        Uses differential drive kinematics.

        Args:
            left_count: Current left encoder count
            right_count: Current right encoder count

        Returns:
            Updated pose
        """
        with self._lock:
            # Calculate count differences
            delta_left = left_count - self._last_left_count
            delta_right = right_count - self._last_right_count

            # Update last counts
            self._last_left_count = left_count
            self._last_right_count = right_count

            # Convert to distances
            left_dist = delta_left * self._mm_per_count
            right_dist = delta_right * self._mm_per_count

            # Calculate robot movement
            # Linear distance is average of wheel distances
            linear_dist = (left_dist + right_dist) / 2.0

            # Angular change from difference in wheel distances
            delta_theta = (right_dist - left_dist) / self._config.wheel_base_mm

            # Update pose using midpoint integration
            if abs(delta_theta) < 1e-6:
                # Straight line motion
                dx = linear_dist * math.cos(self._pose.theta)
                dy = linear_dist * math.sin(self._pose.theta)
            else:
                # Arc motion
                # Radius of curvature
                radius = linear_dist / delta_theta

                # New position relative to start of arc
                dx = radius * (math.sin(self._pose.theta + delta_theta) -
                              math.sin(self._pose.theta))
                dy = radius * (math.cos(self._pose.theta) -
                              math.cos(self._pose.theta + delta_theta))

            # Update pose
            self._pose.x += dx
            self._pose.y += dy
            self._pose.theta += delta_theta

            # Normalize theta to [-pi, pi]
            self._pose.theta = math.atan2(
                math.sin(self._pose.theta),
                math.cos(self._pose.theta)
            )

            # Track total distance
            self._total_distance += abs(linear_dist)

            return self._pose.copy()

    def get_pose(self) -> Pose2D:
        """
        Get current pose (thread-safe).

        Returns:
            Current pose
        """
        with self._lock:
            return self._pose.copy()

    def reset(self, pose: Optional[Pose2D] = None) -> None:
        """
        Reset odometry to given pose.

        Args:
            pose: Pose to reset to (default: origin)
        """
        with self._lock:
            if pose is None:
                self._pose = Pose2D()
            else:
                self._pose = pose.copy()

            self._last_left_count = 0
            self._last_right_count = 0
            self._total_distance = 0.0

    def reset_with_current_counts(
        self,
        left_count: int,
        right_count: int,
        pose: Optional[Pose2D] = None
    ) -> None:
        """
        Reset odometry while preserving current encoder counts.

        Useful when resetting pose without resetting encoders.

        Args:
            left_count: Current left encoder count
            right_count: Current right encoder count
            pose: Pose to reset to (default: origin)
        """
        with self._lock:
            if pose is None:
                self._pose = Pose2D()
            else:
                self._pose = pose.copy()

            self._last_left_count = left_count
            self._last_right_count = right_count
            self._total_distance = 0.0

    def get_distance_traveled(self) -> float:
        """
        Get total distance traveled in mm.

        Returns:
            Total distance traveled since last reset
        """
        with self._lock:
            return self._total_distance

    def get_displacement(self) -> float:
        """
        Get straight-line displacement from origin in mm.

        Returns:
            Distance from origin
        """
        with self._lock:
            return math.sqrt(self._pose.x ** 2 + self._pose.y ** 2)

    def transform_point(
        self,
        local_x: float,
        local_y: float
    ) -> Tuple[float, float]:
        """
        Transform a point from robot-local to global coordinates.

        Args:
            local_x: X coordinate in robot frame (right)
            local_y: Y coordinate in robot frame (forward)

        Returns:
            Tuple of (global_x, global_y)
        """
        with self._lock:
            cos_theta = math.cos(self._pose.theta)
            sin_theta = math.sin(self._pose.theta)

            global_x = self._pose.x + local_x * cos_theta - local_y * sin_theta
            global_y = self._pose.y + local_x * sin_theta + local_y * cos_theta

            return (global_x, global_y)

    def transform_to_local(
        self,
        global_x: float,
        global_y: float
    ) -> Tuple[float, float]:
        """
        Transform a point from global to robot-local coordinates.

        Args:
            global_x: X coordinate in global frame
            global_y: Y coordinate in global frame

        Returns:
            Tuple of (local_x, local_y)
        """
        with self._lock:
            # Translate to robot origin
            dx = global_x - self._pose.x
            dy = global_y - self._pose.y

            # Rotate to robot frame
            cos_theta = math.cos(-self._pose.theta)
            sin_theta = math.sin(-self._pose.theta)

            local_x = dx * cos_theta - dy * sin_theta
            local_y = dx * sin_theta + dy * cos_theta

            return (local_x, local_y)
