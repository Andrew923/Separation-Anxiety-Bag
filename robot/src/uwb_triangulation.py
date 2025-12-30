"""
UWB triangulation and front/back disambiguation.

Uses two UWB anchors mounted on the robot to triangulate
the angle to a person wearing a UWB tag.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import math
import json
from pathlib import Path


@dataclass
class TriangulationConfig:
    """Configuration for UWB triangulation."""
    # Anchor positions relative to robot center (x, y) in mm
    # x: positive = right, y: positive = forward
    anchor1_position: Tuple[float, float] = (100.0, 50.0)
    anchor2_position: Tuple[float, float] = (-100.0, 50.0)
    # Calibration offset for front/back disambiguation
    front_offset_deg: float = 0.0
    # Whether calibration has been performed
    calibration_valid: bool = False


@dataclass
class TriangulationResult:
    """Result of UWB triangulation."""
    angle_deg: float           # Angle to target (0 = forward, positive = right)
    is_front: bool             # True if target is in front of robot
    confidence: float          # 0-1 confidence score
    range1_mm: float           # Distance from anchor 1
    range2_mm: float           # Distance from anchor 2
    estimated_distance_mm: float  # Estimated distance to target


class UWBTriangulator:
    """
    Triangulates person position from two UWB anchor ranges.

    Geometry:
    - Two anchors mounted at known positions on robot
    - TAG on person returns range to each anchor
    - Circle-circle intersection gives possible positions
    - Calibration determines which intersection is "front"
    """

    def __init__(self, config: TriangulationConfig):
        """
        Initialize triangulator with anchor positions.

        Args:
            config: Triangulation configuration
        """
        self._config = config
        self._baseline = self._compute_baseline()

    def _compute_baseline(self) -> float:
        """Compute distance between anchors."""
        x1, y1 = self._config.anchor1_position
        x2, y2 = self._config.anchor2_position
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def triangulate(
        self,
        range1_mm: float,
        range2_mm: float
    ) -> Optional[TriangulationResult]:
        """
        Compute angle to target from two ranges.

        Uses circle-circle intersection algorithm.

        Args:
            range1_mm: Distance from anchor 1
            range2_mm: Distance from anchor 2

        Returns:
            TriangulationResult or None if geometry invalid
        """
        x1, y1 = self._config.anchor1_position
        x2, y2 = self._config.anchor2_position
        d = self._baseline

        # Check if triangulation is geometrically possible
        if range1_mm + range2_mm < d:
            # Circles don't intersect - ranges too short
            return None
        if abs(range1_mm - range2_mm) > d:
            # One circle inside the other
            return None

        # Circle-circle intersection
        # Reference: https://mathworld.wolfram.com/Circle-CircleIntersection.html

        # Distance along baseline from anchor1 to intersection chord
        a = (range1_mm ** 2 - range2_mm ** 2 + d ** 2) / (2 * d)

        # Check for valid geometry
        h_squared = range1_mm ** 2 - a ** 2
        if h_squared < 0:
            return None

        h = math.sqrt(h_squared)

        # Direction vector from anchor1 to anchor2
        dx = (x2 - x1) / d
        dy = (y2 - y1) / d

        # Point on baseline at intersection chord
        px = x1 + a * dx
        py = y1 + a * dy

        # Perpendicular direction
        nx = -dy
        ny = dx

        # Two intersection points
        # Point 1: "positive" side of baseline
        ix1 = px + h * nx
        iy1 = py + h * ny

        # Point 2: "negative" side of baseline
        ix2 = px - h * nx
        iy2 = py - h * ny

        # Determine which point to use based on calibration
        # Default: assume the one with larger y (more forward) is front
        if self._config.calibration_valid:
            # Use calibrated offset to determine front
            angle1 = math.degrees(math.atan2(ix1, iy1))
            angle2 = math.degrees(math.atan2(ix2, iy2))

            # Choose point closer to calibrated front direction
            diff1 = abs(self._normalize_angle(angle1 - self._config.front_offset_deg))
            diff2 = abs(self._normalize_angle(angle2 - self._config.front_offset_deg))

            if diff1 < diff2:
                target_x, target_y = ix1, iy1
                is_front = True
            else:
                target_x, target_y = ix2, iy2
                is_front = False
        else:
            # No calibration - assume larger Y is front
            if iy1 >= iy2:
                target_x, target_y = ix1, iy1
                is_front = iy1 > 0
            else:
                target_x, target_y = ix2, iy2
                is_front = iy2 > 0

        # Convert to angle (0 = forward, positive = right)
        angle_rad = math.atan2(target_x, target_y)
        angle_deg = math.degrees(angle_rad)

        # Estimate distance (average of ranges, or distance to intersection point)
        estimated_distance = math.sqrt(target_x ** 2 + target_y ** 2)

        # Confidence based on geometry
        # Higher when ranges are similar (target roughly on perpendicular bisector)
        range_ratio = min(range1_mm, range2_mm) / max(range1_mm, range2_mm)
        # Also consider if ranges are reasonable
        geometry_score = min(1.0, h / (d * 0.5))  # Higher h = better geometry
        confidence = range_ratio * 0.5 + geometry_score * 0.5

        return TriangulationResult(
            angle_deg=angle_deg,
            is_front=is_front,
            confidence=confidence,
            range1_mm=range1_mm,
            range2_mm=range2_mm,
            estimated_distance_mm=estimated_distance
        )

    def _normalize_angle(self, angle_deg: float) -> float:
        """Normalize angle to [-180, 180] range."""
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg < -180:
            angle_deg += 360
        return angle_deg

    def set_front_offset(self, offset_deg: float) -> None:
        """
        Set calibration offset for front determination.

        Args:
            offset_deg: Angle offset when person is in front
        """
        self._config.front_offset_deg = offset_deg
        self._config.calibration_valid = True

    def get_config(self) -> TriangulationConfig:
        """Get current configuration."""
        return self._config


class UWBCalibrator:
    """
    Calibration routine for UWB front/back disambiguation.

    Procedure:
    1. User stands in front of robot at known position
    2. System records triangulation angle
    3. This becomes the "front" reference angle
    """

    def __init__(self, triangulator: UWBTriangulator):
        """
        Initialize calibrator.

        Args:
            triangulator: Triangulator to calibrate
        """
        self._triangulator = triangulator
        self._front_samples: List[Tuple[float, float]] = []
        self._back_samples: List[Tuple[float, float]] = []
        self._calibration_complete = False

    def reset(self) -> None:
        """Reset calibration samples."""
        self._front_samples = []
        self._back_samples = []
        self._calibration_complete = False

    def record_front_sample(self, range1: float, range2: float) -> Optional[float]:
        """
        Record a sample when user is in front of robot.

        Args:
            range1: Range from anchor 1
            range2: Range from anchor 2

        Returns:
            Computed angle, or None if triangulation failed
        """
        result = self._triangulator.triangulate(range1, range2)
        if result is not None:
            self._front_samples.append((range1, range2))
            return result.angle_deg
        return None

    def record_back_sample(self, range1: float, range2: float) -> Optional[float]:
        """
        Record a sample when user is behind robot.

        Args:
            range1: Range from anchor 1
            range2: Range from anchor 2

        Returns:
            Computed angle, or None if triangulation failed
        """
        result = self._triangulator.triangulate(range1, range2)
        if result is not None:
            self._back_samples.append((range1, range2))
            return result.angle_deg
        return None

    def get_sample_counts(self) -> Tuple[int, int]:
        """
        Get number of samples collected.

        Returns:
            Tuple of (front_count, back_count)
        """
        return (len(self._front_samples), len(self._back_samples))

    def compute_calibration(self) -> Optional[float]:
        """
        Compute calibration offset from samples.

        Requires at least 3 front samples.

        Returns:
            Front offset in degrees, or None if insufficient samples
        """
        if len(self._front_samples) < 3:
            return None

        # Compute average front angle
        front_angles = []
        for r1, r2 in self._front_samples:
            result = self._triangulator.triangulate(r1, r2)
            if result is not None:
                front_angles.append(result.angle_deg)

        if not front_angles:
            return None

        # Average angle (handle wraparound)
        sin_sum = sum(math.sin(math.radians(a)) for a in front_angles)
        cos_sum = sum(math.cos(math.radians(a)) for a in front_angles)
        avg_angle = math.degrees(math.atan2(sin_sum, cos_sum))

        # Apply to triangulator
        self._triangulator.set_front_offset(avg_angle)
        self._calibration_complete = True

        return avg_angle

    def save_calibration(self, config_path: str) -> bool:
        """
        Save calibration to config file.

        Args:
            config_path: Path to robot_config.yaml

        Returns:
            True if save successful
        """
        if not self._calibration_complete:
            return False

        try:
            import yaml

            # Read existing config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Update calibration section
            if 'uwb' not in config:
                config['uwb'] = {}
            if 'calibration' not in config['uwb']:
                config['uwb']['calibration'] = {}

            config['uwb']['calibration']['front_offset_deg'] = float(
                self._triangulator.get_config().front_offset_deg
            )
            config['uwb']['calibration']['valid'] = True

            # Write back
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            return True

        except Exception as e:
            print(f"Failed to save calibration: {e}")
            return False

    @property
    def is_complete(self) -> bool:
        """Check if calibration is complete."""
        return self._calibration_complete
