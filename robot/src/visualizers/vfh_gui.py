"""
VFH (Vector Field Histogram) visualization using OpenCV.

Provides a polar plot visualization of obstacle histogram with
target direction and recommended heading overlays.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np
import cv2

from ..vfh import VFHResult
from ..navigation import NavigationState


@dataclass
class VisualizerConfig:
    """Configuration for VFH visualizer."""
    polar_size: int = 400                    # Size of polar plot (square)
    camera_fov_deg: float = 60.0             # Camera horizontal FOV
    max_range_display_mm: float = 3000.0     # Max range for display scaling
    num_sectors: int = 72                    # Must match VFH config
    obstacle_threshold: float = 0.3          # Threshold for blocked (must match VFH)
    background_color: Tuple[int, int, int] = (30, 30, 30)    # Dark gray
    safe_color: Tuple[int, int, int] = (0, 180, 0)           # Green
    blocked_color: Tuple[int, int, int] = (0, 0, 200)        # Red
    unknown_color: Tuple[int, int, int] = (80, 80, 80)       # Gray for no-data sectors
    target_color: Tuple[int, int, int] = (255, 150, 0)       # Blue (BGR)
    heading_color: Tuple[int, int, int] = (0, 255, 100)      # Bright green
    fov_color: Tuple[int, int, int] = (100, 100, 100)        # Gray
    robot_color: Tuple[int, int, int] = (200, 200, 200)      # Light gray


class VFHVisualizer:
    """
    Visualizes VFH obstacle histogram as a polar plot.

    Features:
    - Polar histogram with blocked (red) and safe (green) sectors
    - Camera FOV indicator
    - Target direction line (from UWB)
    - Best heading arrow (from VFH)
    - Robot icon at center
    """

    def __init__(self, config: Optional[VisualizerConfig] = None):
        """
        Initialize VFH visualizer.

        Args:
            config: Visualization configuration
        """
        self._config = config or VisualizerConfig()
        self._size = self._config.polar_size
        self._center = (self._size // 2, self._size // 2)
        self._max_radius = int(self._size * 0.45)  # Leave margin for labels

        # Precompute sector angles
        self._sector_width = 360.0 / self._config.num_sectors
        self._sector_angles = np.arange(self._config.num_sectors) * self._sector_width

    def draw_polar_histogram(
        self,
        vfh_result: Optional[VFHResult],
        target_angle_deg: Optional[float] = None,
        target_range_mm: Optional[float] = None,
        nav_state: NavigationState = NavigationState.IDLE
    ) -> np.ndarray:
        """
        Draw complete VFH polar visualization.

        Args:
            vfh_result: VFH computation result (None for empty display)
            target_angle_deg: UWB target angle (None if no target)
            target_range_mm: UWB target range (None if no target)
            nav_state: Current navigation state

        Returns:
            BGR image of polar plot
        """
        # Create blank image
        img = np.full(
            (self._size, self._size, 3),
            self._config.background_color,
            dtype=np.uint8
        )

        # Draw reference circles
        self._draw_reference_circles(img)

        # Draw camera FOV indicator
        self._draw_camera_fov(img)

        # Draw histogram sectors
        if vfh_result is not None:
            self._draw_sectors(img, vfh_result.histogram, vfh_result.blocked_sectors)

            # Draw best heading arrow
            if vfh_result.best_heading_deg is not None:
                self._draw_best_heading(img, vfh_result.best_heading_deg)

        # Draw target direction
        if target_angle_deg is not None:
            self._draw_target_direction(img, target_angle_deg, target_range_mm)

        # Draw robot icon at center
        self._draw_robot_icon(img)

        # Draw forward indicator
        self._draw_forward_indicator(img)

        return img

    def _draw_reference_circles(self, img: np.ndarray) -> None:
        """Draw concentric reference circles."""
        # Draw 3 reference circles at 33%, 66%, 100% of max radius
        for fraction in [0.33, 0.66, 1.0]:
            radius = int(self._max_radius * fraction)
            cv2.circle(
                img,
                self._center,
                radius,
                (60, 60, 60),  # Dark gray
                1,
                cv2.LINE_AA
            )

        # Draw cross at center
        cross_size = 10
        cv2.line(
            img,
            (self._center[0] - cross_size, self._center[1]),
            (self._center[0] + cross_size, self._center[1]),
            (60, 60, 60),
            1
        )
        cv2.line(
            img,
            (self._center[0], self._center[1] - cross_size),
            (self._center[0], self._center[1] + cross_size),
            (60, 60, 60),
            1
        )

    def _draw_camera_fov(self, img: np.ndarray) -> None:
        """Draw camera field of view indicator."""
        fov = self._config.camera_fov_deg
        half_fov = fov / 2

        # FOV edges (0 degrees = up/forward in image coordinates)
        # Convert to image angle: 0 deg = up, positive = clockwise
        left_angle = -half_fov - 90  # -90 to rotate from math coords to image
        right_angle = half_fov - 90

        # Draw FOV arc
        cv2.ellipse(
            img,
            self._center,
            (self._max_radius, self._max_radius),
            0,  # No rotation
            left_angle,
            right_angle,
            self._config.fov_color,
            2,
            cv2.LINE_AA
        )

        # Draw FOV edge lines (dashed effect using short segments)
        for angle in [-half_fov, half_fov]:
            end_x, end_y = self._angle_to_point(angle, self._max_radius)
            # Draw dashed line
            self._draw_dashed_line(
                img,
                self._center,
                (end_x, end_y),
                self._config.fov_color,
                dash_length=10
            )

    def _draw_sectors(
        self,
        img: np.ndarray,
        histogram: np.ndarray,
        blocked_sectors: list
    ) -> None:
        """
        Draw histogram sectors as colored wedges.

        All 72 sectors are drawn:
        - Gray: Outside camera FOV (no data)
        - Red: Blocked (obstacle density above threshold)
        - Green: Safe (intensity varies with density - brighter = safer)

        Args:
            img: Image to draw on
            histogram: Obstacle density per sector
            blocked_sectors: List of blocked sector indices
        """
        num_sectors = len(histogram)
        sector_width = 360.0 / num_sectors
        fov = self._config.camera_fov_deg
        half_fov = fov / 2
        threshold = self._config.obstacle_threshold

        # Always use full radius for all sectors
        radius = self._max_radius

        for i in range(num_sectors):
            # Sector angle (centered on forward = 0 degrees)
            # Sector 0 starts at -180 degrees
            angle_start = -180 + i * sector_width
            angle_center = angle_start + sector_width / 2

            density = histogram[i]

            # Determine if sector is inside camera FOV
            is_inside_fov = abs(angle_center) <= half_fov

            # Determine color based on state
            if not is_inside_fov:
                # Outside FOV - gray (unknown/no data)
                color = self._config.unknown_color
            elif i in blocked_sectors:
                # Blocked - solid red
                color = self._config.blocked_color
            else:
                # Safe - green with intensity based on safety margin
                # Lower density = brighter green (safer)
                # density=0 -> intensity=1.0, density=threshold -> intensity=0.4
                safety_factor = 1.0 - (density / threshold) if threshold > 0 else 1.0
                safety_factor = max(0.0, min(1.0, safety_factor))  # Clamp to [0, 1]
                intensity = 0.4 + 0.6 * safety_factor  # Range: 0.4 to 1.0
                color = tuple(int(c * intensity) for c in self._config.safe_color)

            # Draw filled sector
            # OpenCV ellipse angles: 0 = right (3 o'clock), counter-clockwise
            # We want: 0 = up (12 o'clock), clockwise for positive angles
            # Transform: cv_angle = -(angle + 90)
            cv_start = -(angle_start + sector_width + 90)
            cv_end = -(angle_start + 90)

            # Draw sector as filled ellipse arc
            cv2.ellipse(
                img,
                self._center,
                (radius, radius),
                0,
                cv_start,
                cv_end,
                color,
                -1,  # Filled
                cv2.LINE_AA
            )

            # Draw sector outline
            cv2.ellipse(
                img,
                self._center,
                (radius, radius),
                0,
                cv_start,
                cv_end,
                (50, 50, 50),
                1,
                cv2.LINE_AA
            )

    def _draw_target_direction(
        self,
        img: np.ndarray,
        angle_deg: float,
        range_mm: Optional[float] = None
    ) -> None:
        """
        Draw target direction indicator.

        Args:
            img: Image to draw on
            angle_deg: Target angle in degrees (0 = forward)
            range_mm: Optional target range for scaling
        """
        # Calculate line length based on range if available
        if range_mm is not None and range_mm > 0:
            length = int(self._max_radius * min(1.0, range_mm / self._config.max_range_display_mm))
            length = max(length, int(self._max_radius * 0.3))  # Minimum visibility
        else:
            length = self._max_radius

        # Draw thick line from center toward target
        end_x, end_y = self._angle_to_point(angle_deg, length)

        cv2.line(
            img,
            self._center,
            (end_x, end_y),
            self._config.target_color,
            3,
            cv2.LINE_AA
        )

        # Draw small circle at end
        cv2.circle(img, (end_x, end_y), 6, self._config.target_color, -1, cv2.LINE_AA)

        # Draw "T" label near the end
        label_x = end_x + 10 if end_x > self._center[0] else end_x - 20
        label_y = end_y - 10 if end_y < self._center[1] else end_y + 20
        cv2.putText(
            img,
            "T",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self._config.target_color,
            2,
            cv2.LINE_AA
        )

    def _draw_best_heading(self, img: np.ndarray, angle_deg: float) -> None:
        """
        Draw VFH recommended heading as an arrow.

        Args:
            img: Image to draw on
            angle_deg: Best heading angle in degrees
        """
        length = int(self._max_radius * 0.7)
        end_x, end_y = self._angle_to_point(angle_deg, length)

        # Draw arrow shaft
        cv2.line(
            img,
            self._center,
            (end_x, end_y),
            self._config.heading_color,
            2,
            cv2.LINE_AA
        )

        # Draw arrowhead
        arrow_size = 15
        angle_rad = math.radians(angle_deg - 90)  # Convert to image coords

        # Arrowhead points
        left_angle = angle_rad + math.radians(150)
        right_angle = angle_rad - math.radians(150)

        left_x = int(end_x + arrow_size * math.cos(left_angle))
        left_y = int(end_y + arrow_size * math.sin(left_angle))
        right_x = int(end_x + arrow_size * math.cos(right_angle))
        right_y = int(end_y + arrow_size * math.sin(right_angle))

        pts = np.array([[end_x, end_y], [left_x, left_y], [right_x, right_y]], np.int32)
        cv2.fillPoly(img, [pts], self._config.heading_color, cv2.LINE_AA)

    def _draw_robot_icon(self, img: np.ndarray) -> None:
        """Draw a small robot icon at center."""
        size = 12
        cx, cy = self._center

        # Draw triangle pointing up (forward)
        pts = np.array([
            [cx, cy - size],           # Top (front)
            [cx - size, cy + size],    # Bottom left
            [cx + size, cy + size]     # Bottom right
        ], np.int32)

        cv2.fillPoly(img, [pts], self._config.robot_color, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, (100, 100, 100), 1, cv2.LINE_AA)

    def _draw_forward_indicator(self, img: np.ndarray) -> None:
        """Draw forward direction label."""
        cv2.putText(
            img,
            "FWD",
            (self._center[0] - 15, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1,
            cv2.LINE_AA
        )

    def _draw_dashed_line(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        dash_length: int = 10
    ) -> None:
        """Draw a dashed line between two points."""
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = math.sqrt(dx * dx + dy * dy)

        if length == 0:
            return

        dx /= length
        dy /= length

        num_dashes = int(length / (dash_length * 2))
        for i in range(num_dashes + 1):
            start_dist = i * dash_length * 2
            end_dist = min(start_dist + dash_length, length)

            start_x = int(pt1[0] + dx * start_dist)
            start_y = int(pt1[1] + dy * start_dist)
            end_x = int(pt1[0] + dx * end_dist)
            end_y = int(pt1[1] + dy * end_dist)

            cv2.line(img, (start_x, start_y), (end_x, end_y), color, 1, cv2.LINE_AA)

    def _angle_to_point(self, angle_deg: float, radius: int) -> Tuple[int, int]:
        """
        Convert angle and radius to image point.

        Args:
            angle_deg: Angle in degrees (0 = forward/up)
            radius: Distance from center

        Returns:
            (x, y) image coordinates
        """
        # In image coords: 0 degrees = up, positive = clockwise
        # Convert to standard math: angle_rad = -(angle_deg - 90) in radians
        # Or: x = r * sin(angle), y = -r * cos(angle)
        angle_rad = math.radians(angle_deg)
        x = int(self._center[0] + radius * math.sin(angle_rad))
        y = int(self._center[1] - radius * math.cos(angle_rad))
        return (x, y)


class InfoPanelRenderer:
    """
    Renders info panel with sensor data and status.
    """

    def __init__(
        self,
        width: int = 320,
        height: int = 160,
        background_color: Tuple[int, int, int] = (40, 40, 40)
    ):
        """
        Initialize info panel renderer.

        Args:
            width: Panel width in pixels
            height: Panel height in pixels
            background_color: Background BGR color
        """
        self._width = width
        self._height = height
        self._bg_color = background_color
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.45
        self._line_height = 22

    def render(
        self,
        uwb_enabled: bool,
        range1_mm: Optional[float],
        range2_mm: Optional[float],
        target_angle_deg: Optional[float],
        target_range_mm: Optional[float],
        nav_state: NavigationState,
        linear_vel_mm_s: float,
        angular_vel_deg_s: float,
        fps: float
    ) -> np.ndarray:
        """
        Render info panel with current status.

        Args:
            uwb_enabled: Whether UWB is enabled
            range1_mm: Range from anchor 1
            range2_mm: Range from anchor 2
            target_angle_deg: Target angle
            target_range_mm: Target range
            nav_state: Navigation state
            linear_vel_mm_s: Linear velocity command
            angular_vel_deg_s: Angular velocity command
            fps: Current FPS

        Returns:
            BGR image of info panel
        """
        img = np.full((self._height, self._width, 3), self._bg_color, dtype=np.uint8)

        y = 20
        text_color = (200, 200, 200)
        value_color = (100, 255, 100)
        disabled_color = (100, 100, 100)

        # Line 1: UWB status
        if uwb_enabled:
            r1_str = f"{range1_mm:.0f}" if range1_mm else "---"
            r2_str = f"{range2_mm:.0f}" if range2_mm else "---"
            uwb_text = f"UWB: A1={r1_str}mm A2={r2_str}mm"
            cv2.putText(img, uwb_text, (10, y), self._font, self._font_scale, value_color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "UWB: DISABLED", (10, y), self._font, self._font_scale, disabled_color, 1, cv2.LINE_AA)

        # Line 2: Target
        y += self._line_height
        if target_angle_deg is not None and target_range_mm is not None:
            target_text = f"Target: {target_angle_deg:+.1f}deg @ {target_range_mm:.0f}mm"
            cv2.putText(img, target_text, (10, y), self._font, self._font_scale, value_color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "Target: NO TARGET", (10, y), self._font, self._font_scale, disabled_color, 1, cv2.LINE_AA)

        # Line 3: Nav state
        y += self._line_height
        state_text = f"Nav: {nav_state.name}"
        cv2.putText(img, state_text, (10, y), self._font, self._font_scale, text_color, 1, cv2.LINE_AA)

        # Line 4: Command velocities
        y += self._line_height
        cmd_text = f"Cmd: {linear_vel_mm_s:+.0f}mm/s {angular_vel_deg_s:+.0f}deg/s"
        cv2.putText(img, cmd_text, (10, y), self._font, self._font_scale, text_color, 1, cv2.LINE_AA)

        # Line 5: FPS and hints
        y += self._line_height
        hint_text = f"FPS: {fps:.1f} | E:lines C:cmap Q:quit"
        cv2.putText(img, hint_text, (10, y), self._font, self._font_scale, (150, 150, 150), 1, cv2.LINE_AA)

        return img
