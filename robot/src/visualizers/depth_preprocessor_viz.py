"""
Depth preprocessor visualization helpers.

Provides visualization for floor detection tuning and 1D distance output.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np
import cv2


@dataclass
class DepthVisualizerConfig:
    """Configuration for depth preprocessor visualizer."""
    polar_size: int = 400                    # Size of polar plot (square)
    max_range_mm: float = 3000.0             # Max range for display scaling
    background_color: Tuple[int, int, int] = (30, 30, 30)
    floor_color: Tuple[int, int, int] = (255, 100, 50)      # Blue tint for floor
    ceiling_color: Tuple[int, int, int] = (255, 50, 100)    # Purple for ceiling
    obstacle_color: Tuple[int, int, int] = (50, 50, 255)    # Red for obstacles
    distance_color: Tuple[int, int, int] = (0, 255, 0)      # Green for distance plot
    out_of_range_color: Tuple[int, int, int] = (80, 80, 80) # Gray


class DepthPreprocessorVisualizer:
    """Visualization helpers for depth preprocessing tuning."""

    def __init__(self, config: Optional[DepthVisualizerConfig] = None):
        """
        Initialize visualizer.

        Args:
            config: Visualization configuration
        """
        self._config = config or DepthVisualizerConfig()

    def draw_depth_with_masks(
        self,
        depth_map: np.ndarray,
        floor_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        max_range_mm: Optional[float] = None
    ) -> np.ndarray:
        """
        Draw depth map with floor and obstacle masks overlaid.

        Args:
            depth_map: Depth values in mm (H x W)
            floor_mask: Boolean mask where True = floor/ceiling (ignored)
            obstacle_mask: Boolean mask where True = valid obstacle
            max_range_mm: Max range for colorization (uses config if None)

        Returns:
            BGR image with overlays
        """
        max_range = max_range_mm or self._config.max_range_mm

        # Normalize depth for colorization
        depth_normalized = np.clip(depth_map / max_range, 0.0, 1.0)

        # Create colorized depth using TURBO colormap
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)

        # Mark invalid (zero depth) as black
        invalid_mask = depth_map <= 0
        depth_colored[invalid_mask] = [0, 0, 0]

        # Overlay floor mask (blue tint)
        floor_overlay = np.zeros_like(depth_colored)
        floor_overlay[floor_mask] = self._config.floor_color
        depth_colored = cv2.addWeighted(depth_colored, 0.7, floor_overlay, 0.3, 0)

        # Overlay obstacle mask (red outline/tint)
        obstacle_edges = self._get_mask_edges(obstacle_mask)
        depth_colored[obstacle_edges] = self._config.obstacle_color

        return depth_colored

    def draw_virtual_lidar(
        self,
        distances: np.ndarray,
        sector_angles: np.ndarray,
        valid_sectors: Optional[np.ndarray] = None,
        max_range_mm: Optional[float] = None,
        show_grid: bool = True
    ) -> np.ndarray:
        """
        Draw 1D distance array as polar plot (virtual LIDAR scan).

        Args:
            distances: Min distance per sector in mm
            sector_angles: Center angle of each sector in degrees
            valid_sectors: Boolean mask of sectors with valid data
            max_range_mm: Max range for scaling (uses config if None)
            show_grid: Whether to draw reference grid

        Returns:
            BGR image of polar plot
        """
        size = self._config.polar_size
        center = (size // 2, size // 2)
        max_radius = int(size * 0.45)
        max_range = max_range_mm or self._config.max_range_mm

        # Create blank image
        img = np.full((size, size, 3), self._config.background_color, dtype=np.uint8)

        # Draw reference grid
        if show_grid:
            self._draw_reference_grid(img, center, max_radius, max_range)

        # Draw distance points
        n = len(distances)
        for i in range(n):
            angle_deg = sector_angles[i]
            dist = distances[i]

            # Check if sector has valid data
            is_valid = valid_sectors[i] if valid_sectors is not None else dist < max_range

            if not is_valid or dist >= max_range:
                color = self._config.out_of_range_color
                radius = max_radius
            else:
                color = self._config.distance_color
                radius = int(max_radius * (dist / max_range))
                radius = max(5, radius)  # Minimum visibility

            # Convert angle to image coordinates
            x, y = self._angle_to_point(angle_deg, radius, center)

            # Draw point
            cv2.circle(img, (x, y), 4, color, -1, cv2.LINE_AA)

        # Connect valid points with lines
        prev_point = None
        prev_valid = False
        for i in range(n):
            angle_deg = sector_angles[i]
            dist = distances[i]
            is_valid = valid_sectors[i] if valid_sectors is not None else dist < max_range

            if is_valid and dist < max_range:
                radius = int(max_radius * (dist / max_range))
                radius = max(5, radius)
                x, y = self._angle_to_point(angle_deg, radius, center)
                current_point = (x, y)

                if prev_valid and prev_point is not None:
                    cv2.line(img, prev_point, current_point, self._config.distance_color, 1, cv2.LINE_AA)

                prev_point = current_point
                prev_valid = True
            else:
                prev_valid = False

        # Draw robot icon at center
        self._draw_robot_icon(img, center)

        # Draw forward indicator
        cv2.putText(img, "FWD", (center[0] - 15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        return img

    def draw_height_histogram(
        self,
        heights: np.ndarray,
        floor_threshold_mm: float,
        robot_height_mm: float,
        num_bins: int = 100,
        height_range: Tuple[float, float] = (-200, 800)
    ) -> np.ndarray:
        """
        Draw histogram of pixel heights with threshold lines.

        Useful for understanding height distribution and tuning thresholds.

        Args:
            heights: Height values in mm (flattened or 2D)
            floor_threshold_mm: Current floor threshold
            robot_height_mm: Current robot height cutoff
            num_bins: Number of histogram bins
            height_range: (min, max) height range for histogram

        Returns:
            BGR image of histogram
        """
        width, height = 400, 200
        img = np.full((height, width, 3), (40, 40, 40), dtype=np.uint8)

        # Flatten heights and remove invalid
        h_flat = heights.flatten()
        h_valid = h_flat[np.isfinite(h_flat) & (h_flat != 0)]

        if len(h_valid) == 0:
            cv2.putText(img, "No valid height data", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            return img

        # Compute histogram
        hist, bin_edges = np.histogram(h_valid, bins=num_bins, range=height_range)
        hist = hist.astype(np.float32)

        # Normalize histogram to fit in image
        if hist.max() > 0:
            hist = hist / hist.max()

        # Draw histogram bars
        bar_width = (width - 40) / num_bins
        margin_left = 30
        margin_bottom = 30

        for i in range(num_bins):
            x1 = int(margin_left + i * bar_width)
            x2 = int(margin_left + (i + 1) * bar_width) - 1
            bar_height = int(hist[i] * (height - margin_bottom - 20))
            y1 = height - margin_bottom - bar_height
            y2 = height - margin_bottom

            # Color based on whether bin is in valid range
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            if bin_center < floor_threshold_mm:
                color = self._config.floor_color
            elif bin_center > robot_height_mm:
                color = self._config.ceiling_color
            else:
                color = (100, 200, 100)  # Valid range - green

            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        # Draw threshold lines
        def height_to_x(h):
            return int(margin_left + (h - height_range[0]) / (height_range[1] - height_range[0]) * (width - 40))

        # Floor threshold line
        floor_x = height_to_x(floor_threshold_mm)
        cv2.line(img, (floor_x, 10), (floor_x, height - margin_bottom), (255, 100, 50), 2)
        cv2.putText(img, f"Floor: {floor_threshold_mm:.0f}", (floor_x + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 50), 1)

        # Robot height line
        robot_x = height_to_x(robot_height_mm)
        cv2.line(img, (robot_x, 10), (robot_x, height - margin_bottom), (255, 50, 100), 2)
        cv2.putText(img, f"Robot: {robot_height_mm:.0f}", (robot_x + 5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 50, 100), 1)

        # Draw axis labels
        cv2.putText(img, "Height (mm)", (width // 2 - 40, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # Draw axis ticks
        for h in [0, 200, 400, 600]:
            x = height_to_x(h)
            cv2.line(img, (x, height - margin_bottom), (x, height - margin_bottom + 5), (100, 100, 100), 1)
            cv2.putText(img, str(h), (x - 10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        return img

    def _draw_reference_grid(
        self,
        img: np.ndarray,
        center: Tuple[int, int],
        max_radius: int,
        max_range_mm: float
    ) -> None:
        """Draw reference circles and labels."""
        # Draw concentric circles at 1m, 2m, 3m
        for dist_m in [1.0, 2.0, 3.0]:
            dist_mm = dist_m * 1000
            if dist_mm <= max_range_mm:
                radius = int(max_radius * (dist_mm / max_range_mm))
                cv2.circle(img, center, radius, (60, 60, 60), 1, cv2.LINE_AA)

                # Label
                label_x = center[0] + radius + 5
                label_y = center[1]
                cv2.putText(img, f"{dist_m:.0f}m", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

        # Draw radial lines every 30 degrees
        for angle in range(0, 360, 30):
            x, y = self._angle_to_point(angle, max_radius, center)
            cv2.line(img, center, (x, y), (50, 50, 50), 1, cv2.LINE_AA)

    def _draw_robot_icon(self, img: np.ndarray, center: Tuple[int, int]) -> None:
        """Draw robot triangle at center."""
        size = 10
        cx, cy = center
        pts = np.array([
            [cx, cy - size],
            [cx - size, cy + size],
            [cx + size, cy + size]
        ], np.int32)
        cv2.fillPoly(img, [pts], (180, 180, 180), cv2.LINE_AA)

    def _get_mask_edges(self, mask: np.ndarray, thickness: int = 2) -> np.ndarray:
        """Get edge pixels of a boolean mask."""
        mask_uint8 = mask.astype(np.uint8) * 255
        kernel = np.ones((thickness, thickness), np.uint8)
        dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
        eroded = cv2.erode(mask_uint8, kernel, iterations=1)
        edges = dilated - eroded
        return edges > 0

    def _angle_to_point(
        self,
        angle_deg: float,
        radius: int,
        center: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Convert angle and radius to image point."""
        angle_rad = math.radians(angle_deg)
        x = int(center[0] + radius * math.sin(angle_rad))
        y = int(center[1] - radius * math.cos(angle_rad))
        return (x, y)

    def draw_virtual_lidar_with_heading(
        self,
        distances: np.ndarray,
        sector_angles: np.ndarray,
        valid_sectors: Optional[np.ndarray] = None,
        selected_heading_deg: Optional[float] = None,
        target_heading_deg: Optional[float] = None,
        safety_distance_mm: Optional[float] = None,
        max_range_mm: Optional[float] = None,
        can_proceed: bool = True
    ) -> np.ndarray:
        """
        Draw 1D distance array as polar plot with heading overlays.

        Args:
            distances: Min distance per sector in mm
            sector_angles: Center angle of each sector in degrees
            valid_sectors: Boolean mask of sectors with valid data
            selected_heading_deg: Best heading from path planner (green arrow)
            target_heading_deg: Target direction (cyan arrow)
            safety_distance_mm: Safety threshold to display as circle
            max_range_mm: Max range for scaling
            can_proceed: Whether path is clear (affects title color)

        Returns:
            BGR image of polar plot with overlays
        """
        # Start with base virtual LIDAR
        img = self.draw_virtual_lidar(
            distances, sector_angles, valid_sectors, max_range_mm, show_grid=True
        )

        size = self._config.polar_size
        center = (size // 2, size // 2)
        max_radius = int(size * 0.45)
        max_range = max_range_mm or self._config.max_range_mm

        # Draw safety distance circle if specified
        if safety_distance_mm is not None and safety_distance_mm < max_range:
            safety_radius = int(max_radius * (safety_distance_mm / max_range))
            cv2.circle(img, center, safety_radius, (0, 100, 255), 1, cv2.LINE_AA)

        # Draw target heading (cyan dashed line)
        if target_heading_deg is not None:
            x, y = self._angle_to_point(target_heading_deg, max_radius, center)
            # Draw as dotted/dashed by drawing shorter segments
            self._draw_dashed_line(img, center, (x, y), (255, 255, 0), 1, 8)
            # Small arrowhead
            self._draw_arrowhead(img, center, (x, y), (255, 255, 0), 8)

        # Draw selected heading (green solid arrow) if different from target
        if selected_heading_deg is not None:
            arrow_color = (0, 255, 0) if can_proceed else (0, 100, 255)
            arrow_len = int(max_radius * 0.7)
            x, y = self._angle_to_point(selected_heading_deg, arrow_len, center)
            cv2.arrowedLine(img, center, (x, y), arrow_color, 2, cv2.LINE_AA, tipLength=0.2)

        # Status indicator
        status_color = (0, 255, 0) if can_proceed else (0, 0, 255)
        status_text = "CLEAR" if can_proceed else "BLOCKED"
        cv2.putText(img, status_text, (size - 70, size - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1, cv2.LINE_AA)

        return img

    def draw_planner_debug_panel(
        self,
        planner_result: Optional[object],
        algorithm: str,
        target_angle_deg: Optional[float] = None,
        target_range_mm: Optional[float] = None,
        panel_size: Tuple[int, int] = (320, 240)
    ) -> np.ndarray:
        """
        Draw debug info panel showing planner state and reason for can_proceed.

        Args:
            planner_result: PlannerResult from path planner
            algorithm: Algorithm name ("follow_gap" or "apf")
            target_angle_deg: Current target angle
            target_range_mm: Current target range
            panel_size: (width, height) of panel

        Returns:
            BGR image with debug info
        """
        width, height = panel_size
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        y = 20
        line_height = 20

        # Title
        cv2.putText(panel, f"Planner: {algorithm.upper()}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_height + 5

        if planner_result is None:
            cv2.putText(panel, "No planner result", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)
            return panel

        # Access attributes safely (duck typing for PlannerResult)
        can_proceed = getattr(planner_result, 'can_proceed', None)
        best_heading = getattr(planner_result, 'best_heading_deg', None)
        debug_info = getattr(planner_result, 'debug_info', {}) or {}

        # Can proceed status (prominent)
        if can_proceed is not None:
            status_color = (0, 255, 0) if can_proceed else (0, 0, 255)
            status_text = "CAN PROCEED: YES" if can_proceed else "CAN PROCEED: NO"
            cv2.putText(panel, status_text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
            y += line_height

        # Reason (when blocked)
        reason = debug_info.get('reason', '')
        if reason:
            reason_text = f"Reason: {reason.replace('_', ' ')}"
            cv2.putText(panel, reason_text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 255), 1, cv2.LINE_AA)
            y += line_height

        # Selected heading
        if best_heading is not None:
            cv2.putText(panel, f"Heading: {best_heading:+.1f} deg", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            y += line_height

        # Target info
        if target_angle_deg is not None:
            cv2.putText(panel, f"Target: {target_angle_deg:+.1f} deg", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1, cv2.LINE_AA)
            y += line_height

        y += 5  # Small gap before algorithm-specific info

        # Algorithm-specific debug info
        if algorithm == 'follow_gap':
            self._draw_follow_gap_debug(panel, debug_info, y, line_height)
        elif algorithm == 'apf':
            self._draw_apf_debug(panel, debug_info, y, line_height)

        return panel

    def _draw_follow_gap_debug(
        self,
        panel: np.ndarray,
        debug_info: dict,
        y: int,
        line_height: int
    ) -> None:
        """Draw Follow-the-Gap specific debug info."""
        gaps_found = debug_info.get('gaps_found', '?')
        passable_gaps = debug_info.get('passable_gaps', '?')

        cv2.putText(panel, f"Gaps found: {gaps_found}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        y += line_height

        color = (0, 255, 0) if passable_gaps and passable_gaps > 0 else (100, 100, 255)
        cv2.putText(panel, f"Passable gaps: {passable_gaps}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        y += line_height

        # Selected gap details
        gap_width = debug_info.get('selected_gap_width_deg')
        gap_depth = debug_info.get('selected_gap_depth_mm')
        if gap_width is not None:
            cv2.putText(panel, f"Gap width: {gap_width:.1f} deg", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 150), 1, cv2.LINE_AA)
            y += line_height
        if gap_depth is not None:
            cv2.putText(panel, f"Gap depth: {gap_depth:.0f} mm", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 150), 1, cv2.LINE_AA)

    def _draw_apf_debug(
        self,
        panel: np.ndarray,
        debug_info: dict,
        y: int,
        line_height: int
    ) -> None:
        """Draw APF specific debug info."""
        min_dist = debug_info.get('min_distance_mm')
        if min_dist is not None:
            color = (0, 255, 0) if min_dist > 200 else (0, 100, 255)
            cv2.putText(panel, f"Min distance: {min_dist:.0f} mm", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            y += line_height

        # Force magnitudes
        f_att = debug_info.get('f_attractive')
        f_rep = debug_info.get('f_repulsive')
        f_total = debug_info.get('f_total')
        magnitude = debug_info.get('resultant_magnitude')

        if f_att is not None:
            cv2.putText(panel, f"F_attract: ({f_att[0]:.2f}, {f_att[1]:.2f})", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1, cv2.LINE_AA)
            y += line_height

        if f_rep is not None:
            cv2.putText(panel, f"F_repel: ({f_rep[0]:.2f}, {f_rep[1]:.2f})", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 255), 1, cv2.LINE_AA)
            y += line_height

        if magnitude is not None:
            cv2.putText(panel, f"Resultant: {magnitude:.3f}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

        # Emergency stop indicator
        if debug_info.get('emergency_stop'):
            cv2.putText(panel, "EMERGENCY STOP", (10, panel.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        elif debug_info.get('local_minimum'):
            cv2.putText(panel, "LOCAL MINIMUM", (10, panel.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1, cv2.LINE_AA)

    def _draw_dashed_line(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 1,
        dash_length: int = 5
    ) -> None:
        """Draw a dashed line between two points."""
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 1:
            return

        num_dashes = int(dist / (dash_length * 2))
        for i in range(num_dashes):
            t1 = (i * 2 * dash_length) / dist
            t2 = ((i * 2 + 1) * dash_length) / dist
            t2 = min(t2, 1.0)

            x1 = int(pt1[0] + dx * t1)
            y1 = int(pt1[1] + dy * t1)
            x2 = int(pt1[0] + dx * t2)
            y2 = int(pt1[1] + dy * t2)

            cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    def _draw_arrowhead(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        size: int = 10
    ) -> None:
        """Draw arrowhead at pt2 pointing from pt1."""
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 1:
            return

        # Normalize
        dx /= dist
        dy /= dist

        # Perpendicular
        px = -dy
        py = dx

        # Arrowhead points
        p1 = (int(pt2[0] - size * dx + size * 0.5 * px),
              int(pt2[1] - size * dy + size * 0.5 * py))
        p2 = (int(pt2[0] - size * dx - size * 0.5 * px),
              int(pt2[1] - size * dy - size * 0.5 * py))

        cv2.line(img, pt2, p1, color, 1, cv2.LINE_AA)
        cv2.line(img, pt2, p2, color, 1, cv2.LINE_AA)
