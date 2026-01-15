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
