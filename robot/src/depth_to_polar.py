"""
Convert depth map to polar obstacle representation.

Takes stereo depth map and converts to angular sectors with
height filtering to ignore floor and above-robot obstacles.

DEPRECATED: Use DepthPreprocessor instead for new code.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import math
import numpy as np


@dataclass
class DepthToPolarConfig:
    """Configuration for depth to polar conversion."""
    # Camera parameters
    horizontal_fov_deg: float = 60.0    # Horizontal field of view
    vertical_fov_deg: float = 45.0      # Vertical field of view
    image_width: int = 320
    image_height: int = 240
    # Camera mounting
    camera_height_mm: float = 200.0     # Height above ground
    camera_tilt_deg: float = 0.0        # Tilt angle (positive = down)


class DepthToPolar:
    """
    Converts stereo depth map to polar obstacle representation.

    Key operations:
    1. Filter by height (remove floor, ceiling)
    2. Convert to polar coordinates (angle, distance)
    3. Aggregate into angular sectors for path planning

    DEPRECATED: Use DepthPreprocessor instead.
    """

    def __init__(self, config: DepthToPolarConfig):
        """
        Initialize converter with camera parameters.

        Args:
            config: Conversion configuration
        """
        self._config = config
        self._angle_lut: Optional[np.ndarray] = None
        self._height_lut: Optional[np.ndarray] = None
        self._precompute_lookup_tables()

    def _precompute_lookup_tables(self) -> None:
        """Precompute pixel-to-angle and pixel-to-height mappings."""
        w = self._config.image_width
        h = self._config.image_height
        h_fov = math.radians(self._config.horizontal_fov_deg)
        v_fov = math.radians(self._config.vertical_fov_deg)
        tilt = math.radians(self._config.camera_tilt_deg)

        # Create meshgrid of pixel coordinates
        cols = np.arange(w)
        rows = np.arange(h)
        col_grid, row_grid = np.meshgrid(cols, rows)

        # Horizontal angle for each column
        # Center of image is 0, left is negative, right is positive
        normalized_x = (col_grid - w / 2) / (w / 2)
        self._angle_lut = np.degrees(normalized_x * (h_fov / 2))

        # Vertical angle for each row (relative to camera optical axis)
        # Top of image is positive angle (looking up), bottom is negative
        normalized_y = (h / 2 - row_grid) / (h / 2)
        vertical_angle = normalized_y * (v_fov / 2)

        # Store for height calculation (combine with tilt at runtime)
        self._vertical_angle_lut = vertical_angle - tilt

    def compute_heights(
        self,
        depth_map: np.ndarray
    ) -> np.ndarray:
        """
        Compute height of each pixel above ground.

        Args:
            depth_map: Depth values in mm

        Returns:
            Height map in mm (relative to ground)
        """
        # Height = camera_height + depth * sin(vertical_angle)
        # (positive = above camera level, negative = below)
        heights = (
            self._config.camera_height_mm +
            depth_map * np.sin(self._vertical_angle_lut)
        )
        return heights

    def convert(
        self,
        depth_map: np.ndarray,
        min_height_mm: float = 50.0,
        max_height_mm: float = 500.0,
        min_range_mm: float = 200.0,
        max_range_mm: float = 3000.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth map to polar coordinates with height filtering.

        Args:
            depth_map: Depth values in mm
            min_height_mm: Minimum height to consider (filter floor)
            max_height_mm: Maximum height to consider
            min_range_mm: Minimum range to consider
            max_range_mm: Maximum range to consider

        Returns:
            Tuple of (angles, distances) for valid obstacle points
        """
        # Compute heights
        heights = self.compute_heights(depth_map)

        # Create validity mask
        valid = (
            (depth_map > min_range_mm) &
            (depth_map < max_range_mm) &
            (heights > min_height_mm) &
            (heights < max_height_mm)
        )

        # Extract valid points
        angles = self._angle_lut[valid]
        distances = depth_map[valid]

        return (angles, distances)

    def compute_sector_histogram(
        self,
        depth_map: np.ndarray,
        num_sectors: int,
        min_height_mm: float = 50.0,
        max_height_mm: float = 500.0,
        min_range_mm: float = 200.0,
        max_range_mm: float = 3000.0
    ) -> np.ndarray:
        """
        Compute obstacle density histogram directly from depth map.

        Optimized path combining conversion and histogram creation.

        Args:
            depth_map: Depth values in mm
            num_sectors: Number of angular sectors
            min_height_mm: Minimum height to consider
            max_height_mm: Maximum height to consider
            min_range_mm: Minimum range to consider
            max_range_mm: Maximum range to consider

        Returns:
            Array of obstacle densities per sector (0-1 range)
        """
        # Sector width in degrees
        h_fov = self._config.horizontal_fov_deg
        sector_width = h_fov / num_sectors

        # Find which sector each column maps to
        # Columns at center map to center sector
        center_sector = num_sectors // 2
        angles = self._angle_lut[0, :]  # Same for all rows
        sector_indices = (angles / sector_width + center_sector).astype(int)
        sector_indices = np.clip(sector_indices, 0, num_sectors - 1)

        # Compute heights
        heights = self.compute_heights(depth_map)

        # Create validity mask
        valid = (
            (depth_map > min_range_mm) &
            (depth_map < max_range_mm) &
            (heights > min_height_mm) &
            (heights < max_height_mm)
        )

        # Initialize histogram
        histogram = np.zeros(num_sectors)
        counts = np.zeros(num_sectors)

        # Accumulate obstacle density per sector
        # Weight by inverse distance (closer = more important)
        for col in range(depth_map.shape[1]):
            sector = sector_indices[col]
            col_valid = valid[:, col]

            if np.any(col_valid):
                col_depths = depth_map[:, col][col_valid]
                # Weight by inverse distance
                weights = 1.0 / (col_depths / 1000.0)  # Normalize to meters
                histogram[sector] += np.sum(weights)
                counts[sector] += len(col_depths)

        # Normalize by number of valid points per sector
        # This gives a density measure
        max_points = depth_map.shape[0]  # Max possible points per column
        cols_per_sector = depth_map.shape[1] / num_sectors

        normalization = max_points * cols_per_sector
        if normalization > 0:
            histogram = histogram / (normalization * 0.1)  # Scale factor

        # Clip to 0-1 range
        histogram = np.clip(histogram, 0.0, 1.0)

        return histogram

    def compute_sector_min_distances(
        self,
        depth_map: np.ndarray,
        num_sectors: int,
        min_height_mm: float = 50.0,
        max_height_mm: float = 500.0,
        min_range_mm: float = 200.0,
        max_range_mm: float = 3000.0
    ) -> np.ndarray:
        """
        Compute minimum distance to obstacle in each sector.

        Args:
            depth_map: Depth values in mm
            num_sectors: Number of angular sectors
            min_height_mm: Minimum height to consider
            max_height_mm: Maximum height to consider
            min_range_mm: Minimum range to consider
            max_range_mm: Maximum range to consider

        Returns:
            Array of minimum distances per sector (max_range if no obstacle)
        """
        # Sector width in degrees
        h_fov = self._config.horizontal_fov_deg
        sector_width = h_fov / num_sectors

        # Find which sector each column maps to
        center_sector = num_sectors // 2
        angles = self._angle_lut[0, :]
        sector_indices = (angles / sector_width + center_sector).astype(int)
        sector_indices = np.clip(sector_indices, 0, num_sectors - 1)

        # Compute heights
        heights = self.compute_heights(depth_map)

        # Create validity mask
        valid = (
            (depth_map > min_range_mm) &
            (depth_map < max_range_mm) &
            (heights > min_height_mm) &
            (heights < max_height_mm)
        )

        # Initialize with max range
        min_distances = np.full(num_sectors, max_range_mm)

        # Find minimum distance in each sector
        for col in range(depth_map.shape[1]):
            sector = sector_indices[col]
            col_valid = valid[:, col]

            if np.any(col_valid):
                col_depths = depth_map[:, col][col_valid]
                col_min = np.min(col_depths)
                min_distances[sector] = min(min_distances[sector], col_min)

        return min_distances

    def get_sector_angles(self, num_sectors: int) -> np.ndarray:
        """
        Get the center angle for each sector.

        Args:
            num_sectors: Number of sectors

        Returns:
            Array of sector center angles in degrees
        """
        h_fov = self._config.horizontal_fov_deg
        sector_width = h_fov / num_sectors
        center_sector = num_sectors // 2

        angles = np.arange(num_sectors)
        angles = (angles - center_sector) * sector_width + sector_width / 2

        return angles
