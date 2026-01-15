"""
Depth map preprocessing for obstacle detection.

Converts stereo depth maps to 1D distance arrays (virtual LIDAR scan)
with floor/ceiling filtering. Algorithm-agnostic output suitable for
VFH, Follow-the-Gap, Artificial Potential Fields, etc.
"""

from dataclasses import dataclass
from typing import Optional
import math

import numpy as np

from .floor_detection import (
    FloorDetector,
    HeightBasedFloorDetector,
    HeightBasedFloorConfig,
    AdaptiveFloorDetector,
    AdaptiveFloorConfig,
)


@dataclass
class DepthPreprocessorConfig:
    """Configuration for depth preprocessing pipeline."""

    # Camera geometry
    horizontal_fov_deg: float = 60.0
    vertical_fov_deg: float = 45.0
    image_width: int = 640
    image_height: int = 480
    camera_height_mm: float = 200.0
    camera_tilt_deg: float = 0.0

    # Range filtering
    min_range_mm: float = 200.0
    max_range_mm: float = 3000.0

    # Height filtering (used if no floor_detector provided)
    floor_threshold_mm: float = 50.0
    robot_height_mm: float = 500.0

    # Output configuration
    num_sectors: int = 72

    # Floor detection method: "height" or "adaptive"
    floor_detection_method: str = "height"


@dataclass
class PreprocessorResult:
    """Result of depth preprocessing."""

    distances: np.ndarray       # 1D array of min distances per sector (mm)
    sector_angles: np.ndarray   # Center angle of each sector (degrees)
    valid_sectors: np.ndarray   # Boolean mask of sectors with valid data
    floor_mask: np.ndarray      # 2D mask of floor/ceiling pixels (for viz)
    obstacle_mask: np.ndarray   # 2D mask of obstacle pixels (for viz)


class DepthPreprocessor:
    """
    Depth map preprocessing pipeline.

    Converts stereo depth map to 1D distance array (virtual LIDAR scan)
    with pluggable floor detection.

    Pipeline:
    1. Range filtering (min/max distance)
    2. Floor/ceiling detection (height-based or adaptive)
    3. Polar projection (pixel angle lookup)
    4. Sector aggregation (min distance per sector)
    """

    def __init__(
        self,
        config: DepthPreprocessorConfig,
        floor_detector: Optional[FloorDetector] = None
    ):
        """
        Initialize preprocessor.

        Args:
            config: Preprocessing configuration
            floor_detector: Floor detection strategy (created from config if None)
        """
        self._config = config
        self._angle_lut: Optional[np.ndarray] = None
        self._vertical_angle_lut: Optional[np.ndarray] = None
        self._sector_angles: Optional[np.ndarray] = None
        self._current_image_size: tuple = (0, 0)  # (height, width)

        # Create floor detector if not provided
        if floor_detector is None:
            self._floor_detector = self._create_floor_detector()
        else:
            self._floor_detector = floor_detector

        # LUTs will be built lazily on first process() call

    def _create_floor_detector(self) -> FloorDetector:
        """Create floor detector based on config."""
        method = self._config.floor_detection_method

        if method == "adaptive":
            return AdaptiveFloorDetector(AdaptiveFloorConfig(
                base_threshold_mm=self._config.floor_threshold_mm,
                robot_height_mm=self._config.robot_height_mm,
            ))
        else:  # Default to "height"
            return HeightBasedFloorDetector(HeightBasedFloorConfig(
                floor_threshold_mm=self._config.floor_threshold_mm,
                robot_height_mm=self._config.robot_height_mm,
            ))

    def _precompute_lookup_tables(self, height: int, width: int) -> None:
        """
        Precompute pixel-to-angle and sector mappings for given image size.

        Args:
            height: Image height in pixels
            width: Image width in pixels
        """
        h_fov = math.radians(self._config.horizontal_fov_deg)
        v_fov = math.radians(self._config.vertical_fov_deg)
        tilt = math.radians(self._config.camera_tilt_deg)

        # Create meshgrid of pixel coordinates
        cols = np.arange(width)
        rows = np.arange(height)
        col_grid, row_grid = np.meshgrid(cols, rows)

        # Horizontal angle for each column
        # Center of image is 0, left is negative, right is positive
        normalized_x = (col_grid - width / 2) / (width / 2)
        self._angle_lut = np.degrees(normalized_x * (h_fov / 2))

        # Vertical angle for each row (for height calculation)
        normalized_y = (height / 2 - row_grid) / (height / 2)
        vertical_angle = normalized_y * (v_fov / 2)
        self._vertical_angle_lut = vertical_angle - tilt

        # Precompute sector angles (center of each sector)
        n = self._config.num_sectors
        fov = self._config.horizontal_fov_deg
        sector_width = fov / n
        self._sector_angles = np.linspace(
            -fov / 2 + sector_width / 2,
            fov / 2 - sector_width / 2,
            n
        )

        # Precompute column-to-sector mapping
        col_angles = self._angle_lut[0, :]  # Same for all rows
        sector_indices = ((col_angles + fov / 2) / sector_width).astype(int)
        self._col_to_sector = np.clip(sector_indices, 0, n - 1)

        # Store current size
        self._current_image_size = (height, width)

    def _ensure_lut_size(self, depth_map: np.ndarray) -> None:
        """Rebuild LUTs if image size changed."""
        h, w = depth_map.shape[:2]
        if self._current_image_size != (h, w):
            self._precompute_lookup_tables(h, w)

    def compute_heights(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Compute height of each pixel above ground.

        Args:
            depth_map: Depth values in mm (H x W)

        Returns:
            Height map in mm (relative to ground)
        """
        # Ensure LUT matches image size
        self._ensure_lut_size(depth_map)

        heights = (
            self._config.camera_height_mm +
            depth_map * np.sin(self._vertical_angle_lut)
        )
        return heights

    def process(self, depth_map: np.ndarray) -> PreprocessorResult:
        """
        Process depth map to 1D distance array with debug info.

        Args:
            depth_map: Depth values in mm (H x W)

        Returns:
            PreprocessorResult with distances and visualization masks
        """
        # Ensure LUT matches image size
        self._ensure_lut_size(depth_map)

        config = self._config
        n_sectors = config.num_sectors

        # Step 1: Range filtering
        in_range = (
            (depth_map > config.min_range_mm) &
            (depth_map < config.max_range_mm)
        )

        # Step 2: Compute heights and detect floor/ceiling
        heights = self.compute_heights(depth_map)
        floor_mask = self._floor_detector.detect(depth_map, heights)

        # Step 3: Combine masks -> valid obstacle pixels
        obstacle_mask = in_range & ~floor_mask

        # Step 4: Compute min distance per sector
        distances = np.full(n_sectors, config.max_range_mm)
        valid_sectors = np.zeros(n_sectors, dtype=bool)

        # Process column by column for efficiency
        for col in range(depth_map.shape[1]):
            sector = self._col_to_sector[col]
            col_obstacle = obstacle_mask[:, col]

            if np.any(col_obstacle):
                col_depths = depth_map[:, col][col_obstacle]
                col_min = np.min(col_depths)
                if len(col_depths) > 5:
                    # Ignore the closest 10% of points (noise/floor sparkles)
                    # and take the next closest value.
                    col_min = np.percentile(col_depths, 10) 
                if col_min < distances[sector]:
                    distances[sector] = col_min
                valid_sectors[sector] = True

        return PreprocessorResult(
            distances=distances,
            sector_angles=self._sector_angles.copy(),
            valid_sectors=valid_sectors,
            floor_mask=floor_mask,
            obstacle_mask=obstacle_mask,
        )

    def get_distances(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Simple interface returning only the 1D distance array.

        For production use where debug info is not needed.

        Args:
            depth_map: Depth values in mm (H x W)

        Returns:
            1D array of min distances per sector (mm)
        """
        result = self.process(depth_map)
        return result.distances

    def get_sector_angles(self) -> np.ndarray:
        """
        Get center angle for each sector.

        Returns:
            Array of sector center angles in degrees
        """
        return self._sector_angles.copy()

    def set_floor_detector(self, detector: FloorDetector) -> None:
        """
        Swap floor detection strategy at runtime.

        Args:
            detector: New floor detection strategy
        """
        self._floor_detector = detector

    def get_floor_detector(self) -> FloorDetector:
        """Get current floor detector."""
        return self._floor_detector

    @property
    def config(self) -> DepthPreprocessorConfig:
        """Get current configuration."""
        return self._config

    @property
    def num_sectors(self) -> int:
        """Get number of output sectors."""
        return self._config.num_sectors

    @property
    def horizontal_fov_deg(self) -> float:
        """Get horizontal field of view."""
        return self._config.horizontal_fov_deg


# Backward compatibility alias
DepthToPolarConfig = DepthPreprocessorConfig
