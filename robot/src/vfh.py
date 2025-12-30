"""
Vector Field Histogram (VFH) obstacle avoidance.

Converts depth map obstacles to polar histogram and determines
safe traversal directions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from .depth_to_polar import DepthToPolar, DepthToPolarConfig


@dataclass
class VFHConfig:
    """VFH algorithm configuration."""
    num_sectors: int = 72           # 5-degree sectors (360/72 = 5)
    min_range_mm: float = 200.0     # Ignore closer (inside robot body)
    max_range_mm: float = 3000.0    # Max detection range
    min_height_mm: float = 50.0     # Ignore floor
    max_height_mm: float = 500.0    # Ignore above robot obstacles
    obstacle_threshold: float = 0.3  # Density to consider blocked
    safety_margin_mm: float = 150.0  # Extra clearance
    wide_valley_threshold: int = 3   # Sectors for "wide" valley
    narrow_valley_threshold: int = 1 # Minimum passable valley
    smoothing_kernel_size: int = 3   # Histogram smoothing


@dataclass
class VFHResult:
    """Result of VFH computation."""
    histogram: np.ndarray              # Obstacle density per sector
    safe_sectors: List[int]            # Indices of safe sectors
    blocked_sectors: List[int]         # Indices of blocked sectors
    valleys: List[Tuple[int, int, int]]  # List of (start, end, width)
    best_heading_deg: Optional[float]  # Best heading toward target
    best_sector: Optional[int]         # Best sector index
    can_proceed: bool                  # True if path exists


class VectorFieldHistogram:
    """
    VFH obstacle avoidance algorithm.

    Converts depth map obstacles to polar histogram and determines
    safe traversal directions.
    """

    def __init__(
        self,
        config: VFHConfig,
        depth_converter: Optional[DepthToPolar] = None
    ):
        """
        Initialize VFH with configuration.

        Args:
            config: VFH configuration
            depth_converter: Depth to polar converter (created if None)
        """
        self._config = config
        self._histogram = np.zeros(config.num_sectors)
        self._min_distances = np.full(config.num_sectors, config.max_range_mm)

        # Create depth converter if not provided
        if depth_converter is None:
            self._depth_converter = DepthToPolar(DepthToPolarConfig())
        else:
            self._depth_converter = depth_converter

        # Precompute sector angles
        self._sector_angles = self._compute_sector_angles()

        # Smoothing kernel
        k = config.smoothing_kernel_size
        self._smoothing_kernel = np.ones(k) / k

    def _compute_sector_angles(self) -> np.ndarray:
        """Compute center angle for each sector."""
        n = self._config.num_sectors
        # Sectors span the camera FOV, centered on forward (0 degrees)
        # With 72 sectors and 60 degree FOV, each sector is ~0.83 degrees
        # But we want sectors to cover full 360 for general VFH
        # For camera-based VFH, we only fill the visible portion
        sector_width = 360.0 / n
        angles = np.arange(n) * sector_width - 180.0 + sector_width / 2
        return angles

    def update_from_depth(
        self,
        depth_map: np.ndarray,
        camera_fov_h_deg: float = 60.0
    ) -> None:
        """
        Update histogram from depth map.

        Args:
            depth_map: Depth values in mm (height x width)
            camera_fov_h_deg: Horizontal field of view
        """
        config = self._config

        # Compute obstacle density histogram
        histogram = self._depth_converter.compute_sector_histogram(
            depth_map,
            num_sectors=config.num_sectors,
            min_height_mm=config.min_height_mm,
            max_height_mm=config.max_height_mm,
            min_range_mm=config.min_range_mm,
            max_range_mm=config.max_range_mm
        )

        # Compute minimum distances
        min_distances = self._depth_converter.compute_sector_min_distances(
            depth_map,
            num_sectors=config.num_sectors,
            min_height_mm=config.min_height_mm,
            max_height_mm=config.max_height_mm,
            min_range_mm=config.min_range_mm,
            max_range_mm=config.max_range_mm
        )

        # Map camera FOV sectors to full histogram
        # Camera sees only a portion of the full 360 degrees
        visible_sectors = int(camera_fov_h_deg / (360.0 / config.num_sectors))
        center_sector = config.num_sectors // 2

        # Reset histogram (areas outside camera view are unknown)
        self._histogram = np.zeros(config.num_sectors)
        self._min_distances = np.full(config.num_sectors, config.max_range_mm)

        # Fill visible portion
        start = center_sector - visible_sectors // 2
        end = center_sector + visible_sectors // 2

        if start >= 0 and end < config.num_sectors:
            visible_range = slice(start, end)
            source_range = slice(0, min(len(histogram), end - start))
            self._histogram[visible_range] = histogram[source_range]
            self._min_distances[visible_range] = min_distances[source_range]

        # Apply smoothing
        self._histogram = np.convolve(
            self._histogram,
            self._smoothing_kernel,
            mode='same'
        )

    def update_from_histogram(self, histogram: np.ndarray) -> None:
        """
        Update from pre-computed histogram.

        Args:
            histogram: Obstacle density per sector
        """
        if len(histogram) != self._config.num_sectors:
            raise ValueError(
                f"Histogram size {len(histogram)} doesn't match "
                f"config {self._config.num_sectors}"
            )
        self._histogram = histogram.copy()

        # Apply smoothing
        self._histogram = np.convolve(
            self._histogram,
            self._smoothing_kernel,
            mode='same'
        )

    def get_histogram(self) -> np.ndarray:
        """Get current obstacle histogram."""
        return self._histogram.copy()

    def get_min_distances(self) -> np.ndarray:
        """Get minimum distances per sector."""
        return self._min_distances.copy()

    def find_safe_direction(
        self,
        target_heading_deg: float = 0.0
    ) -> VFHResult:
        """
        Find safest direction toward target.

        Args:
            target_heading_deg: Desired heading (0 = forward)

        Returns:
            VFHResult with recommended heading
        """
        config = self._config

        # Threshold histogram to binary
        blocked = self._histogram > config.obstacle_threshold
        safe = ~blocked

        # Get lists of safe and blocked sectors
        safe_sectors = np.where(safe)[0].tolist()
        blocked_sectors = np.where(blocked)[0].tolist()

        # Find valleys (continuous safe sectors)
        valleys = self._find_valleys(safe)

        # Find target sector
        target_sector = self.angle_to_sector(target_heading_deg)

        # Select best valley
        best_sector = self._select_best_sector(
            valleys,
            safe_sectors,
            target_sector
        )

        # Convert to heading
        if best_sector is not None:
            best_heading_deg = self.sector_to_angle(best_sector)
            can_proceed = True
        else:
            best_heading_deg = None
            can_proceed = False

        return VFHResult(
            histogram=self._histogram.copy(),
            safe_sectors=safe_sectors,
            blocked_sectors=blocked_sectors,
            valleys=valleys,
            best_heading_deg=best_heading_deg,
            best_sector=best_sector,
            can_proceed=can_proceed
        )

    def _find_valleys(
        self,
        safe: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """
        Find continuous valleys (safe sectors) in histogram.

        Args:
            safe: Boolean array of safe sectors

        Returns:
            List of (start_sector, end_sector, width) tuples
        """
        valleys = []
        n = len(safe)
        in_valley = False
        valley_start = 0

        # Scan through sectors (with wraparound)
        for i in range(n + 1):
            idx = i % n
            is_safe = safe[idx]

            if is_safe and not in_valley:
                # Starting new valley
                valley_start = idx
                in_valley = True
            elif not is_safe and in_valley:
                # Ending valley
                valley_end = (idx - 1) % n

                # Calculate width (handling wraparound)
                if valley_end >= valley_start:
                    width = valley_end - valley_start + 1
                else:
                    width = (n - valley_start) + valley_end + 1

                # Only add if meets minimum threshold
                if width >= self._config.narrow_valley_threshold:
                    valleys.append((valley_start, valley_end, width))

                in_valley = False

        return valleys

    def _select_best_sector(
        self,
        valleys: List[Tuple[int, int, int]],
        safe_sectors: List[int],
        target_sector: int
    ) -> Optional[int]:
        """
        Select best sector toward target.

        Prefers:
        1. Target sector if safe
        2. Closest sector in wide valley near target
        3. Closest safe sector to target

        Args:
            valleys: List of valleys
            safe_sectors: List of safe sector indices
            target_sector: Target sector index

        Returns:
            Best sector index or None
        """
        if not safe_sectors:
            return None

        n = self._config.num_sectors
        wide_threshold = self._config.wide_valley_threshold

        # Check if target sector is in a wide valley
        for start, end, width in valleys:
            if width >= wide_threshold:
                if self._sector_in_range(target_sector, start, end, n):
                    return target_sector

        # Find closest safe sector to target
        best_sector = None
        best_distance = float('inf')

        for sector in safe_sectors:
            # Calculate angular distance (handling wraparound)
            dist = abs(sector - target_sector)
            dist = min(dist, n - dist)

            if dist < best_distance:
                best_distance = dist
                best_sector = sector

        return best_sector

    def _sector_in_range(
        self,
        sector: int,
        start: int,
        end: int,
        n: int
    ) -> bool:
        """Check if sector is within range [start, end] with wraparound."""
        if start <= end:
            return start <= sector <= end
        else:
            return sector >= start or sector <= end

    def sector_to_angle(self, sector: int) -> float:
        """
        Convert sector index to angle in degrees.

        Args:
            sector: Sector index

        Returns:
            Angle in degrees (0 = forward)
        """
        return self._sector_angles[sector]

    def angle_to_sector(self, angle_deg: float) -> int:
        """
        Convert angle to sector index.

        Args:
            angle_deg: Angle in degrees

        Returns:
            Sector index
        """
        n = self._config.num_sectors
        sector_width = 360.0 / n

        # Normalize angle to [-180, 180]
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg < -180:
            angle_deg += 360

        # Convert to sector
        sector = int((angle_deg + 180) / sector_width)
        sector = sector % n

        return sector

    def is_direction_safe(self, angle_deg: float) -> bool:
        """
        Check if a specific direction is safe.

        Args:
            angle_deg: Direction to check

        Returns:
            True if direction is safe
        """
        sector = self.angle_to_sector(angle_deg)
        return self._histogram[sector] <= self._config.obstacle_threshold

    def get_clearance(self, angle_deg: float) -> float:
        """
        Get minimum distance to obstacle in given direction.

        Args:
            angle_deg: Direction to check

        Returns:
            Distance to nearest obstacle in mm
        """
        sector = self.angle_to_sector(angle_deg)
        return self._min_distances[sector]
