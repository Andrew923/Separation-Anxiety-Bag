"""
Follow-the-Gap path planning algorithm.

Finds the largest gap between obstacles and selects the best one
based on proximity to target, width, and depth.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .path_planner import PathPlanner, PlannerResult


@dataclass
class Gap:
    """Represents a traversable gap between obstacles."""
    start_idx: int          # Starting sector index
    end_idx: int            # Ending sector index
    start_angle_deg: float  # Starting angle
    end_angle_deg: float    # Ending angle
    center_angle_deg: float # Center of gap
    width_deg: float        # Angular width
    min_depth_mm: float     # Minimum depth within gap
    avg_depth_mm: float     # Average depth within gap


@dataclass
class FollowGapConfig:
    """Configuration for Follow-the-Gap algorithm."""
    min_gap_width_deg: float = 15.0     # Minimum gap width in degrees
    robot_width_mm: float = 300.0        # Robot width for clearance check
    safety_margin_mm: float = 100.0      # Extra clearance on each side
    max_range_mm: float = 3000.0         # Max detection range
    min_range_mm: float = 200.0          # Min detection range

    # Gap scoring weights
    target_weight: float = 0.5           # Weight for proximity to target
    width_weight: float = 0.3            # Weight for gap width
    depth_weight: float = 0.2            # Weight for gap depth (clearance)


class FollowTheGap(PathPlanner):
    """
    Follow-the-Gap obstacle avoidance algorithm.

    Algorithm steps:
    1. Find all gaps (continuous sectors with clearance > threshold)
    2. Filter gaps that are wide enough for robot to fit
    3. Score each gap based on target proximity, width, and depth
    4. Return center of best gap
    """

    def __init__(self, config: FollowGapConfig):
        """
        Initialize Follow-the-Gap algorithm.

        Args:
            config: Algorithm configuration
        """
        self._config = config

    @property
    def name(self) -> str:
        return "follow_gap"

    @property
    def config(self) -> FollowGapConfig:
        """Get current configuration."""
        return self._config

    def compute(
        self,
        distances: np.ndarray,
        sector_angles: np.ndarray,
        target_heading_deg: float = 0.0
    ) -> PlannerResult:
        """Compute best heading using Follow-the-Gap algorithm."""
        config = self._config

        # Step 1: Find all gaps
        gaps = self._find_gaps(distances, sector_angles)

        # Step 2: Filter gaps that robot can fit through
        passable_gaps = self._filter_passable_gaps(gaps, distances)

        if not passable_gaps:
            return PlannerResult(
                best_heading_deg=None,
                can_proceed=False,
                debug_info={
                    'gaps_found': len(gaps),
                    'passable_gaps': 0,
                    'reason': 'no_passable_gaps'
                }
            )

        # Step 3: Score and select best gap
        best_gap = self._select_best_gap(passable_gaps, target_heading_deg)

        return PlannerResult(
            best_heading_deg=best_gap.center_angle_deg,
            can_proceed=True,
            debug_info={
                'gaps_found': len(gaps),
                'passable_gaps': len(passable_gaps),
                'selected_gap_width_deg': best_gap.width_deg,
                'selected_gap_depth_mm': best_gap.avg_depth_mm,
                'selected_gap_center_deg': best_gap.center_angle_deg
            }
        )

    def _find_gaps(
        self,
        distances: np.ndarray,
        sector_angles: np.ndarray
    ) -> List[Gap]:
        """
        Find all gaps in the distance data.

        A gap is a continuous sequence of sectors where the distance
        exceeds the threshold (indicating no close obstacle).
        """
        config = self._config
        n = len(distances)
        gaps = []

        # Threshold for considering a sector "open"
        open_threshold = config.min_range_mm + config.safety_margin_mm

        # Boolean mask of open sectors
        is_open = distances > open_threshold

        # Find continuous runs of open sectors
        in_gap = False
        gap_start = 0

        for i in range(n):
            if is_open[i] and not in_gap:
                # Start of new gap
                gap_start = i
                in_gap = True
            elif not is_open[i] and in_gap:
                # End of gap
                gap = self._create_gap(
                    gap_start, i - 1, distances, sector_angles
                )
                if gap is not None:
                    gaps.append(gap)
                in_gap = False

        # Handle gap that extends to end
        if in_gap:
            gap = self._create_gap(
                gap_start, n - 1, distances, sector_angles
            )
            if gap is not None:
                gaps.append(gap)

        return gaps

    def _create_gap(
        self,
        start_idx: int,
        end_idx: int,
        distances: np.ndarray,
        sector_angles: np.ndarray
    ) -> Optional[Gap]:
        """Create a Gap object from sector indices."""
        if end_idx < start_idx:
            return None

        gap_distances = distances[start_idx:end_idx + 1]

        # Calculate width accounting for sector angle spacing
        width_deg = abs(sector_angles[end_idx] - sector_angles[start_idx])
        if len(sector_angles) > 1:
            # Add one sector width since we measure edge-to-edge
            sector_spacing = abs(sector_angles[1] - sector_angles[0])
            width_deg += sector_spacing

        return Gap(
            start_idx=start_idx,
            end_idx=end_idx,
            start_angle_deg=float(sector_angles[start_idx]),
            end_angle_deg=float(sector_angles[end_idx]),
            center_angle_deg=float((sector_angles[start_idx] + sector_angles[end_idx]) / 2),
            width_deg=width_deg,
            min_depth_mm=float(np.min(gap_distances)),
            avg_depth_mm=float(np.mean(gap_distances))
        )

    def _filter_passable_gaps(
        self,
        gaps: List[Gap],
        distances: np.ndarray
    ) -> List[Gap]:
        """Filter gaps that the robot can physically fit through."""
        config = self._config
        passable = []

        for gap in gaps:
            # Check if gap is wide enough angularly
            if gap.width_deg < config.min_gap_width_deg:
                continue

            # Check if gap depth provides enough clearance
            # At the gap's depth, the angular width translates to physical width
            # physical_width ≈ 2 * depth * tan(width_deg / 2)
            # For small angles: physical_width ≈ depth * width_rad
            import math
            width_rad = math.radians(gap.width_deg)
            physical_width_mm = gap.min_depth_mm * width_rad

            # Robot needs: robot_width + safety_margin on each side
            required_width = config.robot_width_mm + 2 * config.safety_margin_mm

            if physical_width_mm >= required_width:
                passable.append(gap)

        return passable

    def _select_best_gap(
        self,
        gaps: List[Gap],
        target_heading_deg: float
    ) -> Gap:
        """
        Select best gap based on weighted scoring.

        Score = target_weight * target_proximity + width_weight * width + depth_weight * depth
        """
        config = self._config
        best_gap = None
        best_score = float('-inf')

        # Normalize factors
        max_width = max(g.width_deg for g in gaps) if gaps else 1.0
        max_depth = max(g.avg_depth_mm for g in gaps) if gaps else 1.0

        # Prevent division by zero
        max_width = max(max_width, 1.0)
        max_depth = max(max_depth, 1.0)

        for gap in gaps:
            # Target proximity score (0 to 1, higher is better)
            # 0 degrees difference = 1.0, 180 degrees = 0.0
            angle_diff = abs(gap.center_angle_deg - target_heading_deg)
            # Handle wraparound
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            target_score = 1.0 - (angle_diff / 180.0)

            # Width score (normalized, higher is better)
            width_score = gap.width_deg / max_width

            # Depth score (normalized, higher is better)
            depth_score = gap.avg_depth_mm / max_depth

            # Combined score
            score = (
                config.target_weight * target_score +
                config.width_weight * width_score +
                config.depth_weight * depth_score
            )

            if score > best_score:
                best_score = score
                best_gap = gap

        return best_gap

    def get_gaps(
        self,
        distances: np.ndarray,
        sector_angles: np.ndarray
    ) -> List[Gap]:
        """
        Get all detected gaps (for visualization/debugging).

        Args:
            distances: 1D array of min distances per sector (mm)
            sector_angles: Center angle of each sector (degrees)

        Returns:
            List of all gaps found
        """
        return self._find_gaps(distances, sector_angles)

    def get_passable_gaps(
        self,
        distances: np.ndarray,
        sector_angles: np.ndarray
    ) -> List[Gap]:
        """
        Get only passable gaps (for visualization/debugging).

        Args:
            distances: 1D array of min distances per sector (mm)
            sector_angles: Center angle of each sector (degrees)

        Returns:
            List of gaps robot can fit through
        """
        gaps = self._find_gaps(distances, sector_angles)
        return self._filter_passable_gaps(gaps, distances)
