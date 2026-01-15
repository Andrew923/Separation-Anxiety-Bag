"""
Artificial Potential Fields (APF) path planning algorithm.

Uses attractive force from target and repulsive forces from obstacles
to compute a resultant heading direction.
"""

from dataclasses import dataclass
from typing import Tuple
import math

import numpy as np

from .path_planner import PathPlanner, PlannerResult


@dataclass
class APFConfig:
    """Configuration for Artificial Potential Fields algorithm."""
    # Attractive force parameters
    attractive_gain: float = 1.0         # Strength of attraction to target

    # Repulsive force parameters
    repulsive_gain: float = 500.0        # Strength of obstacle repulsion
    repulsion_radius_mm: float = 1500.0  # Distance at which repulsion starts
    min_range_mm: float = 200.0          # Minimum sensing range
    max_range_mm: float = 3000.0         # Maximum sensing range

    # Falloff type: "linear", "quadratic", "exponential"
    repulsion_falloff: str = "quadratic"

    # Safety thresholds
    emergency_stop_distance_mm: float = 150.0  # Stop if any obstacle closer
    min_resultant_magnitude: float = 0.1       # Threshold to consider stuck


class ArtificialPotentialFields(PathPlanner):
    """
    Artificial Potential Fields obstacle avoidance.

    Algorithm:
    1. Compute attractive force vector toward target
    2. For each obstacle sector, compute repulsive force vector away from obstacle
    3. Sum all forces to get resultant vector
    4. Return direction of resultant vector as heading

    Coordinate system:
    - x positive = right
    - y positive = forward
    - angle 0 = forward, positive = clockwise (right)
    """

    def __init__(self, config: APFConfig):
        """
        Initialize APF algorithm.

        Args:
            config: Algorithm configuration
        """
        self._config = config

    @property
    def name(self) -> str:
        return "apf"

    @property
    def config(self) -> APFConfig:
        """Get current configuration."""
        return self._config

    def compute(
        self,
        distances: np.ndarray,
        sector_angles: np.ndarray,
        target_heading_deg: float = 0.0
    ) -> PlannerResult:
        """Compute best heading using APF algorithm."""
        config = self._config

        # Check for emergency stop condition
        min_distance = float(np.min(distances))
        if min_distance < config.emergency_stop_distance_mm:
            return PlannerResult(
                best_heading_deg=None,
                can_proceed=False,
                debug_info={
                    'min_distance_mm': min_distance,
                    'emergency_stop': True,
                    'reason': 'emergency_stop'
                }
            )

        # Step 1: Compute attractive force toward target
        f_att = self._compute_attractive_force(target_heading_deg)

        # Step 2: Compute repulsive forces from all obstacles
        f_rep = self._compute_repulsive_forces(distances, sector_angles)

        # Step 3: Sum forces
        f_total = (f_att[0] + f_rep[0], f_att[1] + f_rep[1])

        # Step 4: Convert resultant to heading
        magnitude = math.sqrt(f_total[0]**2 + f_total[1]**2)

        if magnitude < config.min_resultant_magnitude:
            # Forces cancel out - stuck in local minimum
            return PlannerResult(
                best_heading_deg=None,
                can_proceed=False,
                debug_info={
                    'resultant_magnitude': magnitude,
                    'local_minimum': True,
                    'reason': 'local_minimum',
                    'f_attractive': f_att,
                    'f_repulsive': f_rep,
                    'f_total': f_total
                }
            )

        # Convert (x, y) to heading angle
        # x = right, y = forward
        # atan2(x, y) gives angle from forward (0°), positive clockwise
        heading_rad = math.atan2(f_total[0], f_total[1])
        heading_deg = math.degrees(heading_rad)

        return PlannerResult(
            best_heading_deg=heading_deg,
            can_proceed=True,
            debug_info={
                'resultant_magnitude': magnitude,
                'f_attractive': f_att,
                'f_repulsive': f_rep,
                'f_total': f_total,
                'min_distance_mm': min_distance
            }
        )

    def _compute_attractive_force(
        self,
        target_heading_deg: float
    ) -> Tuple[float, float]:
        """
        Compute attractive force toward target.

        Returns (fx, fy) force vector where:
        - x positive = right
        - y positive = forward
        """
        config = self._config

        # Convert heading to radians (0 = forward, positive = clockwise)
        theta = math.radians(target_heading_deg)

        # Unit vector toward target, scaled by gain
        # sin(theta) = x component (right)
        # cos(theta) = y component (forward)
        fx = config.attractive_gain * math.sin(theta)
        fy = config.attractive_gain * math.cos(theta)

        return (fx, fy)

    def _compute_repulsive_forces(
        self,
        distances: np.ndarray,
        sector_angles: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute sum of repulsive forces from all obstacles.

        Each obstacle pushes the robot away with force inversely
        proportional to distance (based on falloff type).
        """
        config = self._config
        fx_total = 0.0
        fy_total = 0.0

        for dist, angle_deg in zip(distances, sector_angles):
            # Skip if obstacle is too far to affect
            if dist >= config.repulsion_radius_mm:
                continue

            # Skip if distance indicates no obstacle (at max range)
            if dist >= config.max_range_mm:
                continue

            # Compute repulsion magnitude based on falloff type
            magnitude = self._compute_repulsion_magnitude(float(dist))

            # Direction: away from obstacle (opposite of obstacle direction)
            theta = math.radians(float(angle_deg))

            # Force pushes AWAY from obstacle (negative of obstacle direction)
            # Obstacle at angle theta -> repulsion in direction (theta + 180°)
            fx_total -= magnitude * math.sin(theta)
            fy_total -= magnitude * math.cos(theta)

        return (fx_total, fy_total)

    def _compute_repulsion_magnitude(self, distance: float) -> float:
        """
        Compute repulsion force magnitude based on distance.

        Closer obstacles produce stronger repulsion.
        """
        config = self._config

        # Clamp distance to valid range
        distance = max(config.min_range_mm, min(distance, config.repulsion_radius_mm))

        # Normalize distance to [0, 1] range
        # 0 = at repulsion_radius (no repulsion)
        # 1 = at min_range (max repulsion)
        normalized = 1.0 - (distance - config.min_range_mm) / (
            config.repulsion_radius_mm - config.min_range_mm
        )
        normalized = max(0.0, min(1.0, normalized))

        if config.repulsion_falloff == "linear":
            return config.repulsive_gain * normalized

        elif config.repulsion_falloff == "quadratic":
            return config.repulsive_gain * (normalized ** 2)

        elif config.repulsion_falloff == "exponential":
            # Exponential: grows faster as obstacle gets closer
            return config.repulsive_gain * (math.exp(normalized) - 1) / (math.e - 1)

        else:
            # Default to quadratic
            return config.repulsive_gain * (normalized ** 2)

    def get_force_components(
        self,
        distances: np.ndarray,
        sector_angles: np.ndarray,
        target_heading_deg: float = 0.0
    ) -> dict:
        """
        Get detailed force breakdown (for visualization/debugging).

        Args:
            distances: 1D array of min distances per sector (mm)
            sector_angles: Center angle of each sector (degrees)
            target_heading_deg: Target direction

        Returns:
            Dictionary with force components:
            - attractive: (fx, fy) attractive force
            - repulsive: (fx, fy) total repulsive force
            - total: (fx, fy) resultant force
            - per_sector: list of (angle, magnitude) for each repulsive force
        """
        f_att = self._compute_attractive_force(target_heading_deg)
        f_rep = self._compute_repulsive_forces(distances, sector_angles)
        f_total = (f_att[0] + f_rep[0], f_att[1] + f_rep[1])

        # Per-sector repulsion for visualization
        config = self._config
        per_sector = []
        for dist, angle_deg in zip(distances, sector_angles):
            if dist < config.repulsion_radius_mm and dist < config.max_range_mm:
                magnitude = self._compute_repulsion_magnitude(float(dist))
                per_sector.append((float(angle_deg), magnitude))

        return {
            'attractive': f_att,
            'repulsive': f_rep,
            'total': f_total,
            'per_sector': per_sector
        }
