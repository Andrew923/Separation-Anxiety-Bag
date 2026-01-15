"""
Factory for creating path planning algorithms from configuration.
"""

from typing import Dict, Any

from .path_planner import PathPlanner
from .follow_gap import FollowTheGap, FollowGapConfig
from .apf import ArtificialPotentialFields, APFConfig


def create_path_planner(config: Dict[str, Any]) -> PathPlanner:
    """
    Create a path planner from YAML configuration.

    Args:
        config: Dictionary from robot_config.yaml containing:
            - path_planning.algorithm: "follow_gap" or "apf"
            - follow_gap: Follow-the-Gap config (if algorithm == "follow_gap")
            - apf: APF-specific config (if algorithm == "apf")
            - robot.robot_width_mm: Robot width (used by follow_gap)

    Returns:
        PathPlanner instance

    Raises:
        ValueError: If unknown algorithm specified
    """
    path_planning = config.get('path_planning', {})
    algorithm = path_planning.get('algorithm', 'follow_gap')

    if algorithm == 'follow_gap':
        return _create_follow_gap(config)

    elif algorithm == 'apf':
        return _create_apf(config)

    else:
        raise ValueError(
            f"Unknown path planning algorithm: {algorithm}. "
            f"Valid options: follow_gap, apf"
        )


def _create_follow_gap(config: Dict[str, Any]) -> FollowTheGap:
    """Create Follow-the-Gap planner from config."""
    fg_cfg = config.get('follow_gap', {})
    robot_cfg = config.get('robot', {})
    depth_cfg = config.get('depth_preprocessing', {})

    follow_gap_config = FollowGapConfig(
        min_gap_width_deg=fg_cfg.get('min_gap_width_deg', 15.0),
        robot_width_mm=robot_cfg.get('robot_width_mm', 300.0),
        safety_margin_mm=fg_cfg.get('safety_margin_mm', 100.0),
        max_range_mm=fg_cfg.get('max_range_mm', depth_cfg.get('max_range_mm', 3000.0)),
        min_range_mm=fg_cfg.get('min_range_mm', depth_cfg.get('min_range_mm', 200.0)),
        target_weight=fg_cfg.get('target_weight', 0.5),
        width_weight=fg_cfg.get('width_weight', 0.3),
        depth_weight=fg_cfg.get('depth_weight', 0.2)
    )

    return FollowTheGap(follow_gap_config)


def _create_apf(config: Dict[str, Any]) -> ArtificialPotentialFields:
    """Create APF planner from config."""
    apf_cfg = config.get('apf', {})
    depth_cfg = config.get('depth_preprocessing', {})

    apf_config = APFConfig(
        attractive_gain=apf_cfg.get('attractive_gain', 1.0),
        repulsive_gain=apf_cfg.get('repulsive_gain', 500.0),
        repulsion_radius_mm=apf_cfg.get('repulsion_radius_mm', 1500.0),
        min_range_mm=apf_cfg.get('min_range_mm', depth_cfg.get('min_range_mm', 200.0)),
        max_range_mm=apf_cfg.get('max_range_mm', depth_cfg.get('max_range_mm', 3000.0)),
        repulsion_falloff=apf_cfg.get('repulsion_falloff', 'quadratic'),
        emergency_stop_distance_mm=apf_cfg.get('emergency_stop_distance_mm', 150.0),
        min_resultant_magnitude=apf_cfg.get('min_resultant_magnitude', 0.1)
    )

    return ArtificialPotentialFields(apf_config)


def get_available_algorithms() -> list:
    """Get list of available path planning algorithms."""
    return ['follow_gap', 'apf']
