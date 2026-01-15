"""
Factory for creating path planning algorithms from configuration.
"""

from typing import Dict, Any

from .path_planner import PathPlanner
from .vfh import VFHConfig
from .vfh_adapter import VFHAdapter
from .follow_gap import FollowTheGap, FollowGapConfig
from .apf import ArtificialPotentialFields, APFConfig


def create_path_planner(config: Dict[str, Any]) -> PathPlanner:
    """
    Create a path planner from YAML configuration.

    Args:
        config: Dictionary from robot_config.yaml containing:
            - path_planning.algorithm: "vfh", "follow_gap", or "apf"
            - vfh: VFH-specific config (if algorithm == "vfh")
            - follow_gap: Follow-the-Gap config (if algorithm == "follow_gap")
            - apf: APF-specific config (if algorithm == "apf")
            - robot.robot_width_mm: Robot width (used by follow_gap)
            - camera.horizontal_fov_deg: Camera FOV (used by vfh)

    Returns:
        PathPlanner instance

    Raises:
        ValueError: If unknown algorithm specified
    """
    path_planning = config.get('path_planning', {})
    algorithm = path_planning.get('algorithm', 'vfh')

    # Get camera FOV for VFH
    camera_cfg = config.get('camera', {})
    camera_fov_deg = camera_cfg.get('horizontal_fov_deg', 60.0)

    if algorithm == 'vfh':
        return _create_vfh(config, camera_fov_deg)

    elif algorithm == 'follow_gap':
        return _create_follow_gap(config)

    elif algorithm == 'apf':
        return _create_apf(config)

    else:
        raise ValueError(
            f"Unknown path planning algorithm: {algorithm}. "
            f"Valid options: vfh, follow_gap, apf"
        )


def _create_vfh(config: Dict[str, Any], camera_fov_deg: float) -> VFHAdapter:
    """Create VFH adapter from config."""
    vfh_cfg = config.get('vfh', {})

    vfh_config = VFHConfig(
        num_sectors=vfh_cfg.get('num_sectors', 72),
        min_range_mm=vfh_cfg.get('min_range_mm', 200.0),
        max_range_mm=vfh_cfg.get('max_range_mm', 3000.0),
        min_height_mm=vfh_cfg.get('min_height_mm', 50.0),
        max_height_mm=vfh_cfg.get('max_height_mm', 500.0),
        obstacle_threshold=vfh_cfg.get('obstacle_threshold', 0.3),
        safety_margin_mm=vfh_cfg.get('safety_margin_mm', 150.0),
        wide_valley_threshold=vfh_cfg.get('wide_valley_threshold', 3),
        narrow_valley_threshold=vfh_cfg.get('narrow_valley_threshold', 1),
        smoothing_kernel_size=vfh_cfg.get('smoothing_kernel_size', 3)
    )

    return VFHAdapter(vfh_config, camera_fov_deg=camera_fov_deg)


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
    return ['vfh', 'follow_gap', 'apf']
