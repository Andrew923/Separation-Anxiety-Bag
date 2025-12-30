"""
Robot utility functions.
"""

from typing import Dict, Any
from pathlib import Path
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_default_config_path(config_name: str = 'robot_config.yaml') -> str:
    """
    Get path to default configuration file.

    Args:
        config_name: Name of config file

    Returns:
        Full path to config file
    """
    return str(Path(__file__).parent.parent / 'config' / config_name)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value to range.

    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def normalize_angle(angle_deg: float) -> float:
    """
    Normalize angle to [-180, 180] range.

    Args:
        angle_deg: Angle in degrees

    Returns:
        Normalized angle
    """
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg < -180:
        angle_deg += 360
    return angle_deg


def mm_to_m(mm: float) -> float:
    """Convert millimeters to meters."""
    return mm / 1000.0


def m_to_mm(m: float) -> float:
    """Convert meters to millimeters."""
    return m * 1000.0
