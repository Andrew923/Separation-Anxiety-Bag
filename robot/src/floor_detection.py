"""
Floor detection strategies for depth preprocessing.

Pluggable floor detection to filter out ground plane from depth maps.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


class FloorDetector(ABC):
    """Abstract base class for floor detection strategies."""

    @abstractmethod
    def detect(
        self,
        depth_map: np.ndarray,
        heights: np.ndarray
    ) -> np.ndarray:
        """
        Detect floor pixels in depth map.

        Args:
            depth_map: Depth values in mm (H x W)
            heights: Precomputed height above ground for each pixel (H x W)

        Returns:
            Boolean mask where True = floor pixel (should be ignored)
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return current tunable parameters."""
        pass

    @abstractmethod
    def set_parameters(self, **kwargs) -> None:
        """Update tunable parameters."""
        pass


@dataclass
class HeightBasedFloorConfig:
    """Configuration for height-based floor detection."""
    floor_threshold_mm: float = 50.0    # Ignore points below this height (floor)
    robot_height_mm: float = 500.0      # Ignore points above this height (ceiling)


class HeightBasedFloorDetector(FloorDetector):
    """
    Simple height-based floor detection.

    Filters pixels based on their computed height above ground:
    - Below floor_threshold_mm -> floor (ignored)
    - Above robot_height_mm -> ceiling/over robot (ignored)
    - In between -> obstacle (kept)

    Good for flat floors with known camera mounting.
    """

    def __init__(self, config: HeightBasedFloorConfig = None):
        """
        Initialize detector with configuration.

        Args:
            config: Detection configuration (uses defaults if None)
        """
        self._config = config or HeightBasedFloorConfig()

    def detect(
        self,
        depth_map: np.ndarray,
        heights: np.ndarray
    ) -> np.ndarray:
        """
        Detect floor and ceiling pixels.

        Args:
            depth_map: Depth values in mm (H x W)
            heights: Height above ground for each pixel (H x W)

        Returns:
            Boolean mask where True = should be ignored (floor or ceiling)
        """
        floor_mask = heights < self._config.floor_threshold_mm
        ceiling_mask = heights > self._config.robot_height_mm

        # Combine: ignore floor and ceiling
        ignore_mask = floor_mask | ceiling_mask

        return ignore_mask

    def get_parameters(self) -> Dict[str, Any]:
        """Return current tunable parameters."""
        return {
            'floor_threshold_mm': self._config.floor_threshold_mm,
            'robot_height_mm': self._config.robot_height_mm,
        }

    def set_parameters(self, **kwargs) -> None:
        """
        Update tunable parameters.

        Args:
            floor_threshold_mm: New floor threshold (optional)
            robot_height_mm: New robot height cutoff (optional)
        """
        if 'floor_threshold_mm' in kwargs:
            self._config.floor_threshold_mm = float(kwargs['floor_threshold_mm'])
        if 'robot_height_mm' in kwargs:
            self._config.robot_height_mm = float(kwargs['robot_height_mm'])

    @property
    def floor_threshold_mm(self) -> float:
        """Get floor threshold."""
        return self._config.floor_threshold_mm

    @floor_threshold_mm.setter
    def floor_threshold_mm(self, value: float) -> None:
        """Set floor threshold."""
        self._config.floor_threshold_mm = value

    @property
    def robot_height_mm(self) -> float:
        """Get robot height cutoff."""
        return self._config.robot_height_mm

    @robot_height_mm.setter
    def robot_height_mm(self, value: float) -> None:
        """Set robot height cutoff."""
        self._config.robot_height_mm = value


@dataclass
class AdaptiveFloorConfig:
    """Configuration for adaptive floor detection."""
    base_threshold_mm: float = 50.0     # Base floor threshold at close range
    depth_scaling_factor: float = 0.02  # How much threshold grows with depth
    robot_height_mm: float = 500.0      # Upper height cutoff


class AdaptiveFloorDetector(FloorDetector):
    """
    Adaptive threshold floor detection.

    Floor threshold increases with distance to account for:
    - Stereo matching errors at far range
    - Perspective effects
    - Depth noise increasing with distance

    threshold(depth) = base_threshold + depth_scaling_factor * depth

    Example: At 2000mm depth with defaults:
    threshold = 50 + 0.02 * 2000 = 90mm
    """

    def __init__(self, config: AdaptiveFloorConfig = None):
        """
        Initialize detector with configuration.

        Args:
            config: Detection configuration (uses defaults if None)
        """
        self._config = config or AdaptiveFloorConfig()

    def detect(
        self,
        depth_map: np.ndarray,
        heights: np.ndarray
    ) -> np.ndarray:
        """
        Detect floor pixels with adaptive threshold.

        Args:
            depth_map: Depth values in mm (H x W)
            heights: Height above ground for each pixel (H x W)

        Returns:
            Boolean mask where True = should be ignored
        """
        # Compute adaptive floor threshold based on depth
        adaptive_threshold = (
            self._config.base_threshold_mm +
            self._config.depth_scaling_factor * depth_map
        )

        floor_mask = heights < adaptive_threshold
        ceiling_mask = heights > self._config.robot_height_mm

        ignore_mask = floor_mask | ceiling_mask

        return ignore_mask

    def get_parameters(self) -> Dict[str, Any]:
        """Return current tunable parameters."""
        return {
            'base_threshold_mm': self._config.base_threshold_mm,
            'depth_scaling_factor': self._config.depth_scaling_factor,
            'robot_height_mm': self._config.robot_height_mm,
        }

    def set_parameters(self, **kwargs) -> None:
        """
        Update tunable parameters.

        Args:
            base_threshold_mm: New base threshold (optional)
            depth_scaling_factor: New scaling factor (optional)
            robot_height_mm: New robot height cutoff (optional)
        """
        if 'base_threshold_mm' in kwargs:
            self._config.base_threshold_mm = float(kwargs['base_threshold_mm'])
        if 'depth_scaling_factor' in kwargs:
            self._config.depth_scaling_factor = float(kwargs['depth_scaling_factor'])
        if 'robot_height_mm' in kwargs:
            self._config.robot_height_mm = float(kwargs['robot_height_mm'])

    @property
    def base_threshold_mm(self) -> float:
        """Get base threshold."""
        return self._config.base_threshold_mm

    @base_threshold_mm.setter
    def base_threshold_mm(self, value: float) -> None:
        """Set base threshold."""
        self._config.base_threshold_mm = value

    @property
    def depth_scaling_factor(self) -> float:
        """Get depth scaling factor."""
        return self._config.depth_scaling_factor

    @depth_scaling_factor.setter
    def depth_scaling_factor(self, value: float) -> None:
        """Set depth scaling factor."""
        self._config.depth_scaling_factor = value

    @property
    def robot_height_mm(self) -> float:
        """Get robot height cutoff."""
        return self._config.robot_height_mm

    @robot_height_mm.setter
    def robot_height_mm(self, value: float) -> None:
        """Set robot height cutoff."""
        self._config.robot_height_mm = value
