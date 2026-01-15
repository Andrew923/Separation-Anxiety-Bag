"""
Abstract base class for path planning algorithms.

All path planners (Follow-the-Gap, APF) implement this interface
to allow algorithm selection via configuration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np


@dataclass
class PlannerResult:
    """
    Unified result from any path planning algorithm.

    Compatible with NavigationController.update() expectations.
    """
    best_heading_deg: Optional[float]   # Recommended direction (None if stuck)
    can_proceed: bool                    # True if any safe path exists
    debug_info: Dict[str, Any] = field(default_factory=dict)  # Algorithm-specific debug data


class PathPlanner(ABC):
    """
    Abstract base class for path planning algorithms.

    All implementations receive preprocessed 1D distance data and
    output a recommended heading direction.
    """

    @abstractmethod
    def compute(
        self,
        distances: np.ndarray,
        sector_angles: np.ndarray,
        target_heading_deg: float = 0.0
    ) -> PlannerResult:
        """
        Compute best heading given obstacle distances.

        Args:
            distances: 1D array of min distances per sector (mm)
            sector_angles: Center angle of each sector (degrees)
            target_heading_deg: Desired heading toward goal (0 = forward)

        Returns:
            PlannerResult with recommended heading
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm name for logging/debugging."""
        pass
