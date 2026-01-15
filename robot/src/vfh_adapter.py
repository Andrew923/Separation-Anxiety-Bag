"""
Adapter to make existing VFH conform to PathPlanner interface.

Wraps VectorFieldHistogram to implement PathPlanner ABC.
"""

import numpy as np

from .path_planner import PathPlanner, PlannerResult
from .vfh import VectorFieldHistogram, VFHConfig


class VFHAdapter(PathPlanner):
    """
    Adapter wrapping VectorFieldHistogram to PathPlanner interface.

    This allows VFH to be used interchangeably with FollowTheGap and APF.
    """

    def __init__(self, config: VFHConfig, camera_fov_deg: float = 60.0):
        """
        Initialize VFH adapter.

        Args:
            config: VFH configuration
            camera_fov_deg: Camera horizontal field of view
        """
        self._vfh = VectorFieldHistogram(config)
        self._config = config
        self._camera_fov_deg = camera_fov_deg

    @property
    def name(self) -> str:
        return "vfh"

    @property
    def config(self) -> VFHConfig:
        """Get VFH configuration."""
        return self._config

    def compute(
        self,
        distances: np.ndarray,
        sector_angles: np.ndarray,
        target_heading_deg: float = 0.0
    ) -> PlannerResult:
        """Compute best heading using VFH algorithm."""
        # Update VFH with distance data
        self._vfh.update_from_distances(
            distances,
            camera_fov_h_deg=self._camera_fov_deg
        )

        # Get VFH result
        vfh_result = self._vfh.find_safe_direction(target_heading_deg)

        # Convert to PlannerResult
        return PlannerResult(
            best_heading_deg=vfh_result.best_heading_deg,
            can_proceed=vfh_result.can_proceed,
            debug_info={
                'safe_sectors': vfh_result.safe_sectors,
                'blocked_sectors': vfh_result.blocked_sectors,
                'valleys': vfh_result.valleys,
                'best_sector': vfh_result.best_sector,
                'histogram': vfh_result.histogram.tolist() if vfh_result.histogram is not None else None
            }
        )

    def get_vfh(self) -> VectorFieldHistogram:
        """Access underlying VFH for backward compatibility."""
        return self._vfh

    def get_vfh_result(
        self,
        distances: np.ndarray,
        sector_angles: np.ndarray,
        target_heading_deg: float = 0.0
    ):
        """
        Get raw VFHResult for visualization compatibility.

        Args:
            distances: 1D array of min distances per sector (mm)
            sector_angles: Center angle of each sector (degrees)
            target_heading_deg: Target direction

        Returns:
            VFHResult from underlying VFH algorithm
        """
        self._vfh.update_from_distances(
            distances,
            camera_fov_h_deg=self._camera_fov_deg
        )
        return self._vfh.find_safe_direction(target_heading_deg)
