"""
Stereo matching module using StereoSGBM algorithm.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any


@dataclass
class SGBMParams:
    """StereoSGBM parameters with sensible defaults."""
    min_disparity: int = 0
    num_disparities: int = 64       # Must be divisible by 16
    block_size: int = 5             # Odd number, 3-11 range
    P1: Optional[int] = None        # Auto-calculated if None
    P2: Optional[int] = None        # Auto-calculated if None
    disp12_max_diff: int = 1
    pre_filter_cap: int = 63
    uniqueness_ratio: int = 10      # 5-15 range
    speckle_window_size: int = 100  # 50-200 range
    speckle_range: int = 32
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY

    def __post_init__(self):
        """Calculate P1 and P2 based on block_size if not provided."""
        self.calculate_penalties()

    def calculate_penalties(self):
        """Calculate P1 and P2 based on block_size."""
        cn = 3  # Number of channels (assuming BGR)
        if self.P1 is None:
            self.P1 = 8 * cn * self.block_size ** 2
        if self.P2 is None:
            self.P2 = 32 * cn * self.block_size ** 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for creating SGBM object."""
        return {
            'minDisparity': self.min_disparity,
            'numDisparities': self.num_disparities,
            'blockSize': self.block_size,
            'P1': self.P1,
            'P2': self.P2,
            'disp12MaxDiff': self.disp12_max_diff,
            'preFilterCap': self.pre_filter_cap,
            'uniquenessRatio': self.uniqueness_ratio,
            'speckleWindowSize': self.speckle_window_size,
            'speckleRange': self.speckle_range,
            'mode': self.mode
        }


class StereoMatcher:
    """
    Stereo matching using StereoSGBM algorithm.
    """

    COLORMAP_OPTIONS = {
        'JET': cv2.COLORMAP_JET,
        'TURBO': cv2.COLORMAP_TURBO,
        'MAGMA': cv2.COLORMAP_MAGMA,
        'INFERNO': cv2.COLORMAP_INFERNO,
        'PLASMA': cv2.COLORMAP_PLASMA,
        'VIRIDIS': cv2.COLORMAP_VIRIDIS,
    }

    def __init__(
        self,
        calibration_data: Dict[str, Any],
        params: Optional[SGBMParams] = None
    ):
        """
        Initialize stereo matcher with calibration data.

        Args:
            calibration_data: Calibration data dictionary (from load_calibration)
            params: StereoSGBM parameters (uses defaults if None)
        """
        self.calibration = calibration_data
        self.params = params or SGBMParams()

        # Extract rectification maps
        rect = calibration_data['rectification']
        self.map1_left = rect['map1_left']
        self.map2_left = rect['map2_left']
        self.map1_right = rect['map1_right']
        self.map2_right = rect['map2_right']
        self.Q = rect['Q']

        # Create SGBM matcher
        self._create_matcher()

        # Colormap for visualization
        self.colormap = cv2.COLORMAP_JET

    def _create_matcher(self):
        """Create or recreate the SGBM matcher with current params."""
        params = self.params.to_dict()
        self.stereo = cv2.StereoSGBM_create(**params)

    def rectify(
        self,
        left: np.ndarray,
        right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply rectification to image pair.

        Args:
            left: Left camera image
            right: Right camera image

        Returns:
            Tuple of (left_rectified, right_rectified)
        """
        left_rect = cv2.remap(
            left,
            self.map1_left,
            self.map2_left,
            cv2.INTER_LINEAR
        )
        right_rect = cv2.remap(
            right,
            self.map1_right,
            self.map2_right,
            cv2.INTER_LINEAR
        )
        return left_rect, right_rect

    def compute_disparity(
        self,
        left_rect: np.ndarray,
        right_rect: np.ndarray
    ) -> np.ndarray:
        """
        Compute disparity map from rectified images.

        Args:
            left_rect: Rectified left image
            right_rect: Rectified right image

        Returns:
            Disparity map (int16, values scaled by 16)
        """
        disparity = self.stereo.compute(left_rect, right_rect)
        return disparity

    def process_frame(
        self,
        left: np.ndarray,
        right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a stereo frame pair: rectify and compute disparity.

        Args:
            left: Left camera image
            right: Right camera image

        Returns:
            Tuple of (left_rectified, right_rectified, disparity)
        """
        left_rect, right_rect = self.rectify(left, right)
        disparity = self.compute_disparity(left_rect, right_rect)
        return left_rect, right_rect, disparity

    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity to depth using Q matrix.

        Args:
            disparity: Disparity map (int16, scaled by 16)

        Returns:
            Depth map in same units as calibration (mm)
        """
        # Convert disparity to float
        disp_float = disparity.astype(np.float32) / 16.0

        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disp_float, self.Q)

        # Extract Z (depth)
        depth = points_3d[:, :, 2]

        return depth

    def normalize_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """
        Normalize disparity map to 0-255 range.

        Args:
            disparity: Raw disparity from SGBM (int16, scaled by 16)

        Returns:
            Normalized disparity as uint8
        """
        disp_float = disparity.astype(np.float32) / 16.0

        # Mask invalid disparities
        valid_mask = disp_float > 0

        normalized = np.zeros_like(disp_float, dtype=np.uint8)
        if np.any(valid_mask):
            min_disp = disp_float[valid_mask].min()
            max_disp = disp_float[valid_mask].max()
            if max_disp > min_disp:
                normalized[valid_mask] = (
                    255 * (disp_float[valid_mask] - min_disp) / (max_disp - min_disp)
                ).astype(np.uint8)

        return normalized

    def get_colorized_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity to colorized visualization.

        Args:
            disparity: Raw disparity from SGBM (int16, scaled by 16)

        Returns:
            Colorized disparity image (BGR)
        """
        normalized = self.normalize_disparity(disparity)
        return cv2.applyColorMap(normalized, self.colormap)

    def set_colormap(self, colormap_name: str) -> None:
        """
        Set colormap for disparity visualization.

        Args:
            colormap_name: Name of colormap ('JET', 'TURBO', etc.)
        """
        if colormap_name in self.COLORMAP_OPTIONS:
            self.colormap = self.COLORMAP_OPTIONS[colormap_name]

    def update_params(self, **kwargs) -> None:
        """
        Update SGBM parameters at runtime.

        Args:
            **kwargs: Parameter names and values to update
        """
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)

        # Recalculate P1/P2 if block_size changed
        if 'block_size' in kwargs:
            self.params.P1 = None
            self.params.P2 = None
            self.params.calculate_penalties()

        # Recreate matcher with new params
        self._create_matcher()

    def get_param(self, name: str) -> Any:
        """Get current parameter value."""
        return getattr(self.params, name, None)


def create_parameter_trackbars(window_name: str, matcher: StereoMatcher) -> None:
    """
    Create trackbars for adjusting SGBM parameters.

    Args:
        window_name: OpenCV window name
        matcher: StereoMatcher instance
    """
    def on_num_disp(val):
        val = max(16, (val // 16) * 16)  # Must be divisible by 16
        matcher.update_params(num_disparities=val)

    def on_block_size(val):
        val = max(1, val)
        if val % 2 == 0:
            val += 1  # Must be odd
        matcher.update_params(block_size=val)

    def on_uniqueness(val):
        matcher.update_params(uniqueness_ratio=val)

    def on_speckle_window(val):
        matcher.update_params(speckle_window_size=val)

    def on_speckle_range(val):
        matcher.update_params(speckle_range=val)

    cv2.createTrackbar('Num Disp', window_name, matcher.params.num_disparities, 256, on_num_disp)
    cv2.createTrackbar('Block Size', window_name, matcher.params.block_size, 21, on_block_size)
    cv2.createTrackbar('Uniqueness', window_name, matcher.params.uniqueness_ratio, 25, on_uniqueness)
    cv2.createTrackbar('Speckle Win', window_name, matcher.params.speckle_window_size, 200, on_speckle_window)
    cv2.createTrackbar('Speckle Rng', window_name, matcher.params.speckle_range, 64, on_speckle_range)
