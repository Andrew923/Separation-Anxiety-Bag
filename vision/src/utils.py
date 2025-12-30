"""
Utility functions for stereo vision system.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple


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


def create_output_dirs(base_path: str) -> None:
    """
    Create directory structure for calibration data.

    Args:
        base_path: Base path for data directory
    """
    base = Path(base_path)
    dirs = [
        base / 'calibration_images' / 'left',
        base / 'calibration_images' / 'right',
        base / 'calibration_data'
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def draw_epipolar_lines(
    left: np.ndarray,
    right: np.ndarray,
    num_lines: int = 20,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw horizontal epipolar lines on stacked rectified images.
    Useful for verifying rectification quality.

    Args:
        left: Left rectified image
        right: Right rectified image
        num_lines: Number of horizontal lines to draw
        color: Line color (BGR)

    Returns:
        Combined image with epipolar lines
    """
    # Stack images horizontally
    combined = np.hstack([left, right])
    height = combined.shape[0]

    # Draw horizontal lines
    for i in range(num_lines):
        y = int(height * (i + 1) / (num_lines + 1))
        cv2.line(combined, (0, y), (combined.shape[1], y), color, 1)

    return combined


def resize_for_display(
    image: np.ndarray,
    max_width: int = 1280,
    max_height: int = 720
) -> np.ndarray:
    """
    Resize image for display while maintaining aspect ratio.

    Args:
        image: Input image
        max_width: Maximum display width
        max_height: Maximum display height

    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)

    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image


def compute_calibration_quality(calibration: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute quality metrics for calibration.

    Args:
        calibration: Calibration data dictionary

    Returns:
        Quality metrics dictionary
    """
    metrics = {
        'left_rmse': calibration['left_camera']['rmse'],
        'right_rmse': calibration['right_camera']['rmse'],
        'stereo_rmse': calibration['stereo']['rmse'],
        'baseline_mm': calibration['baseline_mm']
    }

    # Quality assessment
    if metrics['left_rmse'] < 0.5 and metrics['right_rmse'] < 0.5:
        metrics['individual_quality'] = 'excellent'
    elif metrics['left_rmse'] < 1.0 and metrics['right_rmse'] < 1.0:
        metrics['individual_quality'] = 'good'
    else:
        metrics['individual_quality'] = 'fair'

    if metrics['stereo_rmse'] < 0.3:
        metrics['stereo_quality'] = 'excellent'
    elif metrics['stereo_rmse'] < 0.5:
        metrics['stereo_quality'] = 'good'
    else:
        metrics['stereo_quality'] = 'fair'

    return metrics


def normalize_disparity(
    disparity: np.ndarray,
    num_disparities: int
) -> np.ndarray:
    """
    Normalize disparity map to 0-255 range for visualization.

    Args:
        disparity: Raw disparity map from SGBM (int16, scaled by 16)
        num_disparities: Number of disparities used

    Returns:
        Normalized disparity as uint8
    """
    # SGBM returns disparity * 16
    disp_float = disparity.astype(np.float32) / 16.0

    # Mask invalid disparities
    valid_mask = disp_float > 0

    # Normalize to 0-255
    normalized = np.zeros_like(disp_float, dtype=np.uint8)
    if np.any(valid_mask):
        min_disp = disp_float[valid_mask].min()
        max_disp = disp_float[valid_mask].max()
        if max_disp > min_disp:
            normalized[valid_mask] = (
                255 * (disp_float[valid_mask] - min_disp) / (max_disp - min_disp)
            ).astype(np.uint8)

    return normalized


def colorize_disparity(
    disparity: np.ndarray,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Apply colormap to disparity for visualization.

    Args:
        disparity: Normalized disparity (uint8)
        colormap: OpenCV colormap constant

    Returns:
        Colorized disparity image
    """
    return cv2.applyColorMap(disparity, colormap)
