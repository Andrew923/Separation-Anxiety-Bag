"""
Stereo camera calibration module.
Handles individual camera calibration, stereo calibration, and rectification.
"""

import cv2
import numpy as np
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime


@dataclass
class CheckerboardConfig:
    """Configuration for checkerboard calibration pattern."""
    rows: int = 7           # Internal corners (vertical)
    cols: int = 10          # Internal corners (horizontal)
    square_size_mm: float = 25.0

    @property
    def pattern_size(self) -> Tuple[int, int]:
        """Get pattern size as (cols, rows) for OpenCV."""
        return (self.cols, self.rows)

    def get_object_points(self) -> np.ndarray:
        """
        Generate 3D object points for the checkerboard.

        Returns:
            Array of shape (rows*cols, 3) with Z=0
        """
        objp = np.zeros((self.rows * self.cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        objp *= self.square_size_mm
        return objp


@dataclass
class CalibrationResult:
    """Container for calibration results."""
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rmse: float
    rvecs: List[np.ndarray] = field(default_factory=list)
    tvecs: List[np.ndarray] = field(default_factory=list)


class StereoCalibrator:
    """
    Performs stereo camera calibration from image pairs.
    """

    def __init__(
        self,
        checkerboard: CheckerboardConfig,
        image_size: Tuple[int, int]
    ):
        """
        Initialize stereo calibrator.

        Args:
            checkerboard: Checkerboard configuration
            image_size: Single camera image size (width, height)
        """
        self.checkerboard = checkerboard
        self.image_size = image_size

        # Storage for calibration points
        self.object_points: List[np.ndarray] = []
        self.left_image_points: List[np.ndarray] = []
        self.right_image_points: List[np.ndarray] = []

        # Corner detection flags
        self.find_flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH |
            cv2.CALIB_CB_FAST_CHECK |
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        # Subpixel refinement criteria
        self.subpix_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
        self.subpix_winsize = (11, 11)

    def find_corners(
        self,
        image: np.ndarray,
        draw: bool = False
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find checkerboard corners in an image.

        Args:
            image: Input image (color or grayscale)
            draw: If True, return image with drawn corners

        Returns:
            Tuple of (found, corners, drawn_image)
            - found: True if corners detected
            - corners: Refined corner coordinates (Nx1x2) or None
            - drawn_image: Image with drawn corners (if draw=True) or None
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Find corners
        found, corners = cv2.findChessboardCorners(
            gray,
            self.checkerboard.pattern_size,
            self.find_flags
        )

        drawn_image = None

        if found:
            # Refine corner positions
            corners = cv2.cornerSubPix(
                gray,
                corners,
                self.subpix_winsize,
                (-1, -1),
                self.subpix_criteria
            )

            if draw:
                drawn_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(
                    drawn_image,
                    self.checkerboard.pattern_size,
                    corners,
                    found
                )
        elif draw:
            drawn_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return found, corners, drawn_image

    def add_calibration_pair(
        self,
        left_corners: np.ndarray,
        right_corners: np.ndarray
    ) -> None:
        """
        Add a valid corner pair for calibration.

        Args:
            left_corners: Corners from left image
            right_corners: Corners from right image
        """
        self.object_points.append(self.checkerboard.get_object_points())
        self.left_image_points.append(left_corners)
        self.right_image_points.append(right_corners)

    def get_num_pairs(self) -> int:
        """Get number of calibration pairs added."""
        return len(self.object_points)

    def calibrate_individual(self) -> Tuple[CalibrationResult, CalibrationResult]:
        """
        Calibrate each camera individually.

        Returns:
            Tuple of (left_result, right_result)
        """
        if len(self.object_points) < 3:
            raise ValueError("Need at least 3 calibration pairs")

        # Calibrate left camera
        left_rmse, left_mtx, left_dist, left_rvecs, left_tvecs = cv2.calibrateCamera(
            self.object_points,
            self.left_image_points,
            self.image_size,
            None,
            None
        )

        left_result = CalibrationResult(
            camera_matrix=left_mtx,
            dist_coeffs=left_dist,
            rmse=left_rmse,
            rvecs=left_rvecs,
            tvecs=left_tvecs
        )

        # Calibrate right camera
        right_rmse, right_mtx, right_dist, right_rvecs, right_tvecs = cv2.calibrateCamera(
            self.object_points,
            self.right_image_points,
            self.image_size,
            None,
            None
        )

        right_result = CalibrationResult(
            camera_matrix=right_mtx,
            dist_coeffs=right_dist,
            rmse=right_rmse,
            rvecs=right_rvecs,
            tvecs=right_tvecs
        )

        return left_result, right_result

    def calibrate_stereo(
        self,
        left_calib: CalibrationResult,
        right_calib: CalibrationResult
    ) -> Dict[str, Any]:
        """
        Perform stereo calibration.

        Args:
            left_calib: Left camera calibration result
            right_calib: Right camera calibration result

        Returns:
            Dictionary with stereo calibration results
        """
        stereo_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-5
        )

        # Use fixed intrinsics from individual calibration
        flags = cv2.CALIB_FIX_INTRINSIC

        rmse, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            self.object_points,
            self.left_image_points,
            self.right_image_points,
            left_calib.camera_matrix,
            left_calib.dist_coeffs,
            right_calib.camera_matrix,
            right_calib.dist_coeffs,
            self.image_size,
            criteria=stereo_criteria,
            flags=flags
        )

        return {
            'R': R,
            'T': T,
            'E': E,
            'F': F,
            'rmse': rmse
        }

    def compute_rectification(
        self,
        left_calib: CalibrationResult,
        right_calib: CalibrationResult,
        stereo_calib: Dict[str, Any],
        alpha: float = 0.0
    ) -> Dict[str, Any]:
        """
        Compute rectification transforms.

        Args:
            left_calib: Left camera calibration
            right_calib: Right camera calibration
            stereo_calib: Stereo calibration results
            alpha: Rectification alpha (0=crop, 1=full)

        Returns:
            Dictionary with rectification data including precomputed maps
        """
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            left_calib.camera_matrix,
            left_calib.dist_coeffs,
            right_calib.camera_matrix,
            right_calib.dist_coeffs,
            self.image_size,
            stereo_calib['R'],
            stereo_calib['T'],
            alpha=alpha
        )

        # Precompute rectification maps
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            left_calib.camera_matrix,
            left_calib.dist_coeffs,
            R1,
            P1,
            self.image_size,
            cv2.CV_16SC2
        )

        map1_right, map2_right = cv2.initUndistortRectifyMap(
            right_calib.camera_matrix,
            right_calib.dist_coeffs,
            R2,
            P2,
            self.image_size,
            cv2.CV_16SC2
        )

        return {
            'R1': R1,
            'R2': R2,
            'P1': P1,
            'P2': P2,
            'Q': Q,
            'roi1': roi1,
            'roi2': roi2,
            'map1_left': map1_left,
            'map2_left': map2_left,
            'map1_right': map1_right,
            'map2_right': map2_right
        }

    def run_full_calibration(
        self,
        alpha: float = 0.0
    ) -> Dict[str, Any]:
        """
        Run complete calibration pipeline.

        Args:
            alpha: Rectification alpha (0=crop, 1=full)

        Returns:
            Complete calibration data dictionary
        """
        print(f"Running calibration with {len(self.object_points)} image pairs...")

        # Individual calibration
        print("Calibrating left camera...")
        left_calib, right_calib = self.calibrate_individual()
        print(f"  Left RMSE: {left_calib.rmse:.4f}")
        print(f"  Right RMSE: {right_calib.rmse:.4f}")

        # Stereo calibration
        print("Running stereo calibration...")
        stereo_calib = self.calibrate_stereo(left_calib, right_calib)
        print(f"  Stereo RMSE: {stereo_calib['rmse']:.4f}")

        # Compute baseline
        baseline = np.linalg.norm(stereo_calib['T'])
        print(f"  Baseline: {baseline:.2f} mm")

        # Rectification
        print("Computing rectification...")
        rect = self.compute_rectification(left_calib, right_calib, stereo_calib, alpha)

        return {
            'calibration_date': datetime.now().isoformat(),
            'image_size': list(self.image_size),
            'checkerboard': {
                'rows': self.checkerboard.rows,
                'cols': self.checkerboard.cols,
                'square_size_mm': self.checkerboard.square_size_mm
            },
            'left_camera': {
                'camera_matrix': left_calib.camera_matrix,
                'dist_coeffs': left_calib.dist_coeffs,
                'rmse': left_calib.rmse
            },
            'right_camera': {
                'camera_matrix': right_calib.camera_matrix,
                'dist_coeffs': right_calib.dist_coeffs,
                'rmse': right_calib.rmse
            },
            'stereo': stereo_calib,
            'rectification': rect,
            'baseline_mm': baseline
        }


def save_calibration(calibration_data: Dict[str, Any], output_dir: str) -> None:
    """
    Save calibration data to files.

    Args:
        calibration_data: Complete calibration dictionary
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare JSON-serializable data
    json_data = {
        'calibration_date': calibration_data['calibration_date'],
        'image_size': calibration_data['image_size'],
        'checkerboard': calibration_data['checkerboard'],
        'left_camera': {
            'camera_matrix': calibration_data['left_camera']['camera_matrix'].tolist(),
            'dist_coeffs': calibration_data['left_camera']['dist_coeffs'].tolist(),
            'rmse': calibration_data['left_camera']['rmse']
        },
        'right_camera': {
            'camera_matrix': calibration_data['right_camera']['camera_matrix'].tolist(),
            'dist_coeffs': calibration_data['right_camera']['dist_coeffs'].tolist(),
            'rmse': calibration_data['right_camera']['rmse']
        },
        'stereo': {
            'R': calibration_data['stereo']['R'].tolist(),
            'T': calibration_data['stereo']['T'].tolist(),
            'E': calibration_data['stereo']['E'].tolist(),
            'F': calibration_data['stereo']['F'].tolist(),
            'rmse': calibration_data['stereo']['rmse']
        },
        'rectification': {
            'R1': calibration_data['rectification']['R1'].tolist(),
            'R2': calibration_data['rectification']['R2'].tolist(),
            'P1': calibration_data['rectification']['P1'].tolist(),
            'P2': calibration_data['rectification']['P2'].tolist(),
            'Q': calibration_data['rectification']['Q'].tolist(),
            'roi1': list(calibration_data['rectification']['roi1']),
            'roi2': list(calibration_data['rectification']['roi2'])
        },
        'baseline_mm': calibration_data['baseline_mm']
    }

    # Save JSON
    json_path = output_path / 'calibration.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved calibration parameters to {json_path}")

    # Save rectification maps (numpy arrays)
    maps_path = output_path / 'stereo_maps.npz'
    np.savez(
        maps_path,
        map1_left=calibration_data['rectification']['map1_left'],
        map2_left=calibration_data['rectification']['map2_left'],
        map1_right=calibration_data['rectification']['map1_right'],
        map2_right=calibration_data['rectification']['map2_right']
    )
    print(f"Saved rectification maps to {maps_path}")


def load_calibration(calibration_dir: str) -> Dict[str, Any]:
    """
    Load calibration data from files.

    Args:
        calibration_dir: Directory containing calibration files

    Returns:
        Calibration data dictionary with numpy arrays
    """
    calib_path = Path(calibration_dir)

    # Load JSON
    json_path = calib_path / 'calibration.json'
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # Load rectification maps
    maps_path = calib_path / 'stereo_maps.npz'
    maps = np.load(maps_path)

    # Convert lists back to numpy arrays
    calibration_data = {
        'calibration_date': json_data['calibration_date'],
        'image_size': tuple(json_data['image_size']),
        'checkerboard': json_data['checkerboard'],
        'left_camera': {
            'camera_matrix': np.array(json_data['left_camera']['camera_matrix']),
            'dist_coeffs': np.array(json_data['left_camera']['dist_coeffs']),
            'rmse': json_data['left_camera']['rmse']
        },
        'right_camera': {
            'camera_matrix': np.array(json_data['right_camera']['camera_matrix']),
            'dist_coeffs': np.array(json_data['right_camera']['dist_coeffs']),
            'rmse': json_data['right_camera']['rmse']
        },
        'stereo': {
            'R': np.array(json_data['stereo']['R']),
            'T': np.array(json_data['stereo']['T']),
            'E': np.array(json_data['stereo']['E']),
            'F': np.array(json_data['stereo']['F']),
            'rmse': json_data['stereo']['rmse']
        },
        'rectification': {
            'R1': np.array(json_data['rectification']['R1']),
            'R2': np.array(json_data['rectification']['R2']),
            'P1': np.array(json_data['rectification']['P1']),
            'P2': np.array(json_data['rectification']['P2']),
            'Q': np.array(json_data['rectification']['Q']),
            'roi1': tuple(json_data['rectification']['roi1']),
            'roi2': tuple(json_data['rectification']['roi2']),
            'map1_left': maps['map1_left'],
            'map2_left': maps['map2_left'],
            'map1_right': maps['map1_right'],
            'map2_right': maps['map2_right']
        },
        'baseline_mm': json_data['baseline_mm']
    }

    return calibration_data
