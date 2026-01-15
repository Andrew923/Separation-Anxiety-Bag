#!/usr/bin/env python3
"""
Real-time stereo depth estimation using StereoSGBM with improvements.
Works with standard opencv-python (no contrib required for basic version).
"""

import sys
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera import StereoCamera
from src.calibration import load_calibration
from src.stereo_matcher import StereoMatcher, SGBMParams, create_parameter_trackbars
from src.utils import draw_epipolar_lines, resize_for_display


class ImprovedStereoMatcher:
    """Enhanced stereo matcher with pre/post-processing"""
    
    def __init__(self, calibration_data, params, use_wls=True):
        self.calibration_data = calibration_data
        self.params = params
        self.colormap_name = 'JET'
        self.use_wls = use_wls
        
        # Create improved SGBM matcher
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=params.num_disparities,
            blockSize=params.block_size,
            P1=8 * 3 * params.block_size**2,
            P2=32 * 3 * params.block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # Try to enable WLS if requested and available
        if use_wls:
            try:
                self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
                self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
                self.wls_filter.setLambda(8000)
                self.wls_filter.setSigmaColor(1.5)
                self.wls_available = True
                print("WLS filtering enabled (opencv-contrib detected)")
            except AttributeError:
                print("Warning: opencv-contrib not installed, WLS filtering disabled")
                print("Install with: pip install opencv-contrib-python")
                self.wls_available = False
                self.use_wls = False
        else:
            self.wls_available = False
        
        # Compute rectification maps
        self._compute_rectification_maps()
        
    def _compute_rectification_maps(self):
        """Compute rectification maps from calibration"""
        calib = self.calibration_data
        
        # Extract parameters from nested structure
        # Try different possible key names for distortion coefficients
        K1 = calib['left_camera']['camera_matrix']
        if 'distortion' in calib['left_camera']:
            D1 = calib['left_camera']['distortion']
        elif 'distortion_coefficients' in calib['left_camera']:
            D1 = calib['left_camera']['distortion_coefficients']
        elif 'dist_coeffs' in calib['left_camera']:
            D1 = calib['left_camera']['dist_coeffs']
        else:
            raise KeyError(f"Cannot find distortion in left_camera. Available keys: {list(calib['left_camera'].keys())}")
        
        K2 = calib['right_camera']['camera_matrix']
        if 'distortion' in calib['right_camera']:
            D2 = calib['right_camera']['distortion']
        elif 'distortion_coefficients' in calib['right_camera']:
            D2 = calib['right_camera']['distortion_coefficients']
        elif 'dist_coeffs' in calib['right_camera']:
            D2 = calib['right_camera']['dist_coeffs']
        else:
            D2 = D1  # Fallback, though this shouldn't happen
        
        R1 = calib['rectification']['R1']
        R2 = calib['rectification']['R2']
        P1 = calib['rectification']['P1']
        P2 = calib['rectification']['P2']
        
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1,
            tuple(calib['image_size']), cv2.CV_16SC2
        )
        
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2,
            tuple(calib['image_size']), cv2.CV_16SC2
        )
    
    def preprocess_image(self, img):
        """Pre-process image for better stereo matching"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Histogram equalization for better contrast
        equalized = cv2.equalizeHist(gray)
        
        # Slight Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        
        return blurred
    
    def post_process_disparity(self, disparity):
        """Post-process disparity without WLS (basic filtering)"""
        # Convert to float
        disp = disparity.astype(np.float32) / 16.0
        
        # Simple median filter to reduce noise
        disp = cv2.medianBlur(disp.astype(np.uint8), 5).astype(np.float32)
        
        # Bilateral filter for edge-preserving smoothing
        disp = cv2.bilateralFilter(disp, 5, 50, 50)
        
        return disp
    
    def process_frame(self, left, right, use_wls=None):
        """Process stereo frame with improvements"""
        if use_wls is None:
            use_wls = self.use_wls
        
        # Rectify images
        left_rect = cv2.remap(left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        
        # Pre-process for better matching
        left_proc = self.preprocess_image(left_rect)
        right_proc = self.preprocess_image(right_rect)
        
        # Compute disparity
        disparity_left = self.left_matcher.compute(left_proc, right_proc)
        
        if use_wls and self.wls_available:
            # Compute right disparity for WLS filtering
            disparity_right = self.right_matcher.compute(right_proc, left_proc)
            
            # Apply WLS filter
            disparity = self.wls_filter.filter(
                disparity_left, left_proc, None, disparity_right
            )
            disparity = disparity.astype(np.float32) / 16.0
        else:
            # Use basic post-processing instead
            disparity = self.post_process_disparity(disparity_left)
        
        return left_rect, right_rect, disparity
    
    def get_colorized_disparity(self, disparity):
        """Colorize disparity map"""
        # Normalize to 0-255
        disp_normalized = cv2.normalize(
            disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        
        # Apply colormap
        colormap = getattr(cv2, f'COLORMAP_{self.colormap_name}')
        disp_color = cv2.applyColorMap(disp_normalized, colormap)
        
        return disp_color
    
    def set_colormap(self, colormap_name):
        """Set colormap for visualization"""
        self.colormap_name = colormap_name
    
    def update_parameters(self, **kwargs):
        """Update SGBM parameters dynamically"""
        if 'num_disparities' in kwargs:
            self.params.num_disparities = kwargs['num_disparities']
            self.left_matcher.setNumDisparities(kwargs['num_disparities'])
            if self.wls_available:
                self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
                self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
            
        if 'block_size' in kwargs:
            self.params.block_size = kwargs['block_size']
            self.left_matcher.setBlockSize(kwargs['block_size'])
            
        if 'uniqueness_ratio' in kwargs:
            self.left_matcher.setUniquenessRatio(kwargs['uniqueness_ratio'])
            
        if 'speckle_window_size' in kwargs:
            self.left_matcher.setSpeckleWindowSize(kwargs['speckle_window_size'])
            
        if 'speckle_range' in kwargs:
            self.left_matcher.setSpeckleRange(kwargs['speckle_range'])


def main():
    parser = argparse.ArgumentParser(
        description='Improved real-time stereo depth estimation'
    )
    parser.add_argument(
        '--calibration', type=str, default='data/calibration_data',
        help='Path to calibration directory'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='Camera device ID'
    )
    parser.add_argument(
        '--resolution', choices=['high', 'medium', 'low'], default='medium',
        help='Resolution: high=2560x960, medium=1280x480, low=640x240'
    )
    parser.add_argument(
        '--num-disparities', type=int, default=128,
        help='Number of disparities (must be divisible by 16)'
    )
    parser.add_argument(
        '--block-size', type=int, default=9,
        help='SGBM block size (odd number, 3-11)'
    )
    parser.add_argument(
        '--wls', action='store_true',
        help='Enable WLS filtering (requires opencv-contrib-python)'
    )
    parser.add_argument(
        '--colormap', type=str, default='JET',
        choices=['JET', 'TURBO', 'MAGMA', 'INFERNO', 'PLASMA', 'VIRIDIS'],
        help='Colormap for disparity visualization'
    )
    args = parser.parse_args()

    # Resolution mapping
    resolutions = {
        'high': (2560, 960),
        'medium': (1280, 480),
        'low': (640, 240)
    }
    resolution = resolutions[args.resolution]

    # Load calibration
    calib_path = Path(__file__).parent.parent / args.calibration
    print(f"Loading calibration from {calib_path}...")

    try:
        calibration_data = load_calibration(str(calib_path))
    except FileNotFoundError as e:
        print(f"Error: Calibration files not found at {calib_path}")
        print("Run capture_calibration.py and run_calibration.py first")
        return 1

    # Check resolution compatibility
    calib_size = tuple(calibration_data['image_size'])
    single_res = (resolution[0] // 2, resolution[1])

    if single_res != calib_size:
        print(f"Warning: Requested resolution {single_res} differs from calibration {calib_size}")
        print("Rectification maps may not work correctly")

    # Setup SGBM parameters
    params = SGBMParams(
        num_disparities=args.num_disparities,
        block_size=args.block_size
    )

    # Create improved stereo matcher
    matcher = ImprovedStereoMatcher(calibration_data, params, use_wls=args.wls)
    matcher.set_colormap(args.colormap)

    use_filtering = True

    print(f"Starting improved stereo depth estimation")
    print(f"  Camera: /dev/video{args.device}")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")
    print(f"  Num disparities: {args.num_disparities}")
    print(f"  Block size: {args.block_size}")
    print(f"  Filtering: {'WLS' if matcher.wls_available else 'Basic (median+bilateral)'}")
    print()
    print("Controls:")
    print("  Q - Quit")
    print("  E - Toggle epipolar lines")
    print("  C - Cycle colormap")
    print("  F - Toggle filtering")
    print()

    # Open camera
    camera = StereoCamera(args.device, resolution, fps=30)
    if not camera.open():
        print("Error: Could not open camera")
        return 1

    # Create windows
    cv2.namedWindow('Improved Stereo Depth', cv2.WINDOW_NORMAL)

    # State
    show_epipolar = False
    colormap_idx = 0
    colormaps = ['JET', 'TURBO', 'MAGMA', 'INFERNO', 'PLASMA', 'VIRIDIS']

    # FPS tracking
    frame_times = []
    fps = 0

    print("Running... Press Q to quit")

    while True:
        start_time = time.time()

        success, left, right = camera.read()
        if not success:
            continue

        # Process frame with improvements
        left_rect, right_rect, disparity = matcher.process_frame(
            left, right, use_wls=use_filtering
        )

        # Get colorized disparity
        disp_color = matcher.get_colorized_disparity(disparity)

        # Create display
        if show_epipolar:
            stereo_view = draw_epipolar_lines(left_rect, right_rect, num_lines=15)
        else:
            stereo_view = np.hstack([left_rect, right_rect])

        # Resize disparity to match stereo view width
        disp_resized = cv2.resize(disp_color, (stereo_view.shape[1], disp_color.shape[0]))

        # Stack vertically
        display = np.vstack([stereo_view, disp_resized])

        # Add info overlay
        cv2.rectangle(display, (0, 0), (550, 30), (40, 40, 40), -1)
        filter_status = 'WLS' if (use_filtering and matcher.wls_available) else 'Basic' if use_filtering else 'OFF'
        info_text = f"FPS: {fps:.1f} | Disp: {matcher.params.num_disparities} | Block: {matcher.params.block_size} | Filter: {filter_status}"
        cv2.putText(display, info_text, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Resize for display
        display = resize_for_display(display, max_width=1400, max_height=900)

        cv2.imshow('Improved Stereo Depth', display)

        # Handle input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('e'):
            show_epipolar = not show_epipolar
            print(f"Epipolar lines: {'ON' if show_epipolar else 'OFF'}")
        elif key == ord('c'):
            colormap_idx = (colormap_idx + 1) % len(colormaps)
            matcher.set_colormap(colormaps[colormap_idx])
            print(f"Colormap: {colormaps[colormap_idx]}")
        elif key == ord('f'):
            use_filtering = not use_filtering
            print(f"Filtering: {'ON' if use_filtering else 'OFF'}")

        # Update FPS
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times))

    camera.release()
    cv2.destroyAllWindows()

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())