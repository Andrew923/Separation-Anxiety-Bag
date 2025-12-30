"""
Stereo camera capture and frame splitting module.
Handles side-by-side stereo cameras that output concatenated frames.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class StereoCamera:
    """
    Manages a side-by-side stereo camera.
    Handles capture and splitting of concatenated frames.
    """

    # Supported resolutions (full frame width x height)
    RESOLUTIONS = {
        'high': (2560, 960),      # 1280x960 per camera
        'medium': (1280, 480),    # 640x480 per camera
        'low': (640, 240),        # 320x240 per camera
    }

    def __init__(
        self,
        device_id: int = 0,
        resolution: Tuple[int, int] = (2560, 960),
        fps: int = 30
    ):
        """
        Initialize stereo camera.

        Args:
            device_id: V4L2 device ID (e.g., 0 for /dev/video0)
            resolution: Full frame resolution (width, height) - side-by-side
            fps: Target frames per second
        """
        self.device_id = device_id
        self.resolution = resolution
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_open = False

    def open(self) -> bool:
        """
        Open camera with specified settings.

        Returns:
            True if camera opened successfully
        """
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            return False

        # Set MJPG codec for better performance
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Set framerate
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Verify settings were applied
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width != self.resolution[0] or actual_height != self.resolution[1]:
            print(f"Warning: Requested {self.resolution}, got ({actual_width}, {actual_height})")
            self.resolution = (actual_width, actual_height)

        self._is_open = True
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read and split a stereo frame.

        Returns:
            Tuple of (success, left_image, right_image)
            If success is False, images will be None
        """
        if not self._is_open or self.cap is None:
            return False, None, None

        ret, frame = self.cap.read()

        if not ret or frame is None:
            return False, None, None

        left, right = self._split_frame(frame)
        return True, left, right

    def read_raw(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the raw concatenated frame without splitting.

        Returns:
            Tuple of (success, raw_frame)
        """
        if not self._is_open or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        return ret, frame

    def _split_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split concatenated frame into left and right images.

        Args:
            frame: Concatenated stereo frame

        Returns:
            Tuple of (left_image, right_image)
        """
        height, width = frame.shape[:2]
        mid = width // 2
        left = frame[:, :mid]
        right = frame[:, mid:]
        return left, right

    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._is_open = False

    def is_opened(self) -> bool:
        """Check if camera is open."""
        return self._is_open and self.cap is not None and self.cap.isOpened()

    def get_single_resolution(self) -> Tuple[int, int]:
        """Get the resolution of a single camera (half width)."""
        return (self.resolution[0] // 2, self.resolution[1])

    def set_resolution(self, resolution: Tuple[int, int]) -> bool:
        """
        Change camera resolution.

        Args:
            resolution: New full frame resolution (width, height)

        Returns:
            True if resolution was changed successfully
        """
        if not self._is_open or self.cap is None:
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.resolution = (actual_width, actual_height)
        return actual_width == resolution[0] and actual_height == resolution[1]

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
