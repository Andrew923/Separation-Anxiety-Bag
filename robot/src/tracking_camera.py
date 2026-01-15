"""
Dedicated tracking camera for brightness-based target detection.

This camera runs at a fixed low exposure to isolate bright light sources
(e.g., phone flashlight) from ambient light. It is separate from the
stereo camera used for depth estimation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class TrackingCameraConfig:
    """Configuration for the tracking camera."""

    device_id: int = 2
    resolution: Tuple[int, int] = (640, 480)  # Min MJPEG resolution
    fps: int = 30
    exposure: int = 20
    auto_exposure: bool = False
    horizontal_fov_deg: float = 60.0


class TrackingCamera:
    """
    Single camera dedicated to brightness-based target tracking.

    Runs at fixed low exposure to isolate bright light sources
    (e.g., phone flashlight) from ambient light. The low exposure
    suppresses ambient reflections while keeping the flashlight visible.

    Unlike the stereo camera, this is a simple single-frame camera
    that does not require splitting or stereo matching.
    """

    def __init__(self, config: TrackingCameraConfig):
        """
        Initialize tracking camera.

        Args:
            config: Camera configuration
        """
        self._config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_open = False

    def open(self) -> bool:
        """
        Open camera with configured settings.

        Sets up MJPEG codec, resolution, framerate, and exposure.

        Returns:
            True if camera opened successfully
        """
        self._cap = cv2.VideoCapture(self._config.device_id, cv2.CAP_V4L2)

        if not self._cap.isOpened():
            return False

        # Set MJPEG codec for better performance
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.resolution[1])

        # Set framerate
        self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)

        # Verify settings were applied
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if (actual_width != self._config.resolution[0] or
                actual_height != self._config.resolution[1]):
            print(f"TrackingCamera: Requested {self._config.resolution}, "
                  f"got ({actual_width}, {actual_height})")

        # Set exposure (low exposure for brightness detection)
        self._apply_exposure()

        # Discard initial frames to let camera settle
        for _ in range(5):
            self._cap.read()

        self._is_open = True
        return True

    def _apply_exposure(self) -> None:
        """Apply configured exposure settings."""
        if self._cap is None:
            return

        if self._config.auto_exposure:
            # V4L2 auto exposure: 3 = auto
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        else:
            # V4L2 auto exposure: 1 = manual
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self._cap.set(cv2.CAP_PROP_EXPOSURE, self._config.exposure)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.

        Returns:
            Tuple of (success, frame). Frame is BGR format.
        """
        if not self._is_open or self._cap is None:
            return False, None

        ret, frame = self._cap.read()
        return ret, frame

    def release(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False

    def is_opened(self) -> bool:
        """Check if camera is open and ready."""
        return self._is_open and self._cap is not None and self._cap.isOpened()

    def set_exposure(self, exposure: float) -> bool:
        """
        Set manual exposure value.

        Args:
            exposure: Exposure value (camera-dependent units)

        Returns:
            True if setting was applied
        """
        if not self._is_open or self._cap is None:
            return False

        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual mode
        self._cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        return True

    def get_exposure(self) -> Optional[float]:
        """
        Get current exposure value.

        Returns:
            Current exposure value, or None if unavailable
        """
        if not self._is_open or self._cap is None:
            return None

        return self._cap.get(cv2.CAP_PROP_EXPOSURE)

    def get_resolution(self) -> Tuple[int, int]:
        """
        Get the current resolution.

        Returns:
            Tuple of (width, height)
        """
        if self._cap is not None:
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return self._config.resolution

    @property
    def horizontal_fov_deg(self) -> float:
        """Get horizontal field of view in degrees."""
        return self._config.horizontal_fov_deg

    @property
    def config(self) -> TrackingCameraConfig:
        """Get camera configuration."""
        return self._config

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
