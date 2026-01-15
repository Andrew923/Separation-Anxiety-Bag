#!/usr/bin/env python3
"""
Interactive brightness detection tuning script.

Allows real-time adjustment of camera exposure and detection
parameters to find optimal settings for flashlight tracking.

Supports both stereo cameras (device 0) and single cameras (device 1+).

Usage:
    python robot/scripts/tune_brightness_detection.py [--device 1] [--resolution low]

Keyboard Controls (adjacent keys for down/up):
    e/r     Exposure -/+ (fine ±1)
    w/f     Exposure -/+ (coarse ±50)
    t/y     Threshold -/+ (fine ±5)
    g/h     Threshold -/+ (coarse ±25)
    u/i     Gain -/+ (fine ±0.05)
    j/k     Gain -/+ (coarse ±0.2)
    o/p     Blur kernel -/+ (±2)
    z/x     Min area -/+ (fine ±5)
    c/v     Min area -/+ (coarse ±50)
    b/n     Max area -/+ (fine ±100)
    ,/.     Max area -/+ (coarse ±1000)
    [/]     Settle frames -/+ (±1)
    1/2/3   Resolution: low/medium/high
    a       Toggle auto-exposure
    0       Reset to defaults
    /       Print settings to console
    s       Save snapshot (write to disk)
    q       Quit (prints final settings)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import cv2
import numpy as np

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vision.src.camera import StereoCamera


class SingleCamera:
    """
    Simple wrapper for a single (non-stereo) USB camera.

    Provides the same interface as StereoCamera for compatibility,
    but works with regular webcams that output a single frame.
    """

    # Common resolutions for single cameras
    RESOLUTIONS = {
        'high': (1280, 720),
        'medium': (640, 480),
        'low': (320, 240),
    }

    def __init__(
        self,
        device_id: int = 1,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30
    ):
        """
        Initialize single camera.

        Args:
            device_id: V4L2 device ID (e.g., 1 for /dev/video1)
            resolution: Frame resolution (width, height)
            fps: Target frames per second
        """
        self.device_id = device_id
        self.resolution = resolution
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_open = False

    def open(self) -> bool:
        """Open camera with specified settings."""
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

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.

        Returns:
            Tuple of (success, frame)
        """
        if not self._is_open or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        return ret, frame

    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._is_open = False

    def is_opened(self) -> bool:
        """Check if camera is open."""
        return self._is_open and self.cap is not None and self.cap.isOpened()

    def get_resolution(self) -> Tuple[int, int]:
        """Get the current resolution."""
        return self.resolution

    def set_resolution(self, resolution: Tuple[int, int]) -> bool:
        """Change camera resolution."""
        if not self._is_open or self.cap is None:
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.resolution = (actual_width, actual_height)
        return actual_width == resolution[0] and actual_height == resolution[1]

    def set_auto_exposure(self, enabled: bool) -> bool:
        """Enable or disable auto exposure."""
        if not self._is_open or self.cap is None:
            return False

        # V4L2 auto exposure: 1 = manual, 3 = auto
        mode = 3 if enabled else 1
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, mode)
        return True

    def set_exposure(self, exposure: float) -> bool:
        """Set manual exposure value."""
        if not self._is_open or self.cap is None:
            return False

        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        return True

    def get_exposure(self) -> Optional[float]:
        """Get current exposure value."""
        if not self._is_open or self.cap is None:
            return None

        return self.cap.get(cv2.CAP_PROP_EXPOSURE)

    def get_auto_exposure(self) -> Optional[bool]:
        """Get current auto exposure state."""
        if not self._is_open or self.cap is None:
            return None

        mode = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        # V4L2: 1 = manual, 3 = auto
        return mode == 3


@dataclass
class TuningParams:
    """Current tuning parameters."""
    exposure: int = 20
    threshold: int = 230
    gain: float = 1.60
    blur_kernel: int = 1
    min_area: int = 20
    max_area: int = 200
    settle_frames: int = 2
    auto_exposure: bool = False

    # Limits
    exposure_min: int = 3
    exposure_max: int = 2047
    threshold_min: int = 50
    threshold_max: int = 255
    gain_min: float = 0.1
    gain_max: float = 2.0
    blur_min: int = 1
    blur_max: int = 15
    min_area_min: int = 1
    min_area_max: int = 500
    max_area_min: int = 100
    max_area_max: int = 50000
    settle_min: int = 0
    settle_max: int = 10

    def clamp(self) -> None:
        """Clamp all values to valid ranges."""
        self.exposure = max(self.exposure_min, min(self.exposure_max, self.exposure))
        self.threshold = max(self.threshold_min, min(self.threshold_max, self.threshold))
        self.gain = max(self.gain_min, min(self.gain_max, self.gain))
        self.blur_kernel = max(self.blur_min, min(self.blur_max, self.blur_kernel))
        # Ensure blur kernel is odd
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1
        self.min_area = max(self.min_area_min, min(self.min_area_max, self.min_area))
        self.max_area = max(self.max_area_min, min(self.max_area_max, self.max_area))
        self.settle_frames = max(self.settle_min, min(self.settle_max, self.settle_frames))

    def reset(self) -> None:
        """Reset to defaults."""
        self.exposure = 20
        self.threshold = 230
        self.gain = 1.60
        self.blur_kernel = 1
        self.min_area = 20
        self.max_area = 200
        self.settle_frames = 2
        self.auto_exposure = False


@dataclass
class BlobInfo:
    """Information about a detected blob."""
    contour: np.ndarray
    cx: int
    cy: int
    brightness: int
    area: int

    def __eq__(self, other):
        """Compare blobs by position (avoid numpy array comparison)."""
        if not isinstance(other, BlobInfo):
            return False
        return self.cx == other.cx and self.cy == other.cy


def detect_brightness(
    frame: np.ndarray,
    params: TuningParams
) -> Tuple[np.ndarray, List[BlobInfo]]:
    """
    Run brightness detection with current params.

    Args:
        frame: BGR image
        params: Current tuning parameters

    Returns:
        Tuple of (binary_mask, list of BlobInfo)
    """
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # Apply gain
    darkened = (gray.astype(np.float32) * params.gain).clip(0, 255).astype(np.uint8)

    # Apply blur
    kernel = params.blur_kernel if params.blur_kernel % 2 == 1 else params.blur_kernel + 1
    blurred = cv2.GaussianBlur(darkened, (kernel, kernel), 0)

    # Threshold
    _, binary = cv2.threshold(blurred, params.threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area and collect blob info
    blobs = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if params.min_area <= area <= params.max_area:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Sample brightness from original grayscale
                if 0 <= cy < gray.shape[0] and 0 <= cx < gray.shape[1]:
                    brightness = int(gray[cy, cx])
                    blobs.append(BlobInfo(contour, cx, cy, brightness, area))

    # Sort by brightness (descending)
    blobs.sort(key=lambda b: b.brightness, reverse=True)

    return binary, blobs


def draw_overlay(
    frame: np.ndarray,
    binary: np.ndarray,
    blobs: List[BlobInfo],
    params: TuningParams,
    resolution_name: str
) -> np.ndarray:
    """
    Create side-by-side display with detection overlay.

    Args:
        frame: Original BGR frame
        binary: Binary threshold mask
        blobs: Detected blobs
        params: Current parameters
        resolution_name: Current resolution name

    Returns:
        Combined image for display
    """
    # Create detection overlay (binary as BGR + contours)
    detection = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Draw all contours in blue
    for blob in blobs:
        cv2.drawContours(detection, [blob.contour], -1, (255, 100, 0), 2)

    # Draw best blob (if any) in green with crosshair
    if blobs:
        best = blobs[0]
        cv2.drawContours(detection, [best.contour], -1, (0, 255, 0), 2)
        cv2.circle(detection, (best.cx, best.cy), 8, (0, 255, 0), 2)
        cv2.line(detection, (best.cx - 15, best.cy), (best.cx + 15, best.cy), (0, 255, 0), 2)
        cv2.line(detection, (best.cx, best.cy - 15), (best.cx, best.cy + 15), (0, 255, 0), 2)

    # Also draw detection on original frame
    frame_overlay = frame.copy()
    for blob in blobs:
        color = (0, 255, 0) if blob == blobs[0] else (255, 100, 0)
        cv2.drawContours(frame_overlay, [blob.contour], -1, color, 2)
        if blob == blobs[0]:
            cv2.circle(frame_overlay, (blob.cx, blob.cy), 8, color, 2)

    # Stack horizontally
    combined = np.hstack([frame_overlay, detection])

    # Add text overlay
    h, w = combined.shape[:2]
    
    # Background for text
    cv2.rectangle(combined, (0, h - 70), (w, h), (0, 0, 0), -1)

    # Line 1: Exposure, threshold, gain, blur, settle
    auto_str = "ON" if params.auto_exposure else "OFF"
    line1 = f"Exp: {params.exposure} ({params.exposure_min}-{params.exposure_max}) | " \
            f"Thresh: {params.threshold} | Gain: {params.gain:.2f} | " \
            f"Blur: {params.blur_kernel} | Settle: {params.settle_frames}"
    cv2.putText(combined, line1, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Line 2: Area range, auto-exposure, resolution
    line2 = f"Area: {params.min_area}-{params.max_area} | Auto-exp: {auto_str} | Res: {resolution_name}"
    cv2.putText(combined, line2, (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Line 3: Detection results
    if blobs:
        best = blobs[0]
        line3 = f"Blobs: {len(blobs)} | Best: ({best.cx}, {best.cy}) bright={best.brightness} area={best.area}"
    else:
        line3 = "Blobs: 0 | No detection"
    cv2.putText(combined, line3, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Add labels for left/right panels
    cv2.putText(combined, "Raw Camera", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(combined, "Detection", (w // 2 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return combined


def print_settings(params: TuningParams) -> None:
    """Print current settings to console."""
    print("\n" + "=" * 60)
    print("Current Brightness Detection Settings")
    print("=" * 60)
    print(f"  Exposure:       {params.exposure}")
    print(f"  Threshold:      {params.threshold}")
    print(f"  Gain:           {params.gain:.2f}")
    print(f"  Blur kernel:    {params.blur_kernel}")
    print(f"  Min area:       {params.min_area}")
    print(f"  Max area:       {params.max_area}")
    print(f"  Settle frames:  {params.settle_frames}")
    print(f"  Auto-exposure:  {params.auto_exposure}")
    print("=" * 60 + "\n")


def print_final_settings(params: TuningParams) -> None:
    """Print final settings in copyable format."""
    print("\n" + "=" * 80)
    print("Tuned Brightness Detection Settings")
    print("=" * 80)
    print("""
# For robot_config.yaml:
target_detection:
  pattern_type: "brightness"
  brightness:
    threshold: {threshold}
    gain: {gain:.2f}
    blur_kernel_size: {blur}
    min_area_px: {min_area}
    max_area_px: {max_area}
    use_low_exposure: true
    low_exposure: {exposure}
    settle_frames: {settle}
""".format(
        threshold=params.threshold,
        gain=params.gain,
        blur=params.blur_kernel,
        min_area=params.min_area,
        max_area=params.max_area,
        exposure=params.exposure,
        settle=params.settle_frames
    ))

    print("""# For TargetDetectorConfig:
TargetDetectorConfig(
    pattern_type="brightness",
    brightness_threshold={threshold},
    brightness_gain={gain:.2f},
    brightness_blur_kernel_size={blur},
    brightness_min_area_px={min_area},
    brightness_max_area_px={max_area},
    brightness_use_low_exposure=True,
    brightness_low_exposure={exposure}.0,
    brightness_settle_frames={settle},
)
""".format(
        threshold=params.threshold,
        gain=params.gain,
        blur=params.blur_kernel,
        min_area=params.min_area,
        max_area=params.max_area,
        exposure=params.exposure,
        settle=params.settle_frames
    ))
    print("=" * 80 + "\n")


def save_snapshot(
    frame: np.ndarray,
    combined: np.ndarray,
    params: TuningParams,
    output_dir: Path
) -> None:
    """Save snapshot of current frame and settings."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"brightness_tune_{timestamp}"

    # Save raw frame
    raw_path = output_dir / f"{base_name}_raw.png"
    cv2.imwrite(str(raw_path), frame)

    # Save combined view
    detection_path = output_dir / f"{base_name}_detection.png"
    cv2.imwrite(str(detection_path), combined)

    # Save settings
    settings_path = output_dir / f"{base_name}_settings.txt"
    with open(settings_path, 'w') as f:
        f.write("Brightness Detection Tuning Settings\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'=' * 40}\n\n")
        f.write(f"exposure: {params.exposure}\n")
        f.write(f"threshold: {params.threshold}\n")
        f.write(f"gain: {params.gain:.2f}\n")
        f.write(f"blur_kernel: {params.blur_kernel}\n")
        f.write(f"min_area: {params.min_area}\n")
        f.write(f"max_area: {params.max_area}\n")
        f.write(f"settle_frames: {params.settle_frames}\n")
        f.write(f"auto_exposure: {params.auto_exposure}\n")

    print(f"\nSnapshot saved:")
    print(f"  {raw_path}")
    print(f"  {detection_path}")
    print(f"  {settings_path}\n")


def apply_exposure(camera, params: TuningParams) -> None:
    """Apply current exposure settings to camera."""
    if params.auto_exposure:
        camera.set_auto_exposure(True)
    else:
        camera.set_auto_exposure(False)
        camera.set_exposure(params.exposure)


def handle_key(
    key: int,
    params: TuningParams,
    camera
) -> Tuple[bool, Optional[str]]:
    """
    Handle keyboard input.

    Controls use adjacent keys for down/up:
      e/r: Exposure (fine), w/f: Exposure (coarse)
      t/y: Threshold (fine), g/h: Threshold (coarse)
      u/i: Gain (fine), j/k: Gain (coarse)
      o/p: Blur kernel
      z/x: Min area (fine), c/v: Min area (coarse)
      b/n: Max area (fine), ,/.: Max area (coarse)
      [/]: Settle frames

    Args:
        key: Key code from cv2.waitKey
        params: Current parameters (modified in place)
        camera: Camera instance for exposure control

    Returns:
        Tuple of (should_continue, action or None)
    """
    if key == -1:
        return True, None

    char = chr(key & 0xFF) if key >= 0 else ''

    exposure_changed = False
    new_resolution = None

    # Exposure: e/r (fine ±1), w/f (coarse ±50)
    if char == 'e':
        params.exposure -= 1
        exposure_changed = True
    elif char == 'r':
        params.exposure += 1
        exposure_changed = True
    elif char == 'w':
        params.exposure -= 50
        exposure_changed = True
    elif char == 'f':
        params.exposure += 50
        exposure_changed = True

    # Threshold: t/y (fine ±5), g/h (coarse ±25)
    elif char == 't':
        params.threshold -= 5
    elif char == 'y':
        params.threshold += 5
    elif char == 'g':
        params.threshold -= 25
    elif char == 'h':
        params.threshold += 25

    # Gain: u/i (fine ±0.05), j/k (coarse ±0.2)
    elif char == 'u':
        params.gain -= 0.05
    elif char == 'i':
        params.gain += 0.05
    elif char == 'j':
        params.gain -= 0.2
    elif char == 'k':
        params.gain += 0.2

    # Blur: o/p (±2)
    elif char == 'o':
        params.blur_kernel -= 2
    elif char == 'p':
        params.blur_kernel += 2

    # Min area: z/x (fine ±5), c/v (coarse ±50)
    elif char == 'z':
        params.min_area -= 5
    elif char == 'x':
        params.min_area += 5
    elif char == 'c':
        params.min_area -= 50
    elif char == 'v':
        params.min_area += 50

    # Max area: b/n (fine ±100), ,/. (coarse ±1000)
    elif char == 'b':
        params.max_area -= 100
    elif char == 'n':
        params.max_area += 100
    elif char == ',':
        params.max_area -= 1000
    elif char == '.':
        params.max_area += 1000

    # Settle frames: [/] (±1)
    elif char == '[':
        params.settle_frames -= 1
    elif char == ']':
        params.settle_frames += 1

    # Resolution: 1/2/3
    elif char == '1':
        new_resolution = 'low'
    elif char == '2':
        new_resolution = 'medium'
    elif char == '3':
        new_resolution = 'high'

    # Auto-exposure toggle: a
    elif char == 'a':
        params.auto_exposure = not params.auto_exposure
        exposure_changed = True

    # Reset: 0 (zero)
    elif char == '0':
        params.reset()
        exposure_changed = True

    # Print: /
    elif char == '/':
        print_settings(params)

    # Write snapshot: s
    elif char == 's':
        return True, 'snapshot'

    # Quit: q
    elif char == 'q':
        return False, None

    # Clamp values
    params.clamp()

    # Apply exposure changes
    if exposure_changed:
        apply_exposure(camera, params)
        # Discard settle frames
        for _ in range(params.settle_frames):
            camera.read()

    return True, new_resolution


def main():
    parser = argparse.ArgumentParser(
        description="Interactive brightness detection tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--device', type=int, default=2,
                        help='Camera device ID (default: 2 for dedicated brightness camera)')
    parser.add_argument('--resolution', choices=['low', 'medium', 'high'],
                        default='low', help='Starting resolution (default: low)')
    parser.add_argument('--stereo', action='store_true',
                        help='Use stereo camera mode (for device 0)')
    args = parser.parse_args()

    # Determine if we should use stereo or single camera mode
    # Device 0 is typically the stereo camera, device 1+ are single cameras
    use_stereo = args.stereo or args.device == 0

    if use_stereo:
        # Stereo camera resolutions (side-by-side, full frame)
        resolutions = {
            'low': (640, 240),      # 320x240 per camera
            'medium': (1280, 480),  # 640x480 per camera
            'high': (2560, 960),    # 1280x960 per camera
        }
        camera_type = "stereo"
    else:
        # Single camera resolutions
        resolutions = {
            'low': (320, 240),
            'medium': (640, 480),
            'high': (1280, 720),
        }
        camera_type = "single"

    current_resolution = args.resolution
    resolution = resolutions[current_resolution]

    # Initialize camera
    print(f"Opening {camera_type} camera {args.device} at {resolution}...")

    if use_stereo:
        camera = StereoCamera(device_id=args.device, resolution=resolution, fps=30)
    else:
        camera = SingleCamera(device_id=args.device, resolution=resolution, fps=30)

    if not camera.open():
        print("Failed to open camera!")
        return 1

    print("Camera opened successfully.")

    # Initialize parameters
    params = TuningParams()

    # Apply initial exposure settings
    apply_exposure(camera, params)

    # Discard initial frames for camera to settle
    print("Waiting for camera to settle...")
    for _ in range(10):
        camera.read()

    # Output directory for snapshots
    output_dir = Path(__file__).parent / "captures"

    # Create window
    window_name = "Brightness Detection Tuning"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\nControls (adjacent keys for -/+):")
    print("  e/r: Exposure (fine ±1)     w/f: Exposure (coarse ±50)")
    print("  t/y: Threshold (fine ±5)    g/h: Threshold (coarse ±25)")
    print("  u/i: Gain (fine ±0.05)      j/k: Gain (coarse ±0.2)")
    print("  o/p: Blur kernel (±2)")
    print("  z/x: Min area (fine ±5)     c/v: Min area (coarse ±50)")
    print("  b/n: Max area (fine ±100)   ,/.: Max area (coarse ±1000)")
    print("  [/]: Settle frames (±1)")
    print("  1/2/3: Resolution low/medium/high")
    print("  a: Toggle auto-exposure     0: Reset to defaults")
    print("  /: Print settings           s: Save snapshot")
    print("  q: Quit")
    print("\nStarting tuning loop...\n")

    try:
        while True:
            # Capture frame (different API for stereo vs single camera)
            if use_stereo:
                ret, left, right = camera.read()
                if not ret or left is None:
                    print("Failed to capture frame")
                    continue
                frame = left
            else:
                ret, frame = camera.read()
                if not ret or frame is None:
                    print("Failed to capture frame")
                    continue

            # Run detection
            binary, blobs = detect_brightness(frame, params)

            # Get resolution name string
            if use_stereo:
                single_res = camera.get_single_resolution()
            else:
                single_res = camera.get_resolution()
            res_name = f"{single_res[0]}x{single_res[1]} ({current_resolution})"

            # Create display
            combined = draw_overlay(frame, binary, blobs, params, res_name)

            # Show
            cv2.imshow(window_name, combined)

            # Handle input (30ms wait = ~33fps max)
            key = cv2.waitKey(30)
            should_continue, action = handle_key(key, params, camera)

            if not should_continue:
                break

            if action == 'snapshot':
                save_snapshot(frame, combined, params, output_dir)
            elif action in resolutions:
                # Change resolution
                current_resolution = action
                new_res = resolutions[action]
                print(f"Changing resolution to {action} ({new_res})...")
                camera.set_resolution(new_res)
                # Re-apply exposure after resolution change
                apply_exposure(camera, params)
                for _ in range(5):
                    camera.read()

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        # Print final settings
        print_final_settings(params)

        # Cleanup
        cv2.destroyAllWindows()
        camera.release()
        print("Camera released.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
