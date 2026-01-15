"""
Target detection using configurable pattern recognition.

Supports multiple pattern types:
- Bullseye: Concentric circle patterns using contour hierarchy analysis
- Checkerboard: 2x2 checkerboard with corner detection
- Brightness: Bright spot detection (e.g., phone flashlight)

Used for visual target tracking as secondary localization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple
import math

import numpy as np
import cv2


@dataclass
class TargetDetectorConfig:
    """Configuration for target detection."""

    pattern_type: str = "checkerboard"  # "bullseye", "checkerboard", or "brightness"

    # Shared settings
    depth_sample_radius: int = 3

    # Bullseye settings
    bullseye_min_rings: int = 1
    bullseye_circularity_threshold: float = 0.3
    bullseye_concentricity_threshold_px: int = 5
    bullseye_min_radius_px: int = 10
    bullseye_max_radius_px: int = 150
    bullseye_blur_kernel_size: int = 1
    bullseye_canny_low: int = 50
    bullseye_canny_high: int = 150

    # Checkerboard settings
    checker_sample_size: int = 6        # Pixels to sample in each quadrant
    checker_sample_offset: int = 3      # Offset from corner to avoid blur
    checker_contrast_threshold: float = 0.3  # Min contrast between B/W
    checker_corner_quality: float = 0.1      # Shi-Tomasi quality level
    checker_min_distance: int = 20           # Min distance between corners
    checker_max_corners: int = 50            # Max corners to evaluate
    checker_block_size: int = 7              # Block size for corner detection
    checker_diagonal_tolerance: float = 0.2  # Max diff within diagonal pairs

    # Brightness settings (for flashlight detection)
    brightness_threshold: int = 230          # Min pixel value to consider bright (after gain)
    brightness_min_area_px: int = 20         # Min blob area in pixels
    brightness_max_area_px: int = 5000       # Max blob area in pixels
    brightness_blur_kernel_size: int = 5     # Blur to reduce noise
    brightness_gain: float = 0.8             # Gain multiplier to suppress ambient light


@dataclass
class TargetDetection:
    """Result of target detection (pattern-agnostic)."""

    center_x: int                         # Pixel X coordinate
    center_y: int                         # Pixel Y coordinate
    angle_deg: float                      # Angle from camera center (positive = right)
    range_mm: Optional[float]             # Distance in mm (None if depth unavailable)
    confidence: float                     # Detection confidence (0-1)
    pattern_type: str                     # "bullseye" or "checkerboard"


class PatternDetector(ABC):
    """Abstract base class for pattern detection strategies."""

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        horizontal_fov_deg: float
    ) -> Optional[TargetDetection]:
        """
        Detect pattern in frame.

        Args:
            frame: BGR or grayscale image from camera
            depth_map: Optional depth map in mm (same size as frame)
            horizontal_fov_deg: Camera horizontal field of view

        Returns:
            TargetDetection if found, None otherwise
        """
        pass


class BullseyePattern(PatternDetector):
    """
    Detects bullseye patterns using contour hierarchy analysis.

    The bullseye pattern consists of concentric circles (rings). Detection
    uses OpenCV's contour hierarchy to find nested contours, then validates
    they form a proper bullseye pattern.
    """

    def __init__(self, config: TargetDetectorConfig):
        self._config = config

    def detect(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        horizontal_fov_deg: float
    ) -> Optional[TargetDetection]:
        if frame is None or frame.size == 0:
            return None

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        # Preprocess
        gray = self._preprocess(frame)

        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(
            gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None or len(contours) == 0:
            return None

        hierarchy = hierarchy[0]  # Unpack from (1, N, 4) to (N, 4)

        # Find nested contour chains (potential bullseyes)
        bullseye_candidates = self._find_nested_chains(contours, hierarchy)

        if not bullseye_candidates:
            return None

        # Validate and score each candidate
        best_detection = None
        best_confidence = 0.0

        for chain in bullseye_candidates:
            detection = self._validate_and_create_detection(
                contours, chain, frame_width, frame_height,
                depth_map, horizontal_fov_deg
            )
            if detection and detection.confidence > best_confidence:
                best_detection = detection
                best_confidence = detection.confidence

        return best_detection

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for contour detection."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        kernel_size = self._config.bullseye_blur_kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        edges = cv2.Canny(
            blurred,
            self._config.bullseye_canny_low,
            self._config.bullseye_canny_high
        )

        return edges

    def _find_nested_chains(
        self,
        contours: List,
        hierarchy: np.ndarray
    ) -> List[List[int]]:
        """Find chains of nested contours (parent-child relationships)."""
        chains = []
        visited = set()

        for i in range(len(contours)):
            if i in visited:
                continue

            chain = self._build_chain(i, hierarchy, visited)

            if len(chain) >= self._config.bullseye_min_rings:
                chains.append(chain)

        return chains

    def _build_chain(
        self,
        start_idx: int,
        hierarchy: np.ndarray,
        visited: set
    ) -> List[int]:
        """Build a chain of nested contours starting from given index."""
        chain = []
        current = start_idx

        while current >= 0 and current not in visited:
            visited.add(current)
            chain.append(current)
            first_child = hierarchy[current][2]
            current = first_child

        return chain

    def _validate_and_create_detection(
        self,
        contours: List,
        chain: List[int],
        frame_width: int,
        frame_height: int,
        depth_map: Optional[np.ndarray],
        horizontal_fov_deg: float
    ) -> Optional[TargetDetection]:
        """Validate a contour chain and create detection if valid."""
        valid_contours = []
        centers = []
        radii = []

        for idx in chain:
            contour = contours[idx]

            if len(contour) < 5:
                continue

            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                continue

            circularity = 4 * math.pi * area / (perimeter * perimeter)

            if circularity < self._config.bullseye_circularity_threshold:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)

            if radius < self._config.bullseye_min_radius_px:
                continue
            if radius > self._config.bullseye_max_radius_px:
                continue

            valid_contours.append(idx)
            centers.append((cx, cy))
            radii.append(radius)

        if len(valid_contours) < self._config.bullseye_min_rings:
            return None

        # Check concentricity
        mean_cx = sum(c[0] for c in centers) / len(centers)
        mean_cy = sum(c[1] for c in centers) / len(centers)

        max_offset = 0
        for cx, cy in centers:
            offset = math.sqrt((cx - mean_cx)**2 + (cy - mean_cy)**2)
            max_offset = max(max_offset, offset)

        if max_offset > self._config.bullseye_concentricity_threshold_px:
            return None

        # Compute final center (weighted by inverse radius)
        total_weight = 0
        weighted_cx = 0
        weighted_cy = 0

        for (cx, cy), r in zip(centers, radii):
            weight = 1.0 / (r + 1)
            weighted_cx += cx * weight
            weighted_cy += cy * weight
            total_weight += weight

        final_cx = int(weighted_cx / total_weight)
        final_cy = int(weighted_cy / total_weight)

        # Compute angle
        angle_deg = _pixel_to_angle(final_cx, frame_width, horizontal_fov_deg)

        # Get depth
        range_mm = None
        if depth_map is not None:
            range_mm = _sample_depth(
                depth_map, final_cx, final_cy,
                self._config.depth_sample_radius
            )

        # Compute confidence
        num_rings = len(valid_contours)
        ring_score = min(1.0, num_rings / 5.0)
        concentricity_score = max(
            0.0,
            1.0 - max_offset / self._config.bullseye_concentricity_threshold_px
        )
        confidence = 0.6 * ring_score + 0.4 * concentricity_score

        return TargetDetection(
            center_x=final_cx,
            center_y=final_cy,
            angle_deg=angle_deg,
            range_mm=range_mm,
            confidence=confidence,
            pattern_type="bullseye"
        )


class CheckerboardPattern(PatternDetector):
    """
    Detects 2x2 checkerboard pattern using corner detection.

    Looks for corners where diagonal quadrants have similar intensity
    (both dark or both light) and adjacent quadrants have opposite intensity.
    Expected pattern: black at top-left/bottom-right, white at top-right/bottom-left.
    """

    def __init__(self, config: TargetDetectorConfig):
        self._config = config

    def detect(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        horizontal_fov_deg: float
    ) -> Optional[TargetDetection]:
        if frame is None or frame.size == 0:
            return None

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Find checkerboard corners
        valid_corners = self._detect_checkerboard_corners(gray)

        if not valid_corners:
            return None

        # Select best corner (highest confidence)
        best = max(valid_corners, key=lambda c: c[2])
        cx, cy, confidence = best

        # Compute angle
        angle_deg = _pixel_to_angle(cx, frame_width, horizontal_fov_deg)

        # Get depth
        range_mm = None
        if depth_map is not None:
            range_mm = _sample_depth(
                depth_map, cx, cy,
                self._config.depth_sample_radius
            )

        return TargetDetection(
            center_x=cx,
            center_y=cy,
            angle_deg=angle_deg,
            range_mm=range_mm,
            confidence=confidence,
            pattern_type="checkerboard"
        )

    def _detect_checkerboard_corners(
        self,
        gray: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """
        Find 2x2 checkerboard corners.

        Returns list of (x, y, confidence) for valid checkerboard centers.
        """
        # Find corners using Shi-Tomasi
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self._config.checker_max_corners,
            qualityLevel=self._config.checker_corner_quality,
            minDistance=self._config.checker_min_distance,
            blockSize=self._config.checker_block_size
        )

        if corners is None:
            return []

        valid_corners = []
        sample_size = self._config.checker_sample_size
        offset = self._config.checker_sample_offset
        margin = offset + sample_size  # Total distance from corner to edge of sample
        h, w = gray.shape

        for corner in corners:
            x, y = int(corner[0][0]), int(corner[0][1])

            # Skip corners too close to edge
            if x < margin or y < margin:
                continue
            if x >= w - margin or y >= h - margin:
                continue

            # Sample 4 quadrants with offset from corner to avoid blur
            # Each sample region is offset pixels away from the corner center
            tl = np.mean(gray[y - margin:y - offset, x - margin:x - offset])
            tr = np.mean(gray[y - margin:y - offset, x + offset:x + margin])
            bl = np.mean(gray[y + offset:y + margin, x - margin:x - offset])
            br = np.mean(gray[y + offset:y + margin, x + offset:x + margin])

            # Check pattern: TL~BR (one diagonal), TR~BL (other diagonal)
            # The two diagonals should have high contrast
            diag1 = (tl + br) / 2  # TL-BR diagonal
            diag2 = (tr + bl) / 2  # TR-BL diagonal

            contrast = abs(diag1 - diag2) / 255.0

            if contrast >= self._config.checker_contrast_threshold:
                # Verify diagonals are similar within themselves
                diag1_diff = abs(tl - br) / 255.0
                diag2_diff = abs(tr - bl) / 255.0

                tolerance = self._config.checker_diagonal_tolerance
                if diag1_diff < tolerance and diag2_diff < tolerance:
                    # Confidence based on contrast and diagonal consistency
                    consistency = 1 - max(diag1_diff, diag2_diff)
                    confidence = contrast * consistency
                    valid_corners.append((x, y, confidence))

        return valid_corners


class BrightnessPattern(PatternDetector):
    """
    Detects bright spots such as phone flashlights.

    Uses thresholding to find saturated bright regions, then locates
    the centroid of the brightest blob as the target center.
    """

    def __init__(self, config: TargetDetectorConfig):
        self._config = config

    def detect(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        horizontal_fov_deg: float
    ) -> Optional[TargetDetection]:
        if frame is None or frame.size == 0:
            return None

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Apply gain reduction to suppress ambient light reflections
        # This simulates low exposure - only very bright sources (flashlight) remain visible
        gain = self._config.brightness_gain
        darkened = (gray.astype(np.float32) * gain).clip(0, 255).astype(np.uint8)

        # Apply blur to reduce noise
        kernel_size = self._config.brightness_blur_kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(darkened, (kernel_size, kernel_size), 0)

        # Threshold to find bright regions
        _, binary = cv2.threshold(
            blurred,
            self._config.brightness_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Find the best bright blob
        best_contour = None
        best_brightness = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self._config.brightness_min_area_px:
                continue
            if area > self._config.brightness_max_area_px:
                continue

            # Compute centroid
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            # Sample brightness at centroid from original grayscale
            brightness = float(gray[cy, cx])

            if brightness > best_brightness:
                best_brightness = brightness
                best_contour = contour

        if best_contour is None:
            return None

        # Get centroid of best contour
        moments = cv2.moments(best_contour)
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        # Compute angle
        angle_deg = _pixel_to_angle(cx, frame_width, horizontal_fov_deg)

        # Get depth
        range_mm = None
        if depth_map is not None:
            range_mm = _sample_depth(
                depth_map, cx, cy,
                self._config.depth_sample_radius
            )

        # Confidence based on brightness (normalized)
        confidence = best_brightness / 255.0

        return TargetDetection(
            center_x=cx,
            center_y=cy,
            angle_deg=angle_deg,
            range_mm=range_mm,
            confidence=confidence,
            pattern_type="brightness"
        )


class TargetDetector:
    """
    Target detector with configurable pattern type.

    Delegates to appropriate pattern detector based on configuration.
    """

    def __init__(self, config: TargetDetectorConfig, horizontal_fov_deg: float):
        """
        Initialize target detector.

        Args:
            config: Detection configuration
            horizontal_fov_deg: Camera horizontal field of view in degrees
        """
        self._config = config
        self._horizontal_fov_deg = horizontal_fov_deg

        # Create pattern detector based on config
        if config.pattern_type == "bullseye":
            self._detector = BullseyePattern(config)
        elif config.pattern_type == "brightness":
            self._detector = BrightnessPattern(config)
        else:  # Default to checkerboard
            self._detector = CheckerboardPattern(config)

    def detect(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> Optional[TargetDetection]:
        """
        Detect target in frame.

        Args:
            frame: BGR or grayscale image from camera
            depth_map: Optional depth map in mm (same size as frame)

        Returns:
            TargetDetection if found, None otherwise
        """
        return self._detector.detect(frame, depth_map, self._horizontal_fov_deg)

    def detect_with_debug(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> Tuple[Optional[TargetDetection], np.ndarray]:
        """
        Detect target and return debug visualization.

        Args:
            frame: Input frame
            depth_map: Optional depth map

        Returns:
            Tuple of (detection, debug_image)
        """
        detection = self.detect(frame, depth_map)

        # Create debug visualization
        if len(frame.shape) == 2:
            debug = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            debug = frame.copy()

        if detection:
            cx, cy = detection.center_x, detection.center_y
            color = (0, 255, 0)  # Green

            cv2.circle(debug, (cx, cy), 10, color, 2)
            cv2.line(debug, (cx - 20, cy), (cx + 20, cy), color, 2)
            cv2.line(debug, (cx, cy - 20), (cx, cy + 20), color, 2)

            # Draw pattern type indicator
            pattern_label = detection.pattern_type.upper()[:6]
            info = f"{pattern_label} Conf: {detection.confidence:.2f}"
            cv2.putText(debug, info, (cx + 15, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            angle_text = f"Angle: {detection.angle_deg:.1f}deg"
            cv2.putText(debug, angle_text, (cx + 15, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if detection.range_mm:
                range_text = f"Range: {detection.range_mm:.0f}mm"
                cv2.putText(debug, range_text, (cx + 15, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return detection, debug

    @property
    def pattern_type(self) -> str:
        """Get current pattern type."""
        return self._config.pattern_type

    @property
    def config(self) -> TargetDetectorConfig:
        """Get current configuration."""
        return self._config


# Helper functions

def _pixel_to_angle(center_x: int, frame_width: int, fov_deg: float) -> float:
    """Convert pixel X coordinate to angle from camera center."""
    normalized_x = (center_x - frame_width / 2) / (frame_width / 2)
    return normalized_x * (fov_deg / 2)


def _sample_depth(
    depth_map: np.ndarray,
    cx: int,
    cy: int,
    radius: int
) -> Optional[float]:
    """Sample depth at center with averaging."""
    h, w = depth_map.shape[:2]

    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius + 1)
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius + 1)

    region = depth_map[y1:y2, x1:x2]

    valid = region[(region > 0) & (region < 65535)]

    if len(valid) == 0:
        return None

    return float(np.median(valid))
