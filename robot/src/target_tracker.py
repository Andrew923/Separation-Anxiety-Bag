"""
Target tracker with sensor fusion for UWB and visual detection.

Fuses UWB ranging and visual target detection to provide robust
target tracking with smooth transitions between sources.
"""

from dataclasses import dataclass
from typing import Optional
import time

from .target_detector import TargetDetection
from .uwb_tracker import EMAFilter, AngleFilter


@dataclass
class TargetTrackerConfig:
    """Configuration for target tracking and sensor fusion."""
    # EMA filter settings (shared between UWB and visual)
    ema_alpha: float = 0.25               # Range smoothing factor (0-1)
    outlier_threshold_mm: float = 200.0   # Reject range changes larger than this
    angle_ema_alpha: float = 0.3          # Angle smoothing factor

    # Visual preference settings
    visual_range_threshold_mm: float = 3000.0   # Prefer visual when closer than this
    visual_confidence_threshold: float = 0.5    # Minimum detection confidence to use

    # Timeout settings
    visual_timeout_ms: float = 500.0      # Consider visual stale after this
    uwb_timeout_ms: float = 500.0         # Consider UWB stale after this

    # Minimum samples before output is valid
    min_samples: int = 3


@dataclass
class TargetState:
    """Current target state from sensor fusion."""
    angle_deg: Optional[float] = None     # Filtered angle to target
    range_mm: Optional[float] = None      # Filtered distance to target
    source: str = "none"                  # Active source: "uwb", "visual", or "none"
    uwb_valid: bool = False               # Whether UWB data is current
    visual_valid: bool = False            # Whether visual data is current
    confidence: float = 0.0               # Overall tracking confidence


class TargetTracker:
    """
    Fuses UWB and visual target tracking with shared EMA filtering.

    Priority logic:
    1. If visual detection available AND (close range OR high confidence): use visual
    2. Else if UWB available: use UWB
    3. Else: return last filtered values or None

    Both sources feed into the same EMA filters, ensuring smooth transitions
    when switching between sources. This prevents jumps in the output when
    the robot moves from far (UWB only) to close (visual available).

    Usage:
        tracker = TargetTracker(config)

        # In control loop:
        uwb_angle, uwb_range = get_uwb_data()  # From triangulator
        visual = target_detector.detect(frame, depth)

        state = tracker.update(uwb_angle, uwb_range, visual)

        # Use state.angle_deg, state.range_mm for navigation
    """

    def __init__(self, config: TargetTrackerConfig):
        """
        Initialize target tracker.

        Args:
            config: Tracker configuration
        """
        self._config = config

        # Shared EMA filters - both sources feed into these
        self._range_filter = EMAFilter(
            alpha=config.ema_alpha,
            outlier_threshold=config.outlier_threshold_mm
        )
        self._angle_filter = AngleFilter(alpha=config.angle_ema_alpha)

        # Timestamps for timeout detection
        self._last_uwb_time: float = 0.0
        self._last_visual_time: float = 0.0

        # Track which source is active
        self._active_source: str = "none"

        # Sample counter for minimum samples check
        self._sample_count: int = 0

    def update(
        self,
        uwb_angle: Optional[float],
        uwb_range: Optional[float],
        visual_detection: Optional[TargetDetection]
    ) -> TargetState:
        """
        Update tracker with new measurements from both sources.

        Args:
            uwb_angle: Angle from UWB triangulation (degrees, None if invalid)
            uwb_range: Range from UWB triangulation (mm, None if invalid)
            visual_detection: Target detection result (None if not detected)

        Returns:
            TargetState with fused tracking data
        """
        current_time = time.time()

        # Update timestamps if we have valid data
        uwb_valid = uwb_angle is not None and uwb_range is not None
        visual_valid = (
            visual_detection is not None and
            visual_detection.confidence >= self._config.visual_confidence_threshold
        )

        if uwb_valid:
            self._last_uwb_time = current_time
        if visual_valid:
            self._last_visual_time = current_time

        # Check for stale data
        uwb_timeout = self._config.uwb_timeout_ms / 1000.0
        visual_timeout = self._config.visual_timeout_ms / 1000.0

        uwb_current = uwb_valid or (current_time - self._last_uwb_time < uwb_timeout)
        visual_current = visual_valid or (current_time - self._last_visual_time < visual_timeout)

        # Select source based on priority
        source = self._select_source(
            uwb_valid, uwb_range,
            visual_valid, visual_detection
        )

        # Get measurements from selected source
        angle: Optional[float] = None
        range_mm: Optional[float] = None
        confidence: float = 0.0

        if source == "visual" and visual_detection:
            angle = visual_detection.angle_deg
            range_mm = visual_detection.range_mm
            confidence = visual_detection.confidence
        elif source == "uwb" and uwb_valid:
            angle = uwb_angle
            range_mm = uwb_range
            confidence = 0.7  # UWB has moderate confidence

        # Update filters with new measurement
        filtered_angle: Optional[float] = None
        filtered_range: Optional[float] = None

        if angle is not None:
            filtered_angle = self._angle_filter.update(angle)
            self._sample_count += 1

        if range_mm is not None:
            filtered_range = self._range_filter.update(range_mm)

        # If no new measurement, use last filtered values
        if filtered_angle is None:
            filtered_angle = self._angle_filter.get_value()
        if filtered_range is None:
            filtered_range = self._range_filter.get_value()

        # Check minimum samples requirement
        if self._sample_count < self._config.min_samples:
            filtered_angle = None
            filtered_range = None

        self._active_source = source

        return TargetState(
            angle_deg=filtered_angle,
            range_mm=filtered_range,
            source=source,
            uwb_valid=uwb_current,
            visual_valid=visual_current,
            confidence=confidence
        )

    def _select_source(
        self,
        uwb_valid: bool,
        uwb_range: Optional[float],
        visual_valid: bool,
        visual_detection: Optional[TargetDetection]
    ) -> str:
        """
        Select which source to use based on priority rules.

        Priority:
        1. Visual if close range and high confidence
        2. UWB if available
        3. Visual as fallback (even lower confidence)
        4. None if nothing available

        Args:
            uwb_valid: Whether UWB data is valid
            uwb_range: UWB range measurement
            visual_valid: Whether visual detection is valid
            visual_detection: Visual detection result

        Returns:
            Source name: "visual", "uwb", or "none"
        """
        # Check if visual should be preferred
        if visual_valid and visual_detection:
            visual_range = visual_detection.range_mm
            visual_conf = visual_detection.confidence

            # Use visual if:
            # 1. Close range (visual more accurate up close)
            # 2. High confidence detection
            if visual_range and visual_range < self._config.visual_range_threshold_mm:
                return "visual"
            if visual_conf > 0.8:
                return "visual"

        # Use UWB if available
        if uwb_valid:
            return "uwb"

        # Fallback to visual even at lower confidence
        if visual_detection is not None:
            return "visual"

        return "none"

    def reset(self) -> None:
        """
        Reset tracker state.

        Call this when target is lost for extended period
        or when restarting tracking.
        """
        self._range_filter.reset()
        self._angle_filter.reset()
        self._last_uwb_time = 0.0
        self._last_visual_time = 0.0
        self._active_source = "none"
        self._sample_count = 0

    def get_filter_stats(self) -> dict:
        """
        Get debugging information about filter state.

        Returns:
            Dict with filter statistics
        """
        return {
            'active_source': self._active_source,
            'sample_count': self._sample_count,
            'range_filter_value': self._range_filter.get_value(),
            'angle_filter_value': self._angle_filter.get_value(),
            'range_filter_samples': self._range_filter.sample_count,
        }

    @property
    def active_source(self) -> str:
        """Get currently active tracking source."""
        return self._active_source

    @property
    def is_tracking(self) -> bool:
        """Check if actively tracking a target."""
        return self._active_source != "none"
