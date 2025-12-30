"""
Quadrature encoder interface using GPIO interrupts.

FIT0186 encoder specifications:
- 16 PPR on motor shaft
- 43.8:1 gearbox
- ~700 CPR at output shaft (16 * 43.8 = 700.8)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import threading
import time

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


@dataclass
class EncoderConfig:
    """Configuration for a quadrature encoder."""
    channel_a_pin: int
    channel_b_pin: int
    counts_per_revolution: int = 700
    inverted: bool = False


class QuadratureEncoder:
    """
    Reads quadrature encoder using GPIO interrupts.

    Uses both edges of channel A for 2x resolution.
    Direction determined by state of channel B at each A edge.
    """

    def __init__(self, config: EncoderConfig):
        """
        Initialize encoder with interrupt handlers.

        Args:
            config: Encoder configuration
        """
        self._config = config
        self._count: int = 0
        self._lock = threading.Lock()
        self._last_time: float = time.time()
        self._last_count: int = 0
        self._velocity: float = 0.0
        self._initialized = False

        if GPIO_AVAILABLE:
            self._setup_gpio()

    def _setup_gpio(self) -> None:
        """Configure GPIO pins and interrupt handlers."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Setup encoder pins as inputs with pull-up resistors
        GPIO.setup(self._config.channel_a_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self._config.channel_b_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # Add interrupt on channel A (both edges for 2x resolution)
        GPIO.add_event_detect(
            self._config.channel_a_pin,
            GPIO.BOTH,
            callback=self._on_channel_a_change
        )

        self._initialized = True

    def _on_channel_a_change(self, channel: int) -> None:
        """
        Interrupt handler for channel A edges.

        Determines direction based on channel B state.
        """
        a_state = GPIO.input(self._config.channel_a_pin)
        b_state = GPIO.input(self._config.channel_b_pin)

        # Quadrature decoding logic
        # A rising + B low = forward, A rising + B high = reverse
        # A falling + B high = forward, A falling + B low = reverse
        if a_state == b_state:
            direction = -1
        else:
            direction = 1

        if self._config.inverted:
            direction = -direction

        with self._lock:
            self._count += direction

    def get_count(self) -> int:
        """
        Get current encoder count (thread-safe).

        Returns:
            Cumulative encoder count
        """
        with self._lock:
            return self._count

    def get_velocity(self) -> float:
        """
        Get current velocity in counts per second.

        Computes velocity based on count change since last call.

        Returns:
            Velocity in counts per second
        """
        current_time = time.time()
        current_count = self.get_count()

        with self._lock:
            dt = current_time - self._last_time
            if dt > 0:
                dcount = current_count - self._last_count
                self._velocity = dcount / dt

            self._last_time = current_time
            self._last_count = current_count

            return self._velocity

    def get_rpm(self) -> float:
        """
        Get velocity in RPM.

        Returns:
            Rotational velocity in revolutions per minute
        """
        counts_per_sec = self.get_velocity()
        revs_per_sec = counts_per_sec / self._config.counts_per_revolution
        return revs_per_sec * 60.0

    def get_revolutions(self) -> float:
        """
        Get total revolutions since reset.

        Returns:
            Number of revolutions
        """
        return self.get_count() / self._config.counts_per_revolution

    def reset(self) -> None:
        """Reset encoder count to zero."""
        with self._lock:
            self._count = 0
            self._last_count = 0
            self._velocity = 0.0
            self._last_time = time.time()

    def cleanup(self) -> None:
        """Remove interrupt handlers and cleanup GPIO."""
        if self._initialized and GPIO_AVAILABLE:
            try:
                GPIO.remove_event_detect(self._config.channel_a_pin)
            except Exception:
                pass
            self._initialized = False


class DualEncoders:
    """Manages both wheel encoders."""

    def __init__(
        self,
        left_config: EncoderConfig,
        right_config: EncoderConfig
    ):
        """
        Initialize both encoders.

        Args:
            left_config: Configuration for left encoder
            right_config: Configuration for right encoder
        """
        self._left = QuadratureEncoder(left_config)
        self._right = QuadratureEncoder(right_config)

    @property
    def left(self) -> QuadratureEncoder:
        """Get left encoder."""
        return self._left

    @property
    def right(self) -> QuadratureEncoder:
        """Get right encoder."""
        return self._right

    def get_counts(self) -> Tuple[int, int]:
        """
        Get counts from both encoders.

        Returns:
            Tuple of (left_count, right_count)
        """
        return (self._left.get_count(), self._right.get_count())

    def get_velocities(self) -> Tuple[float, float]:
        """
        Get velocities from both encoders in counts/sec.

        Returns:
            Tuple of (left_velocity, right_velocity)
        """
        return (self._left.get_velocity(), self._right.get_velocity())

    def get_rpms(self) -> Tuple[float, float]:
        """
        Get velocities in RPM.

        Returns:
            Tuple of (left_rpm, right_rpm)
        """
        return (self._left.get_rpm(), self._right.get_rpm())

    def get_revolutions(self) -> Tuple[float, float]:
        """
        Get total revolutions.

        Returns:
            Tuple of (left_revolutions, right_revolutions)
        """
        return (self._left.get_revolutions(), self._right.get_revolutions())

    def reset(self) -> None:
        """Reset both encoder counts."""
        self._left.reset()
        self._right.reset()

    def cleanup(self) -> None:
        """Cleanup both encoders."""
        self._left.cleanup()
        self._right.cleanup()
