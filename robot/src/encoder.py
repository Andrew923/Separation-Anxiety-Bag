"""
Quadrature encoder interface using libgpiod edge detection.

FIT0186 encoder specifications:
- 16 PPR on motor shaft
- 43.8:1 gearbox
- ~700 CPR at output shaft (16 * 43.8 = 700.8)

Wiring notes:
- Encoder VCC to 5V, GND to Pi GND
- Encoder A/B outputs connect directly to Pi GPIO (no voltage divider needed)
- Internal pull-ups are enabled - works with open-collector encoder outputs
- The encoder pulls LOW when active, Pi pull-up provides 3.3V HIGH

Uses libgpiod for reliable edge detection (lgpio callbacks don't work on kernel 6.x).
"""

import sys
# Ensure system gpiod is available
sys.path.insert(0, '/usr/lib/python3/dist-packages')

from dataclasses import dataclass
from typing import Optional, Tuple
import threading
import time

try:
    import gpiod
    GPIOD_AVAILABLE = True
except ImportError:
    GPIOD_AVAILABLE = False

GPIOCHIP = 'gpiochip0'


@dataclass
class EncoderConfig:
    """Configuration for a quadrature encoder."""
    channel_a_pin: int
    channel_b_pin: int
    counts_per_revolution: int = 700
    inverted: bool = False


class QuadratureEncoder:
    """
    Reads quadrature encoder using gpiod edge detection in a background thread.

    Counts pulses on rising edge of channel A.
    Direction determined by state of channel B at rising edge.
    """

    def __init__(self, config: EncoderConfig):
        """
        Initialize encoder with edge detection thread.

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
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # gpiod objects
        self._chip: Optional[gpiod.Chip] = None
        self._line_a: Optional[gpiod.Line] = None
        self._line_b: Optional[gpiod.Line] = None

        if GPIOD_AVAILABLE:
            self._setup_gpiod()

    def _setup_gpiod(self) -> None:
        """Configure GPIO pins using libgpiod."""
        try:
            self._chip = gpiod.Chip(GPIOCHIP)
            
            # Get lines
            self._line_a = self._chip.get_line(self._config.channel_a_pin)
            self._line_b = self._chip.get_line(self._config.channel_b_pin)
            
            # Request line A for rising edge events with pull-up
            self._line_a.request(
                consumer="encoder_a",
                type=gpiod.LINE_REQ_EV_RISING_EDGE,
                flags=gpiod.LINE_REQ_FLAG_BIAS_PULL_UP
            )
            
            # Request line B as input with pull-up
            self._line_b.request(
                consumer="encoder_b",
                type=gpiod.LINE_REQ_DIR_IN,
                flags=gpiod.LINE_REQ_FLAG_BIAS_PULL_UP
            )
            
            self._initialized = True
            
            # Start edge detection thread
            self._running = True
            self._thread = threading.Thread(target=self._edge_loop, daemon=True)
            self._thread.start()
            
        except Exception as e:
            print(f"Warning: gpiod setup failed for encoder on pin {self._config.channel_a_pin}: {e}")
            self._cleanup_gpiod()
            self._initialized = False

    def _edge_loop(self) -> None:
        """Background thread that waits for edge events."""
        inverted = self._config.inverted
        
        while self._running:
            try:
                # Wait for rising edge event with 50ms timeout
                if self._line_a.event_wait(nsec=50_000_000):
                    # Read and discard the event (clears the event queue)
                    self._line_a.event_read()
                    
                    # Read B state for direction
                    b_state = self._line_b.get_value()
                    
                    # Direction based on B state at A rising edge
                    # B LOW = forward, B HIGH = reverse
                    if b_state == 0:
                        direction = 1
                    else:
                        direction = -1
                    
                    if inverted:
                        direction = -direction
                    
                    with self._lock:
                        self._count += direction
                        
            except Exception:
                # Handle case where lines are released during shutdown
                if not self._running:
                    break

    def _cleanup_gpiod(self) -> None:
        """Release gpiod resources."""
        if self._line_a is not None:
            try:
                self._line_a.release()
            except Exception:
                pass
            self._line_a = None
            
        if self._line_b is not None:
            try:
                self._line_b.release()
            except Exception:
                pass
            self._line_b = None
            
        if self._chip is not None:
            try:
                self._chip.close()
            except Exception:
                pass
            self._chip = None

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
        """Stop edge detection thread and cleanup GPIO."""
        # Stop the thread
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.2)
            self._thread = None

        self._cleanup_gpiod()
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if encoder was successfully initialized."""
        return self._initialized


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

    @property
    def is_initialized(self) -> bool:
        """Check if both encoders were successfully initialized."""
        return self._left.is_initialized and self._right.is_initialized
