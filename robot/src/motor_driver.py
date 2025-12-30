"""
Cytron MDD10A motor driver interface.

The MDD10A is a dual channel 10A motor driver that uses:
- PWM pin: Speed control (0-100% duty cycle)
- DIR pin: Direction (HIGH=forward, LOW=reverse)
"""

from dataclasses import dataclass
from typing import Optional

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


@dataclass
class MotorDriverConfig:
    """Configuration for a single motor channel."""
    pwm_pin: int
    dir_pin: int
    pwm_frequency: int = 20000
    inverted: bool = False


class MotorDriver:
    """
    Controls a single motor channel on Cytron MDD10A.

    Hardware interface:
    - PWM pin: Speed control (0-100% duty cycle)
    - DIR pin: Direction (HIGH=forward, LOW=reverse)
    """

    def __init__(self, config: MotorDriverConfig):
        """
        Initialize motor driver with GPIO.

        Args:
            config: Motor driver configuration
        """
        self._config = config
        self._pwm: Optional[object] = None
        self._current_speed: float = 0.0
        self._initialized = False

        if GPIO_AVAILABLE:
            self._setup_gpio()

    def _setup_gpio(self) -> None:
        """Configure GPIO pins for motor control."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Setup direction pin
        GPIO.setup(self._config.dir_pin, GPIO.OUT)
        GPIO.output(self._config.dir_pin, GPIO.LOW)

        # Setup PWM pin
        GPIO.setup(self._config.pwm_pin, GPIO.OUT)
        self._pwm = GPIO.PWM(self._config.pwm_pin, self._config.pwm_frequency)
        self._pwm.start(0)

        self._initialized = True

    def set_speed(self, speed: float) -> None:
        """
        Set motor speed.

        Args:
            speed: Speed from -100 to +100
                   Positive = forward, Negative = reverse
        """
        # Clamp speed to valid range
        speed = max(-100.0, min(100.0, speed))

        # Apply inversion if configured
        if self._config.inverted:
            speed = -speed

        self._current_speed = speed

        if not self._initialized:
            return

        # Set direction
        if speed >= 0:
            GPIO.output(self._config.dir_pin, GPIO.HIGH)
            duty_cycle = speed
        else:
            GPIO.output(self._config.dir_pin, GPIO.LOW)
            duty_cycle = -speed

        # Set PWM duty cycle
        self._pwm.ChangeDutyCycle(duty_cycle)

    def get_speed(self) -> float:
        """Get current speed setting."""
        return self._current_speed

    def stop(self) -> None:
        """Stop motor (coast - no active braking)."""
        self.set_speed(0)

    def brake(self) -> None:
        """
        Active brake by setting both motor terminals low.
        Note: MDD10A may not support true active braking via software.
        This implementation just stops the motor.
        """
        self.stop()

    def cleanup(self) -> None:
        """Release GPIO resources."""
        if self._initialized and self._pwm is not None:
            self._pwm.stop()
            self._initialized = False


class DualMotorDriver:
    """
    Controls both motor channels on Cytron MDD10A.

    Provides convenience methods for differential drive control.
    """

    def __init__(
        self,
        left_config: MotorDriverConfig,
        right_config: MotorDriverConfig
    ):
        """
        Initialize dual motor driver.

        Args:
            left_config: Configuration for left motor
            right_config: Configuration for right motor
        """
        self._left = MotorDriver(left_config)
        self._right = MotorDriver(right_config)

    @property
    def left(self) -> MotorDriver:
        """Get left motor driver."""
        return self._left

    @property
    def right(self) -> MotorDriver:
        """Get right motor driver."""
        return self._right

    def set_speeds(self, left_speed: float, right_speed: float) -> None:
        """
        Set both motor speeds.

        Args:
            left_speed: Left motor speed (-100 to +100)
            right_speed: Right motor speed (-100 to +100)
        """
        self._left.set_speed(left_speed)
        self._right.set_speed(right_speed)

    def get_speeds(self) -> tuple:
        """
        Get current speed settings.

        Returns:
            Tuple of (left_speed, right_speed)
        """
        return (self._left.get_speed(), self._right.get_speed())

    def arcade_drive(self, throttle: float, turn: float) -> None:
        """
        Arcade-style drive control.

        Args:
            throttle: Forward/backward (-100 to +100)
            turn: Left/right turn (-100 to +100, positive = turn right)
        """
        # Mix throttle and turn
        left_speed = throttle + turn
        right_speed = throttle - turn

        # Scale down if either exceeds limits
        max_magnitude = max(abs(left_speed), abs(right_speed))
        if max_magnitude > 100:
            scale = 100.0 / max_magnitude
            left_speed *= scale
            right_speed *= scale

        self.set_speeds(left_speed, right_speed)

    def differential_drive(self, linear: float, angular: float) -> None:
        """
        Differential drive control (same as arcade_drive).

        Args:
            linear: Forward/backward speed (-100 to +100)
            angular: Turning rate (-100 to +100, positive = clockwise/right)
        """
        self.arcade_drive(linear, angular)

    def stop(self) -> None:
        """Stop both motors."""
        self._left.stop()
        self._right.stop()

    def cleanup(self) -> None:
        """Release all GPIO resources."""
        self._left.cleanup()
        self._right.cleanup()
        if GPIO_AVAILABLE:
            # Don't call GPIO.cleanup() as it affects all pins
            # Individual pins are released by stopping PWM
            pass
