"""
3-pin H-bridge motor driver interface.

Uses IN1, IN2, and ENA pins for motor control:
- IN1=HIGH, IN2=LOW,  ENA=PWM  -> Forward at speed
- IN1=LOW,  IN2=HIGH, ENA=PWM  -> Reverse at speed
- IN1=HIGH, IN2=HIGH, ENA=0    -> Coast (floating)
- IN1=LOW,  IN2=LOW,  ENA=any  -> Brake

Uses lgpio for kernel 6.x compatibility (no root required).
"""

from dataclasses import dataclass
from typing import Optional

try:
    import lgpio
    LGPIO_AVAILABLE = True
except ImportError:
    LGPIO_AVAILABLE = False

# Fallback to RPi.GPIO if lgpio not available
if not LGPIO_AVAILABLE:
    try:
        import RPi.GPIO as GPIO
        GPIO_AVAILABLE = True
    except ImportError:
        GPIO_AVAILABLE = False
else:
    GPIO_AVAILABLE = False

from .gpio_manager import GPIOManager


@dataclass
class MotorDriverConfig:
    """Configuration for a single motor channel (3-pin H-bridge)."""
    in1_pin: int
    in2_pin: int
    ena_pin: int
    pwm_frequency: int = 10000
    inverted: bool = False


class MotorDriver:
    """
    Controls a single motor channel via 3-pin H-bridge interface.

    Hardware interface:
    - IN1 pin: Direction control 1
    - IN2 pin: Direction control 2
    - ENA pin: PWM speed control

    Control logic:
    - Forward:  IN1=HIGH, IN2=LOW,  ENA=PWM (0-100%)
    - Reverse:  IN1=LOW,  IN2=HIGH, ENA=PWM (0-100%)
    - Coast:    IN1=HIGH, IN2=HIGH, ENA=0
    - Brake:    IN1=LOW,  IN2=LOW,  ENA=any
    """

    def __init__(self, config: MotorDriverConfig):
        """
        Initialize motor driver with GPIO.

        Args:
            config: Motor driver configuration
        """
        self._config = config
        self._current_speed: float = 0.0
        self._initialized = False
        self._using_lgpio = False
        self._handle: Optional[int] = None

        if GPIOManager.is_available():
            self._setup_lgpio()
        elif GPIO_AVAILABLE:
            self._setup_rpi_gpio()

    def _setup_lgpio(self) -> None:
        """Configure GPIO pins using lgpio (kernel 6.x compatible)."""
        try:
            self._handle = GPIOManager.get_handle()
            if self._handle is None:
                return

            # Setup IN1 pin (direction control 1)
            lgpio.gpio_claim_output(self._handle, self._config.in1_pin, 1)  # Start HIGH (coast)

            # Setup IN2 pin (direction control 2)
            lgpio.gpio_claim_output(self._handle, self._config.in2_pin, 1)  # Start HIGH (coast)

            # Setup ENA pin with PWM
            lgpio.gpio_claim_output(self._handle, self._config.ena_pin, 0)  # Start with 0% duty
            
            self._initialized = True
            self._using_lgpio = True

        except Exception as e:
            print(f"Warning: lgpio motor setup failed: {e}")
            self._initialized = False

    def _setup_rpi_gpio(self) -> None:
        """Configure GPIO pins using RPi.GPIO (fallback, needs root)."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Setup IN1 pin (direction control 1)
        GPIO.setup(self._config.in1_pin, GPIO.OUT)
        GPIO.output(self._config.in1_pin, GPIO.HIGH)

        # Setup IN2 pin (direction control 2)
        GPIO.setup(self._config.in2_pin, GPIO.OUT)
        GPIO.output(self._config.in2_pin, GPIO.HIGH)

        # Setup ENA pin (PWM speed control)
        GPIO.setup(self._config.ena_pin, GPIO.OUT)
        self._pwm = GPIO.PWM(self._config.ena_pin, self._config.pwm_frequency)
        self._pwm.start(0)

        self._initialized = True

    def set_speed(self, speed: float) -> None:
        """
        Set motor speed.

        Args:
            speed: Speed from -100 to +100
                   Positive = forward, Negative = reverse
                   Zero = coast (motor floats)
        """
        # Clamp speed to valid range
        speed = max(-100.0, min(100.0, speed))

        # Apply inversion if configured
        if self._config.inverted:
            speed = -speed

        self._current_speed = speed

        if not self._initialized:
            return

        if self._using_lgpio:
            self._set_speed_lgpio(speed)
        else:
            self._set_speed_rpi_gpio(speed)

    def _set_speed_lgpio(self, speed: float) -> None:
        """Set speed using lgpio."""
        if self._handle is None:
            return

        h = self._handle

        if speed > 0:
            # Forward: IN1=HIGH, IN2=LOW, ENA=PWM
            lgpio.gpio_write(h, self._config.in1_pin, 1)
            lgpio.gpio_write(h, self._config.in2_pin, 0)
            # PWM duty cycle (0-100 maps to 0-100%)
            lgpio.tx_pwm(h, self._config.ena_pin, self._config.pwm_frequency, speed)
        elif speed < 0:
            # Reverse: IN1=LOW, IN2=HIGH, ENA=PWM
            lgpio.gpio_write(h, self._config.in1_pin, 0)
            lgpio.gpio_write(h, self._config.in2_pin, 1)
            lgpio.tx_pwm(h, self._config.ena_pin, self._config.pwm_frequency, -speed)
        else:
            # Coast: IN1=HIGH, IN2=HIGH, ENA=0
            lgpio.gpio_write(h, self._config.in1_pin, 1)
            lgpio.gpio_write(h, self._config.in2_pin, 1)
            lgpio.tx_pwm(h, self._config.ena_pin, self._config.pwm_frequency, 0)

    def _set_speed_rpi_gpio(self, speed: float) -> None:
        """Set speed using RPi.GPIO."""
        if speed > 0:
            # Forward: IN1=HIGH, IN2=LOW, ENA=PWM
            GPIO.output(self._config.in1_pin, GPIO.HIGH)
            GPIO.output(self._config.in2_pin, GPIO.LOW)
            self._pwm.ChangeDutyCycle(speed)
        elif speed < 0:
            # Reverse: IN1=LOW, IN2=HIGH, ENA=PWM
            GPIO.output(self._config.in1_pin, GPIO.LOW)
            GPIO.output(self._config.in2_pin, GPIO.HIGH)
            self._pwm.ChangeDutyCycle(-speed)
        else:
            # Coast: IN1=HIGH, IN2=HIGH, ENA=0
            GPIO.output(self._config.in1_pin, GPIO.HIGH)
            GPIO.output(self._config.in2_pin, GPIO.HIGH)
            self._pwm.ChangeDutyCycle(0)

    def get_speed(self) -> float:
        """Get current speed setting."""
        return self._current_speed

    def stop(self) -> None:
        """Stop motor (coast - no active braking)."""
        self.set_speed(0)

    def brake(self) -> None:
        """
        Active brake by setting both IN1 and IN2 low.

        This shorts the motor terminals for rapid deceleration.
        """
        self._current_speed = 0.0

        if not self._initialized:
            return

        if self._using_lgpio:
            if self._handle is not None:
                lgpio.gpio_write(self._handle, self._config.in1_pin, 0)
                lgpio.gpio_write(self._handle, self._config.in2_pin, 0)
                lgpio.tx_pwm(self._handle, self._config.ena_pin, self._config.pwm_frequency, 0)
        else:
            GPIO.output(self._config.in1_pin, GPIO.LOW)
            GPIO.output(self._config.in2_pin, GPIO.LOW)
            self._pwm.ChangeDutyCycle(0)

    def cleanup(self) -> None:
        """Release GPIO resources."""
        if self._initialized:
            if self._using_lgpio:
                if self._handle is not None:
                    # Stop PWM and set pins low
                    try:
                        lgpio.tx_pwm(self._handle, self._config.ena_pin, 0, 0)
                        lgpio.gpio_write(self._handle, self._config.in1_pin, 0)
                        lgpio.gpio_write(self._handle, self._config.in2_pin, 0)
                        lgpio.gpio_free(self._handle, self._config.in1_pin)
                        lgpio.gpio_free(self._handle, self._config.in2_pin)
                        lgpio.gpio_free(self._handle, self._config.ena_pin)
                    except Exception:
                        pass
                GPIOManager.release_handle()
                self._handle = None
            else:
                if hasattr(self, '_pwm') and self._pwm is not None:
                    self._pwm.stop()
            self._initialized = False


class DualMotorDriver:
    """
    Controls both motor channels via 3-pin H-bridge interfaces.

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
        """Stop both motors (coast)."""
        self._left.stop()
        self._right.stop()

    def brake(self) -> None:
        """Active brake both motors."""
        self._left.brake()
        self._right.brake()

    def cleanup(self) -> None:
        """Release all GPIO resources."""
        self._left.cleanup()
        self._right.cleanup()
