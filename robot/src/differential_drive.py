"""
Differential drive controller with PID speed control.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import math
import time
import threading

from .motor_driver import DualMotorDriver
from .encoder import DualEncoders


@dataclass
class PIDConfig:
    """PID controller configuration."""
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 100.0
    output_limit: float = 100.0


@dataclass
class DriveConfig:
    """Differential drive configuration."""
    wheel_diameter_mm: float = 52.0
    wheel_base_mm: float = 200.0
    encoder_cpr: int = 700
    max_rpm: float = 251.0
    deadband: float = 0.0
    left_pid: PIDConfig = field(default_factory=PIDConfig)
    right_pid: PIDConfig = field(default_factory=PIDConfig)
    heading_pid: PIDConfig = field(default_factory=PIDConfig)
    control_rate_hz: float = 100.0
    heading_correction_enabled: bool = True


class PIDController:
    """
    Generic PID controller with anti-windup and derivative filtering.
    """

    def __init__(self, config: PIDConfig):
        """
        Initialize PID with configuration.

        Args:
            config: PID configuration parameters
        """
        self._config = config
        self._integral: float = 0.0
        self._last_error: float = 0.0
        self._last_time: Optional[float] = None
        self._derivative_filter: float = 0.0
        self._alpha: float = 0.1  # Low-pass filter coefficient for derivative

    def compute(self, setpoint: float, measured: float) -> float:
        """
        Compute PID output.

        Args:
            setpoint: Desired value
            measured: Current measured value

        Returns:
            Control output (clamped to output_limit)
        """
        current_time = time.time()

        if self._last_time is None:
            self._last_time = current_time
            self._last_error = setpoint - measured
            return 0.0

        dt = current_time - self._last_time
        if dt <= 0:
            return 0.0

        # Error calculation
        error = setpoint - measured

        # Proportional term
        p_term = self._config.kp * error

        # Integral term with anti-windup
        self._integral += error * dt
        self._integral = max(
            -self._config.integral_limit,
            min(self._config.integral_limit, self._integral)
        )
        i_term = self._config.ki * self._integral

        # Derivative term with low-pass filter (reduces noise)
        derivative = (error - self._last_error) / dt
        self._derivative_filter = (
            self._alpha * derivative +
            (1 - self._alpha) * self._derivative_filter
        )
        d_term = self._config.kd * self._derivative_filter

        # Total output
        output = p_term + i_term + d_term

        # Clamp output
        output = max(
            -self._config.output_limit,
            min(self._config.output_limit, output)
        )

        # Anti-windup: reduce integral accumulation when saturated
        if abs(output) >= self._config.output_limit and error * self._integral > 0:
            self._integral -= error * dt * 0.5

        # Update state
        self._last_error = error
        self._last_time = current_time

        return output

    def reset(self) -> None:
        """Reset integral and derivative state."""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = None
        self._derivative_filter = 0.0

    def update_gains(self, kp: float, ki: float, kd: float) -> None:
        """
        Update PID gains at runtime.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        self._config.kp = kp
        self._config.ki = ki
        self._config.kd = kd

    def get_gains(self) -> Tuple[float, float, float]:
        """Get current PID gains."""
        return (self._config.kp, self._config.ki, self._config.kd)


class DifferentialDriveController:
    """
    Closed-loop differential drive controller.

    Converts velocity commands to motor PWM with encoder feedback.
    Runs PID control loop in a separate thread.
    """

    def __init__(
        self,
        config: DriveConfig,
        motors: DualMotorDriver,
        encoders: DualEncoders
    ):
        """
        Initialize differential drive controller.

        Args:
            config: Drive configuration
            motors: Motor driver instance
            encoders: Encoder reader instance
        """
        self._config = config
        self._motors = motors
        self._encoders = encoders

        # PID controllers for each wheel
        self._left_pid = PIDController(config.left_pid)
        self._right_pid = PIDController(config.right_pid)

        # Heading PID for straight-line correction
        self._heading_pid = PIDController(config.heading_pid)

        # Setpoints (in RPM)
        self._left_setpoint: float = 0.0
        self._right_setpoint: float = 0.0
        self._setpoint_lock = threading.Lock()

        # Measured speeds (cached from control loop to avoid race conditions)
        self._left_actual: float = 0.0
        self._right_actual: float = 0.0
        self._actual_lock = threading.Lock()

        # Heading tracking state
        self._target_heading: float = 0.0  # radians
        self._current_heading: float = 0.0  # radians
        self._last_left_count: int = 0
        self._last_right_count: int = 0
        self._heading_initialized: bool = False

        # Control loop state
        self._control_thread: Optional[threading.Thread] = None
        self._running: bool = False

        # Precompute constants
        self._wheel_circumference_mm = math.pi * config.wheel_diameter_mm
        self._mm_per_count = self._wheel_circumference_mm / config.encoder_cpr

    def _velocity_to_wheel_rpms(
        self,
        linear_mm_s: float,
        angular_deg_s: float
    ) -> Tuple[float, float]:
        """
        Convert linear/angular velocity to wheel RPMs.

        Args:
            linear_mm_s: Forward velocity in mm/s
            angular_deg_s: Angular velocity in deg/s (positive = clockwise)

        Returns:
            Tuple of (left_rpm, right_rpm)
        """
        angular_rad_s = math.radians(angular_deg_s)

        # Wheel linear velocities
        # v_left = v_linear - omega * L/2
        # v_right = v_linear + omega * L/2
        v_left_mm_s = linear_mm_s - (angular_rad_s * self._config.wheel_base_mm / 2)
        v_right_mm_s = linear_mm_s + (angular_rad_s * self._config.wheel_base_mm / 2)

        # Convert to RPM
        left_rpm = (v_left_mm_s / self._wheel_circumference_mm) * 60
        right_rpm = (v_right_mm_s / self._wheel_circumference_mm) * 60

        return (left_rpm, right_rpm)

    def _wheel_rpms_to_velocity(
        self,
        left_rpm: float,
        right_rpm: float
    ) -> Tuple[float, float]:
        """
        Convert wheel RPMs to linear/angular velocity.

        Args:
            left_rpm: Left wheel RPM
            right_rpm: Right wheel RPM

        Returns:
            Tuple of (linear_mm_s, angular_deg_s)
        """
        # Wheel linear velocities
        v_left_mm_s = (left_rpm / 60) * self._wheel_circumference_mm
        v_right_mm_s = (right_rpm / 60) * self._wheel_circumference_mm

        # Robot velocities
        linear_mm_s = (v_left_mm_s + v_right_mm_s) / 2
        angular_rad_s = (v_right_mm_s - v_left_mm_s) / self._config.wheel_base_mm
        angular_deg_s = math.degrees(angular_rad_s)

        return (linear_mm_s, angular_deg_s)

    def set_velocity(self, linear_mm_s: float, angular_deg_s: float) -> None:
        """
        Set desired robot velocity.

        Args:
            linear_mm_s: Forward velocity in mm/s
            angular_deg_s: Angular velocity in deg/s (positive = clockwise)
        """
        left_rpm, right_rpm = self._velocity_to_wheel_rpms(linear_mm_s, angular_deg_s)
        self.set_wheel_speeds(left_rpm, right_rpm)

    def set_wheel_speeds(self, left_rpm: float, right_rpm: float) -> None:
        """
        Set desired wheel speeds directly in RPM.

        Args:
            left_rpm: Left wheel target RPM
            right_rpm: Right wheel target RPM
        """
        # Clamp to max RPM
        left_rpm = max(-self._config.max_rpm, min(self._config.max_rpm, left_rpm))
        right_rpm = max(-self._config.max_rpm, min(self._config.max_rpm, right_rpm))

        with self._setpoint_lock:
            self._left_setpoint = left_rpm
            self._right_setpoint = right_rpm

    def get_setpoints(self) -> Tuple[float, float]:
        """
        Get current wheel speed setpoints.

        Returns:
            Tuple of (left_rpm, right_rpm)
        """
        with self._setpoint_lock:
            return (self._left_setpoint, self._right_setpoint)

    def get_actual_wheel_speeds(self) -> Tuple[float, float]:
        """
        Get actual wheel speeds (cached from control loop).

        Returns:
            Tuple of (left_rpm, right_rpm)
        """
        with self._actual_lock:
            return (self._left_actual, self._right_actual)

    def get_actual_velocity(self) -> Tuple[float, float]:
        """
        Get actual robot velocity (cached from control loop).

        Returns:
            Tuple of (linear_mm_s, angular_deg_s)
        """
        left_rpm, right_rpm = self.get_actual_wheel_speeds()
        return self._wheel_rpms_to_velocity(left_rpm, right_rpm)

    def start(self) -> None:
        """Start the control loop thread."""
        if self._running:
            return

        self._running = True
        self._left_pid.reset()
        self._right_pid.reset()
        self._heading_pid.reset()

        # Reset heading tracking
        self._heading_initialized = False
        self._current_heading = 0.0
        self._target_heading = 0.0

        self._control_thread = threading.Thread(
            target=self._control_loop,
            daemon=True
        )
        self._control_thread.start()

    def stop(self) -> None:
        """Stop the control loop and motors."""
        self._running = False

        if self._control_thread is not None:
            self._control_thread.join(timeout=0.5)
            self._control_thread = None

        # Stop motors
        self._motors.stop()

        # Reset PIDs
        self._left_pid.reset()
        self._right_pid.reset()
        self._heading_pid.reset()

        # Reset setpoints
        with self._setpoint_lock:
            self._left_setpoint = 0.0
            self._right_setpoint = 0.0

    def _apply_deadband(self, output: float) -> float:
        """Apply deadband compensation to motor output."""
        deadband = self._config.deadband
        if abs(output) < 1.0:
            return 0.0
        
        if output > 0:
            return min(100.0, output + deadband)
        else:
            return max(-100.0, output - deadband)

    def _control_loop(self) -> None:
        """Internal PID control loop (runs in thread)."""
        period = 1.0 / self._config.control_rate_hz

        while self._running:
            loop_start = time.time()

            # Get setpoints
            with self._setpoint_lock:
                left_setpoint = self._left_setpoint
                right_setpoint = self._right_setpoint

            # Get actual speeds
            left_actual, right_actual = self._encoders.get_rpms()

            # Cache for external readers (avoids race condition with get_actual_wheel_speeds)
            with self._actual_lock:
                self._left_actual = left_actual
                self._right_actual = right_actual

            # Update heading tracking from encoder counts
            left_count, right_count = self._encoders.get_counts()
            if not self._heading_initialized:
                self._last_left_count = left_count
                self._last_right_count = right_count
                self._heading_initialized = True
            else:
                # Compute heading change from encoder differences
                delta_left = left_count - self._last_left_count
                delta_right = right_count - self._last_right_count
                self._last_left_count = left_count
                self._last_right_count = right_count

                left_dist = delta_left * self._mm_per_count
                right_dist = delta_right * self._mm_per_count
                delta_heading = (right_dist - left_dist) / self._config.wheel_base_mm
                self._current_heading += delta_heading

            # Apply velocity deadzone to filter encoder noise
            VELOCITY_DEADZONE_RPM = 5.0
            if abs(left_actual) < VELOCITY_DEADZONE_RPM:
                left_actual = 0.0
            if abs(right_actual) < VELOCITY_DEADZONE_RPM:
                right_actual = 0.0

            # Apply heading correction to setpoints when driving straight
            # (must be done BEFORE velocity PID, not after)
            if (self._config.heading_correction_enabled and
                abs(left_setpoint - right_setpoint) < 1.0 and
                abs(left_setpoint) > 1.0):
                # Driving straight - compute heading correction
                heading_error = self._target_heading - self._current_heading
                heading_correction = self._heading_pid.compute(0.0, -math.degrees(heading_error))

                # Apply correction to setpoints: positive correction turns left
                # (slow left setpoint, increase right setpoint)
                left_setpoint -= heading_correction
                right_setpoint += heading_correction
            else:
                # Turning or stopped - update target heading to current
                self._target_heading = self._current_heading
                self._heading_pid.reset()

            # Reset PID integral when stationary to prevent windup
            if abs(left_setpoint) < 1.0 and abs(left_actual) < VELOCITY_DEADZONE_RPM:
                self._left_pid.reset()
            if abs(right_setpoint) < 1.0 and abs(right_actual) < VELOCITY_DEADZONE_RPM:
                self._right_pid.reset()

            # Compute PID outputs
            left_output = self._left_pid.compute(left_setpoint, left_actual)
            right_output = self._right_pid.compute(right_setpoint, right_actual)

            # Apply deadband compensation
            left_output = self._apply_deadband(left_output)
            right_output = self._apply_deadband(right_output)

            # Apply to motors
            self._motors.set_speeds(left_output, right_output)

            # Maintain loop timing
            elapsed = time.time() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def update_pid_gains(
        self,
        left_gains: Optional[Tuple[float, float, float]] = None,
        right_gains: Optional[Tuple[float, float, float]] = None
    ) -> None:
        """
        Update PID gains at runtime.

        Args:
            left_gains: Tuple of (kp, ki, kd) for left wheel
            right_gains: Tuple of (kp, ki, kd) for right wheel
        """
        if left_gains is not None:
            self._left_pid.update_gains(*left_gains)
        if right_gains is not None:
            self._right_pid.update_gains(*right_gains)

    def get_pid_gains(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get current PID gains.

        Returns:
            Tuple of (left_gains, right_gains) where each is (kp, ki, kd)
        """
        return (self._left_pid.get_gains(), self._right_pid.get_gains())

    @property
    def is_running(self) -> bool:
        """Check if control loop is running."""
        return self._running
