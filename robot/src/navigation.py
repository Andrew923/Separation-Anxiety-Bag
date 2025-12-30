"""
High-level navigation controller for person following.

Fuses UWB target tracking with VFH obstacle avoidance.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import time
import math

from .vfh import VFHResult


class NavigationState(Enum):
    """Robot navigation states."""
    IDLE = auto()           # Not moving, waiting for target
    FOLLOWING = auto()      # Actively following target
    AVOIDING = auto()       # Avoiding obstacle while following
    SPINNING = auto()       # Spinning to find clear path
    LOST_TARGET = auto()    # Target lost, waiting
    STOPPED = auto()        # Emergency stop


@dataclass
class NavigationConfig:
    """Navigation controller configuration."""
    target_follow_distance_mm: float = 1500.0
    follow_distance_tolerance_mm: float = 200.0
    angular_tolerance_deg: float = 10.0
    spin_threshold_deg: float = 45.0
    stop_if_blocked_timeout_s: float = 3.0
    max_linear_speed_mm_s: float = 500.0
    max_angular_speed_deg_s: float = 90.0
    approach_speed_factor: float = 0.5
    lost_target_timeout_s: float = 5.0


@dataclass
class NavigationCommand:
    """Output command from navigation controller."""
    linear_velocity_mm_s: float = 0.0
    angular_velocity_deg_s: float = 0.0
    state: NavigationState = NavigationState.IDLE


class NavigationController:
    """
    Fuses UWB target tracking with VFH obstacle avoidance.

    Decision logic:
    1. Get target angle from UWB
    2. Check if target direction is safe via VFH
    3. If safe: drive toward target
    4. If blocked: find nearest safe direction
    5. If all blocked: spin in place
    6. If target lost: stop and wait
    """

    def __init__(self, config: NavigationConfig):
        """
        Initialize navigation controller.

        Args:
            config: Navigation configuration
        """
        self._config = config
        self._state = NavigationState.IDLE
        self._spin_start_time: Optional[float] = None
        self._lost_target_time: Optional[float] = None
        self._last_target_angle: float = 0.0
        self._spin_direction: float = 1.0  # 1 = clockwise, -1 = counter-clockwise

    def update(
        self,
        target_angle_deg: Optional[float],
        target_range_mm: Optional[float],
        vfh_result: VFHResult
    ) -> NavigationCommand:
        """
        Compute navigation command.

        Args:
            target_angle_deg: Angle to person (None if lost)
            target_range_mm: Distance to person (None if lost)
            vfh_result: VFH obstacle analysis

        Returns:
            NavigationCommand with velocities and state
        """
        # Handle lost target
        if target_angle_deg is None or target_range_mm is None:
            return self._handle_lost_target()

        # Target found - reset lost timer
        self._lost_target_time = None
        self._last_target_angle = target_angle_deg

        # Check if target direction is safe
        target_safe = vfh_result.best_heading_deg is not None

        if not target_safe and not vfh_result.can_proceed:
            # All directions blocked
            return self._handle_all_blocked()

        # Reset spin timer if we can proceed
        if vfh_result.can_proceed:
            self._spin_start_time = None

        # Determine heading and state
        if target_safe and abs(target_angle_deg - (vfh_result.best_heading_deg or 0)) < self._config.spin_threshold_deg:
            # Target direction is relatively clear
            return self._follow_target(
                target_angle_deg,
                target_range_mm,
                vfh_result
            )
        else:
            # Need to avoid obstacles
            return self._avoid_obstacles(
                target_angle_deg,
                target_range_mm,
                vfh_result
            )

    def _follow_target(
        self,
        target_angle: float,
        target_range: float,
        vfh_result: VFHResult
    ) -> NavigationCommand:
        """
        Follow target directly.

        Args:
            target_angle: Angle to target
            target_range: Distance to target
            vfh_result: VFH result

        Returns:
            Navigation command
        """
        self._state = NavigationState.FOLLOWING
        config = self._config

        # Compute linear speed based on distance
        linear_speed = self._compute_linear_speed(target_range, target_angle)

        # Compute angular speed to turn toward target
        angular_speed = self._compute_angular_speed(target_angle)

        return NavigationCommand(
            linear_velocity_mm_s=linear_speed,
            angular_velocity_deg_s=angular_speed,
            state=self._state
        )

    def _avoid_obstacles(
        self,
        target_angle: float,
        target_range: float,
        vfh_result: VFHResult
    ) -> NavigationCommand:
        """
        Avoid obstacles while trying to reach target.

        Args:
            target_angle: Angle to target
            target_range: Distance to target
            vfh_result: VFH result

        Returns:
            Navigation command
        """
        self._state = NavigationState.AVOIDING
        config = self._config

        # Use VFH's recommended heading
        if vfh_result.best_heading_deg is not None:
            heading = vfh_result.best_heading_deg
        else:
            # No safe direction - shouldn't reach here
            return self._handle_all_blocked()

        # Reduce speed when avoiding
        linear_speed = self._compute_linear_speed(target_range, heading) * 0.5

        # Turn toward safe direction
        angular_speed = self._compute_angular_speed(heading)

        return NavigationCommand(
            linear_velocity_mm_s=linear_speed,
            angular_velocity_deg_s=angular_speed,
            state=self._state
        )

    def _handle_all_blocked(self) -> NavigationCommand:
        """
        Handle case where all directions are blocked.

        Spins in place to find a clear path.
        """
        config = self._config

        # Start spin timer if not already spinning
        if self._spin_start_time is None:
            self._spin_start_time = time.time()
            # Choose spin direction based on last known target
            self._spin_direction = 1.0 if self._last_target_angle >= 0 else -1.0

        # Check if we've been spinning too long
        spin_duration = time.time() - self._spin_start_time
        if spin_duration > config.stop_if_blocked_timeout_s:
            # Give up and stop
            self._state = NavigationState.STOPPED
            return NavigationCommand(
                linear_velocity_mm_s=0.0,
                angular_velocity_deg_s=0.0,
                state=self._state
            )

        # Continue spinning
        self._state = NavigationState.SPINNING
        return NavigationCommand(
            linear_velocity_mm_s=0.0,
            angular_velocity_deg_s=config.max_angular_speed_deg_s * self._spin_direction,
            state=self._state
        )

    def _handle_lost_target(self) -> NavigationCommand:
        """Handle case where target is lost."""
        config = self._config

        # Start lost timer if not already timing
        if self._lost_target_time is None:
            self._lost_target_time = time.time()

        # Check if we've been lost too long
        lost_duration = time.time() - self._lost_target_time
        if lost_duration > config.lost_target_timeout_s:
            # Stop completely
            self._state = NavigationState.STOPPED
        else:
            # Still searching
            self._state = NavigationState.LOST_TARGET

        return NavigationCommand(
            linear_velocity_mm_s=0.0,
            angular_velocity_deg_s=0.0,
            state=self._state
        )

    def _compute_linear_speed(
        self,
        target_range: float,
        heading_error: float
    ) -> float:
        """
        Compute forward speed based on distance and heading error.

        Args:
            target_range: Distance to target in mm
            heading_error: Heading error in degrees

        Returns:
            Linear speed in mm/s
        """
        config = self._config

        # Distance error (positive = too far, negative = too close)
        distance_error = target_range - config.target_follow_distance_mm

        # Base speed from distance error
        if abs(distance_error) < config.follow_distance_tolerance_mm:
            # Within tolerance - minimal speed
            base_speed = 0.0
        elif distance_error > 0:
            # Too far - move forward
            # Scale speed with distance, max out at max_linear_speed
            speed_factor = min(1.0, distance_error / 1000.0)
            base_speed = config.max_linear_speed_mm_s * speed_factor
        else:
            # Too close - move backward
            speed_factor = min(1.0, abs(distance_error) / 500.0)
            base_speed = -config.max_linear_speed_mm_s * speed_factor * 0.5

        # Reduce speed when heading error is large
        heading_factor = max(0.3, 1.0 - abs(heading_error) / 90.0)
        base_speed *= heading_factor

        # Apply approach speed factor when close
        if target_range < config.target_follow_distance_mm * 1.5:
            base_speed *= config.approach_speed_factor

        return base_speed

    def _compute_angular_speed(self, heading_error: float) -> float:
        """
        Compute turning speed to correct heading error.

        Args:
            heading_error: Heading error in degrees (positive = turn right)

        Returns:
            Angular speed in deg/s
        """
        config = self._config

        # Simple proportional control
        if abs(heading_error) < config.angular_tolerance_deg:
            return 0.0

        # Scale angular speed with error
        kp = 2.0  # Proportional gain
        angular_speed = kp * heading_error

        # Clamp to max speed
        angular_speed = max(
            -config.max_angular_speed_deg_s,
            min(config.max_angular_speed_deg_s, angular_speed)
        )

        return angular_speed

    def get_state(self) -> NavigationState:
        """Get current navigation state."""
        return self._state

    def reset(self) -> None:
        """Reset navigation to IDLE state."""
        self._state = NavigationState.IDLE
        self._spin_start_time = None
        self._lost_target_time = None
        self._last_target_angle = 0.0

    def set_state(self, state: NavigationState) -> None:
        """
        Manually set navigation state.

        Args:
            state: New navigation state
        """
        self._state = state
        if state == NavigationState.IDLE:
            self.reset()
