"""
Gamepad input handler using direct js0 device reading.

Supports ShanWan USB WirelessGamepad and similar controllers.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import struct
import threading
import time
import select
import os
import fcntl


# Linux joystick event format
JS_EVENT_FMT = "<IhBB"
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FMT)

# Event types
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80


@dataclass
class ControllerConfig:
    """Configuration for gamepad controller."""
    deadzone: float = 0.1
    device_path: str = "/dev/input/js0"
    invert_throttle: bool = False  # ShanWan: up=forward


class Gamepad:
    """
    Gamepad input reader using direct js0 device.

    Thread-safe, non-blocking access to controller state.
    """

    def __init__(self, config: Optional[ControllerConfig] = None):
        """
        Initialize gamepad.

        Args:
            config: Controller configuration (uses defaults if None)
        """
        if config is None:
            self._config = ControllerConfig()
        else:
            self._config = config

        # Ensure device_path has a default
        if self._config.device_path is None:
            self._config.device_path = "/dev/input/js0"

        self._device = None

        self._running = False
        self._read_thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()

        # Axis values (raw, -32767 to +32767)
        self._axis_values = {
            0: 0,  # Left stick X
            1: 0,  # Left stick Y (throttle)
            2: 0,  # Right stick X
            3: 0,  # Right stick Y
            4: 0,  # D-pad X
            5: 0,  # D-pad Y
        }

        # Button states
        self._button_states = {i: False for i in range(10)}  # Buttons 0-9

        self._connect_device()

    def _connect_device(self) -> None:
        """Connect to gamepad device."""
        try:
            self._device = open(self._config.device_path, 'rb')

            # Set non-blocking
            flags = fcntl.fcntl(self._device.fileno(), fcntl.F_GETFL)
            fcntl.fcntl(
                self._device.fileno(),
                fcntl.F_SETFL,
                flags | os.O_NONBLOCK
            )

            # Try to get controller name
            try:
                name_bytes = fcntl.ioctl(self._device.fileno(), 0x80006113 + (0x10000 << 8), b'\x00' * 128)
                name = name_bytes.decode('utf-8').rstrip('\x00')
                print(f"Connected to: {self._config.device_path} ({name})")
            except:
                print(f"Connected to: {self._config.device_path}")

        except FileNotFoundError:
            raise RuntimeError(
                f"Gamepad not found: {self._config.device_path}. "
                "Is controller connected?"
            )
        except PermissionError:
            raise RuntimeError(
                f"Permission denied: {self._config.device_path}. "
                "Try: sudo usermod -a -G input $USER"
            )

    def _read_loop(self) -> None:
        """Background thread for reading gamepad events."""
        while self._running:
            try:
                # Check if data available
                if self._device is None:
                    time.sleep(0.01)
                    continue

                rlist, _, _ = select.select([self._device], [], [], 0.01)

                if rlist:
                    data = self._device.read(JS_EVENT_SIZE)
                    if len(data) < JS_EVENT_SIZE:
                        continue

                    timestamp, value, evt_type, number = struct.unpack(
                        JS_EVENT_FMT, data
                    )

                    # Strip init flag
                    evt_type &= ~JS_EVENT_INIT

                    with self._state_lock:
                        if evt_type == JS_EVENT_AXIS:
                            self._axis_values[number] = value
                        elif evt_type == JS_EVENT_BUTTON:
                            self._button_states[number] = (value == 1)

            except (OSError, IOError):
                if self._running:
                    time.sleep(0.01)

    def get_state(self) -> Tuple[float, float]:
        """
        Get normalized throttle and turn values.

        Returns:
            Tuple of (throttle, turn) in [-1.0, 1.0] range
            throttle: -1.0 (back) to +1.0 (forward)
            turn: -1.0 (left) to +1.0 (right)
        """
        with self._state_lock:
            raw_x = self._axis_values.get(0, 0)      # Left stick X
            raw_y = self._axis_values.get(1, 0)      # Left stick Y

        # Normalize from [-32767, 32767] to [-1, 1]
        turn = raw_x / 32767.0
        throttle = raw_y / 32767.0

        # Invert throttle if configured
        if self._config.invert_throttle:
            throttle = -throttle

        # Apply deadzone
        if abs(turn) < self._config.deadzone:
            turn = 0.0
        if abs(throttle) < self._config.deadzone:
            throttle = 0.0

        return (throttle, turn)

    def is_button_pressed(self, button_id: int) -> bool:
        """
        Check if a button is currently pressed.

        Args:
            button_id: Button number (0-9)

            ShanWan mapping:
                0 = Triangle
                1 = Circle
                2 = Cross
                3 = Square
                8 = Select
                9 = Start

        Returns:
            True if button is pressed, False otherwise
        """
        with self._state_lock:
            return self._button_states.get(button_id, False)

    def get_dpad(self) -> Tuple[int, int]:
        """
        Get D-pad state.

        Returns:
            Tuple of (x, y) where each is -1, 0, or 1
            x: -1=left, 0=center, 1=right
            y: -1=up, 0=center, 1=down
        """
        with self._state_lock:
            raw_x = self._axis_values.get(4, 0)
            raw_y = self._axis_values.get(5, 0)

        # Normalize from raw axis values to -1, 0, 1
        # Raw values are -32767 to +32767, threshold at half
        threshold = 16000
        x = -1 if raw_x < -threshold else (1 if raw_x > threshold else 0)
        y = -1 if raw_y < -threshold else (1 if raw_y > threshold else 0)

        return (x, y)

    def start(self) -> None:
        """Start reading events in background thread."""
        if self._running:
            return

        self._running = True
        self._read_thread = threading.Thread(
            target=self._read_loop,
            daemon=True
        )
        self._read_thread.start()

    def stop(self) -> None:
        """Stop reading events."""
        self._running = False
        if self._read_thread is not None:
            self._read_thread.join(timeout=0.5)
            self._read_thread = None

    def cleanup(self) -> None:
        """Release gamepad device."""
        self.stop()
        if self._device is not None:
            try:
                self._device.close()
            except:
                pass
            self._device = None

    @property
    def name(self) -> str:
        """Get controller name (if available)."""
        try:
            # Try to read name from js device
            import fcntl
            if self._device is not None and hasattr(self._device, 'fileno'):
                name_bytes = fcntl.ioctl(
                    self._device.fileno(),
                    0x80006113 + (0x10000 << 8),
                    b'\x00' * 128
                )
                return name_bytes.decode('utf-8').rstrip('\x00')
        except:
            pass
        return "Unknown"

    @property
    def path(self) -> str:
        """Get device path."""
        return self._config.device_path if self._config.device_path else "/dev/input/js0"


# Backward compatibility - PS2Controller is now Gamepad
PS2Controller = Gamepad
