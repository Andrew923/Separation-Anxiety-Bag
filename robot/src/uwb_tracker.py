"""
RYUW122 UWB module UART interface.

The RYUW122 is a UWB (Ultra-Wideband) transceiver module that supports:
- Distance measurement with ~10cm accuracy
- AT command interface via UART
- ANCHOR/TAG modes for ranging
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import threading
import time
import re
import math

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


@dataclass
class RangeFilterConfig:
    """Configuration for range filtering."""
    enabled: bool = True
    # EMA alpha: higher = more responsive, lower = more smoothing
    # alpha = 2/(N+1) where N is roughly equivalent window size
    # 0.3 ≈ 6 sample window, 0.2 ≈ 9 sample window, 0.1 ≈ 19 sample window
    ema_alpha: float = 0.3
    # Outlier rejection: ignore readings that deviate more than this from EMA
    outlier_threshold_mm: float = 200.0
    # Minimum samples before filter output is valid
    min_samples: int = 3


class EMAFilter:
    """
    Exponential Moving Average filter for range smoothing.

    EMA provides a good balance between noise reduction and responsiveness.
    The formula is: EMA_new = alpha * value + (1 - alpha) * EMA_old
    """

    def __init__(self, alpha: float = 0.3, outlier_threshold: float = 200.0):
        """
        Initialize EMA filter.

        Args:
            alpha: Smoothing factor (0-1). Higher = more responsive.
            outlier_threshold: Reject values deviating more than this from current EMA.
        """
        self._alpha = alpha
        self._outlier_threshold = outlier_threshold
        self._ema: Optional[float] = None
        self._sample_count = 0

    def update(self, value: float) -> float:
        """
        Update filter with new value and return filtered result.

        Args:
            value: New measurement

        Returns:
            Filtered value
        """
        if self._ema is None:
            # First sample - initialize
            self._ema = value
            self._sample_count = 1
            return value

        # Outlier rejection
        if self._outlier_threshold > 0:
            if abs(value - self._ema) > self._outlier_threshold:
                # Outlier detected - use reduced alpha to slowly adapt
                # This prevents sudden jumps but still allows tracking
                effective_alpha = self._alpha * 0.1
            else:
                effective_alpha = self._alpha
        else:
            effective_alpha = self._alpha

        # EMA update
        self._ema = effective_alpha * value + (1 - effective_alpha) * self._ema
        self._sample_count += 1

        return self._ema

    def get_value(self) -> Optional[float]:
        """Get current filtered value without updating."""
        return self._ema

    def reset(self) -> None:
        """Reset filter state."""
        self._ema = None
        self._sample_count = 0

    @property
    def sample_count(self) -> int:
        """Number of samples processed."""
        return self._sample_count

    @property
    def is_initialized(self) -> bool:
        """Whether filter has received enough samples."""
        return self._ema is not None


class AngleFilter:
    """
    Filter for angular values that handles wraparound correctly.

    Uses circular mean for proper angle averaging.
    """

    def __init__(self, alpha: float = 0.3, window_size: int = 5):
        """
        Initialize angle filter.

        Args:
            alpha: EMA smoothing factor for primary filter
            window_size: Size of median pre-filter window (0 to disable)
        """
        self._alpha = alpha
        self._window_size = window_size
        self._window: List[float] = []
        self._ema_sin: Optional[float] = None
        self._ema_cos: Optional[float] = None

    def update(self, angle_deg: float) -> float:
        """
        Update filter with new angle and return filtered result.

        Args:
            angle_deg: New angle measurement in degrees

        Returns:
            Filtered angle in degrees
        """
        # Convert to radians for circular math
        angle_rad = math.radians(angle_deg)
        sin_val = math.sin(angle_rad)
        cos_val = math.cos(angle_rad)

        if self._ema_sin is None:
            # First sample
            self._ema_sin = sin_val
            self._ema_cos = cos_val
        else:
            # EMA update on sin and cos components separately
            self._ema_sin = self._alpha * sin_val + (1 - self._alpha) * self._ema_sin
            self._ema_cos = self._alpha * cos_val + (1 - self._alpha) * self._ema_cos

        # Convert back to angle
        filtered_rad = math.atan2(self._ema_sin, self._ema_cos)
        return math.degrees(filtered_rad)

    def get_value(self) -> Optional[float]:
        """Get current filtered angle without updating."""
        if self._ema_sin is None or self._ema_cos is None:
            return None
        filtered_rad = math.atan2(self._ema_sin, self._ema_cos)
        return math.degrees(filtered_rad)

    def reset(self) -> None:
        """Reset filter state."""
        self._window = []
        self._ema_sin = None
        self._ema_cos = None


@dataclass
class UWBModuleConfig:
    """Configuration for a UWB module."""
    uart_port: str = "/dev/ttyAMA0"
    baud_rate: int = 115200
    timeout_ms: int = 100
    network_id: str = "0x1234"  # 8 bytes ASCII network group ID
    address: str = "ANCHOR01"  # 8 bytes ASCII device address
    reset_pin: Optional[int] = None  # GPIO pin for hardware reset (active low)
    target_address: str = "TAG001"  # Address of TAG to communicate with


class RYUW122:
    """
    Interface for RYUW122 UWB module.

    AT Command Protocol:
    - AT+MODE=<ANCHOR|TAG>  Set operating mode
    - AT+NETWORKID=<id>     Set network ID
    - AT+ANCHOR_SEND=<id>   Request range from specific anchor (TAG mode)
    - Distance output format: "+RANGE:<id>,<distance_cm>"

    For our setup:
    - Robot modules are ANCHORs
    - Human module is TAG
    - ANCHORs measure distance when TAG is in range
    """

    # AT Command constants (RYUW122 AT command set)
    CMD_TEST = "AT"
    CMD_RESET = "AT+RESET"
    CMD_MODE = "AT+MODE"  # 0=TAG, 1=ANCHOR, 2=Sleep
    CMD_NETWORKID = "AT+NETWORKID"
    CMD_ADDRESS = "AT+ADDRESS"
    CMD_ANCHOR_SEND = "AT+ANCHOR_SEND"  # Send from anchor to tag
    CMD_RSSI = "AT+RSSI"
    RESPONSE_OK = "+OK"
    RESPONSE_ERROR = "ERROR"
    RESPONSE_READY = "+READY"

    # Response patterns
    # +ANCHOR_RCV=<Addr>,<Len>,<Data>,<Dist>,<RSSI>
    ANCHOR_RCV_PATTERN = re.compile(
        r'\+ANCHOR_RCV=([^,]+),(\d+),([^,]*),(\d+)(?:,(-?\d+))?'
    )
    # Legacy pattern for backwards compatibility
    RANGE_PATTERN = re.compile(r'\+RANGE:(\d+),(\d+)')

    def __init__(self, config: UWBModuleConfig):
        """
        Initialize UWB module connection.

        Args:
            config: Module configuration
        """
        self._config = config
        self._serial: Optional['serial.Serial'] = None
        self._lock = threading.Lock()
        self._last_range_mm: Optional[float] = None
        self._last_range_time: float = 0.0
        self._connected = False
        self._reset_pin_initialized = False

        # Setup reset pin if configured
        if config.reset_pin is not None and GPIO_AVAILABLE:
            self._setup_reset_pin()

    def _setup_reset_pin(self) -> None:
        """Setup GPIO for hardware reset."""
        if self._config.reset_pin is None:
            return

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            # Reset is active low, so set HIGH initially (not resetting)
            GPIO.setup(self._config.reset_pin, GPIO.OUT, initial=GPIO.HIGH)
            self._reset_pin_initialized = True
        except Exception as e:
            print(f"Failed to setup reset pin: {e}")

    def hardware_reset(self, reset_time_ms: int = 100, recovery_time_ms: int = 500) -> bool:
        """
        Perform hardware reset by toggling reset pin.

        Args:
            reset_time_ms: Time to hold reset low (milliseconds)
            recovery_time_ms: Time to wait after reset (milliseconds)

        Returns:
            True if reset was performed
        """
        if not self._reset_pin_initialized or self._config.reset_pin is None:
            print("No reset pin configured")
            return False

        try:
            print(f"Resetting UWB module (GPIO {self._config.reset_pin})...")
            # Pull reset LOW
            GPIO.output(self._config.reset_pin, GPIO.LOW)
            time.sleep(reset_time_ms / 1000.0)

            # Release reset (HIGH)
            GPIO.output(self._config.reset_pin, GPIO.HIGH)
            time.sleep(recovery_time_ms / 1000.0)

            print("Reset complete")
            return True
        except Exception as e:
            print(f"Hardware reset failed: {e}")
            return False

    def connect(self, do_reset: bool = True) -> bool:
        """
        Open serial connection and configure module.

        Args:
            do_reset: If True, perform hardware reset before connecting

        Returns:
            True if connection successful
        """
        if not SERIAL_AVAILABLE:
            print("Warning: pyserial not available, UWB in simulation mode")
            return False

        # Perform hardware reset if configured and requested
        if do_reset and self._reset_pin_initialized:
            self.hardware_reset()

        try:
            self._serial = serial.Serial(
                port=self._config.uart_port,
                baudrate=self._config.baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self._config.timeout_ms / 1000.0
            )

            # Clear any pending data
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()

            # Small delay for module to stabilize
            time.sleep(0.1)

            self._connected = True
            return True

        except Exception as e:
            print(f"Failed to connect to UWB module: {e}")
            return False

    def configure_as_anchor(self, address: Optional[str] = None) -> bool:
        """
        Configure module as ANCHOR for ranging.

        Args:
            address: Device address (8 bytes ASCII), uses config default if None

        Returns:
            True if configuration successful
        """
        if not self._connected:
            return False

        # Test module is responsive
        if not self._send_command(self.CMD_TEST):
            print("Module not responding to AT command")
            return False

        # Set mode to ANCHOR (1)
        if not self._send_command(f"{self.CMD_MODE}=1"):
            print("Failed to set ANCHOR mode")
            return False

        # Set network ID (must match TAG)
        network_id = self._config.network_id
        if not self._send_command(f"{self.CMD_NETWORKID}={network_id}"):
            print(f"Failed to set network ID: {network_id}")
            return False

        # Set device address
        addr = address or self._config.address
        if not self._send_command(f"{self.CMD_ADDRESS}={addr}"):
            print(f"Failed to set address: {addr}")
            return False

        # Enable RSSI in received messages
        if not self._send_command(f"{self.CMD_RSSI}=1"):
            print("Failed to enable RSSI")
            # Non-fatal, continue

        print(f"Configured as ANCHOR: network={network_id}, address={addr}")
        return True

    def configure_as_tag(self, address: Optional[str] = None) -> bool:
        """
        Configure module as TAG for ranging.

        Args:
            address: Device address (8 bytes ASCII), uses config default if None

        Returns:
            True if configuration successful
        """
        if not self._connected:
            return False

        # Test module is responsive
        if not self._send_command(self.CMD_TEST):
            print("Module not responding to AT command")
            return False

        # Set mode to TAG (0)
        if not self._send_command(f"{self.CMD_MODE}=0"):
            print("Failed to set TAG mode")
            return False

        # Set network ID (must match ANCHOR)
        network_id = self._config.network_id
        if not self._send_command(f"{self.CMD_NETWORKID}={network_id}"):
            print(f"Failed to set network ID: {network_id}")
            return False

        # Set device address
        addr = address or self._config.address
        if not self._send_command(f"{self.CMD_ADDRESS}={addr}"):
            print(f"Failed to set address: {addr}")
            return False

        print(f"Configured as TAG: network={network_id}, address={addr}")
        return True

    def get_range(self) -> Optional[float]:
        """
        Get distance to TAG in mm.

        Reads available data from serial port and parses range values.
        Parses +ANCHOR_RCV=<Addr>,<Len>,<Data>,<Dist>,<RSSI> format.

        Returns:
            Distance in mm, or None if no reading available
        """
        if not self._connected or self._serial is None:
            return None

        with self._lock:
            try:
                # Check if data available
                if self._serial.in_waiting == 0:
                    # Return cached value if recent
                    if time.time() - self._last_range_time < 0.5:
                        return self._last_range_mm
                    return None

                # Read available data
                data = self._serial.read(self._serial.in_waiting)
                text = data.decode('utf-8', errors='ignore')

                # Parse +ANCHOR_RCV format: +ANCHOR_RCV=<Addr>,<Len>,<Data>,<Dist>,<RSSI>
                # Distance is in cm
                matches = self.ANCHOR_RCV_PATTERN.findall(text)
                if matches:
                    # Take most recent match
                    # Groups: (addr, len, data, dist_cm, rssi)
                    addr, length, payload, distance_cm, rssi = matches[-1]
                    distance_mm = int(distance_cm) * 10  # Convert cm to mm

                    self._last_range_mm = float(distance_mm)
                    self._last_range_time = time.time()

                    return self._last_range_mm

                # Fallback: try legacy pattern
                legacy_matches = self.RANGE_PATTERN.findall(text)
                if legacy_matches:
                    tag_id, distance_cm = legacy_matches[-1]
                    distance_mm = int(distance_cm) * 10

                    self._last_range_mm = float(distance_mm)
                    self._last_range_time = time.time()

                    return self._last_range_mm

                return None

            except Exception as e:
                print(f"UWB read error: {e}")
                return None

    def poll_range(self, target_address: Optional[str] = None) -> Optional[float]:
        """
        Actively poll for range measurement by sending data to TAG.

        Sends AT+ANCHOR_SEND to the target TAG, which triggers TWR (Two-Way Ranging).
        The response +ANCHOR_RCV includes the measured distance.

        Args:
            target_address: TAG address to range to (uses config default if None)

        Returns:
            Distance in mm, or None if no response
        """
        if not self._connected or self._serial is None:
            return None

        target = target_address or self._config.target_address

        with self._lock:
            try:
                # Clear input buffer
                self._serial.reset_input_buffer()

                # Send data to TAG to trigger ranging
                # AT+ANCHOR_SEND=<Addr>,<Len>,<Data>
                # Using "PING" as a simple ranging trigger
                ping_data = "PING"
                cmd = f"{self.CMD_ANCHOR_SEND}={target},{len(ping_data)},{ping_data}\r\n"
                self._serial.write(cmd.encode('utf-8'))

                # Wait for TWR to complete and response to arrive
                time.sleep(0.1)

                # Read response - should get +OK followed by +ANCHOR_RCV
                if self._serial.in_waiting > 0:
                    data = self._serial.read(self._serial.in_waiting)
                    text = data.decode('utf-8', errors='ignore')

                    # Parse +ANCHOR_RCV format
                    matches = self.ANCHOR_RCV_PATTERN.findall(text)
                    if matches:
                        addr, length, payload, distance_cm, rssi = matches[-1]
                        distance_mm = int(distance_cm) * 10

                        self._last_range_mm = float(distance_mm)
                        self._last_range_time = time.time()

                        return self._last_range_mm

                    # Fallback: try legacy pattern
                    legacy_matches = self.RANGE_PATTERN.findall(text)
                    if legacy_matches:
                        tag_id, distance_cm = legacy_matches[-1]
                        distance_mm = int(distance_cm) * 10

                        self._last_range_mm = float(distance_mm)
                        self._last_range_time = time.time()

                        return self._last_range_mm

                return None

            except Exception as e:
                print(f"UWB poll error: {e}")
                return None

    def send_to_tag(
        self,
        data: str,
        target_address: Optional[str] = None
    ) -> Tuple[bool, Optional[float]]:
        """
        Send data to TAG and get distance measurement.

        Args:
            data: ASCII data to send (max 12 characters)
            target_address: TAG address (uses config default if None)

        Returns:
            Tuple of (success, distance_mm) - distance is None if not received
        """
        if not self._connected or self._serial is None:
            return (False, None)

        target = target_address or self._config.target_address

        # Truncate data to max 12 characters
        if len(data) > 12:
            data = data[:12]

        with self._lock:
            try:
                self._serial.reset_input_buffer()

                # AT+ANCHOR_SEND=<Addr>,<Len>,<Data>
                cmd = f"{self.CMD_ANCHOR_SEND}={target},{len(data)},{data}\r\n"
                self._serial.write(cmd.encode('utf-8'))

                # Wait for response
                time.sleep(0.15)

                if self._serial.in_waiting > 0:
                    response = self._serial.read(self._serial.in_waiting)
                    text = response.decode('utf-8', errors='ignore')

                    # Check for +OK
                    if self.RESPONSE_OK not in text:
                        return (False, None)

                    # Parse distance from +ANCHOR_RCV
                    matches = self.ANCHOR_RCV_PATTERN.findall(text)
                    if matches:
                        addr, length, payload, distance_cm, rssi = matches[-1]
                        distance_mm = int(distance_cm) * 10

                        self._last_range_mm = float(distance_mm)
                        self._last_range_time = time.time()

                        return (True, distance_mm)

                    return (True, None)  # Send succeeded but no range yet

                return (False, None)

            except Exception as e:
                print(f"UWB send error: {e}")
                return (False, None)

    def _send_command(self, command: str) -> bool:
        """
        Send AT command and check for OK response.

        Args:
            command: AT command to send

        Returns:
            True if command acknowledged with OK
        """
        if self._serial is None:
            return False

        with self._lock:
            try:
                # Send command
                self._serial.write(f"{command}\r\n".encode('utf-8'))

                # Wait for response
                time.sleep(0.2)

                # Read response
                response = ""
                if self._serial.in_waiting > 0:
                    data = self._serial.read(self._serial.in_waiting)
                    response = data.decode('utf-8', errors='ignore')

                return self.RESPONSE_OK in response

            except Exception as e:
                print(f"UWB command error: {e}")
                return False

    def reset(self) -> bool:
        """Reset the module."""
        return self._send_command(self.CMD_RESET)

    def close(self) -> None:
        """Close serial connection."""
        with self._lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None
            self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if module is connected."""
        return self._connected


class DualUWBAnchors:
    """
    Manages two UWB anchors for triangulation.

    Supports optional filtering to reduce noise in range measurements.
    """

    def __init__(
        self,
        anchor1_config: UWBModuleConfig,
        anchor2_config: UWBModuleConfig,
        filter_config: Optional[RangeFilterConfig] = None
    ):
        """
        Initialize both anchors.

        Args:
            anchor1_config: Configuration for first anchor
            anchor2_config: Configuration for second anchor
            filter_config: Optional filtering configuration (None = no filtering)
        """
        self._anchor1 = RYUW122(anchor1_config)
        self._anchor2 = RYUW122(anchor2_config)

        # Setup filters if enabled
        self._filter_config = filter_config
        self._filter1: Optional[EMAFilter] = None
        self._filter2: Optional[EMAFilter] = None

        if filter_config and filter_config.enabled:
            self._filter1 = EMAFilter(
                alpha=filter_config.ema_alpha,
                outlier_threshold=filter_config.outlier_threshold_mm
            )
            self._filter2 = EMAFilter(
                alpha=filter_config.ema_alpha,
                outlier_threshold=filter_config.outlier_threshold_mm
            )

    @property
    def anchor1(self) -> RYUW122:
        """Get first anchor module."""
        return self._anchor1

    @property
    def anchor2(self) -> RYUW122:
        """Get second anchor module."""
        return self._anchor2

    @property
    def filtering_enabled(self) -> bool:
        """Check if filtering is enabled."""
        return self._filter1 is not None

    def connect(self) -> bool:
        """
        Connect and configure both anchors.

        Returns:
            True if both connections successful
        """
        success1 = self._anchor1.connect()
        success2 = self._anchor2.connect()

        if success1:
            self._anchor1.configure_as_anchor()
        if success2:
            self._anchor2.configure_as_anchor()

        return success1 and success2

    def get_ranges(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get ranges from both anchors (passive read).

        Returns:
            Tuple of (range1_mm, range2_mm) - None if reading failed
        """
        range1 = self._anchor1.get_range()
        range2 = self._anchor2.get_range()
        return self._apply_filters(range1, range2)

    def poll_ranges(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Actively poll ranges from both anchors.

        Returns:
            Tuple of (range1_mm, range2_mm) - None if reading failed
        """
        range1 = self._anchor1.poll_range()
        range2 = self._anchor2.poll_range()
        return self._apply_filters(range1, range2)

    def poll_ranges_raw(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Actively poll ranges without filtering.

        Useful for debugging or when you need raw values.

        Returns:
            Tuple of (range1_mm, range2_mm) - None if reading failed
        """
        range1 = self._anchor1.poll_range()
        range2 = self._anchor2.poll_range()
        return (range1, range2)

    def _apply_filters(
        self,
        range1: Optional[float],
        range2: Optional[float]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Apply filtering to range measurements.

        Args:
            range1: Raw range from anchor 1
            range2: Raw range from anchor 2

        Returns:
            Filtered ranges (or raw if filtering disabled)
        """
        if self._filter1 is None or self._filter2 is None:
            return (range1, range2)

        filtered1 = self._filter1.update(range1) if range1 is not None else self._filter1.get_value()
        filtered2 = self._filter2.update(range2) if range2 is not None else self._filter2.get_value()

        # Check minimum samples requirement
        if self._filter_config and self._filter_config.min_samples > 0:
            if self._filter1.sample_count < self._filter_config.min_samples:
                filtered1 = None
            if self._filter2.sample_count < self._filter_config.min_samples:
                filtered2 = None

        return (filtered1, filtered2)

    def reset_filters(self) -> None:
        """Reset filter state (e.g., after losing track of target)."""
        if self._filter1:
            self._filter1.reset()
        if self._filter2:
            self._filter2.reset()

    def get_filter_stats(self) -> dict:
        """
        Get filter statistics for debugging.

        Returns:
            Dict with filter state information
        """
        stats = {
            'filtering_enabled': self.filtering_enabled,
        }
        if self._filter1 and self._filter2:
            stats['filter1_samples'] = self._filter1.sample_count
            stats['filter2_samples'] = self._filter2.sample_count
            stats['filter1_value'] = self._filter1.get_value()
            stats['filter2_value'] = self._filter2.get_value()
        return stats

    def close(self) -> None:
        """Close both anchor connections."""
        self._anchor1.close()
        self._anchor2.close()

    @property
    def is_connected(self) -> bool:
        """Check if both anchors are connected."""
        return self._anchor1.is_connected and self._anchor2.is_connected
