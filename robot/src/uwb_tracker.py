"""
RYUW122 UWB module UART interface.

The RYUW122 is a UWB (Ultra-Wideband) transceiver module that supports:
- Distance measurement with ~10cm accuracy
- AT command interface via UART
- ANCHOR/TAG modes for ranging
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import threading
import time
import re

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


@dataclass
class UWBModuleConfig:
    """Configuration for a UWB module."""
    uart_port: str = "/dev/ttyAMA0"
    baud_rate: int = 115200
    timeout_ms: int = 100
    network_id: int = 0x1234


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

    # AT Command constants
    CMD_RESET = "AT+RESET"
    CMD_MODE = "AT+MODE"
    CMD_NETWORKID = "AT+NETWORKID"
    CMD_SEND = "AT+SEND"
    RESPONSE_OK = "OK"
    RESPONSE_ERROR = "ERROR"

    # Response patterns
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

    def connect(self) -> bool:
        """
        Open serial connection and configure module.

        Returns:
            True if connection successful
        """
        if not SERIAL_AVAILABLE:
            print("Warning: pyserial not available, UWB in simulation mode")
            return False

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

    def configure_as_anchor(self) -> bool:
        """
        Configure module as ANCHOR for ranging.

        Returns:
            True if configuration successful
        """
        if not self._connected:
            return False

        # Set mode to ANCHOR
        if not self._send_command(f"{self.CMD_MODE}=ANCHOR"):
            return False

        # Set network ID
        network_id = self._config.network_id
        if not self._send_command(f"{self.CMD_NETWORKID}={network_id:04X}"):
            return False

        return True

    def configure_as_tag(self) -> bool:
        """
        Configure module as TAG for ranging.

        Returns:
            True if configuration successful
        """
        if not self._connected:
            return False

        # Set mode to TAG
        if not self._send_command(f"{self.CMD_MODE}=TAG"):
            return False

        # Set network ID
        network_id = self._config.network_id
        if not self._send_command(f"{self.CMD_NETWORKID}={network_id:04X}"):
            return False

        return True

    def get_range(self) -> Optional[float]:
        """
        Get distance to TAG in mm.

        Reads available data from serial port and parses range values.

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

                # Parse range values
                matches = self.RANGE_PATTERN.findall(text)
                if matches:
                    # Take most recent match
                    tag_id, distance_cm = matches[-1]
                    distance_mm = int(distance_cm) * 10  # Convert cm to mm

                    self._last_range_mm = float(distance_mm)
                    self._last_range_time = time.time()

                    return self._last_range_mm

                return None

            except Exception as e:
                print(f"UWB read error: {e}")
                return None

    def poll_range(self) -> Optional[float]:
        """
        Actively poll for range measurement.

        Sends a range request and waits for response.

        Returns:
            Distance in mm, or None if no response
        """
        if not self._connected or self._serial is None:
            return None

        with self._lock:
            try:
                # Clear input buffer
                self._serial.reset_input_buffer()

                # Send range request (this triggers ranging in some firmware)
                self._serial.write(b"AT+RANGE?\r\n")

                # Wait for response
                time.sleep(0.05)

                # Read response
                if self._serial.in_waiting > 0:
                    data = self._serial.read(self._serial.in_waiting)
                    text = data.decode('utf-8', errors='ignore')

                    matches = self.RANGE_PATTERN.findall(text)
                    if matches:
                        tag_id, distance_cm = matches[-1]
                        distance_mm = int(distance_cm) * 10

                        self._last_range_mm = float(distance_mm)
                        self._last_range_time = time.time()

                        return self._last_range_mm

                return None

            except Exception as e:
                print(f"UWB poll error: {e}")
                return None

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
                time.sleep(0.1)

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
    """

    def __init__(
        self,
        anchor1_config: UWBModuleConfig,
        anchor2_config: UWBModuleConfig
    ):
        """
        Initialize both anchors.

        Args:
            anchor1_config: Configuration for first anchor
            anchor2_config: Configuration for second anchor
        """
        self._anchor1 = RYUW122(anchor1_config)
        self._anchor2 = RYUW122(anchor2_config)

    @property
    def anchor1(self) -> RYUW122:
        """Get first anchor module."""
        return self._anchor1

    @property
    def anchor2(self) -> RYUW122:
        """Get second anchor module."""
        return self._anchor2

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
        Get ranges from both anchors.

        Returns:
            Tuple of (range1_mm, range2_mm) - None if reading failed
        """
        range1 = self._anchor1.get_range()
        range2 = self._anchor2.get_range()
        return (range1, range2)

    def poll_ranges(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Actively poll ranges from both anchors.

        Returns:
            Tuple of (range1_mm, range2_mm) - None if reading failed
        """
        range1 = self._anchor1.poll_range()
        range2 = self._anchor2.poll_range()
        return (range1, range2)

    def close(self) -> None:
        """Close both anchor connections."""
        self._anchor1.close()
        self._anchor2.close()

    @property
    def is_connected(self) -> bool:
        """Check if both anchors are connected."""
        return self._anchor1.is_connected and self._anchor2.is_connected
