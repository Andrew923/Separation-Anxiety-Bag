#!/usr/bin/env python3
"""
UART loopback test to verify GPIO 14/15 are working.

Usage:
1. Disconnect UWB module
2. Connect GPIO 14 directly to GPIO 15 with a jumper wire
3. Run: python uart_loopback_test.py
"""

import serial
import time
import sys


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyAMA0'

    print(f"=== UART Loopback Test on {port} ===")
    print()
    print("Instructions:")
    print("1. Disconnect any devices from GPIO 14/15")
    print("2. Connect GPIO 14 (TX) directly to GPIO 15 (RX) with a wire")
    print()
    input("Press Enter when ready...")

    try:
        ser = serial.Serial(port, 115200, timeout=1)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception as e:
        print(f"ERROR: Could not open {port}: {e}")
        return 1

    test_data = b"UART_LOOPBACK_TEST_1234567890"

    print(f"\nSending: {test_data}")
    ser.write(test_data)
    time.sleep(0.2)

    received = ser.read(len(test_data) + 10)
    print(f"Received: {received}")

    if received == test_data:
        print("\n[SUCCESS] Loopback working - UART TX/RX are functioning!")
        print("The problem is with the UWB module or its connections.")
        result = 0
    elif received:
        print(f"\n[PARTIAL] Received {len(received)} bytes, expected {len(test_data)}")
        print("Check for loose connections.")
        result = 1
    else:
        print("\n[FAILURE] No data received")
        print("Possible causes:")
        print("  - Loopback wire not connected")
        print("  - Wrong GPIO pins")
        print("  - UART not properly enabled (try reboot)")
        result = 1

    ser.close()
    return result


if __name__ == '__main__':
    sys.exit(main())
