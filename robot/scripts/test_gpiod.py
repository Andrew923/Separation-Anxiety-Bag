#!/usr/bin/env python3
"""
Test libgpiod edge detection for encoders.
"""

import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')

import gpiod
import time
import threading

LEFT_A = 17
LEFT_B = 27
CHIP = 'gpiochip0'

print("Testing libgpiod edge detection")
print(f"Monitoring pin {LEFT_A} (channel A)")
print(f"Reading pin {LEFT_B} (channel B) for direction")
print()

# Open chip
chip = gpiod.Chip(CHIP)
print(f"Opened {CHIP}")

# Get lines
line_a = chip.get_line(LEFT_A)
line_b = chip.get_line(LEFT_B)

# Request line A for edge events with pull-up
line_a.request(
    consumer="encoder_test",
    type=gpiod.LINE_REQ_EV_RISING_EDGE,
    flags=gpiod.LINE_REQ_FLAG_BIAS_PULL_UP
)

# Request line B as input with pull-up  
line_b.request(
    consumer="encoder_test",
    type=gpiod.LINE_REQ_DIR_IN,
    flags=gpiod.LINE_REQ_FLAG_BIAS_PULL_UP
)

print(f"Line A configured for rising edge events")
print(f"Line B configured as input")

count = 0

print("\nSpin the wheel - counts should change")
print("Press Ctrl+C to exit")
print()

try:
    while True:
        # Wait for event with timeout (returns True if event occurred)
        if line_a.event_wait(nsec=10_000_000):  # 10ms timeout
            event = line_a.event_read()
            
            # Read B state for direction
            b_state = line_b.get_value()
            
            if b_state == 0:
                count += 1
            else:
                count -= 1
            
            print(f"\rCount: {count:+6d}  (B={b_state})  ", end="", flush=True)

except KeyboardInterrupt:
    print(f"\n\nFinal count: {count}")
finally:
    line_a.release()
    line_b.release()
    chip.close()
    print("Done.")
