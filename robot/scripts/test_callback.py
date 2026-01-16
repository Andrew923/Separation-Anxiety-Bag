#!/usr/bin/env python3
"""
Test lgpio callback mechanism directly.
"""

import time
import lgpio

LEFT_A = 17

print("Testing lgpio callback mechanism")
print(f"Monitoring pin {LEFT_A}")
print()

h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_input(h, LEFT_A, lgpio.SET_PULL_UP)

# Use default tally callback (no function specified)
cb = lgpio.callback(h, LEFT_A, lgpio.BOTH_EDGES)

print("Callback registered with default tally counter")
print("Spin the left wheel - tally should increase")
print("Press Ctrl+C to exit")
print()

try:
    while True:
        tally = cb.tally()
        level = lgpio.gpio_read(h, LEFT_A)
        print(f"\rLevel: {level} | Tally: {tally}  ", end="", flush=True)
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n\nInterrupted.")
finally:
    cb.cancel()
    lgpio.gpiochip_close(h)
    print("Done.")
