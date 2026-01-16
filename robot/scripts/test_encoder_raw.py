#!/usr/bin/env python3
"""
Raw encoder test - minimal lgpio test for debugging.

Tests encoder pins directly using polling (not callbacks).
"""

import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import lgpio

# Encoder pins from gpio_pins.yaml
LEFT_A = 17
LEFT_B = 27
RIGHT_A = 22
RIGHT_B = 23


def main():
    print("Raw encoder test - lgpio (polling mode)")
    print(f"Left encoder:  A={LEFT_A}, B={LEFT_B}")
    print(f"Right encoder: A={RIGHT_A}, B={RIGHT_B}")
    print()
    
    # Open GPIO chip
    h = lgpio.gpiochip_open(0)
    print(f"Opened gpiochip0, handle={h}")
    
    # Setup all encoder pins as inputs with pull-ups
    lgpio.gpio_claim_input(h, LEFT_A, lgpio.SET_PULL_UP)
    lgpio.gpio_claim_input(h, LEFT_B, lgpio.SET_PULL_UP)
    lgpio.gpio_claim_input(h, RIGHT_A, lgpio.SET_PULL_UP)
    lgpio.gpio_claim_input(h, RIGHT_B, lgpio.SET_PULL_UP)
    print("Configured pins as inputs with pull-ups")
    
    # Read initial states
    last_la = lgpio.gpio_read(h, LEFT_A)
    last_ra = lgpio.gpio_read(h, RIGHT_A)
    print(f"Initial states: LA={last_la} RA={last_ra}")
    print()
    print("Spin the wheels manually - counts should change")
    print("Press Ctrl+C to exit")
    print()
    
    # Counters
    left_count = 0
    right_count = 0
    
    try:
        while True:
            # Read current pin states
            la = lgpio.gpio_read(h, LEFT_A)
            lb = lgpio.gpio_read(h, LEFT_B)
            ra = lgpio.gpio_read(h, RIGHT_A)
            rb = lgpio.gpio_read(h, RIGHT_B)
            
            # Detect rising edge on left channel A
            if la == 1 and last_la == 0:
                # Rising edge detected - check B for direction
                if lb == 0:
                    left_count += 1
                else:
                    left_count -= 1
            
            # Detect rising edge on right channel A
            if ra == 1 and last_ra == 0:
                # Rising edge detected - check B for direction
                if rb == 0:
                    right_count += 1
                else:
                    right_count -= 1
            
            last_la = la
            last_ra = ra
            
            # Show state and counts
            print(f"\rPins: LA={la} LB={lb} RA={ra} RB={rb} | "
                  f"Counts: L={left_count:+6d} R={right_count:+6d}  ",
                  end="", flush=True)
            
            # Poll fast to catch edges
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        lgpio.gpiochip_close(h)
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
