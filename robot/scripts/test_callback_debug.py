#!/usr/bin/env python3
"""
Debug lgpio callback mechanism.
"""

import time
import threading
import lgpio

LEFT_A = 17

print("Debug lgpio callback mechanism")
print(f"Monitoring pin {LEFT_A}")
print()

# Check internal state
print("lgpio internal threads:")
print(f"  _callback_thread class: {lgpio._callback_thread}")
print(f"  _notify_thread: {lgpio._notify_thread}")
print()

h = lgpio.gpiochip_open(0)
print(f"Opened handle: {h}")

# Claim input with pull-up
lgpio.gpio_claim_input(h, LEFT_A, lgpio.SET_PULL_UP)
print(f"Claimed pin {LEFT_A} as input with pull-up")

# Check _notify_thread after opening
print(f"_notify_thread after open: {lgpio._notify_thread}")

# Custom callback with debug
callback_count = 0
def my_callback(chip, gpio, level, timestamp):
    global callback_count
    callback_count += 1
    print(f"CALLBACK! chip={chip} gpio={gpio} level={level} ts={timestamp}")

# Register callback
print("\nRegistering callback on BOTH_EDGES...")
cb = lgpio.callback(h, LEFT_A, lgpio.BOTH_EDGES, my_callback)
print(f"Callback object: {cb}")
print(f"Callback type: {type(cb)}")

# Check callback attributes
print(f"Callback attributes: {[a for a in dir(cb) if not a.startswith('_')]}")

# Check _notify_thread after callback registration
print(f"\n_notify_thread after callback: {lgpio._notify_thread}")
if hasattr(lgpio._notify_thread, 'is_alive'):
    print(f"_notify_thread.is_alive(): {lgpio._notify_thread.is_alive()}")

# List all running threads
print(f"\nAll threads: {threading.enumerate()}")

print("\n" + "="*60)
print("Spin the wheel - watching for callbacks...")
print("Also polling pin state for comparison")
print("Press Ctrl+C to exit")
print("="*60 + "\n")

try:
    last_state = lgpio.gpio_read(h, LEFT_A)
    poll_edges = 0
    
    while True:
        state = lgpio.gpio_read(h, LEFT_A)
        if state != last_state:
            poll_edges += 1
            print(f"Poll detected edge: {last_state} -> {state} (total poll: {poll_edges}, callback: {callback_count})")
        last_state = state
        
        time.sleep(0.001)  # 1ms poll
        
except KeyboardInterrupt:
    print(f"\n\nFinal: poll_edges={poll_edges}, callback_count={callback_count}")
finally:
    cb.cancel()
    lgpio.gpiochip_close(h)
    print("Done.")
