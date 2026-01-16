"""
Shared lgpio handle manager for motor PWM output.

Provides a single lgpio handle shared across motor drivers.
This avoids conflicts from multiple gpiochip_open() calls.

Note: Encoders use libgpiod instead of lgpio because lgpio callbacks
don't work reliably on kernel 6.x. The two libraries can coexist since
they use different GPIO pins.
"""

import threading
from typing import Optional

try:
    import lgpio
    LGPIO_AVAILABLE = True
except ImportError:
    LGPIO_AVAILABLE = False


class GPIOManager:
    """
    Singleton manager for shared lgpio handle.

    All GPIO users should get their handle from here to avoid conflicts.
    """

    _handle: Optional[int] = None
    _refcount: int = 0
    _lock = threading.Lock()

    @classmethod
    def get_handle(cls) -> Optional[int]:
        """
        Get shared lgpio handle, opening if necessary.

        Returns:
            lgpio handle, or None if lgpio not available
        """
        if not LGPIO_AVAILABLE:
            return None

        with cls._lock:
            if cls._handle is None:
                cls._handle = lgpio.gpiochip_open(0)
            cls._refcount += 1
            return cls._handle

    @classmethod
    def release_handle(cls) -> None:
        """
        Release a reference to the shared handle.

        Closes the handle when refcount reaches zero.
        """
        if not LGPIO_AVAILABLE:
            return

        with cls._lock:
            cls._refcount -= 1
            if cls._refcount <= 0 and cls._handle is not None:
                try:
                    lgpio.gpiochip_close(cls._handle)
                except Exception:
                    pass
                cls._handle = None
                cls._refcount = 0

    @classmethod
    def is_available(cls) -> bool:
        """Check if lgpio is available."""
        return LGPIO_AVAILABLE
