# AGENTS.md - Coding Agent Guidelines

This document provides guidelines for AI coding agents working in the Separation Anxiety Bag codebase - a person-following robot using UWB tracking and stereo vision obstacle avoidance.

## Project Overview

- **Language**: Python 3.8+
- **Platform**: Raspberry Pi 4B
- **Package Manager**: pip with requirements.txt files
- **No formal build system** (no setup.py, pyproject.toml)

### Key Components

| Module | Purpose |
|--------|---------|
| `robot/` | Motor control, navigation, UWB tracking |
| `vision/` | Stereo camera calibration and depth estimation |

## Build & Install Commands

```bash
# Install vision dependencies
pip install -r vision/requirements.txt

# Install robot dependencies  
pip install -r robot/requirements.txt

# Both modules can be installed in same environment
```

## Test Commands

This project uses manual hardware test scripts (no pytest/unittest framework).

```bash
# Test all motors and encoders
python robot/scripts/test_motors.py --test all

# Test motors only
python robot/scripts/test_motors.py --test motors

# Test encoders only
python robot/scripts/test_motors.py --test encoders

# Test drive (motors + encoders together)
python robot/scripts/test_motors.py --test drive

# Test UWB modules
python robot/scripts/test_uwb.py

# Calibrate UWB
python robot/scripts/calibrate_uwb.py
```

### Running Scripts

```bash
# Main robot control loop
python robot/scripts/run_robot.py
python robot/scripts/run_robot.py --no-vision  # UWB-only mode

# Stereo camera calibration
python vision/scripts/capture_calibration.py --target 20
python vision/scripts/run_calibration.py

# Real-time depth visualization
python vision/scripts/stereo_depth.py
```

## Code Style Guidelines

### Imports

1. **Standard library first**, then third-party, then local imports
2. **Use relative imports** within a module (`from .vfh import VFHResult`)
3. **Use absolute imports** from scripts (`from robot.src.motor_driver import MotorDriver`)
4. Scripts add parent to path: `sys.path.insert(0, str(Path(__file__).parent.parent.parent))`

```python
# Example import order
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

from .local_module import SomeClass
```

### Type Hints

- **Always use type hints** for function signatures
- Use `Optional[T]` for nullable types
- Use `Tuple[...]` for fixed-length sequences
- Use `Dict[K, V]` for dictionaries
- Return type annotations required: `def foo() -> None:`

```python
def compute_speed(
    target_range: float,
    heading_error: float
) -> float:
    """Compute forward speed."""
    ...
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `NavigationController` |
| Functions | snake_case | `compute_linear_speed` |
| Variables | snake_case | `target_angle_deg` |
| Constants | UPPER_CASE | `GPIO_AVAILABLE` |
| Private members | Leading underscore | `_config`, `_state` |
| Units in names | Suffix with unit | `speed_mm_s`, `angle_deg`, `timeout_ms` |

### Docstrings

- **All modules** have a docstring explaining purpose
- **All classes** have a docstring explaining behavior
- **All public methods** have docstrings with Args/Returns sections
- Use Google-style docstrings

```python
"""
Module docstring at top of file.

Brief description of what this module does.
"""

class MyClass:
    """
    Class docstring explaining purpose.

    More detailed description if needed.
    """

    def my_method(self, arg1: float, arg2: Optional[str] = None) -> bool:
        """
        Brief description of method.

        Args:
            arg1: Description of arg1
            arg2: Description of arg2

        Returns:
            Description of return value
        """
```

### Data Classes & Enums

- **Use `@dataclass`** for configuration and data objects
- **Use `Enum`** for state machines and fixed choices
- Provide sensible defaults in dataclasses

```python
@dataclass
class NavigationConfig:
    """Navigation controller configuration."""
    target_follow_distance_mm: float = 1500.0
    max_linear_speed_mm_s: float = 500.0

class NavigationState(Enum):
    """Robot navigation states."""
    IDLE = auto()
    FOLLOWING = auto()
    STOPPED = auto()
```

### Error Handling

1. **Graceful hardware unavailability**: Check if hardware is available before using
2. **Use try/finally** for resource cleanup
3. **Provide fallback behavior** when hardware isn't present

```python
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

class MotorDriver:
    def __init__(self, config):
        self._initialized = False
        if GPIO_AVAILABLE:
            self._setup_gpio()

    def set_speed(self, speed: float) -> None:
        if not self._initialized:
            return  # Silent no-op when hardware unavailable
        # ... actual implementation
```

### Resource Management

- Implement `cleanup()` methods for GPIO/hardware resources
- Use try/finally in scripts to ensure cleanup
- Don't call `GPIO.cleanup()` globally (affects all pins)

```python
try:
    # Main loop
    ...
except KeyboardInterrupt:
    print("Interrupted.")
finally:
    motors.stop()
    motors.cleanup()
    encoders.cleanup()
```

### Configuration

- **YAML files** for all hardware/algorithm parameters
- Configuration files in `*/config/` directories
- Use dataclasses to parse and validate config

Key config files:
- `robot/config/robot_config.yaml` - PID, navigation, VFH parameters
- `robot/config/gpio_pins.yaml` - GPIO pin assignments (BCM numbering)
- `vision/config/default_config.yaml` - Camera and SGBM parameters

## Architecture Patterns

### Module Exports

Each module has `__init__.py` that exports public API:

```python
from .motor_driver import MotorDriver, DualMotorDriver
from .navigation import NavigationController, NavigationState

__all__ = [
    'MotorDriver', 'DualMotorDriver',
    'NavigationController', 'NavigationState',
]
```

### Class Structure

- Private attributes with underscore prefix
- Public properties for read access
- Configuration passed via dataclass in `__init__`

```python
class NavigationController:
    def __init__(self, config: NavigationConfig):
        self._config = config
        self._state = NavigationState.IDLE

    def get_state(self) -> NavigationState:
        """Get current navigation state."""
        return self._state
```

## File Structure

```
robot/
├── config/           # YAML configuration files
├── src/              # Source modules (imported as robot.src.*)
│   └── __init__.py   # Public exports
└── scripts/          # Executable scripts

vision/
├── config/           # YAML configuration files
├── src/              # Source modules (imported as vision.src.*)
│   └── __init__.py   # Public exports
├── scripts/          # Executable scripts
└── data/             # Calibration data, saved files
```

## Common Gotchas

1. **GPIO BCM numbering**: All pin numbers use BCM mode, not physical pin numbers
2. **Units**: Always include units in variable names (`_mm`, `_deg`, `_s`, `_ms`)
3. **NumPy arrays**: Use explicit dtype when creating arrays
4. **OpenCV images**: BGR format, not RGB
5. **Disparity scaling**: StereoSGBM returns int16 scaled by 16 (divide by 16.0 for float)
