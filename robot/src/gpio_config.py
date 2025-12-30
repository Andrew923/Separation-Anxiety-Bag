"""
GPIO configuration management for robot hardware.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import yaml


@dataclass
class MotorPins:
    """Pin configuration for a single motor."""
    pwm_pin: int
    dir_pin: int


@dataclass
class EncoderPins:
    """Pin configuration for a quadrature encoder."""
    channel_a: int
    channel_b: int


@dataclass
class UWBConfig:
    """Configuration for a UWB module."""
    uart_port: str


@dataclass
class GPIOConfig:
    """Complete GPIO configuration container."""
    left_motor: MotorPins
    right_motor: MotorPins
    left_encoder: EncoderPins
    right_encoder: EncoderPins
    uwb_anchor1: UWBConfig
    uwb_anchor2: UWBConfig
    pwm_frequency: int = 20000

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'GPIOConfig':
        """Create GPIOConfig from configuration dictionary."""
        motors = config.get('motors', {})
        encoders = config.get('encoders', {})
        uwb = config.get('uwb', {})
        pwm = config.get('pwm', {})

        left_motor = MotorPins(
            pwm_pin=motors.get('left', {}).get('pwm_pin', 12),
            dir_pin=motors.get('left', {}).get('dir_pin', 5)
        )
        right_motor = MotorPins(
            pwm_pin=motors.get('right', {}).get('pwm_pin', 13),
            dir_pin=motors.get('right', {}).get('dir_pin', 6)
        )
        left_encoder = EncoderPins(
            channel_a=encoders.get('left', {}).get('channel_a', 17),
            channel_b=encoders.get('left', {}).get('channel_b', 27)
        )
        right_encoder = EncoderPins(
            channel_a=encoders.get('right', {}).get('channel_a', 22),
            channel_b=encoders.get('right', {}).get('channel_b', 23)
        )
        uwb_anchor1 = UWBConfig(
            uart_port=uwb.get('anchor1', {}).get('uart_port', '/dev/ttyAMA0')
        )
        uwb_anchor2 = UWBConfig(
            uart_port=uwb.get('anchor2', {}).get('uart_port', '/dev/ttyUSB0')
        )

        return cls(
            left_motor=left_motor,
            right_motor=right_motor,
            left_encoder=left_encoder,
            right_encoder=right_encoder,
            uwb_anchor1=uwb_anchor1,
            uwb_anchor2=uwb_anchor2,
            pwm_frequency=pwm.get('frequency', 20000)
        )

    @classmethod
    def from_yaml(cls, config_path: str) -> 'GPIOConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)


def load_gpio_config(config_path: Optional[str] = None) -> GPIOConfig:
    """
    Load GPIO configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses default.

    Returns:
        GPIOConfig instance
    """
    if config_path is None:
        config_path = str(
            Path(__file__).parent.parent / 'config' / 'gpio_pins.yaml'
        )
    return GPIOConfig.from_yaml(config_path)


def load_robot_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load robot configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses default.

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = str(
            Path(__file__).parent.parent / 'config' / 'robot_config.yaml'
        )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
