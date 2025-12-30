# Separation Anxiety Bag

A person-following robot using stereo vision for obstacle avoidance and UWB for person tracking.

## Overview

This robot autonomously follows a person using:
- **UWB (Ultra-Wideband) tracking**: Two RYUW122 modules on the robot triangulate the position of a tag worn by the person
- **Stereo vision obstacle avoidance**: Depth maps from stereo cameras feed into a Vector Field Histogram (VFH) algorithm
- **Differential drive**: PID-controlled motors with encoder feedback for precise movement

## Hardware

| Component | Model | Purpose |
|-----------|-------|---------|
| Computer | Raspberry Pi 4B | Main controller |
| Stereo Camera | Side-by-side USB camera | Depth perception |
| UWB Modules | REYAX RYUW122 (x3) | Person tracking (2 on robot, 1 on person) |
| Motor Driver | Cytron MDD10A | Dual 10A motor control |
| Motors | DFRobot FIT0186 (x2) | 12V DC with encoders |
| Wheels | 52mm diameter | Differential drive |

## Project Structure

```
├── vision/                  # Stereo vision system
│   ├── config/
│   │   └── default_config.yaml
│   ├── src/
│   │   ├── camera.py        # Stereo camera capture
│   │   ├── calibration.py   # Camera calibration
│   │   ├── stereo_matcher.py # Depth estimation (StereoSGBM)
│   │   └── utils.py
│   ├── scripts/
│   │   ├── capture_calibration.py
│   │   ├── run_calibration.py
│   │   └── stereo_depth.py  # Real-time depth visualization
│   └── data/
│       └── calibration_data/
│
├── robot/                   # Robot control system
│   ├── config/
│   │   ├── gpio_pins.yaml   # GPIO pin assignments
│   │   └── robot_config.yaml # Robot parameters
│   ├── src/
│   │   ├── motor_driver.py  # Cytron MDD10A driver
│   │   ├── encoder.py       # Quadrature encoder reading
│   │   ├── differential_drive.py # PID control
│   │   ├── odometry.py      # Wheel odometry
│   │   ├── uwb_tracker.py   # RYUW122 communication
│   │   ├── uwb_triangulation.py # Position triangulation
│   │   ├── depth_to_polar.py # Depth to polar conversion
│   │   ├── vfh.py           # Vector Field Histogram
│   │   └── navigation.py    # Navigation state machine
│   └── scripts/
│       ├── run_robot.py     # Main control loop
│       ├── calibrate_uwb.py # UWB calibration
│       ├── test_motors.py   # Motor testing
│       └── test_uwb.py      # UWB testing
│
└── README.md
```

## Installation

### Prerequisites

- Raspberry Pi 4B with Raspberry Pi OS
- Python 3.8+

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Separation-Anxiety-Bag.git
cd Separation-Anxiety-Bag

# Install vision dependencies
pip install -r vision/requirements.txt

# Install robot dependencies
pip install -r robot/requirements.txt
```

### Hardware Wiring

Default GPIO configuration (BCM numbering):

| Function | GPIO Pin |
|----------|----------|
| Left Motor PWM | 12 |
| Left Motor DIR | 5 |
| Right Motor PWM | 13 |
| Right Motor DIR | 6 |
| Left Encoder A | 17 |
| Left Encoder B | 27 |
| Right Encoder A | 22 |
| Right Encoder B | 23 |
| UWB Anchor 1 | /dev/ttyAMA0 |
| UWB Anchor 2 | /dev/ttyUSB0 |

Edit `robot/config/gpio_pins.yaml` to change pin assignments.

## Usage

### 1. Calibrate Stereo Camera

```bash
# Capture calibration images (use 7x10 checkerboard, 25mm squares)
python vision/scripts/capture_calibration.py --target 20

# Process calibration
python vision/scripts/run_calibration.py
```

### 2. Test Hardware

```bash
# Test motors and encoders
python robot/scripts/test_motors.py

# Test UWB modules
python robot/scripts/test_uwb.py
```

### 3. Calibrate UWB

The robot needs to know which direction is "front" for the UWB triangulation:

```bash
python robot/scripts/calibrate_uwb.py
```

Follow the prompts to stand in front of the robot while it records samples.

### 4. Run the Robot

```bash
python robot/scripts/run_robot.py
```

Options:
- `--no-vision`: Disable stereo vision (UWB-only tracking)
- `--config PATH`: Use custom config file

## Configuration

### Robot Parameters (`robot/config/robot_config.yaml`)

Key settings:

```yaml
robot:
  wheel_diameter_mm: 52.0
  wheel_base_mm: 200.0      # Adjust to your robot
  encoder_cpr: 700          # Counts per revolution

pid:
  left: {kp: 1.5, ki: 0.5, kd: 0.05}
  right: {kp: 1.5, ki: 0.5, kd: 0.05}

navigation:
  target_follow_distance_mm: 1500  # How far to stay from person
  max_linear_speed_mm_s: 500
  max_angular_speed_deg_s: 90

vfh:
  num_sectors: 72           # 5-degree sectors
  min_height_mm: 50         # Ignore floor
  max_height_mm: 500        # Ignore above robot
```

## How It Works

### Navigation State Machine

1. **FOLLOWING**: Person is visible and path is clear - drive toward them
2. **AVOIDING**: Obstacle in direct path - steer to nearest safe direction
3. **SPINNING**: All forward directions blocked - rotate to find clear path
4. **LOST_TARGET**: UWB signal lost - stop and wait
5. **STOPPED**: Emergency stop (blocked too long or manual stop)

### VFH Obstacle Avoidance

The Vector Field Histogram algorithm:
1. Converts depth map to polar obstacle histogram (72 sectors, 5° each)
2. Filters by height to ignore floor and overhead obstacles
3. Identifies "valleys" (continuous safe sectors)
4. Selects the valley closest to the target direction

### UWB Triangulation

With two UWB anchors on the robot:
1. Each anchor measures distance to the person's tag
2. Circle-circle intersection computes the angle to the person
3. Calibration distinguishes front from back

## Troubleshooting

### Motors not moving
- Check 12V power supply to MDD10A
- Verify GND connection between Pi and motor driver
- Test with `python robot/scripts/test_motors.py`

### UWB not reading
- Ensure correct UART ports in `gpio_pins.yaml`
- Check that all modules have the same network ID
- Verify baud rate (default: 115200)

### Poor depth quality
- Re-run stereo calibration with more image pairs
- Ensure good lighting conditions
- Check that cameras are rigidly mounted

## License

MIT License
