#!/usr/bin/env python3
"""
PID Tuning Script for Differential Drive Robot.

Performs a step response test to tune velocity PID controllers.
Logs data to CSV for analysis.

Usage:
    python tune_pid.py --kp 1.5 --ki 0.5 --kd 0.05 --speed 50 --duration 2.0
"""

import sys
import time
import argparse
import csv
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from robot.src.gpio_config import load_gpio_config, load_robot_config
from robot.src.motor_driver import DualMotorDriver, MotorDriverConfig
from robot.src.encoder import DualEncoders, EncoderConfig
from robot.src.differential_drive import DifferentialDriveController, DriveConfig, PIDConfig
from robot.src.gpio_manager import GPIOManager

def run_step_test(drive, target_rpm, duration, log_file):
    """
    Run a step response test.
    
    Args:
        drive: DifferentialDriveController instance
        target_rpm: Target speed in RPM
        duration: Duration to hold speed in seconds
        log_file: CSV writer object
    """
    print(f"Starting Step Response Test: 0 -> {target_rpm} RPM")
    print("Recording data...")
    
    # Header
    log_file.writerow(["Time", "Target_L", "Actual_L", "Target_R", "Actual_R"])
    
    start_time = time.time()
    
    # Start controller
    drive.start()
    
    # 1. Idle for 1s
    drive.set_wheel_speeds(0, 0)
    time.sleep(1.0)
    
    # 2. Step up
    drive.set_wheel_speeds(target_rpm, target_rpm)
    
    # Record loop
    step_start = time.time()
    while (time.time() - step_start) < duration:
        current_time = time.time() - start_time
        
        # Get data
        target_l, target_r = drive.get_setpoints()
        actual_l, actual_r = drive.get_actual_wheel_speeds()
        
        # Log
        log_file.writerow([f"{current_time:.3f}", target_l, actual_l, target_r, actual_r])
        
        # Print status every 0.2s
        if int(current_time * 10) % 2 == 0:
            print(f"\rTime: {current_time:.1f}s | Target: {target_rpm:.0f} | "
                  f"L: {actual_l:.0f} R: {actual_r:.0f}", end="")
            
        time.sleep(0.05)  # 20Hz logging
        
    # 3. Step down
    drive.set_wheel_speeds(0, 0)
    time.sleep(1.0)
    
    drive.stop()
    print("\nTest Complete.")

def main():
    parser = argparse.ArgumentParser(description="PID Tuning Utility")
    parser.add_argument("--kp", type=float, default=None, help="Proportional gain")
    parser.add_argument("--ki", type=float, default=None, help="Integral gain")
    parser.add_argument("--kd", type=float, default=None, help="Derivative gain")
    parser.add_argument("--speed", type=float, default=50.0, help="Target speed (% of max)")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration of step (s)")
    args = parser.parse_args()
    
    try:
        # Load configs
        gpio_cfg = load_gpio_config()
        robot_cfg = load_robot_config()
        
        # Override PID if provided args
        pid_cfg = robot_cfg['pid']['left'] # Default to config
        kp = args.kp if args.kp is not None else pid_cfg['kp']
        ki = args.ki if args.ki is not None else pid_cfg['ki']
        kd = args.kd if args.kd is not None else pid_cfg['kd']
        
        print(f"Configuration:")
        print(f"  PID: Kp={kp}, Ki={ki}, Kd={kd}")
        print(f"  Deadband: {robot_cfg['motors'].get('deadband', 0)}%")
        
        # Setup Hardware
        left_motor_cfg = MotorDriverConfig(
            in1_pin=gpio_cfg.left_motor.in1_pin,
            in2_pin=gpio_cfg.left_motor.in2_pin,
            ena_pin=gpio_cfg.left_motor.ena_pin,
            pwm_frequency=gpio_cfg.pwm_frequency
        )
        right_motor_cfg = MotorDriverConfig(
            in1_pin=gpio_cfg.right_motor.in1_pin,
            in2_pin=gpio_cfg.right_motor.in2_pin,
            ena_pin=gpio_cfg.right_motor.ena_pin,
            pwm_frequency=gpio_cfg.pwm_frequency
        )
        motors = DualMotorDriver(left_motor_cfg, right_motor_cfg)
        
        left_enc_cfg = EncoderConfig(
            channel_a_pin=gpio_cfg.left_encoder.channel_a,
            channel_b_pin=gpio_cfg.left_encoder.channel_b,
            counts_per_revolution=robot_cfg['robot']['encoder_cpr']
        )
        right_enc_cfg = EncoderConfig(
            channel_a_pin=gpio_cfg.right_encoder.channel_a,
            channel_b_pin=gpio_cfg.right_encoder.channel_b,
            counts_per_revolution=robot_cfg['robot']['encoder_cpr']
        )
        encoders = DualEncoders(left_enc_cfg, right_enc_cfg)
        
        # Setup Drive Controller
        pid_config = PIDConfig(
            kp=kp, ki=ki, kd=kd,
            integral_limit=pid_cfg['integral_limit'],
            output_limit=pid_cfg['output_limit']
        )

        # Heading PID for straight-line correction
        heading_cfg = robot_cfg['pid']['heading']
        heading_pid_config = PIDConfig(
            kp=heading_cfg['kp'],
            ki=heading_cfg['ki'],
            kd=heading_cfg['kd'],
            integral_limit=heading_cfg['integral_limit'],
            output_limit=heading_cfg['output_limit']
        )

        drive_cfg = DriveConfig(
            wheel_diameter_mm=robot_cfg['robot']['wheel_diameter_mm'],
            wheel_base_mm=robot_cfg['robot']['wheel_base_mm'],
            encoder_cpr=robot_cfg['robot']['encoder_cpr'],
            max_rpm=robot_cfg['robot']['max_rpm'],
            deadband=robot_cfg['motors'].get('deadband', 0.0),
            left_pid=pid_config,
            right_pid=pid_config,
            heading_pid=heading_pid_config,
            control_rate_hz=robot_cfg['control']['motor_loop_hz']
        )
        
        drive = DifferentialDriveController(drive_cfg, motors, encoders)
        
        # Wait for initialization
        time.sleep(1.0)
        
        # Calculate target RPM
        target_rpm = (args.speed / 100.0) * robot_cfg['robot']['max_rpm']
        
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pid_test_{timestamp}_kp{kp}_ki{ki}_kd{kd}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            run_step_test(drive, target_rpm, args.duration, writer)
            
        print(f"\nData saved to {filename}")
        print("Analyze this file to check rise time and overshoot.")
        
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if 'drive' in locals():
            drive.stop()
        if 'motors' in locals():
            motors.cleanup()
        if 'encoders' in locals():
            encoders.cleanup()
        if GPIOManager.is_available():
            GPIOManager.release_handle()

if __name__ == "__main__":
    main()
