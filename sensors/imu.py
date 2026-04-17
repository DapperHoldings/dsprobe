"""
IMU (Inertial Measurement Unit) simulation and integration.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime, timezone

@dataclass
class IMUReading:
    """Single IMU measurement"""
    timestamp: datetime
    accelerometer: np.ndarray  # km/s^2 in body frame
    gyroscope: np.ndarray  # rad/s in body frame
    temperature: float = 20.0  # Celsius
    
class IMU:
    """
    IMU sensor model with bias, noise, and drift.
    """
    
    def __init__(self, 
                 accel_noise_std: float = 0.01,  # km/s^2
                 gyro_noise_std: float = 0.001,  # rad/s
                 accel_bias: np.ndarray = None,
                 gyro_bias: np.ndarray = None,
                 bias_instability: float = 1e-5,  # random walk
                 correlation_time: float = 1000.0,  # seconds
                 sample_rate: float = 100.0):  # Hz
        """
        Initialize IMU model.
        
        Args:
            accel_noise_std: Accelerometer white noise std
            gyro_noise_std: Gyroscope white noise std
            accel_bias: Initial accelerometer bias (km/s^2)
            gyro_bias: Initial gyroscope bias (rad/s)
            bias_instability: Bias random walk coefficient
            correlation_time: Bias correlation time (for Gauss-Markov)
            sample_rate: Measurement rate (Hz)
        """
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.bias_instability = bias_instability
        self.correlation_time = correlation_time
        self.dt = 1.0 / sample_rate
        
        # Initialize biases (random walk)
        if accel_bias is None:
            accel_bias = np.zeros(3)
        if gyro_bias is None:
            gyro_bias = np.zeros(3)
            
        self.accel_bias = accel_bias.copy()
        self.gyro_bias = gyro_bias.copy()
        
        # For correlated bias simulation
        self.accel_bias_prev = self.accel_bias.copy()
        self.gyro_bias_prev = self.gyro_bias.copy()
        
    def generate_reading(self, 
                        true_accel: np.ndarray,
                        true_gyro: np.ndarray,
                        timestamp: datetime) -> IMUReading:
        """
        Generate a noisy IMU reading from true values.
        
        Args:
            true_accel: True specific force (km/s^2) in body frame
            true_gyro: True angular velocity (rad/s) in body frame
            timestamp: Current time
            
        Returns:
            IMUReading with noise and bias added
        """
        # Update biases (first-order Gauss-Markov)
        alpha = np.exp(-self.dt / self.correlation_time)
        self.accel_bias = alpha * self.accel_bias + \
                         (1-alpha) * np.random.normal(0, self.bias_instability, 3)
        self.gyro_bias = alpha * self.gyro_bias + \
                         (1-alpha) * np.random.normal(0, self.bias_instability, 3)
        
        # Add white noise
        accel_noise = np.random.normal(0, self.accel_noise_std, 3)
        gyro_noise = np.random.normal(0, self.gyro_noise_std, 3)
        
        measured_accel = true_accel + self.accel_bias + accel_noise
        measured_gyro = true_gyro + self.gyro_bias + gyro_noise
        
        return IMUReading(
            timestamp=timestamp,
            accelerometer=measured_accel,
            gyroscope=measured_gyro
        )
    
    def integrate_trajectory(self, 
                           imu_readings: list,
                           initial_state: np.ndarray) -> np.ndarray:
        """
        Integrate IMU readings to get trajectory (dead reckoning).
        
        Args:
            imu_readings: List of IMUReading in chronological order
            initial_state: [x, y, z, vx, vy, vz, qw, qx, qy, qz] (if with attitude)
            
        Returns:
            Array of states at each timestep
        """
        states = [initial_state]
        
        for i in range(1, len(imu_readings)):
            prev = states[-1]
            reading = imu_readings[i]
            dt = (reading.timestamp - imu_readings[i-1].timestamp).total_seconds()
            
            # Extract components
            x = prev[0:3]
            v = prev[3:6]
            if len(prev) >= 10:
                q = prev[6:10]
                q = q / np.linalg.norm(q)
            else:
                q = None
                
            # Specific force in body frame
            f_body = reading.accelerometer
            # Remove bias? In real filter, bias estimated
            # For simulation, we assume bias already in reading
            
            if q is not None:
                # Rotate to inertial
                # Quaternion from body to inertial
                R = self.quat_to_rot_matrix(q)
                f_inertial = R @ f_body
                # Gravity model (simplified)
                # Would need position-dependent gravity
                g = np.array([0, 0, 0])  # placeholder
                a_inertial = f_inertial + g
                
                # Integrate velocity and position
                v_new = v + a_inertial * dt
                x_new = x + v * dt + 0.5 * a_inertial * dt**2
                
                # Integrate attitude from gyro
                omega_body = reading.gyroscope
                # Quaternion derivative: dq/dt = 0.5 * Omega(omega) * q
                # where Omega(omega) = [[0, -omega], [omega, -skew(omega)]]
                # Use simple integration
                dtheta = np.linalg.norm(omega_body) * dt
                if dtheta > 1e-9:
                    axis = omega_body / np.linalg.norm(omega_body)
                    dq = np.array([
                        np.cos(dtheta/2),
                        axis[0]*np.sin(dtheta/2),
                        axis[1]*np.sin(dtheta/2),
                        axis[2]*np.sin(dtheta/2)
                    ])
                    q_new = self.quat_multiply(q, dq)
                else:
                    q_new = q
                    
                states.append(np.hstack([x_new, v_new, q_new/np.linalg.norm(q_new)]))
            else:
                # No attitude
                a_inertial = f_body  # assumes body aligned with inertial (unrealistic)
                v_new = v + a_inertial * dt
                x_new = x + v * dt + 0.5 * a_inertial * dt**2
                states.append(np.hstack([x_new, v_new]))
                
        return np.array(states)
    
    @staticmethod
    def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion multiplication (q1 * q2)"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix (body to inertial)"""
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])