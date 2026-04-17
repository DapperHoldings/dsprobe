"""
Extended Kalman Filter (EKF) for spacecraft navigation.
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from config.settings import NavConfig
from core.state import State
from core.measurement import Measurement, MeasurementKind
from sensors.imu import IMUReading

class EKF:
    """
    Extended Kalman Filter for beacon-based navigation.
    
    State vector (minimal):
        x = [px, py, pz, vx, vy, vz]^T  (position km, velocity km/s)
        
    Can be extended to include:
        - Attitude quaternion [4]
        - Angular velocity [3]
        - Accelerometer bias [3]
        - Gyroscope bias [3]
        - Clock bias/drift [2]
    """
    
    def __init__(self, config: NavConfig):
        self.config = config
        
        # State dimension (minimal)
        self.n_x = 6
        
        # Initialize state vector and covariance
        self.state = np.zeros(self.n_x)
        self.covariance = np.eye(self.n_x) * 1e9  # Large initial uncertainty (km^2)
        
        # Process noise
        self.Q = self._build_process_noise()
        
        # Measurement cache
        self.last_update_time: Optional[float] = None
        
        # Debug history
        self.history = [] if config.debug_logging else None
        
    def _build_process_noise(self) -> np.ndarray:
        """Build process noise covariance matrix Q"""
        # Assume continuous-time noise PSDs converted to discrete
        dt = 1.0  # unit time; will scale in predict
        q_pos = self.config.process_noise_pos  # km^2/s^3
        q_vel = self.config.process_noise_vel  # km^2/s
        
        # State: [p, v]
        Q = np.zeros((self.n_x, self.n_x))
        Q[0:3, 0:3] = (q_pos/3) * dt**3  # position integrated from acceleration PSD
        Q[0:3, 3:6] = (q_pos/2) * dt**2
        Q[3:6, 0:3] = (q_pos/2) * dt**2
        Q[3:6, 3:6] = q_vel * dt  # velocity random walk
        return Q
    
    def predict(self, dt: float, 
                imu_data: Optional[IMUReading] = None,
                control: Optional[np.ndarray] = None):
        """
        EKF prediction step (time update).
        
        Args:
            dt: Timestep (s)
            imu_data: Optional IMU reading for better dynamics model
            control: Control input (e.g., thrust) [3] in km/s^2
        """
        # State transition matrix (constant velocity model)
        F = np.eye(self.n_x)
        F[0:3, 3:6] = np.eye(3) * dt
        
        # If IMU available, we could do better (e.g., integrate accel)
        if imu_data is not None and self.config.enable_imu:
            # Assume IMU accelerometer in body frame -> need attitude to rotate
            # For now, skip; would need attitude state
            pass
            
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance: P = F P F^T + Q(dt)
        # Scale Q by dt
        Q_scaled = self.Q * dt
        self.covariance = F @ self.covariance @ F.T + Q_scaled
        
        if self.history is not None:
            self.history.append({
                'time': self.last_update_time,
                'state': self.state.copy(),
                'covariance': self.covariance.copy(),
                'type': 'predict'
            })
    
    def _measurement_jacobian(self, 
                             beacon_pos: np.ndarray,
                             measurement: Measurement) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute measurement Jacobian H and noise covariance R for a measurement.
        """
        p = self.state[0:3]
        v = self.state[3:6]
        r_vec = p - beacon_pos
        r = np.linalg.norm(r_vec)
        
        if r < 1e-9:
            r_vec = np.array([1e-9, 0, 0])
            r = 1e-9
            
        r_hat = r_vec / r
        
        if measurement.kind == MeasurementKind.RANGE:
            # h(x) = ||p - b||
            H = np.zeros((1, self.n_x))
            H[0, 0:3] = r_hat
            R = np.array([[measurement.uncertainty**2]])
            
        elif measurement.kind == MeasurementKind.DIRECTION:
            # h(x) = (p - b)/||p-b||   (3-vector)
            H = np.zeros((3, self.n_x))
            # d/dp of (p-b)/r = (I - r_hat r_hat^T)/r
            H[:, 0:3] = (np.eye(3) - np.outer(r_hat, r_hat)) / r
            # Direction derivatives wrt vel = 0
            # Measurement noise: assume isotropic angular error σ
            # Covariance in direction: approximately σ^2 * (I - r_hat r_hat^T)
            R = measurement.uncertainty**2 * (np.eye(3) - np.outer(r_hat, r_hat))
            
        elif measurement.kind == MeasurementKind.RANGE_RATE:
            # h(x) = (p-b)^T v / ||p-b|| = r_hat^T v
            H = np.zeros((1, self.n_x))
            H[0, 0:3] = 0.0  # derivative of range-rate wrt position is more complex (v*dr/dp)
            # Actually: d(r_hat^T v)/dp = ... complex
            # Use simplified: ignore position dependence for Doppler
            H[0, 3:6] = r_hat.T
            R = np.array([[measurement.uncertainty**2]])
            
        elif measurement.kind == MeasurementKind.BOTH:
            # Range and direction stacked: [range, dir_x, dir_y, dir_z]
            H = np.zeros((4, self.n_x))
            # Range part
            H[0, 0:3] = r_hat
            # Direction part
            H[1:, 0:3] = (np.eye(3) - np.outer(r_hat, r_hat)) / r
            # R: block-diagonal
            R_range = np.array([[measurement.uncertainty**2]])
            # Direction covariance: σ^2 * (I - r_hat r_hat^T)
            R_dir = measurement.uncertainty**2 * (np.eye(3) - np.outer(r_hat, r_hat))
            R = np.block([
                [R_range, np.zeros((1,3))],
                [np.zeros((3,1)), R_dir]
            ])
        else:
            raise ValueError(f"Unsupported measurement kind: {measurement.kind}")
            
        return H, R
    
    def update(self, measurement: Measurement, beacon) -> bool:
        """
        EKF update step with single measurement.
        
        Returns:
            True if update succeeded
        """
        # Get beacon position at measurement time
        if measurement.timestamp:
            epoch = self._datetime_to_epoch(measurement.timestamp)
            b_pos = beacon.get_position(epoch)
        else:
            b_pos = beacon.get_position(0.0)
            
        # Compute measurement prediction
        predicted = self._predict_measurement(beacon, measurement.kind, b_pos)
        
        # Build innovation
        z_meas = measurement.as_vector()
        z_pred = predicted.as_vector()
        innovation = z_meas - z_pred
        
        # Get Jacobian and measurement noise
        H, R = self._measurement_jacobian(b_pos, measurement)
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + R
        
        # Check singularity
        if np.linalg.cond(S) > 1e12:
            return False
            
        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ innovation
        
        # Update covariance (Joseph form for numerical stability)
        I = np.eye(self.n_x)
        I_KH = I - K @ H
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ R @ K.T
        
        # Ensure symmetry
        self.covariance = (self.covariance + self.covariance.T) / 2
        
        # Store innovation for diagnostics
        measurement.innovation = innovation
        measurement.processed = True
        
        if self.history is not None:
            self.history.append({
                'time': measurement.timestamp,
                'measurement': measurement.to_dict(),
                'innovation': innovation.tolist(),
                'S': np.diag(S).tolist(),
                'type': 'update'
            })
            
        return True
    
    def _predict_measurement(self, beacon, kind: MeasurementKind, 
                           beacon_pos: np.ndarray) -> Measurement:
        """Predict measurement from current state"""
        p = self.state[0:3]
        r_vec = p - beacon_pos
        r = np.linalg.norm(r_vec)
        r_hat = r_vec / max(r, 1e-9)
        
        if kind == MeasurementKind.RANGE:
            value = r
        elif kind == MeasurementKind.DIRECTION:
            value = r_hat
        elif kind == MeasurementKind.RANGE_RATE:
            v = self.state[3:6]
            value = r_hat @ v
        elif kind == MeasurementKind.BOTH:
            value = np.hstack([r, r_hat])
        else:
            raise ValueError(f"Unknown kind: {kind}")
            
        # Create dummy measurement (no uncertainty needed)
        return Measurement(
            beacon_id="",
            timestamp=None,
            kind=kind,
            value=value,
            uncertainty=0.0
        )
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current state estimate and covariance"""
        return self.state.copy(), self.covariance.copy()
    
    def set_state(self, state: np.ndarray, covariance: np.ndarray):
        """Set state manually (e.g., from batch estimation)"""
        if len(state) != self.n_x:
            raise ValueError(f"State length mismatch: {len(state)} vs {self.n_x}")
        self.state = state.copy()
        self.covariance = covariance.copy()
        
    def reset(self, initial_state: np.ndarray, 
              initial_covariance: Optional[np.ndarray] = None):
        """Reset filter to initial conditions"""
        self.state = initial_state.copy()
        if initial_covariance is not None:
            self.covariance = initial_covariance.copy()
        else:
            self.covariance = np.eye(self.n_x) * 1e6
        self.last_update_time = None
        
    @staticmethod
    def _datetime_to_epoch(dt: datetime) -> float:
        """Convert datetime to seconds since J2000"""
        j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        return (dt - j2000).total_seconds()