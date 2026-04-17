"""
Unscented Kalman Filter (UKF) for nonlinear navigation.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from config.settings import NavConfig
from core.state import State
from core.measurement import Measurement, MeasurementKind
from filters.ekf import EKF

class UKF(EKF):
    """
    Unscented Kalman Filter for better nonlinear handling.
    Useful for pulsar timing where measurement model is highly nonlinear.
    """
    
    def __init__(self, config: NavConfig, 
                 alpha: float = 0.1,
                 beta: float = 2.0,
                 kappa: float = 0.0):
        super().__init__(config)
        self.alpha = alpha  # Spread of sigma points (0 < alpha <= 1)
        self.beta = beta    # Prior distribution parameter (2 for Gaussian)
        self.kappa = kappa  # Secondary scaling (usually 0)
        
        self.n = self.n_x  # state dimension
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n
        
        # Sigma point weights
        self.Wm = np.zeros(2*self.n + 1)
        self.Wc = np.zeros(2*self.n + 1)
        
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - alpha**2 + beta)
        for i in range(1, 2*self.n + 1):
            self.Wm[i] = 1.0 / (2 * (self.n + self.lambda_))
            self.Wc[i] = 1.0 / (2 * (self.n + self.lambda_))
            
    def _sigma_points(self) -> np.ndarray:
        """
        Generate sigma points.
        
        Returns:
            Array of shape (2*n+1, n_x)
        """
        chi = np.zeros((2*self.n + 1, self.n_x))
        chi[0] = self.state
        
        # Cholesky decomposition of (n+λ)P
        try:
            P_sqrt = np.linalg.cholesky((self.n + self.lambda_) * self.covariance)
        except np.linalg.LinAlgError:
            # Add jitter for PSD
            P_jitter = self.covariance + np.eye(self.n_x) * 1e-6
            P_sqrt = np.linalg.cholesky((self.n + self.lambda_) * P_jitter)
            
        for i in range(self.n):
            chi[i+1] = self.state + P_sqrt[i]
            chi[self.n+i+1] = self.state - P_sqrt[i]
            
        return chi
    
    def predict(self, dt: float, 
                imu_data: Optional = None,
                control: Optional[np.ndarray] = None):
        """UKF prediction step"""
        if dt <= 0:
            return
            
        # Generate sigma points
        chi = self._sigma_points()
        
        # Propagate each sigma point through process model
        chi_pred = np.zeros_like(chi)
        F = np.eye(self.n_x)
        F[0:3, 3:6] = np.eye(3) * dt
        
        for i in range(2*self.n + 1):
            chi_pred[i] = F @ chi[i]
            
        # Compute predicted mean
        self.state = np.sum(self.Wm[i] * chi_pred[i] for i in range(2*self.n + 1))
        
        # Compute predicted covariance
        self.covariance = np.zeros((self.n_x, self.n_x))
        for i in range(2*self.n + 1):
            diff = chi_pred[i] - self.state
            self.covariance += self.Wc[i] * np.outer(diff, diff)
        self.covariance += self.Q * dt
        
        # Ensure symmetry
        self.covariance = (self.covariance + self.covariance.T) / 2
        
    def update(self, measurement: Measurement, beacon) -> bool:
        """UKF update step"""
        # Get beacon position
        if measurement.timestamp:
            epoch = self._datetime_to_epoch(measurement.timestamp)
            b_pos = beacon.get_position(epoch)
        else:
            b_pos = beacon.get_position(0.0)
            
        # Generate sigma points
        chi = self._sigma_points()
        
        # Transform sigma points through measurement function h(x)
        Z = np.zeros((2*self.n + 1, measurement.get_measurement_matrix_size()))
        for i in range(2*self.n + 1):
            Z[i] = self._h(chi[i], beacon, measurement.kind, b_pos)
            
        # Predicted measurement mean
        z_pred = np.sum(self.Wm[i] * Z[i] for i in range(2*self.n + 1))
        
        # Innovation covariance Pzz
        Pzz = np.zeros((len(z_pred), len(z_pred)))
        for i in range(2*self.n + 1):
            dz = Z[i] - z_pred
            Pzz += self.Wc[i] * np.outer(dz, dz)
            
        # Cross-covariance Pxz
        Pxz = np.zeros((self.n_x, len(z_pred)))
        for i in range(2*self.n + 1):
            dx = chi[i] - self.state
            dz = Z[i] - z_pred
            Pxz += self.Wc[i] * np.outer(dx, dz)
            
        # Measurement noise R
        R = measurement.as_covariance()
        
        # Innovation
        z_meas = measurement.as_vector()
        y = z_meas - z_pred
        
        # Kalman gain
        try:
            K = Pxz @ np.linalg.inv(Pzz + R)
        except np.linalg.LinAlgError:
            return False
            
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance (Utah form)
        P = self.covariance - K @ (Pzz + R) @ K.T
        self.covariance = 0.5 * (P + P.T)  # enforce symmetry
        
        measurement.innovation = y
        measurement.processed = True
        
        return True
    
    def _h(self, 
           state: np.ndarray,
           beacon,
           kind: MeasurementKind,
           beacon_pos: np.ndarray) -> np.ndarray:
        """Measurement function h(x)"""
        p = state[0:3]
        r_vec = p - beacon_pos
        r = np.linalg.norm(r_vec)
        if r < 1e-9:
            r_vec = np.array([1e-9, 0, 0])
            r = 1e-9
        r_hat = r_vec / r
        
        if kind == MeasurementKind.RANGE:
            return np.array([r])
        elif kind == MeasurementKind.DIRECTION:
            return r_hat
        elif kind == MeasurementKind.BOTH:
            return np.hstack([r, r_hat])
        else:
            raise ValueError(f"Kind {kind} not supported in UKF yet")