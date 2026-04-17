"""
Particle Filter for non-Gaussian navigation.
Useful for multimodal distributions (e.g., ambiguous landmark matching).
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from config.settings import NavConfig
from core.measurement import Measurement, MeasurementKind
from filters.ekf import EKF

class ParticleFilter:
    """
    Sequential Monte Carlo Particle Filter.
    """
    
    def __init__(self, config: NavConfig, n_particles: int = 1000):
        self.config = config
        self.n_particles = n_particles
        self.n_x = 6  # state dim
        
        # Initialize particles uniformly in large box
        self.particles = np.random.uniform(-1e8, 1e8, (n_particles, self.n_x))
        self.weights = np.ones(n_particles) / n_particles
        
        # Process noise
        self.Q = np.diag([
            config.process_noise_pos * 0.1,
            config.process_noise_pos * 0.1,
            config.process_noise_pos * 0.1,
            config.process_noise_vel,
            config.process_noise_vel,
            config.process_noise_vel
        ])
        
    def predict(self, dt: float, 
                imu_data: Optional = None,
                control: Optional[np.ndarray] = None):
        """Propagate particles through motion model"""
        # Constant velocity with process noise
        F = np.eye(self.n_x)
        F[0:3, 3:6] = np.eye(3) * dt
        
        for i in range(self.n_particles):
            self.particles[i] = F @ self.particles[i]
            # Add process noise
            self.particles[i] += np.random.multivariate_normal(
                np.zeros(self.n_x), self.Q * dt
            )
            
    def update(self, measurement: Measurement, beacon) -> float:
        """Update particle weights based on measurement likelihood"""
        # Compute log-likelihood for each particle
        log_weights = np.zeros(self.n_particles)
        
        for i in range(self.n_particles):
            pred = self._predict_measurement(self.particles[i], beacon, measurement)
            residual = measurement.as_vector() - pred
            R = measurement.as_covariance()
            
            # Multivariate Gaussian log-likelihood
            try:
                log_prob = -0.5 * (residual.T @ np.linalg.inv(R) @ residual)
                log_prob -= 0.5 * (len(residual) * np.log(2*np.pi) + np.linalg.slogdet(R)[1])
                log_weights[i] = log_prob
            except np.linalg.LinAlgError:
                log_weights[i] = -np.inf
                
        # Convert to weights (log-sum-exp trick)
        max_log = np.max(log_weights)
        weights = np.exp(log_weights - max_log)
        weights /= np.sum(weights)
        
        # Update
        self.weights = weights
        
        # Compute effective sample size
        ess = 1.0 / np.sum(weights**2)
        
        # Resample if needed
        if ess < self.n_particles / 2:
            self._resample()
            
        return ess
    
    def _predict_measurement(self, state: np.ndarray, beacon, kind: MeasurementKind) -> np.ndarray:
        """Predict measurement from state (same as EKF)"""
        p = state[0:3]
        # Use same logic as EKF
        # For brevity, recycle EKF method
        ekf = EKF(self.config)
        ekf.state = state
        # Need beacon position at some time? Use 0
        b_pos = beacon.get_position(0.0)
        dummy_meas = Measurement("", None, kind, 0, 0)
        return ekf._predict_measurement(beacon, kind, b_pos).as_vector()
    
    def _resample(self):
        """Systematic resampling"""
        cumsum = np.cumsum(self.weights)
        cumsum /= cumsum[-1]
        
        # Draw uniformly
        positions = (np.arange(self.n_particles) + np.random.uniform(0, 1)) / self.n_particles
        
        new_particles = np.zeros_like(self.particles)
        j = 0
        for pos in positions:
            while cumsum[j] < pos:
                j += 1
            new_particles[len(new_particles)-1] = self.particles[j]
            
        self.particles = new_particles
        self.weights.fill(1.0 / self.n_particles)
        
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get mean and covariance from particle set"""
        mean = np.average(self.particles, weights=self.weights, axis=0)
        cov = np.cov(self.particles.T, aweights=self.weights)
        return mean, cov