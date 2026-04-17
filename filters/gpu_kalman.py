"""
GPU-accelerated Kalman Filter using CuPy.
For batch processing of many particles or high-rate updates.
"""

import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

from typing import Optional, Tuple, List
from dataclasses import dataclass
from config.settings import NavConfig
from core.measurement import Measurement, MeasurementKind

@dataclass
class GPUConfig:
    """Configuration for GPU acceleration"""
    use_gpu: bool = True
    device_id: int = 0
    batch_size: int = 1024
    stream: Optional[object] = None

class GPU_EKF:
    """
    GPU-accelerated EKF that can process multiple independent filters in parallel.
    Useful for:
        - Monte Carlo simulation
        - Multiple spacecraft fleet
        - Beamforming (multiple beacon combinations)
    """
    
    def __init__(self, config: NavConfig, gpu_config: GPUConfig):
        self.config = config
        self.gpu_config = gpu_config
        
        if not GPU_AVAILABLE and gpu_config.use_gpu:
            print("Warning: CuPy not available, falling back to CPU")
            self.use_gpu = False
        else:
            self.use_gpu = gpu_config.use_gpu and GPU_AVAILABLE
            
        # We'll allocate arrays on GPU when needed
        self.states = None  # (batch, n_x)
        self.covariances = None  # (batch, n_x, n_x)
        self.n_x = 6
        
    def init_batch(self, n_batches: int, 
                   initial_states: Optional[np.ndarray] = None,
                   initial_covariances: Optional[np.ndarray] = None):
        """
        Initialize multiple EKF instances on GPU.
        
        Args:
            n_batches: Number of parallel filters
            initial_states: (n_batches, n_x) array or None for zeros
            initial_covariances: (n_batches, n_x, n_x) or None
        """
        if self.use_gpu:
            device = cp.cuda.Device(self.gpu_config.device_id)
            device.use()
            
            if initial_states is None:
                self.states = cp.zeros((n_batches, self.n_x), dtype=cp.float64)
            else:
                self.states = cp.asarray(initial_states)
                
            if initial_covariances is None:
                self.covariances = cp.zeros((n_batches, self.n_x, self.n_x), dtype=cp.float64)
                for i in range(n_batches):
                    self.covariances[i] = cp.eye(self.n_x) * 1e6
            else:
                self.covariances = cp.asarray(initial_covariances)
        else:
            # CPU fallback
            if initial_states is None:
                self.states = np.zeros((n_batches, self.n_x))
            else:
                self.states = initial_states.copy()
            if initial_covariances is None:
                self.covariances = np.array([np.eye(self.n_x)*1e6 for _ in range(n_batches)])
            else:
                self.covariances = initial_covariances.copy()
                
    def predict_batch(self, dt: float, 
                     control_batch: Optional[np.ndarray] = None):
        """
        Vectorized prediction for all batches.
        
        Args:
            dt: Time step (scalar or array of shape (n_batches,))
            control_batch: Optional control inputs (n_batches, 3)
        """
        if self.use_gpu:
            dt_arr = cp.asarray(dt) if not np.isscalar(dt) else cp.array([dt]*self.states.shape[0])
            # Reshape for broadcasting: (batch, 1, 1) * (1, 3, 3)
            F = cp.eye(self.n_x)
            F = cp.broadcast_to(F, (self.states.shape[0], self.n_x, self.n_x)).copy()
            # Set upper right 3x3 to dt * eye
            F[:, 0:3, 3:6] = dt_arr[:, None, None] * cp.eye(3)
            
            # Predict: x = F x
            self.states = cp.matmul(F, self.states[:, :, None]).squeeze(-1)
            
            # Covariance: P = F P F^T + Q*dt
            Q = self._build_process_noise_gpu(dt_arr)
            # We'll do this with einsum for efficiency
            # P_new = F P F^T + Q
            P = self.covariances
            P_new = cp.matmul(F, cp.matmul(P, F.transpose(0,2,1))) + Q
            self.covariances = P_new
        else:
            # CPU loop (slow)
            for i in range(self.states.shape[0]):
                F = np.eye(self.n_x)
                F[0:3, 3:6] = np.eye(3) * dt
                self.states[i] = F @ self.states[i]
                Q = self._build_process_noise_cpu(dt)
                self.covariances[i] = F @ self.covariances[i] @ F.T + Q
                
    def _build_process_noise_gpu(self, dt: cp.ndarray) -> cp.ndarray:
        """Build Q matrix on GPU (vectorized)"""
        batch_size = dt.shape[0]
        Q = cp.zeros((batch_size, self.n_x, self.n_x), dtype=cp.float64)
        # Diagonal
        q_pos = self.config.process_noise_pos
        q_vel = self.config.process_noise_vel
        
        Q[:, 0:3, 0:3] = (q_pos/3) * dt[:, None, None]**3
        Q[:, 0:3, 3:6] = (q_pos/2) * dt[:, None, None]**2
        Q[:, 3:6, 0:3] = (q_pos/2) * dt[:, None, None]**2
        Q[:, 3:6, 3:6] = q_vel * dt[:, None, None]
        return Q
    
    def _build_process_noise_cpu(self, dt: float) -> np.ndarray:
        """Build Q matrix on CPU"""
        Q = np.zeros((self.n_x, self.n_x))
        q_pos = self.config.process_noise_pos
        q_vel = self.config.process_noise_vel
        Q[0:3, 0:3] = (q_pos/3) * dt**3
        Q[0:3, 3:6] = (q_pos/2) * dt**2
        Q[3:6, 0:3] = (q_pos/2) * dt**2
        Q[3:6, 3:6] = q_vel * dt
        return Q
    
    def update_batch(self, 
                     measurements: List[Measurement],
                     beacon_positions: np.ndarray) -> cp.ndarray:
        """
        Batch update for multiple measurements (potentially different beacons per batch).
        
        Args:
            measurements: List of Measurement objects (length = batch_size)
            beacon_positions: (batch_size, 3) beacon positions at measurement times
            
        Returns:
            Innovations array (batch, meas_dim)
        """
        # For simplicity, assume all are range measurements
        batch_size = len(measurements)
        innovations = []
        
        if self.use_gpu:
            for i in range(batch_size):
                m = measurements[i]
                b_pos = beacon_positions[i]
                # Single update (could vectorize more)
                # For now, call CPU EKF logic on GPU arrays? Complex.
                # Placeholder: would need fully vectorized EKF update
                pass
        else:
            for i in range(batch_size):
                m = measurements[i]
                # CPU version
                # Build H, R, etc.
                # This is a simplified placeholder
                p = self.states[i, 0:3]
                r_vec = p - beacon_positions[i]
                r = np.linalg.norm(r_vec)
                r_hat = r_vec / max(r, 1e-9)
                
                # Range measurement
                z_pred = r
                z_meas = m.value
                innov = z_meas - z_pred
                innovations.append(innov)
                
        return np.array(innovations)