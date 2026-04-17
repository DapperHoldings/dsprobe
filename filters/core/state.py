"""
State representation for spacecraft navigation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class State:
    """
    State vector for navigation.
    Extensible to include attitude, biases, etc.
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # km
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # km/s
    attitude: Optional[np.ndarray] = None  # Quaternion [w, x, y, z] or rotation matrix
    attitude_rate: Optional[np.ndarray] = None  # rad/s
    biases: Optional[np.ndarray] = None  # [accel_bias_xyz, gyro_bias_xyz]
    
    timestamp: float = 0.0  # seconds since J2000
    coordinate_frame: str = "ECLIPJ2000"
    
    def to_vector(self) -> np.ndarray:
        """Convert to flat state vector for filter"""
        vec = []
        vec.extend(self.position.tolist())
        vec.extend(self.velocity.tolist())
        if self.attitude is not None:
            # Ensure quaternion is normalized
            if len(self.attitude) == 4:
                q = self.attitude / np.linalg.norm(self.attitude)
                vec.extend(q.tolist())
            else:
                # Rotation matrix flatten
                vec.extend(self.attitude.flatten().tolist())
        if self.attitude_rate is not None:
            vec.extend(self.attitude_rate.tolist())
        if self.biases is not None:
            vec.extend(self.biases.tolist())
        return np.array(vec)
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, 
                    with_attitude: bool = False,
                    with_biases: bool = False) -> "State":
        """Reconstruct State from vector"""
        idx = 0
        position = vector[idx:idx+3]; idx += 3
        velocity = vector[idx:idx+3]; idx += 3
        
        attitude = None
        if with_attitude:
            attitude = vector[idx:idx+4]; idx += 4  # quaternion
            
        attitude_rate = None
        if with_attitude and with_biases:
            attitude_rate = vector[idx:idx+3]; idx += 3
            
        biases = None
        if with_biases:
            biases = vector[idx:idx+6]; idx += 6  # 3 accel + 3 gyro
            
        return cls(
            position=position,
            velocity=velocity,
            attitude=attitude,
            attitude_rate=attitude_rate,
            biases=biases
        )
    
    def get_position_covariance(self, full_cov: np.ndarray) -> np.ndarray:
        """Extract 3x3 position covariance from full state covariance"""
        return full_cov[0:3, 0:3]
    
    def get_velocity_covariance(self, full_cov: np.ndarray) -> np.ndarray:
        """Extract 3x3 velocity covariance"""
        return full_cov[3:6, 3:6]
    
    def compute_pdop(self, full_cov: np.ndarray) -> float:
        """Compute Position Dilution of Precision"""
        pos_cov = self.get_position_covariance(full_cov)
        return np.sqrt(np.trace(pos_cov))

@dataclass
class Covariance:
    """
    Covariance matrix with metadata.
    """
    matrix: np.ndarray
    timestamp: float
    state_description: str = "position_velocity"
    filter_name: str = "ekf"
    
    def __post_init__(self):
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Covariance must be square matrix")
    
    def get_marginal_covariance(self, indices: Tuple[int, ...]) -> np.ndarray:
        """Get covariance submatrix for given state indices"""
        return self.matrix[np.ix_(indices, indices)]