"""
Configuration classes for the navigation system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import json

from .constants import *

class FilterType(Enum):
    EKF = "ekf"
    UKF = "ukf"
    PARTICLE = "particle"
    GPU_EKF = "gpu_ekf"

class BeaconType(Enum):
    PULSAR = "pulsar"
    RADIO = "radio"
    OPTICAL = "optical"
    LASER = "laser"
    GRAVITY = "gravity"
    MAGNETIC = "magnetic"
    IMU = "imu"
    STAR_TRACKER = "star_tracker"

class MeasurementType(Enum):
    RANGE = "range"
    DIRECTION = "direction"
    RANGE_RATE = "range_rate"  # Doppler
    BOTH = "both"

@dataclass
class BeaconConfig:
    """Configuration for a specific beacon type"""
    beacon_type: BeaconType
    base_range_std: float  # km
    base_dir_std: float  # radians
    max_range: Optional[float] = None  # km
    min_elevation: float = 0.0  # degrees
    update_rate: float = 1.0  # Hz
    latency: float = 0.0  # seconds (signal processing delay)
    
    @classmethod
    def from_preset(cls, preset: str) -> "BeaconConfig":
        presets = {
            "radio": cls(BeaconType.RADIO, 0.01, 0.0001, max_range=1e9, update_rate=10),
            "xray": cls(BeaconType.PULSAR, 1000.0, 0.001, max_range=1e12, update_rate=0.1),
            "optical": cls(BeaconType.OPTICAL, 10.0, 0.001, max_range=1e7, update_rate=1),
            "laser": cls(BeaconType.LASER, 0.001, 0.00001, max_range=1e6, update_rate=100),
        }
        return presets[preset]

@dataclass
class NavConfig:
    """
    Main configuration for navigation system.
    Can be loaded from JSON/YAML or set programmatically.
    """
    # Filter settings
    filter_type: FilterType = FilterType.EKF
    process_noise_pos: float = DEFAULT_PROCESS_NOISE_POS
    process_noise_vel: float = DEFAULT_PROCESS_NOISE_VEL
    process_noise_attitude: float = 1e-6  # rad^2/s for orientation
    process_noise_bias: float = 1e-9  # bias random walk
    
    # Measurement settings
    measurement_outlier_threshold: float = 9.21  # chi-square 95% for 2 DOF
    max_measurements_per_update: int = 20
    outlier_handling: str = "ransac"  # "none", "gating", "ransac", "ml"
    
    # Beacon management
    beacon_selection_method: str = "greedy"  # "greedy", "adaptive", "rl", "random"
    min_beacons_required: int = 3
    max_beacons_tracked: int = 10
    beacon_reliability_decay: float = 0.99  # EMA factor
    
    # Fault detection
    enable_fault_detection: bool = True
    ransac_iterations: int = 100
    ransac_threshold: float = 3.0  # km or radians
    ml_anomaly_threshold: float = 0.95  # probability threshold
    
    # Relativistic corrections
    enable_relativistic: bool = False
    enable_shapiro_delay: bool = False
    enable_solar_aberration: bool = False
    
    # Optical flow
    enable_optical_flow: bool = False
    optical_flow_update_rate: float = 10.0  # Hz
    camera_matrix: Optional[List[List[float]]] = None
    
    # IMU integration
    enable_imu: bool = False
    imu_update_rate: float = 100.0  # Hz
    imu_noise_accel: float = 0.01  # km/s^2
    imu_noise_gyro: float = 0.001  # rad/s
    imu_bias_accel: float = 1e-5  # km/s^2
    imu_bias_gyro: float = 1e-6  # rad/s
    
    # GPU acceleration
    enable_gpu: bool = False
    gpu_device: int = 0
    gpu_batch_size: int = 1000
    
    # Logging & debugging
    debug_logging: bool = False
    log_level: str = "INFO"
    save_measurement_history: bool = True
    max_history_size: int = 10000
    
    # Performance
    max_processing_time_ms: float = 100.0  # max allowed per update
    enable_multithreading: bool = False
    
    # Mission-specific
    mission_name: str = "default"
    start_date: str = "2000-01-01T12:00:00Z"
    end_date: Optional[str] = None
    spacecraft_id: str = "sc-001"
    
    # Safety
    collision_avoidance_enabled: bool = False
    keep_out_zones: List[Dict[str, Any]] = field(default_factory=list)
    min_safe_altitude: float = 100.0  # km above planetary surfaces
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif hasattr(value, 'value'):  # Another Enum
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    def to_json(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> "NavConfig":
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert string enums back to Enum objects
        if "filter_type" in data:
            data["filter_type"] = FilterType(data["filter_type"])
        
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate configuration, return list of errors"""
        errors = []
        
        if self.process_noise_pos <= 0:
            errors.append("process_noise_pos must be positive")
        if self.process_noise_vel <= 0:
            errors.append("process_noise_vel must be positive")
        if self.min_beacons_required < 3:
            errors.append("min_beacons_required must be at least 3 for 3D positioning")
        if self.max_beacons_tracked < self.min_beacons_required:
            errors.append("max_beacons_tracked must be >= min_beacons_required")
        if self.measurement_outlier_threshold <= 0:
            errors.append("measurement_outlier_threshold must be positive")
            
        return errors