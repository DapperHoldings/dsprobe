"""
Measurement dataclass representing a single navigation observation.
"""

import numpy as np
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import json

class MeasurementKind(Enum):
    """Type of measurement"""
    RANGE = "range"  # Distance only
    DIRECTION = "direction"  # Unit vector
    RANGE_RATE = "range_rate"  # Doppler (radial velocity)
    ANGULAR = "angular"  # Two angles (azimuth, elevation)
    PULSE_TIMING = "pulse_timing"  # X-ray pulsar TOA

@dataclass
class Measurement:
    """
    A single navigation measurement from a beacon.
    
    Attributes:
        beacon_id: Unique identifier of the beacon
        timestamp: UTC time of measurement
        kind: Type of measurement
        value: The measured value(s)
            - RANGE: scalar (km)
            - DIRECTION: 3D unit vector (or 2D if using az/el)
            - RANGE_RATE: scalar (km/s)
            - PULSE_TIMING: time-of-arrival (seconds since some epoch)
        uncertainty: Standard deviation (scalar or vector depending on kind)
        quality: Data quality flag [0-1], 1 is perfect
        metadata: Additional info (e.g., signal strength, SNR)
    """
    beacon_id: str
    timestamp: datetime
    kind: MeasurementKind
    value: Any
    uncertainty: Any  # float or np.ndarray
    quality: float = 1.0  # 0.0-1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Computed fields (not user-set)
    processed: bool = False
    outlier_score: float = 0.0  # 0=inlier, 1=outlier
    innovation: Optional[np.ndarray] = None  # Set during filter update
    
    def __post_init__(self):
        """Validate measurement data"""
        if self.quality < 0 or self.quality > 1:
            raise ValueError("Quality must be in [0,1]")
        if self.timestamp.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for logging)"""
        d = {
            "beacon_id": self.beacon_id,
            "timestamp": self.timestamp.isoformat(),
            "kind": self.kind.value,
            "value": self.value.tolist() if isinstance(self.value, np.ndarray) else self.value,
            "uncertainty": self.uncertainty.tolist() if isinstance(self.uncertainty, np.ndarray) else self.uncertainty,
            "quality": self.quality,
            "metadata": self.metadata,
            "processed": self.processed,
            "outlier_score": self.outlier_score,
        }
        if self.innovation is not None:
            d["innovation"] = self.innovation.tolist()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Measurement":
        """Reconstruct from dictionary"""
        # Convert timestamp
        timestamp = datetime.fromisoformat(data["timestamp"])
        # Convert value
        if isinstance(data["value"], list):
            value = np.array(data["value"])
        else:
            value = data["value"]
        # Convert uncertainty
        if isinstance(data["uncertainty"], list):
            uncertainty = np.array(data["uncertainty"])
        else:
            uncertainty = data["uncertainty"]
        # Convert innovation if present
        if "innovation" in data:
            innovation = np.array(data["innovation"])
        else:
            innovation = None
            
        return cls(
            beacon_id=data["beacon_id"],
            timestamp=timestamp,
            kind=MeasurementKind(data["kind"]),
            value=value,
            uncertainty=uncertainty,
            quality=data.get("quality", 1.0),
            metadata=data.get("metadata", {}),
            processed=data.get("processed", False),
            outlier_score=data.get("outlier_score", 0.0),
        )
    
    def get_measurement_matrix_size(self) -> int:
        """Get dimension of measurement vector"""
        if self.kind == MeasurementKind.RANGE:
            return 1
        elif self.kind == MeasurementKind.DIRECTION:
            return 3  # Assuming 3D unit vector
        elif self.kind == MeasurementKind.RANGE_RATE:
            return 1
        elif self.kind == MeasurementKind.PULSE_TIMING:
            return 1
        else:
            raise ValueError(f"Unknown measurement kind: {self.kind}")
    
    def as_vector(self) -> np.ndarray:
        """Get measurement as a flat numpy array"""
        if isinstance(self.value, np.ndarray):
            return self.value.flatten()
        else:
            return np.array([self.value])
    
    def as_covariance(self) -> np.ndarray:
        """Get measurement covariance matrix"""
        if isinstance(self.uncertainty, np.ndarray):
            if self.uncertainty.ndim == 1:
                return np.diag(self.uncertainty**2)
            else:
                return self.uncertainty
        else:
            return np.array([[self.uncertainty**2]])
    
    def is_valid(self) -> bool:
        """Check if measurement is valid (non-NaN, finite)"""
        val = self.as_vector()
        return np.all(np.isfinite(val)) and self.quality > 0.1