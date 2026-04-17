"""
Beacon class representing navigation beacons (natural or artificial).
"""

import numpy as np
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from config.constants import SPEED_OF_LIGHT, AU_IN_KM

class BeaconType(Enum):
    """Types of navigation beacons"""
    PULSAR = "pulsar"  # X-ray pulsar navigation
    RADIO = "radio"    # Radio beacon (e.g., DSN, relay)
    OPTICAL = "optical"  # Planet/moon/asteroid imaging
    LASER = "laser"    # Laser retroreflector
    GRAVITY = "gravity"  # Gravitational anomaly (theoretical)
    MAGNETIC = "magnetic"  # Magnetic anomaly (theoretical)
    IMU = "imu"  # Inertial measurement unit (特殊处理)
    STAR_TRACKER = "star_tracker"  # Star pattern (direction only)

class Ephemeris:
    """
    Base class for ephemeris calculations.
    For maximum accuracy, should interface with SPICE kernels.
    """
    
    def __init__(self, source: str = "builtin"):
        self.source = source  # "builtin", "spice", "custom"
        
    def get_position(self, epoch: float, frame: str = "ECLIPJ2000") -> np.ndarray:
        """
        Get position at epoch (seconds since J2000).
        
        Args:
            epoch: Seconds since J2000.0 (TDB)
            frame: Coordinate frame (ECLIPJ2000, ICRF, etc.)
            
        Returns:
            3D position vector in km
        """
        raise NotImplementedError
        
    def get_velocity(self, epoch: float, frame: str = "ECLIPJ2000") -> np.ndarray:
        """Get velocity at epoch (km/s)"""
        raise NotImplementedError
        
    def get_attitude(self, epoch: float) -> np.ndarray:
        """For non-spherical bodies or rotating beacons"""
        raise NotImplementedError

@dataclass
class Beacon:
    """
    Represents a navigation beacon in space.
    Can be natural (pulsar, planet) or artificial (radio relay, laser beacon).
    """
    id: str
    name: str
    beacon_type: BeaconType
    
    # Position model
    ephemeris: Optional[Ephemeris] = None
    fixed_position: Optional[np.ndarray] = None  # for static beacons (km)
    
    # Signal characteristics
    frequency: Optional[float] = None  # Hz (for radio/X-ray)
    wavelength: Optional[float] = None  # km (derived from frequency)
    signal_bandwidth: float = 1.0  # Hz
    transmit_power: float = 1.0  # W (relative)
    
    # Navigation accuracy model
    base_uncertainty: Tuple[float, float] = (10.0, 0.001)  # (range_std, dir_std)
    range_std_func: Optional[Callable[[float, np.ndarray], float]] = None
    dir_std_func: Optional[Callable[[float, np.ndarray], float]] = None
    
    # Operational status
    health: float = 1.0  # 0.0-1.0
    reliability: float = 1.0  # estimated based on history
    failure_probability: float = 0.0
    last_successful_measurement: Optional[datetime] = None
    
    # Visibility constraints
    min_elevation: float = 0.0  # degrees above horizon
    max_range: Optional[float] = None  # km
    daylight_visible: bool = True  # can be used in sunlight?
    
    # Metadata
    cataloged: bool = True
    source: str = "natural"  # "natural", "artificial", "simulated"
    notes: str = ""
    
    # For moving beacons: orbital elements (if using Keplerian)
    keplerian_elements: Optional[Dict[str, float]] = None
    central_body_mass: float = SUN_MASS  # km^3/s^2
    
    def __post_init__(self):
        # Convert fixed_position to numpy array if provided
        if self.fixed_position is not None:
            self.fixed_position = np.asarray(self.fixed_position, dtype=float)
            
        # Set wavelength from frequency if provided
        if self.frequency is not None and self.wavelength is None:
            self.wavelength = SPEED_OF_LIGHT / self.frequency  # km
    
    def get_position(self, epoch: float, 
                     observer_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get beacon position at given epoch.
        
        Args:
            epoch: Seconds since J2000.0
            observer_pos: Optional observer position for light-time correction
            
        Returns:
            Position vector in km (in solar system barycenter frame)
        """
        if self.ephemeris is not None:
            pos = self.ephemeris.get_position(epoch)
        elif self.fixed_position is not None:
            pos = self.fixed_position.copy()
        else:
            raise ValueError(f"Beacon {self.id} has no position model")
            
        # Apply light-time correction if observer position given
        if observer_pos is not None:
            # Simple: assume beacon is stationary during light travel
            # For moving beacons, would need iterative solution
            range_approx = np.linalg.norm(pos - observer_pos)
            light_time = range_approx / SPEED_OF_LIGHT
            if self.ephemeris is not None:
                # More accurate: reposition beacon at emission time
                pos = self.ephemeris.get_position(epoch - light_time)
            # else: static beacon, no correction needed
                
        return pos
    
    def get_velocity(self, epoch: float) -> np.ndarray:
        """Get velocity vector (km/s)"""
        if self.ephemeris is not None:
            return self.ephemeris.get_velocity(epoch)
        else:
            return np.zeros(3)
    
    def get_range_and_direction(self, 
                               observer_pos: np.ndarray,
                               epoch: float) -> Tuple[float, np.ndarray]:
        """
        Compute range and direction from observer to beacon.
        
        Returns:
            range_km: Distance in km
            direction: Unit vector from observer to beacon
        """
        beacon_pos = self.get_position(epoch, observer_pos)
        delta = beacon_pos - observer_pos
        range_km = np.linalg.norm(delta)
        
        if range_km < 1e-9:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = delta / range_km
            
        return range_km, direction
    
    def get_uncertainty(self, 
                       range_km: float,
                       observer_pos: np.ndarray) -> Tuple[float, float]:
        """
        Get measurement uncertainty for current geometry.
        
        Returns:
            range_std, direction_std (radians)
        """
        base_range_std, base_dir_std = self.base_uncertainty
        
        # Apply custom functions if provided
        if self.range_std_func is not None:
            range_std = self.range_std_func(range_km, observer_pos)
        else:
            # Default: range std increases with distance for some types
            if self.beacon_type == BeaconType.PULSAR:
                # Pulsar TOA uncertainty roughly constant in time, 
                # so range uncertainty increases linearly with distance?
                # Actually: Δt ~ constant => Δd = c*Δt constant => range_std independent of range?
                # In practice, pulse shape deconvolution gives ~1ms TOA => 300 km range std
                # Doesn't scale with range! So independent of range.
                range_std = base_range_std
            elif self.beacon_type == BeaconType.OPTICAL:
                # Optical centroiding: constant angle error => range_std ~ range * angle_std
                range_std = range_km * base_dir_std * 10  # conservative multiplier
            else:
                range_std = base_range_std * (1.0 + range_km / AU_IN_KM * 0.1)  # slight increase with range
                
        if self.dir_std_func is not None:
            dir_std = self.dir_std_func(range_km, observer_pos)
        else:
            dir_std = base_dir_std
            
        # Scale by health and reliability
        health_factor = max(0.1, self.health * self.reliability)
        range_std /= health_factor
        dir_std /= health_factor
        
        # Scale by failure probability
        range_std *= (1.0 + 10 * self.failure_probability)
        dir_std *= (1.0 + 10 * self.failure_probability)
        
        return range_std, dir_std
    
    def is_visible(self, 
                   observer_pos: np.ndarray,
                   pointing_dir: Optional[np.ndarray] = None,
                   fov_half_angle: Optional[float] = None) -> bool:
        """
        Check if beacon is visible from observer position.
        
        Args:
            observer_pos: Position of observer (km)
            pointing_dir: Optional direction camera/antenna is pointing (unit vector)
            fov_half_angle: Optional field of view half-angle (radians)
            
        Returns:
            True if beacon is visible
        """
        # Health check
        if self.health < 0.2:
            return False
            
        # Get beacon position at current time (need epoch - but not provided here)
        # This method assumes epoch is handled externally; better to pass epoch
        # For now, skip range check
        return True
    
    def __repr__(self) -> str:
        return (f"Beacon(id={self.id}, name={self.name}, type={self.beacon_type.value}, "
                f"health={self.health:.2f})")