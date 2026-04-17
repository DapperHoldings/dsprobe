"""
Ephemeris implementations for celestial bodies and artificial beacons.
"""

import numpy as np
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, Callable
from abc import ABC, abstractmethod
import warnings

from config.constants import SPEED_OF_LIGHT, AU_IN_KM, SECONDS_PER_DAY

class Ephemeris(ABC):
    """Abstract base class for all ephemerides"""
    
    @abstractmethod
    def get_position(self, epoch: float, frame: str = "ECLIPJ2000") -> np.ndarray:
        """Get position in km at epoch (seconds since J2000)"""
        pass
    
    @abstractmethod
    def get_velocity(self, epoch: float, frame: str = "ECLIPJ2000") -> np.ndarray:
        """Get velocity in km/s at epoch"""
        pass
    
    def get_light_time_corrected_position(self, 
                                          observer_pos: np.ndarray,
                                          epoch_receiver: float,
                                          max_iter: int = 3) -> Tuple[np.ndarray, float]:
        """
        Compute beacon position at emission time (accounting for light travel time).
        
        Returns:
            corrected_position, light_time (seconds)
        """
        # Initial guess: assume beacon stationary during light travel
        pos_guess = self.get_position(epoch_receiver)
        delta = pos_guess - observer_pos
        light_time = np.linalg.norm(delta) / SPEED_OF_LIGHT
        
        # Iterate for moving beacons
        for _ in range(max_iter):
            emission_time = epoch_receiver - light_time
            pos_emission = self.get_position(emission_time)
            delta = pos_emission - observer_pos
            new_light_time = np.linalg.norm(delta) / SPEED_OF_LIGHT
            if abs(new_light_time - light_time) < 1e-9:
                break
            light_time = new_light_time
            
        return self.get_position(epoch_receiver - light_time), light_time

class KeplerianEphemeris(Ephemeris):
    """
    Ephemeris based on Keplerian orbital elements.
    For simplicity, assumes two-body, elliptical orbits in ecliptic plane.
    """
    
    def __init__(self, 
                 semi_major_axis: float,  # km
                 eccentricity: float = 0.0,
                 inclination: float = 0.0,  # degrees
                 longitude_of_ascending_node: float = 0.0,  # degrees
                 argument_of_periapsis: float = 0.0,  # degrees
                 mean_anomaly_at_epoch: float = 0.0,  # degrees
                 central_body_mass: float = 1.32712440018e11,  # km^3/s^2 (Sun)
                 epoch: float = 0.0):  # epoch of elements (seconds since J2000)
        
        self.a = semi_major_axis
        self.e = eccentricity
        self.i = np.radians(inclination)
        self.Omega = np.radians(longitude_of_ascending_node)
        self.omega = np.radians(argument_of_periapsis)
        self.M0 = np.radians(mean_anomaly_at_epoch)
        self.mu = central_body_mass
        self.epoch = epoch
        
        # Precompute some constants
        self.n = np.sqrt(self.mu / self.a**3)  # mean motion (rad/s)
        
    def get_position(self, epoch: float, frame: str = "ECLIPJ2000") -> np.ndarray:
        """Get position in orbital frame at epoch"""
        # Mean anomaly at epoch
        M = self.M0 + self.n * (epoch - self.epoch)
        M = np.mod(M, 2*np.pi)
        
        # Solve Kepler's equation for eccentric anomaly E
        # M = E - e*sin(E)
        E = M  # initial guess
        for _ in range(10):
            E = E - (E - self.e * np.sin(E) - M) / (1 - self.e * np.cos(E))
            
        # True anomaly
        cos_nu = (np.cos(E) - self.e) / (1 - self.e * np.cos(E))
        sin_nu = (np.sqrt(1 - self.e**2) * np.sin(E)) / (1 - self.e * np.cos(E))
        nu = np.arctan2(sin_nu, cos_nu)
        
        # Distance
        r = self.a * (1 - self.e * np.cos(E))
        
        # Position in perifocal frame
        x_peri = r * np.cos(nu)
        y_peri = r * np.sin(nu)
        z_peri = 0.0
        
        # Rotation to ecliptic (if inclination, Omega, omega provided)
        # Build rotation matrix: R = Rz(-Omega) * Rx(-i) * Rz(-omega)
        cO, sO = np.cos(self.Omega), np.sin(self.Omega)
        ci, si = np.cos(self.i), np.sin(self.i)
        cw, sw = np.cos(self.omega), np.sin(self.omega)
        
        R = np.array([
            [cO*cw - sO*ci*sw, -cO*sw - sO*ci*cw, sO*si],
            [sO*cw + cO*ci*sw, -sO*sw + cO*ci*cw, -cO*si],
            [si*sw, si*cw, ci]
        ])
        
        pos_peri = np.array([x_peri, y_peri, z_peri])
        pos_ecliptic = R @ pos_peri
        
        return pos_ecliptic
    
    def get_velocity(self, epoch: float, frame: str = "ECLIPJ2000") -> np.ndarray:
        """Get velocity at epoch (km/s)"""
        # Mean anomaly
        M = self.M0 + self.n * (epoch - self.epoch)
        M = np.mod(M, 2*np.pi)
        
        # Solve for E
        E = M
        for _ in range(10):
            E = E - (E - self.e * np.sin(E) - M) / (1 - self.e * np.cos(E))
            
        # Radial and transverse velocity components
        factor = np.sqrt(self.mu * self.a) / r  # r from get_position would need E again
        # Actually compute r:
        r = self.a * (1 - self.e * np.cos(E))
        vr = (self.mu / self.a)**0.5 * (self.e * np.sin(E)) / (1 - self.e*np.cos(E))
        vt = (self.mu / self.a)**0.5 * np.sqrt(1 - self.e**2) / (1 - self.e*np.cos(E))
        
        # In perifocal:
        v_peri = np.array([vr * np.cos(E) - vt * np.sin(E),
                          vr * np.sin(E) + vt * np.cos(E),
                          0.0])
        
        # Rotate to ecliptic (same as position)
        cO, sO = np.cos(self.Omega), np.sin(self.Omega)
        ci, si = np.cos(self.i), np.sin(self.i)
        cw, sw = np.cos(self.omega), np.sin(self.omega)
        R = np.array([
            [cO*cw - sO*ci*sw, -cO*sw - sO*ci*cw, sO*si],
            [sO*cw + cO*ci*sw, -sO*sw + cO*ci*cw, -cO*si],
            [si*sw, si*cw, ci]
        ])
        
        return R @ v_peri

def create_planet_ephemeris(planet_name: str) -> Ephemeris:
    """
    Factory for simplified planetary ephemerides.
    Uses approximate orbital elements (J2000).
    For high accuracy, use SPICE integration instead.
    """
    # Approximate orbital parameters (semi-major axis, ecc, inc, etc.)
    # Source: NASA planetary factsheets (simplified circular for demo)
    elements = {
        "earth": (1.000, 0.0167, 0.0, 0.0, 102.9372, 0.0),
        "mars": (1.524, 0.0934, 1.85, 0.0, 100.464, 0.0),
        "jupiter": (5.203, 0.0489, 1.30, 0.0, 100.456, 0.0),
        "saturn": (9.537, 0.0565, 2.49, 0.0, 113.664, 0.0),
        "venus": (0.723, 0.0068, 3.39, 0.0, 76.680, 0.0),
    }
    
    if planet_name.lower() not in elements:
        raise ValueError(f"Unknown planet: {planet_name}")
    
    a, e, inc, Omega, omega, M0 = elements[planet_name.lower()]
    # Convert AU to km
    a_km = a * AU_IN_KM
    
    return KeplerianEphemeris(
        semi_major_axis=a_km,
        eccentricity=e,
        inclination=inc,
        longitude_of_ascending_node=Omega,
        argument_of_periapsis=omega,
        mean_anomaly_at_epoch=M0,
        central_body_mass=1.32712440018e11,  # Sun GM
        epoch=0.0  # J2000
    )

def create_pulsar_ephemeris(pulsar_name: str, 
                          position_kpc: Tuple[float, float, float],
                          timing_model: Optional[Callable] = None) -> Ephemeris:
    """
    Create ephemeris for a pulsar.
    
    Args:
        pulsar_name: PSR designation
        position_kpc: Position in solar system barycenter (kpc)
        timing_model: Optional function epoch->pulse_phase (for TOA prediction)
    """
    class PulsarEphemeris(Ephemeris):
        def __init__(self, pos_kpc, timing_model):
            self.pos_kpc = np.array(pos_kpc)
            self.timing_model = timing_model
            # Convert kpc to km
            self.pos_km = self.pos_kpc * 3.08567758149137e16
            
        def get_position(self, epoch: float, frame: str = "ECLIPJ2000") -> np.ndarray:
            # Pulsars are so far away, proper motion negligible over mission
            return self.pos_km.copy()
        
        def get_velocity(self, epoch: float, frame: str = "ECLIPJ2000") -> np.ndarray:
            # Assume static (very distant)
            return np.zeros(3)
        
        def get_phase(self, epoch: float) -> float:
            """Get pulse phase at given epoch (for XNAV)"""
            if self.timing_model:
                return self.timing_model(epoch)
            else:
                # Simple: linear
                return epoch / 0.001  # placeholder frequency
                
    return PulsarEphemeris(position_kpc, timing_model)

def create_artificial_beacon_ephemeris(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray = None,
    orbit_params: Optional[Dict] = None,
    reference_time: float = 0.0
) -> Ephemeris:
    """
    Create ephemeris for an artificial beacon (e.g., relay satellite).
    """
    if initial_velocity is None:
        initial_velocity = np.zeros(3)
        
    if orbit_params is None:
        # Simple linear motion (unrealistic but simple)
        class LinearEphemeris(Ephemeris):
            def __init__(self, p0, v0, t0):
                self.p0 = p0
                self.v0 = v0
                self.t0 = t0
            def get_position(self, epoch, frame="ECLIPJ2000"):
                return self.p0 + self.v0 * (epoch - self.t0)
            def get_velocity(self, epoch, frame="ECLIPJ2000"):
                return self.v0.copy()
        return LinearEphemeris(initial_position, initial_velocity, reference_time)
    else:
        # Use Keplerian for orbit
        return KeplerianEphemeris(
            semi_major_axis=orbit_params["a"],
            eccentricity=orbit_params.get("e", 0.0),
            inclination=orbit_params.get("i", 0.0),
            longitude_of_ascending_node=orbit_params.get("Omega", 0.0),
            argument_of_periapsis=orbit_params.get("omega", 0.0),
            mean_anomaly_at_epoch=orbit_params.get("M0", 0.0),
            central_body_mass=orbit_params.get("mu", 1.32712440018e11),
            epoch=reference_time
        )