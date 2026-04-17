"""
X-ray pulsar navigation sensor model.
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from scipy.interpolate import interp1d

@dataclass
class PulsarTimingMeasurement:
    """Pulsar Time-of-Arrival (TOA) measurement"""
    pulsar_id: str
    timestamp: datetime  # detection time
    toa: float  # seconds (measured)
    toa_uncertainty: float  # seconds
    folded_profile: Optional[np.ndarray] = None  # pulse shape
    snr: float = 1.0
    phase: Optional[float] = None  # optional derived phase

class XRayPulsar:
    """
    Model of an X-ray pulsar as a navigation beacon.
    Uses pulse arrival times for positioning.
    """
    
    def __init__(self,
                 pulsar_id: str,
                 barycentric_period: float,  # seconds at solar system barycenter
                 period_derivative: float = 0.0,  # s/s
                 position_bcr: np.ndarray,  # position in solar system barycenter (km)
                 timing_model: Optional[Callable[[float], float]] = None,
                 effective_area: float = 1000.0,  # cm^2
                 background_rate: float = 0.1,  # counts/s
                 energy_band: Tuple[float, float] = (0.2, 2.0),  # keV
                 ):
        """
        Initialize pulsar model.
        
        Args:
            pulsar_id: PSR designation (e.g., "B1937+21")
            barycentric_period: Period at solar system barycenter (s)
            period_derivative: Period derivative (dP/dt)
            position_bcr: Position vector in solar system barycenter (km)
            timing_model: Callable epoch->barycentric TOA model (s)
            effective_area: Detector effective area (cm^2)
            background_rate: Background X-ray count rate (counts/s)
            energy_band: Energy bandpass (keV)
        """
        self.id = pulsar_id
        self.P0 = barycentric_period
        self.P_dot = period_derivative
        self.position_bcr = np.array(position_bcr, dtype=float)
        self.timing_model = timing_model
        
        # Detector parameters
        self.A_eff = effective_area
        self.bkg_rate = background_rate
        self.E_band = energy_band
        
        # Derived: typical flux (would get from catalog)
        self.flux = 1e-11  # erg/cm^2/s, typical for bright millisecond pulsars
        
        # Photon statistics: expected count rate
        # photons/s = flux * area / (E * conversion)
        self.expected_rate = self.flux * self.A_eff / 1e-11  # normalized to ~1
        
    def predict_toa(self, 
                    epoch: float,
                    observer_pos_bcr: np.ndarray) -> float:
        """
        Predict expected Time-of-Arrival at observer.
        
        Args:
            epoch: Observation time (seconds since J2000)
            observer_pos_bcr: Observer position in solar system barycenter (km)
            
        Returns:
            Predicted TOA (seconds of day, or phase)
        """
        # Light travel time from pulsar to solar system barycenter
        r_sc_bcr = np.linalg.norm(observer_pos_bcr)
        # For simplicity, assume pulsar at infinite distance; light-time = 0
        
        if self.timing_model is not None:
            # Use precise timing model
            toa_bcr = self.timing_model(epoch)  # seconds
        else:
            # Simple: phase = (t - t0) / P
            # Need reference epoch t0 where phase=0
            # Let's pick epoch=0 as reference
            phase = (epoch % self.P0) / self.P0  # [0,1)
            toa_bcr = phase * self.P0
            
        # Convert to observer frame? Not needed for phase measurement
        
        return toa_bcr
    
    def generate_measurement(self,
                           epoch: float,
                           observer_pos_bcr: np.ndarray,
                           exposure_time: float,
                           snr_factor: float = 1.0) -> PulsarTimingMeasurement:
        """
        Simulate a pulsar TOA measurement.
        
        Args:
            epoch: Observation start time (s)
            observer_pos_bcr: Observer position in BCR (km)
            exposure_time: Integration time (s)
            snr_factor: Scale factor for signal (e.g., detector degradation)
            
        Returns:
            PulsarTimingMeasurement with noise
        """
        # Expected number of photons
        N_photons = self.expected_rate * exposure_time * snr_factor
        
        # Poisson realization
        observed_counts = np.random.poisson(N_photons + self.bkg_rate * exposure_time)
        signal_counts = observed_counts - self.bkg_rate * exposure_time
        signal_counts = max(0, signal_counts)
        
        # SNR
        snr = signal_counts / np.sqrt(observed_counts) if observed_counts > 0 else 0.1
        
        # TOA uncertainty: roughly (P / (2 * pi * SNR)) for bright pulsars
        # More precisely: σ_TOA ~ (P / (2π * √(N))) * (W/P)^(-3/2) where W=pulse width
        # Simplified: σ ~ P / (2π * SNR)
        if snr > 1:
            toa_uncertainty = self.P0 / (2 * np.pi * snr)
        else:
            toa_uncertainty = self.P0  # very poor
        
        # Add noise to true TOA
        true_toa = self.predict_toa(epoch, observer_pos_bcr)
        measured_toa = true_toa + np.random.normal(0, toa_uncertainty)
        
        return PulsarTimingMeasurement(
            pulsar_id=self.id,
            timestamp=datetime.now(timezone.utc),  # approximate
            toa=measured_toa,
            toa_uncertainty=toa_uncertainty,
            snr=snr
        )