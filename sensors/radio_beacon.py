"""
Radio beacon model (e.g., DSN, relay satellites).
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone

from core.beacon import Beacon, Ephemeris
from core.measurement import Measurement, MeasurementKind

@dataclass
class RadioMeasurement:
    """Specific radio measurement types"""
    beacon_id: str
    timestamp: datetime
    measurement_type: str  # 'range', 'range_rate', 'angles'
    value: Any
    uncertainty: float
    carrier_frequency: float = 0.0  # Hz
    snr: float = 20.0  # dB
    phase: Optional[float] = None  # rad (for carrier phase)

class RadioBeacon(Beacon):
    """
    Radio beacon with two-way ranging and Doppler capabilities.
    
    Models:
    - Two-way ranging: coherent transponder with turnaround ratio
    - One-way Doppler (broadcast)
    - DOR (Delta-Differential One-Way Ranging) for angle
    """
    
    def __init__(self,
                 id: str,
                 name: str,
                 position_model: Any,  # ephemeris or fixed
                 frequency: float = 8.4e9,  # X-band, Hz
                 transmit_power: float = 20.0,  # W (EIRP)
                 antenna_gain: float = 30.0,  # dBi
                 system_noise_temp: float = 20.0,  # K
                 turnaround_ratio: float = 240/221,  # typical for DSN S-band
                 **kwargs):
        """
        Args:
            frequency: Carrier frequency (Hz)
            transmit_power: Transmit power (W)
            antenna_gain: Antenna gain (dBi)
            system_noise_temp: System noise temperature (K)
            turnaround_ratio: Frequency multiplier in transponder (f_return = f_uplink * ratio)
        """
        super().__init__(id=id, name=name, **kwargs)
        self.frequency = frequency
        self.transmit_power = transmit_power
        self.antenna_gain = antenna_gain  # dBi
        self.system_noise_temp = system_noise_temp
        self.turnaround_ratio = turnaround_ratio
        
        # Derived: EIRP
        self.eirp = transmit_power * 10**(antenna_gain/10)  # W
        
        # Boltzmann constant
        self.k_boltzmann = 1.380649e-23  # J/K
        
    def predict_range(self,
                     spacecraft_pos: np.ndarray,
                     epoch: float) -> Tuple[float, float]:
        """
        Predict two-way range measurement.
        
        Two-way range: 2 * distance + transponder delay + relativistic corrections.
        
        Returns:
            range_km, light_time_s
        """
        # Get beacon position
        b_pos = self.get_position(epoch)
        
        # One-way distance
        r = np.linalg.norm(spacecraft_pos - b_pos)
        
        # Two-way
        two_way_range = 2 * r
        
        # Add transponder delay (typically 1 µs)
        transponder_delay = 1e-6  # s
        two_way_range += self.k_boltzmann * 0  # not including; just conceptual
        
        light_time = r / 299792.458  # speed of light in km/s
        
        return two_way_range, light_time
    
    def predict_doppler(self,
                       spacecraft_pos: np.ndarray,
                       spacecraft_vel: np.ndarray,
                       epoch: float) -> Tuple[float, float]:
        """
        Predict two-way Doppler (range rate).
        
        f_observed = f_transmitted * (1 - v_radial/c) for one-way.
        Two-way: f_roundtrip = f_uplink * (1 - v_uplink/c) * ratio * (1 + v_downlink/c)
        
        Returns:
            doppler_shift_hz, light_time_s
        """
        b_pos = self.get_position(epoch)
        b_vel = self.get_velocity(epoch) if hasattr(self, 'get_velocity') else np.zeros(3)
        
        # Velocity of light: c = 299792.458 km/s
        c = 299792.458
        
        # Relative velocity along line of sight
        r_vec = spacecraft_pos - b_pos
        r = np.linalg.norm(r_vec)
        if r < 1e-6:
            return 0.0, 0.0
            
        r_hat = r_vec / r
        
        v_sc_proj = spacecraft_vel @ r_hat
        v_beacon_proj = b_vel @ r_hat if b_vel is not None else 0.0
        
        # One-way Doppler shift (uplink)
        # f_uplink_received = self.frequency * (1 - v_sc_proj/c)
        # Transponder up-converts
        f_rcv = self.frequency * (1 - v_sc_proj/c)
        f_trans = f_rcv * self.turnaround_ratio
        
        # Downlink Doppler
        f_down = f_trans * (1 + v_beacon_proj/c)
        
        # Two-way Doppler relative to transmitted frequency
        doppler_shift = f_down - self.frequency
        
        light_time = r / c
        return doppler_shift, light_time
    
    def generate_measurement(self,
                           spacecraft_state: np.ndarray,  # [pos, vel]
                           epoch: float,
                           snr_db: float = 20.0) -> RadioMeasurement:
        """
        Generate simulated radio measurement.
        
        Args:
            spacecraft_state: [pos(3), vel(3)] in km, km/s
            epoch: Time (seconds since J2000)
            snr_db: Desired signal-to-noise ratio (dB)
            
        Returns:
            RadioMeasurement
        """
        pos = spacecraft_state[0:3]
        vel = spacecraft_state[3:6]
        
        # True range and Doppler
        true_range, light_time = self.predict_range(pos, epoch)
        true_doppler, _ = self.predict_doppler(pos, vel, epoch)
        
        # Measurement noise
        # Range noise: depends on signal bandwidth and SNR
        bandwidth = 1.0  # Hz (typical for DSN two-way)
        # Range resolution = c/(2*bandwidth) ≈ 1.5e8 km for 1 Hz! That's huge
        # Actually: range measurement uses code (e.g., PN sequence) with chip rate
        # For DSN: ~1 m precision with 1 MHz chip rate
        range_std = 0.001  # 1 meter (km) for coherent integration
        
        # Doppler noise: ~ sqrt(T_sys / (2*T_int * E_b/N0))
        # With typical values, ~0.001 Hz for 1000s integration
        doppler_std = 0.001  # Hz
        
        # Add noise
        noisy_range = true_range + np.random.normal(0, range_std)
        noisy_doppler = true_doppler + np.random.normal(0, doppler_std)
        
        # SNR to linear
        snr_linear = 10**(snr_db/10)
        
        measurement = RadioMeasurement(
            beacon_id=self.id,
            timestamp=datetime.now(timezone.utc),
            measurement_type='range_rate',  # default to Doppler
            value=noisy_doppler,
            uncertainty=doppler_std,
            carrier_frequency=self.frequency,
            snr=snr_db
        )
        
        return measurement

# Example integration with Beacon base class
def create_dsn_beacon(station_name: str) -> RadioBeacon:
    """
    Create a DSN ground station beacon (fixed position on Earth).
    
    Args:
        station_name: 'goldstone', 'madrid', 'canberra'
        
    Returns:
        RadioBeacon instance
    """
    # DSN station locations (WGS84)
    dsn_locations = {
        'goldstone': (35.4264, -116.8938, 1000.0),  # lat, lon, alt(km)
        'madrid': (40.4261, -4.2498, 700.0),
        'canberra': (-35.4033, 148.9818, 600.0)
    }
    
    if station_name.lower() not in dsn_locations:
        raise ValueError(f"Unknown DSN station: {station_name}")
        
    lat, lon, alt = dsn_locations[station_name.lower()]
    
    # Convert to ECEF (approximate)
    R_earth = 6378.137  # km
    x = (R_earth + alt) * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    y = (R_earth + alt) * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    z = (R_earth + alt) * np.sin(np.radians(lat))
    
    beacon = RadioBeacon(
        id=f"dsn_{station_name}",
        name=f"DSN {station_name.capitalize()}",
        fixed_position=np.array([x, y, z]),
        beacon_type=BeaconType.RADIO,
        frequency=8.4e9,  # X-band
        transmit_power=20.0,  # kW
        antenna_gain=74.0,  # dBi (70m dish)
        system_noise_temp=20.0,
        base_uncertainty=(0.001, 0.0000001)  # 1 m range, sub-µrad angle from VLBI
    )
    
    return beacon