"""
Tests for beacon and ephemeris implementations.
"""

import numpy as np
import pytest
from datetime import datetime, timezone, timedelta

from core.beacon import Beacon, BeaconType, Ephemeris, KeplerianEphemeris
from core.ephemeris import create_planet_ephemeris, create_pulsar_ephemeris

def test_beacon_creation():
    """Test basic beacon creation"""
    beacon = Beacon(
        id="test",
        name="Test Beacon",
        beacon_type=BeaconType.RADIO,
        fixed_position=np.array([1e5, 2e5, 3e5])
    )
    assert beacon.id == "test"
    assert beacon.name == "Test Beacon"
    assert beacon.beacon_type == BeaconType.RADIO
    assert np.allclose(beacon.fixed_position, [1e5, 2e5, 3e5])

def test_beacon_position_fixed():
    """Test fixed position beacon"""
    pos = np.array([1000.0, 2000.0, 3000.0])
    beacon = Beacon(
        id="fixed",
        name="Fixed Beacon",
        beacon_type=BeaconType.OPTICAL,
        fixed_position=pos
    )
    # Get position at any time should be same
    p1 = beacon.get_position(0.0)
    p2 = beacon.get_position(1e9)  # far future
    assert np.allclose(p1, pos)
    assert np.allclose(p2, pos)

def test_beacon_ephemeris():
    """Test beacon with ephemeris"""
    # Create a simple linear ephemeris
    class LinearEphemeris(Ephemeris):
        def __init__(self, p0, v0):
            self.p0 = np.array(p0)
            self.v0 = np.array(v0)
        def get_position(self, epoch, frame="ECLIPJ2000"):
            return self.p0 + self.v0 * epoch
        def get_velocity(self, epoch, frame="ECLIPJ2000"):
            return self.v0.copy()
    
    ephem = LinearEphemeris([0,0,0], [1,0,0])
    beacon = Beacon(
        id="moving",
        name="Moving Beacon",
        beacon_type=BeaconType.RADIO,
        ephemeris=ephem
    )
    
    # At t=0
    pos0 = beacon.get_position(0.0)
    assert np.allclose(pos0, [0,0,0])
    
    # At t=100 s
    pos100 = beacon.get_position(100.0)
    assert np.allclose(pos100, [100,0,0])

def test_beacon_uncertainty():
    """Test uncertainty model"""
    beacon = Beacon(
        id="test",
        name="Test",
        beacon_type=BeaconType.OPTICAL,
        fixed_position=np.array([1e6, 0, 0]),
        base_uncertainty=(10.0, 0.001)  # 10 km range, 0.001 rad direction
    )
    
    observer_pos = np.array([0,0,0])
    range_std, dir_std = beacon.get_uncertainty(1e6, observer_pos)
    
    # For optical, range uncertainty should scale with range * dir_std
    # But our implementation uses base_range_std directly for range, so:
    assert range_std == 10.0, f"Range std should be 10.0, got {range_std}"
    assert dir_std == 0.001, f"Dir std should be 0.001, got {dir_std}"
    
    # Test with health degradation
    beacon.health = 0.5
    range_std2, dir_std2 = beacon.get_uncertainty(1e6, observer_pos)
    # Health 0.5 should increase uncertainty (divide by health)
    assert range_std2 == 20.0, f"Range std with health 0.5 should be 20.0, got {range_std2}"
    assert dir_std2 == 0.002, f"Dir std with health 0.5 should be 0.002, got {dir_std2}"

def test_planet_ephemeris():
    """Test planet ephemeris factory"""
    earth = create_planet_ephemeris("earth")
    pos = earth.get_position(0.0)  # J2000
    
    # Earth at J2000 should be about 1 AU from Sun in ecliptic plane
    # Actual value: about 149,597,870 km
    distance = np.linalg.norm(pos)
    assert 149e6 < distance < 150e6, f"Earth distance {distance/1e6:.1f} million km, expected ~149.6"

def test_pulsar_ephemeris():
    """Test pulsar ephemeris"""
    def timing_model(t):
        # Simple linear model: phase = t / period
        period = 0.001  # 1 ms
        return t / period
    
    pulsar = create_pulsar_ephemeris(
        "B1937+21",
        position_kpc=(1.0, 0.5, -0.2),
        timing_model=timing_model
    )
    
    pos = pulsar.get_position(0.0)
    # Should be in kpc converted to km
    expected = np.array([1.0, 0.5, -0.2]) * 3.08567758149137e16
    assert np.allclose(pos, expected, rtol=1e-6)
    
    # Check it's far away
    assert np.linalg.norm(pos) > 1e15  # at least 1 pc

def test_keplerian_ephemeris():
    """Test Keplerian orbit calculation"""
    # Circular orbit, 1 AU, 1 year period
    ephem = KeplerianEphemeris(
        semi_major_axis=1.496e8,  # 1 AU in km
        eccentricity=0.0,
        inclination=0.0,
        longitude_of_ascending_node=0.0,
        argument_of_periapsis=0.0,
        mean_anomaly_at_epoch=0.0,
        central_body_mass=1.32712440018e11,  # Sun GM
        epoch=0.0
    )
    
    # At t=0, should be at perihelion (but circular so anywhere)
    pos0 = ephem.get_position(0.0)
    r0 = np.linalg.norm(pos0)
    assert abs(r0 - 1.496e8) < 1000.0, f"Distance {r0/1e3:.1f} km, expected ~149,600,000"
    
    # After half period, should be opposite side
    pos_half = ephem.get_position(0.5 * 365.25 * 86400)  # half year
    # Dot product should be ~ -r^2 (opposite)
    dot = np.dot(pos0, pos_half)
    assert dot < -r0**2 * 0.99, f"Half-period positions should be opposite, got dot={dot:.2e}"

def test_beacon_range_and_direction():
    """Test range and direction calculation"""
    beacon = Beacon(
        id="test",
        name="Test",
        beacon_type=BeaconType.RADIO,
        fixed_position=np.array([1000.0, 0.0, 0.0])
    )
    
    observer = np.array([0.0, 0.0, 0.0])
    r, direction = beacon.get_range_and_direction(observer, 0.0)
    
    assert abs(r - 1000.0) < 1e-6, f"Range should be 1000 km, got {r}"
    assert np.allclose(direction, [1.0, 0.0, 0.0]), f"Direction should be [1,0,0], got {direction}"
    
    # Test with observer not at origin
    observer2 = np.array([100.0, 100.0, 0.0])
    r2, dir2 = beacon.get_range_and_direction(observer2, 0.0)
    expected_r = np.linalg.norm([900.0, -100.0, 0.0])
    assert abs(r2 - expected_r) < 1e-6, f"Range should be {expected_r}, got {r2}"

def test_beacon_visibility():
    """Test beacon visibility checks"""
    beacon = Beacon(
        id="test",
        name="Test",
        beacon_type=BeaconType.RADIO,
        fixed_position=np.array([1e6, 0, 0]),
        health=1.0,
        max_range=2e6
    )
    
    # Within range, healthy -> visible
    observer = np.array([0,0,0])
    assert beacon.is_visible(observer), "Should be visible"
    
    # Too far
    observer_far = np.array([3e6,0,0])
    assert not beacon.is_visible(observer_far), "Should not be visible (too far)"
    
    # Unhealthy
    beacon.health = 0.1
    assert not beacon.is_visible(observer), "Unhealthy beacon not visible"

def test_beacon_repr():
    """Test string representation"""
    beacon = Beacon(
        id="abc123",
        name="My Beacon",
        beacon_type=BeaconType.PULSAR,
        fixed_position=np.array([1,2,3])
    )
    s = repr(beacon)
    assert "abc123" in s
    assert "My Beacon" in s
    assert "pulsar" in s or "PULSAR" in s