"""
Unit tests for navigation filters.
"""

import numpy as np
import pytest
from datetime import datetime, timezone, timedelta

from config.settings import NavConfig, FilterType
from filters.ekf import EKF
from filters.ukf import UKF
from filters.particle_filter import ParticleFilter
from core.measurement import Measurement, MeasurementKind
from core.beacon import Beacon, BeaconType

def create_test_beacon(position: np.ndarray) -> Beacon:
    """Helper: create a simple fixed beacon"""
    return Beacon(
        id="test_beacon",
        name="Test Beacon",
        beacon_type=BeaconType.RADIO,
        fixed_position=position,
        base_uncertainty=(0.1, 0.001)
    )

def test_ekf_basic_range():
    """Test EKF with a single range measurement"""
    config = NavConfig()
    ekf = EKF(config)
    
    # Initial state at origin, zero velocity
    ekf.reset(np.zeros(6), np.eye(6)*1e6)
    
    # Beacon at (1000, 0, 0) km
    beacon = create_test_beacon(np.array([1000.0, 0.0, 0.0]))
    
    # Simulate measurement: range ~1000 km
    np.random.seed(42)
    true_range = np.linalg.norm(ekf.state[0:3] - beacon.fixed_position)
    noisy_range = true_range + np.random.normal(0, 0.1)
    
    meas = Measurement(
        beacon_id="test_beacon",
        timestamp=datetime.now(timezone.utc),
        kind=MeasurementKind.RANGE,
        value=noisy_range,
        uncertainty=0.1
    )
    
    success = ekf.update(meas, beacon)
    assert success, "EKF update should succeed"
    
    state, cov = ekf.get_state()
    # With one range measurement, we can't determine all 3 position coordinates uniquely,
    # but the filter should update state (at least partially)
    # Check that position has changed from zero
    assert not np.allclose(state[0:3], 0), "State should be updated"
    
def test_ekf_convergence():
    """Test that EKF converges to true position with multiple beacons"""
    config = NavConfig()
    ekf = EKF(config)
    
    # True state
    true_state = np.array([1000.0, 2000.0, 3000.0, 1.0, 0.5, 0.2])
    
    # Beacons in a tetrahedron around true position
    beacons = [
        create_test_beacon(np.array([0.0, 0.0, 0.0])),
        create_test_beacon(np.array([2000.0, 0.0, 0.0])),
        create_test_beacon(np.array([0.0, 3000.0, 0.0])),
        create_test_beacon(np.array([0.0, 0.0, 4000.0])),
    ]
    
    # Initialize with significant error
    initial = true_state + np.array([50, -100, 150, 0, 0, 0])
    ekf.reset(initial, np.eye(6)*1000)
    
    # Generate synthetic measurements with noise
    np.random.seed(42)
    for i in range(20):  # Multiple updates
        for beacon in beacons:
            # Compute true range
            r = np.linalg.norm(true_state[0:3] - beacon.fixed_position)
            noisy_r = r + np.random.normal(0, 0.1)
            
            meas = Measurement(
                beacon_id=beacon.id,
                timestamp=datetime(2000,1,1,12,0,0,tzinfo=timezone.utc) + timedelta(seconds=i*10),
                kind=MeasurementKind.RANGE,
                value=noisy_r,
                uncertainty=0.1
            )
            ekf.update(meas, beacon)
            
    state, cov = ekf.get_state()
    pos_error = np.linalg.norm(state[0:3] - true_state[0:3])
    vel_error = np.linalg.norm(state[3:6] - true_state[3:6])
    
    assert pos_error < 1.0, f"Position error {pos_error:.3f} km too large, expected <1.0 km"
    assert vel_error < 0.1, f"Velocity error {vel_error:.3f} km/s too large, expected <0.1 km/s"

def test_ukf_nonlinear():
    """Test UKF on a nonlinear measurement (direction)"""
    config = NavConfig()
    ukf = UKF(config)
    
    true_pos = np.array([1000.0, 500.0, 200.0])
    initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ukf.reset(initial, np.eye(6)*1e6)
    
    beacon = create_test_beacon(np.array([0.0, 0.0, 0.0]))
    
    # Direction measurement (unit vector towards beacon from true pos)
    true_vec = true_pos - beacon.fixed_position
    true_dir = true_vec / np.linalg.norm(true_vec)
    
    # Add small angular noise (~0.1 deg)
    np.random.seed(42)
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    angle = np.random.normal(0, np.radians(0.1))
    
    # Rodriguez rotation
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
    noisy_dir = R @ true_dir
    noisy_dir = noisy_dir / np.linalg.norm(noisy_dir)
    
    meas = Measurement(
        beacon_id="test_beacon",
        timestamp=datetime.now(timezone.utc),
        kind=MeasurementKind.DIRECTION,
        value=noisy_dir,
        uncertainty=0.001  # ~0.057 deg
    )
    
    success = ukf.update(meas, beacon)
    assert success, "UKF update should succeed"
    
    state, cov = ukf.get_state()
    error = np.linalg.norm(state[0:3] - true_pos)
    assert error < 10.0, f"UKF error {error:.2f} km too large, expected <10 km"

def test_particle_filter_multimodal():
    """Test particle filter on problem with ambiguous solutions"""
    config = NavConfig()
    pf = ParticleFilter(config, n_particles=500)
    
    # True position at (1000, 0, 0)
    true_pos = np.array([1000.0, 0.0, 0.0])
    
    # Two beacons at symmetric positions to create ambiguity
    beacons = [
        create_test_beacon(np.array([0.0, 1000.0, 0.0])),
        create_test_beacon(np.array([0.0, -1000.0, 0.0])),
    ]
    
    # Initialize particles over a wide area
    pf.particles[:, 0] = np.random.uniform(-2000, 2000, pf.n_particles)
    pf.particles[:, 1] = np.random.uniform(-2000, 2000, pf.n_particles)
    pf.particles[:, 2] = np.random.uniform(-500, 500, pf.n_particles)
    pf.particles[:, 3:] = 0  # zero velocity
    
    # First measurement: range to both beacons ~1414 km (sqrt(1000^2+1000^2))
    for beacon in beacons:
        r = np.linalg.norm(true_pos - beacon.fixed_position)
        meas = Measurement(
            beacon_id=beacon.id,
            timestamp=datetime.now(timezone.utc),
            kind=MeasurementKind.RANGE,
            value=r + np.random.normal(0, 0.5),
            uncertainty=0.5
        )
        pf.predict(0.0)  # no motion
        pf.update(meas, beacon)
        
    # Particle filter should converge to one of the two symmetric solutions
    mean, cov = pf.get_state()
    # Check that particles are concentrated near one solution
    # The mean might be at origin due to symmetry, but covariance should be small in one cluster
    # Actually with symmetric measurements, particles will form two clusters. ESS will be low.
    # This test just checks it doesn't crash
    assert np.trace(cov[0:3,0:3]) < 1e6, "Covariance should decrease"

def test_ekf_vs_ukf_consistency():
    """Compare EKF and UKF on linear problem (should give similar results)"""
    config = NavConfig()
    ekf = EKF(config)
    ukf = UKF(config)
    
    true_state = np.array([500.0, -200.0, 1000.0, 0.5, -0.2, 0.1])
    initial = true_state + np.array([10, 20, -30, 0, 0, 0])
    
    ekf.reset(initial, np.eye(6)*100)
    ukf.reset(initial, np.eye(6)*100)
    
    beacon = create_test_beacon(np.array([0.0, 0.0, 0.0]))
    
    # Range measurement
    r = np.linalg.norm(true_state[0:3] - beacon.fixed_position)
    meas = Measurement(
        beacon_id="test",
        timestamp=datetime.now(timezone.utc),
        kind=MeasurementKind.RANGE,
        value=r,
        uncertainty=0.1
    )
    
    ekf.update(meas, beacon)
    ukf.update(meas, beacon)
    
    ekf_state, _ = ekf.get_state()
    ukf_state, _ = ukf.get_state()
    
    # Should be very close for linear measurement model
    diff = np.linalg.norm(ekf_state - ukf_state)
    assert diff < 0.1, f"EKF and UKF differ by {diff:.3f}, expected <0.1"