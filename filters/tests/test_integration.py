"""
Integration tests for the full navigation system.
"""

import numpy as np
import pytest
from datetime import datetime, timezone, timedelta

from config.settings import NavConfig
from core.beacon import Beacon, create_planet_ephemeris, BeaconType
from core.measurement import Measurement, MeasurementKind
from navigation.navigator import AdvancedBeaconNavigator
from integration.spice_integration import SPICEIntegration
from sensors.imu import IMU, IMUReading

class TestSPICEIntegration:
    """Test SPICE integration (if available)"""
    
    @pytest.mark.skipif(not SPICE_AVAILABLE, reason="SPICE not installed")
    def test_spice_earth_position(self):
        """Test getting Earth position from SPICE"""
        spice = SPICEIntegration()
        # Load kernels if needed - would need actual kernel files
        # This is a placeholder test
        pass

class TestFullNavigationPipeline:
    """Test complete navigation workflow"""
    
    def test_simple_mission(self):
        """Test a simple mission with 3 beacons"""
        # Create beacons
        earth_ephem = create_planet_ephemeris("earth")
        mars_ephem = create_planet_ephemeris("mars")
        
        beacons = [
            Beacon(
                id="earth",
                name="Earth",
                beacon_type=BeaconType.OPTICAL,
                ephemeris=earth_ephem,
                base_uncertainty=(10.0, 0.001)
            ),
            Beacon(
                id="mars",
                name="Mars",
                beacon_type=BeaconType.OPTICAL,
                ephemeris=mars_ephem,
                base_uncertainty=(10.0, 0.001)
            ),
            Beacon(
                id="relay",
                name="Earth Relay",
                beacon_type=BeaconType.RADIO,
                fixed_position=np.array([1.5e6, 0, 0]),  # Earth L2
                base_uncertainty=(0.01, 0.0001)
            )
        ]
        
        config = NavConfig(
            filter_type="ekf",
            debug_logging=False,
            enable_fault_detection=True
        )
        
        nav = AdvancedBeaconNavigator(beacons, config)
        
        # Start near Earth
        earth_pos = beacons[0].get_position(0.0)
        start_pos = earth_pos + np.array([0, 1000, 500])
        nav.initialize(
            initial_position=start_pos,
            initial_velocity=np.array([0, 0, 0])
        )
        
        # Run for 10 steps
        t0 = 0.0
        dt = 3600.0  # 1 hour steps
        
        for i in range(10):
            current_time = datetime(2000,1,1,12,0,0,tzinfo=timezone.utc) + timedelta(seconds=t0 + i*dt)
            nav.update_time(current_time)
            nav.predict(dt)
            
            # Auto-select beacons
            selected = nav.beacon_selector.select_beacons(
                list(nav.beacons.values()),
                nav.filter,
                nav._datetime_to_epoch(current_time),
                n_select=3
            )
            
            # Get measurements
            measurements = nav.acquire_measurements(selected)
            
            # Process
            state, cov = nav.process_measurements(measurements)
            
            # Check solution
            sol = nav.get_solution()
            assert sol['position'] is not None
            assert sol['pdop'] >= 0
            
            # Should converge toward some reasonable position
            # Not testing exact values because ephemerides are simplified
        
        print(f"Final position: {state[0:3]}")
        print(f"Final PDOP: {sol['pdop']:.2f} km")
        
    def test_imu_integration(self):
        """Test integration with IMU"""
        config = NavConfig(enable_imu=True)
        beacon = Beacon(
            id="test",
            name="Test",
            beacon_type=BeaconType.RADIO,
            fixed_position=np.array([1e6, 0, 0]),
            base_uncertainty=(0.1, 0.001)
        )
        
        nav = AdvancedBeaconNavigator([beacon], config)
        nav.initialize(
            initial_position=np.array([0,0,0]),
            initial_velocity=np.array([0,0,0])
        )
        
        # Create IMU
        imu = IMU(accel_noise_std=0.01, gyro_noise_std=0.001)
        
        # Generate some IMU readings (zero acceleration in inertial frame)
        # But need to account for gravity? Simplified
        imu_readings = []
        for i in range(10):
            # True acceleration would be gravity + thrust; here zero
            reading = imu.generate_reading(
                true_accel=np.array([0,0,0]),
                true_gyro=np.array([0,0,0]),
                timestamp=datetime.now(timezone.utc)
            )
            imu_readings.append(reading)
            
        # Use IMU in predict
        for i, reading in enumerate(imu_readings):
            nav.predict(1.0, imu_reading=reading)
            
        # Should still work
        state, _ = nav.filter.get_state()
        assert state is not None
        
    def test_checkpointing(self):
        """Test save/load checkpoint"""
        import tempfile
        import os
        
        beacon = Beacon(
            id="test",
            name="Test",
            beacon_type=BeaconType.RADIO,
            fixed_position=np.array([1e6, 0, 0])
        )
        
        nav = AdvancedBeaconNavigator([beacon], NavConfig())
        nav.initialize(np.array([100,200,300]), np.array([1,2,3]))
        
        # Add a measurement
        meas = Measurement(
            beacon_id="test",
            timestamp=datetime.now(timezone.utc),
            kind=MeasurementKind.RANGE,
            value=1.0e6 + 10.0,
            uncertainty=0.1
        )
        nav.process_measurements([meas])
        
        # Save
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        nav.save_checkpoint(temp_path)
        
        # Create new navigator and load
        nav2 = AdvancedBeaconNavigator([beacon], NavConfig())
        nav2.load_checkpoint(temp_path)
        
        # Compare states
        state1, cov1 = nav.filter.get_state()
        state2, cov2 = nav2.filter.get_state()
        assert np.allclose(state1, state2)
        assert np.allclose(cov1, cov2)
        
        os.unlink(temp_path)

def test_measurement_io():
    """Test measurement serialization"""
    meas = Measurement(
        beacon_id="test",
        timestamp=datetime(2024,1,1,0,0,0,tzinfo=timezone.utc),
        kind=MeasurementKind.RANGE,
        value=1000.0,
        uncertainty=1.0,
        quality=0.9
    )
    
    # Convert to dict and back
    d = meas.to_dict()
    meas2 = Measurement.from_dict(d)
    
    assert meas2.beacon_id == meas.beacon_id
    assert meas2.value == meas.value
    assert meas2.uncertainty == meas.uncertainty
    assert meas2.quality == meas.quality