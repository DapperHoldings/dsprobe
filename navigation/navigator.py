"""
Main navigation orchestrator - the AdvancedBeaconNavigator.
Combines all modules into a cohesive system.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field

from config.settings import NavConfig, FilterType
from core.beacon import Beacon, BeaconType
from core.measurement import Measurement, MeasurementKind
from core.state import State
from filters import EKF, UKF  # import both, choose at runtime
from sensors.imu import IMU, IMUReading
from sensors.optical_camera import OpticalCamera, ImageMeasurement
from sensors.xray_pulsar import XRayPulsar, PulsarTimingMeasurement
from navigation.beacon_manager import BeaconManager
from navigation.beacon_selector import BeaconSelector
from navigation.fault_detector import FaultDetector
from navigation.collision_avoidance import CollisionAvoidance, KeepOutZone
from integration.spice_integration import SPICEIntegration
from utils.logging import NavLogger
from utils.timing import Timer

class AdvancedBeaconNavigator:
    """
    Central navigation system.
    
    Workflow:
        1. Update time (propagate beacons via ephemeris)
        2. Predict step (using IMU or simply constant velocity)
        3. Select beacons to observe
        4. Acquire measurements (simulated or from real sensors)
        5. Process measurements (fault detection, filter update)
        6. Check collision avoidance
        7. Return navigation solution
    """
    
    def __init__(self,
                 beacons: List[Beacon],
                 config: Optional[NavConfig] = None,
                 filter_type: str = "ekf"):
        
        self.config = config or NavConfig()
        self.beacons = {b.id: b for b in beacons}
        
        # Initialize logger
        self.logger = NavLogger(self.config)
        
        # Initialize filter
        if filter_type.lower() == "ekf":
            self.filter = EKF(self.config)
        elif filter_type.lower() == "ukf":
            self.filter = UKF(self.config)
        else:
            raise ValueError(f"Unsupported filter: {filter_type}")
            
        # Initialize sub-modules
        self.beacon_manager = BeaconManager(self.beacons, self.config)
        self.beacon_selector = BeaconSelector(self.config)
        self.fault_detector = FaultDetector(self.config)
        self.collision_avoidance = CollisionAvoidance(
            keep_out_zones=[],  # can add from config
            min_safe_altitude=self.config.min_safe_altitude
        )
        
        # SPICE integration (optional)
        self.spice = None
        if self.config.enable_spice:
            self.spice = SPICEIntegration()
            
        # Sensors (optional)
        self.imu = None
        self.camera = None
        self.pulsars = {}  # id -> XRayPulsar
        if self.config.enable_imu:
            self.imu = IMU(
                accel_noise_std=self.config.imu_noise_accel,
                gyro_noise_std=self.config.imu_noise_gyro
            )
        if self.config.enable_optical_flow:
            self.camera = OpticalCamera()
            
        # State
        self.current_time: Optional[datetime] = None
        self.measurement_history: List[Measurement] = []
        self.state_history: List[State] = []
        
    def initialize(self, 
                   initial_position: np.ndarray,
                   initial_velocity: np.ndarray = None,
                   initial_attitude: Optional[np.ndarray] = None,
                   initial_covariance: Optional[np.ndarray] = None):
        """Set initial state estimate"""
        if initial_velocity is None:
            initial_velocity = np.zeros(3)
            
        # Build state vector (minimal = pos+vel)
        state = np.hstack([initial_position, initial_velocity])
        
        if initial_covariance is None:
            initial_covariance = np.eye(6) * 1e6  # 1000 km initial uncertainty
            
        self.filter.reset(state, initial_covariance)
        self.logger.info(f"Navigator initialized at position {initial_position/1e3:.1f} km")
        
    def update_time(self, timestamp: datetime):
        """Update current time and propagate beacon positions"""
        self.current_time = timestamp
        # Update beacons via ephemeris
        epoch = self._datetime_to_epoch(timestamp)
        for beacon in self.beacons.values():
            if beacon.ephemeris is not None:
                beacon.fixed_position = beacon.get_position(epoch)
                
        # Update collision avoidance zones that move (e.g., planets)
        # Could update zone centers too
        
    def predict(self, 
                dt: float,
                imu_reading: Optional[IMUReading] = None):
        """
        Prediction step.
        
        Args:
            dt: Time step (seconds)
            imu_reading: If provided, integrate IMU; else use constant velocity
        """
        with Timer("predict", self.logger):
            # If IMU available and enabled, we might use it
            self.filter.predict(dt, imu_data=imu_reading)
            
    def acquire_measurements(self,
                           selected_beacons: Optional[List[Beacon]] = None,
                           sensor_data: Optional[Dict] = None) -> List[Measurement]:
        """
        Simulate or acquire real measurements from selected beacons.
        
        In real spacecraft, this would interface with hardware drivers.
        In simulation, we generate synthetic measurements.
        
        Returns:
            List of Measurement objects
        """
        measurements = []
        
        if selected_beacons is None:
            # Auto-select based on visibility
            state, _ = self.filter.get_state()
            visible = self.beacon_manager.get_visible_beacons(
                state[0:3], self.current_time)
            selected_beacons = self.beacon_selector.select_beacons(
                visible, self.filter, self._datetime_to_epoch(self.current_time))
            
        for beacon in selected_beacons:
            # Simulate measurement
            meas = self._simulate_measurement(beacon, sensor_data)
            measurements.append(meas)
            
        return measurements
    
    def _simulate_measurement(self, 
                            beacon: Beacon,
                            sensor_data: Optional[Dict] = None) -> Measurement:
        """Create synthetic measurement from beacon (for simulation)"""
        state, _ = self.filter.get_state()
        observer_pos = state[0:3]
        
        # Get true range and direction (without filter bias)
        r, direction = beacon.get_range_and_direction(
            observer_pos, self._datetime_to_epoch(self.current_time))
        
        # Determine measurement kind based on beacon type
        if beacon.beacon_type == BeaconType.RADIO:
            kind = MeasurementKind.BOTH  # radio can give range and direction (e.g., DOR)
            range_std, dir_std = beacon.get_uncertainty(r, observer_pos)
        elif beacon.beacon_type == BeaconType.OPTICAL:
            kind = MeasurementKind.DIRECTION  # OpNav usually direction only unless known size
            _, dir_std = beacon.get_uncertainty(r, observer_pos)
            range_std = None
            dir_std = dir_std * 10  # conservative
        elif beacon.beacon_type == BeaconType.PULSAR:
            kind = MeasurementKind.RANGE  # XNAV typically gives range via TOA if pulsar distance known
            range_std, _ = beacon.get_uncertainty(r, observer_pos)
            dir_std = None
        else:
            kind = MeasurementKind.RANGE
            range_std, dir_std = beacon.get_uncertainty(r, observer_pos)
            
        # Add noise
        if kind == MeasurementKind.RANGE:
            noisy_range = r + np.random.normal(0, range_std)
            value = noisy_range
            uncertainty = range_std
        elif kind == MeasurementKind.DIRECTION:
            # Add angular noise (rotation of unit vector)
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            angle = np.random.normal(0, dir_std)
            # Rotation matrix (Rodrigues)
            K = np.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
            noisy_dir = R @ direction
            noisy_dir = noisy_dir / np.linalg.norm(noisy_dir)
            value = noisy_dir
            uncertainty = dir_std
        elif kind == MeasurementKind.BOTH:
            noisy_range = r + np.random.normal(0, range_std)
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            angle = np.random.normal(0, dir_std)
            K = np.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
            noisy_dir = R @ direction
            noisy_dir = noisy_dir / np.linalg.norm(noisy_dir)
            value = np.hstack([noisy_range, noisy_dir])
            uncertainty = np.hstack([range_std, dir_std, dir_std])  # directional per component
        else:
            raise ValueError(f"Can't simulate kind {kind}")
            
        meas = Measurement(
            beacon_id=beacon.id,
            timestamp=self.current_time,
            kind=kind,
            value=value,
            uncertainty=uncertainty,
            quality=beacon.health
        )
        return meas
    
    def process_measurements(self,
                           measurements: List[Measurement],
                           use_fault_detection: bool = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a batch of measurements.
        
        Args:
            measurements: List of Measurement objects
            use_fault_detection: Override config
            
        Returns:
            Updated state and covariance
        """
        if use_fault_detection is None:
            use_fault_detection = self.config.enable_fault_detection
            
        with Timer("measurement_processing", self.logger):
            # Filter valid measurements
            valid = [m for m in measurements if m.is_valid()]
            
            if use_fault_detection and len(valid) > 5:
                # Apply RANSAC or other outlier rejection
                if self.config.outlier_handling == "ransac":
                    inliers = self.fault_detector.ransac_filter(
                        valid, self.beacons, self.filter)
                    self.logger.debug(f"RANSAC: {len(inliers)}/{len(valid)} inliers")
                    valid = inliers
                elif self.config.outlier_handling == "gating":
                    # Will be done per-measurement in update
                    pass
                    
            # Process each measurement
            innovations = []
            for m in valid:
                beacon = self.beacons[m.beacon_id]
                success = self.filter.update(m, beacon)
                if success:
                    self.measurement_history.append(m)
                    # Update beacon reliability
                    if m.innovation is not None:
                        residual = np.linalg.norm(m.innovation)
                        reliability = self.fault_detector.update_beacon_reliability(
                            m.beacon_id, residual, m.kind.value)
                        beacon.reliability = reliability
                        m.outlier_score = 1.0 - reliability
                        
                    innovations.append(m.innovation)
                    
            # Check for anomalies using ML if configured
            if self.config.outlier_handling == "ml" and self.fault_detector.anomaly_detector:
                # Build feature vectors from recent measurements
                pass  # placeholder
                
            # Limit history size
            if len(self.measurement_history) > self.config.max_history_size:
                self.measurement_history = self.measurement_history[-self.config.max_history_size:]
                
            # Get current state
            state, cov = self.filter.get_state()
            
            # Log
            self.logger.debug(f"State: pos={state[0:3]/1e3:.1f} km, "
                            f"pdop={self.filter.get_pdop():.2f} km")
            
            return state, cov
    
    def get_solution(self) -> Dict:
        """Get current navigation solution as dictionary"""
        state, cov = self.filter.get_state()
        pos_cov = self.filter.covariance[0:3,0:3]
        
        # Check for collisions
        trajectory = self.collision_avoidance.generate_trajectory_prediction(state)
        violations = self.collision_avoidance.check_keep_out_zones(
            trajectory, self._datetime_to_epoch(self.current_time))
        
        maneuvers = []
        for zone, t_ca, dist in violations:
            maneuver = self.collision_avoidance.compute_avoidance_maneuver(
                state, zone, t_ca, self._datetime_to_epoch(self.current_time))
            if maneuver:
                maneuvers.append(maneuver)
                
        return {
            "timestamp": self.current_time,
            "position": state[0:3],
            "velocity": state[3:6],
            "position_covariance": pos_cov,
            "velocity_covariance": cov[3:6,3:6],
            "pdop": self.filter.get_pdop(),
            "visible_beacons": len(self.beacon_manager.get_visible_beacons(state[0:3], self.current_time)),
            "measurements_used": len(self.measurement_history),
            "collision_warnings": len(violations),
            "avoidance_maneuvers": maneuvers,
            "filter": self.config.filter_type.value
        }
    
    def save_checkpoint(self, filepath: str):
        """Save full state to file"""
        import json
        state, cov = self.filter.get_state()
        data = {
            "state": state.tolist(),
            "covariance": cov.tolist(),
            "time": self.current_time.isoformat(),
            "measurement_history": [m.to_dict() for m in self.measurement_history],
            "config": self.config.to_dict()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_checkpoint(self, filepath: str):
        """Load state from checkpoint"""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.filter.reset(np.array(data["state"]), np.array(data["covariance"]))
        self.current_time = datetime.fromisoformat(data["time"])
        self.measurement_history = [Measurement.from_dict(m) for m in data["measurement_history"]]
        
    @staticmethod
    def _datetime_to_epoch(dt: datetime) -> float:
        j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        return (dt - j2000).total_seconds()