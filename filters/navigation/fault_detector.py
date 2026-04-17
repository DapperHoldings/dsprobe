"""
Fault detection and outlier rejection for navigation measurements.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy import stats
from sklearn.covariance import MinCovDet

from config.settings import NavConfig
from core.measurement import Measurement
from core.beacon import Beacon
from filters.ekf import EKF

class FaultDetector:
    """
    Detect and reject faulty measurements using multiple methods:
        1. Innovation gating (chi-squared test)
        2. RANSAC (random sample consensus)
        3. Mahalanobis distance
        4. ML-based anomaly detection (if enabled)
    """
    
    def __init__(self, config: NavConfig):
        self.config = config
        self.innovation_history: List[np.ndarray] = []
        self.beacon_stats: Dict[str, Dict] = {}  # per-beacon residuals
        
        # For RANSAC
        self.ransac_threshold = config.ransac_threshold
        self.ransac_iterations = config.ransac_iterations
        
        # For ML anomaly detection
        self.anomaly_detector = None
        if config.outlier_handling == "ml":
            self._init_ml_detector()
            
    def _init_ml_detector(self):
        """Initialize machine learning anomaly detector"""
        try:
            from sklearn.ensemble import IsolationForest
            self.anomaly_detector = IsolationForest(contamination=0.1)
        except ImportError:
            print("scikit-learn not available; falling back to statistical methods")
            self.anomaly_detector = None
            
    def gate_innovation(self, 
                       innovation: np.ndarray,
                       innovation_cov: np.ndarray,
                       measurement: Measurement) -> bool:
        """
        Chi-squared innovation gating.
        
        Returns:
            True if measurement is consistent
        """
        try:
            # Compute chi-squared statistic
            inv_S = np.linalg.inv(innovation_cov)
            chi2 = innovation.T @ inv_S @ innovation
            dof = len(innovation)
            
            # Get threshold from config or from chi2 table
            if isinstance(self.config.measurement_outlier_threshold, dict):
                threshold = self.config.measurement_outlier_threshold.get(dof, 9.21)
            else:
                threshold = self.config.measurement_outlier_threshold
                
            # For 2 DOF (range+direction 2 components? Actually range+2D direction = 3 DOF)
            # But direction has only 2 independent dims if unit vector; we use 3 components with constraint
            # Typically use 3 DOF for range+2D direction
            # For pure range: 1 DOF
            
            return chi2 < threshold
        except np.linalg.LinAlgError:
            return False
        
    def ransac_filter(self,
                     measurements: List[Measurement],
                     beacons: Dict[str, Beacon],
                     navigator: EKF,
                     measurement_type: str = "both") -> List[Measurement]:
        """
        RANSAC to find inlier measurements.
        
        Steps:
            1. Randomly select minimal subset (4 for 3D range, 2 for direction-only)
            2. Estimate position from subset
            3. Score all measurements by residual
            4. Repeat; keep best consensus set
        """
        if len(measurements) < 4:
            return measurements  # not enough for RANSAC
            
        best_inliers = []
        best_score = -np.inf
        
        # Minimal sample size: 4 for range-only 3D; 2 for direction-only
        # If mixed, we need at least 4 total measurements for 3D
        min_sample = 4
        
        for iteration in range(self.ransac_iterations):
            # Random sample
            sample = np.random.choice(measurements, min_sample, replace=False)
            
            # Hypothetical position estimate from sample
            # Copy navigator and update with sample only
            temp_nav = EKF(self.config)
            temp_nav.state = navigator.state.copy()
            temp_nav.covariance = navigator.covariance.copy()
            
            for m in sample:
                beacon = beacons[m.beacon_id]
                success = temp_nav.update(m, beacon)
                if not success:
                    break
            else:
                # Successfully got position
                pos_est = temp_nav.state[0:3]
                
                # Score all measurements
                inliers = []
                for m in measurements:
                    beacon = beacons[m.beacon_id]
                    b_pos = beacon.get_position(0.0)  # epoch not used? Should be at measurement time
                    r_vec = pos_est - b_pos
                    r = np.linalg.norm(r_vec)
                    
                    if m.kind == MeasurementKind.RANGE:
                        err = abs(m.value - r)
                        threshold = self.ransac_threshold  # km
                    elif m.kind == MeasurementKind.DIRECTION:
                        dir_true = r_vec / max(r, 1e-9)
                        # Angular distance
                        dot = np.dot(m.value, dir_true)
                        err = np.arccos(np.clip(dot, -1, 1))  # radians
                        threshold = np.radians(self.ransac_threshold) if self.ransac_threshold < 10 else self.ransac_threshold
                    else:
                        continue
                        
                    if err < threshold:
                        inliers.append(m)
                        
                if len(inliers) > best_score:
                    best_score = len(inliers)
                    best_inliers = inliers
                    
        if not best_inliers:
            # Fallback: all measurements
            return measurements
        return best_inliers
    
    def mahalanobis_filter(self,
                          measurements: List[Measurement],
                          beacons: Dict[str, Beacon],
                          navigator: EKF) -> List[Measurement]:
        """
        Use Mahalanobis distance of innovations to detect outliers.
        Assumes we have state and covariance from previous update.
        """
        if len(self.innovation_history) < 10:
            return measurements  # not enough history
            
        # Estimate innovation covariance from recent history
        innovations = np.array(self.innovation_history[-100:])  # last 100
        if innovations.shape[0] < 5:
            return measurements
            
        # Compute robust covariance (Minimum Covariance Determinant)
        try:
            robust_cov = MinCovDet().fit(innovations).covariance_
        except:
            robust_cov = np.cov(innovations.T)
            
        # Compute Mahalanobis distance for each new measurement's innovation
        inliers = []
        for m in measurements:
            if m.innovation is None:
                continue
            diff = m.innovation
            try:
                inv_cov = np.linalg.inv(robust_cov)
                d2 = diff.T @ inv_cov @ diff
                # Chi-squared threshold with DOF = len(diff)
                from scipy.stats import chi2
                threshold = chi2.ppf(0.99, len(diff))
                if d2 < threshold:
                    inliers.append(m)
            except:
                inliers.append(m)  # keep if can't compute
                
        return inliers if len(inliers) >= 3 else measurements
    
    def update_beacon_reliability(self, 
                                 beacon_id: str,
                                 residual: float,
                                 measurement_type: str):
        """
        Update beacon reliability using exponential moving average.
        """
        if beacon_id not in self.beacon_stats:
            self.beacon_stats[beacon_id] = {
                "residuals": [],
                "mean": 0.0,
                "var": 1.0,
                "count": 0
            }
        stats = self.beacon_stats[beacon_id]
        
        # Update running mean/var (Welford's algorithm)
        stats["count"] += 1
        delta = residual - stats["mean"]
        stats["mean"] += delta / stats["count"]
        delta2 = residual - stats["mean"]
        stats["var"] = 0.9 * stats["var"] + 0.1 * delta * delta2  # EMA
        
        # Reliability = exp(-normalized_residual^2)
        norm_res = abs(residual) / max(np.sqrt(stats["var"]), 1e-6)
        reliability = np.exp(-0.5 * norm_res**2)
        return reliability
        
    def detect_anomaly_ml(self, 
                         measurement: Measurement,
                         context_features: np.ndarray) -> Tuple[bool, float]:
        """
        ML-based anomaly detection.
        Returns (is_anomaly, confidence).
        """
        if self.anomaly_detector is None:
            return False, 0.0
            
        # Feature vector: [residual_magnitude, beacon_health, signal_strength, ...]
        features = context_features.reshape(1, -1)
        try:
            pred = self.anomaly_detector.predict(features)[0]
            score = self.anomaly_detector.decision_function(features)[0]
            return pred == -1, 1.0 - score  # anomaly, confidence
        except:
            return False, 0.0