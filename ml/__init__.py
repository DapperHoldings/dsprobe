"""
Machine Learning extensions for beacon navigation.
Includes:
- Beacon selection via reinforcement learning
- Anomaly detection for faulty measurements
- Adaptive noise tuning with neural networks
- End-to-end differentiable filtering
"""

from .beacon_selection_ml import BeaconSelectionEnv, BeaconSelectionAgent, train_beacon_selector
from .anomaly_detection import MeasurementAnomalyDetector, train_anomaly_detector
from .reinforcement_learning import NavigationRLAgent, NavigationEnv

__all__ = [
    "BeaconSelectionEnv",
    "BeaconSelectionAgent", 
    "train_beacon_selector",
    "MeasurementAnomalyDetector",
    "train_anomaly_detector",
    "NavigationRLAgent",
    "NavigationEnv"
]