"""
Anomaly detection for navigation measurements.
Uses machine learning to identify faulty beacons or sensor malfunctions.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from config.settings import NavConfig
from core.measurement import Measurement
from core.beacon import Beacon

class MeasurementAnomalyDetector:
    """
    Detect anomalous measurements using multiple ML techniques.
    Can be trained on normal operation data.
    """
    
    def __init__(self, 
                 method: str = "isolation_forest",
                 contamination: float = 0.1):
        """
        Args:
            method: "isolation_forest", "svm", "lof", "autoencoder"
            contamination: Expected proportion of outliers
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.is_trained = False
        
        # Feature extraction parameters
        self.scaler_mean = None
        self.scaler_std = None
        
    def extract_features(self, 
                        measurement: Measurement,
                        beacon: Beacon,
                        navigator_state: np.ndarray,
                        innovation_history: List[np.ndarray]) -> np.ndarray:
        """
        Extract feature vector from measurement context.
        
        Features:
            - Measurement value (range/direction components)
            - Beacon health, reliability
            - Time since last measurement from this beacon
            - Innovation magnitude (if available)
            - Innovation history statistics
            - Signal strength (if available)
        """
        features = []
        
        # 1. Measurement value (flatten)
        val = measurement.as_vector()
        features.extend(val.tolist())
        
        # 2. Beacon health
        features.append(beacon.health)
        features.append(beacon.reliability)
        
        # 3. Beacon type one-hot
        type_vec = np.zeros(len(beacon.beacon_type))
        type_vec[list(beacon.beacon_type).index(beacon.beacon_type)] = 1
        features.extend(type_vec.tolist())
        
        # 4. Measurement uncertainty
        unc = measurement.as_covariance()
        if unc.ndim == 2:
            features.extend(np.diag(unc).tolist())
        else:
            features.append(unc)
            
        # 5. Signal strength (from metadata)
        snr = measurement.metadata.get('snr', 1.0)
        features.append(snr)
        
        # 6. Innovation (if available)
        if measurement.innovation is not None:
            features.extend(np.abs(measurement.innovation).tolist())
        else:
            features.extend([0.0] * measurement.get_measurement_matrix_size())
            
        # 7. Innovation history stats
        if len(innovation_history) > 0:
            recent_innov = np.array(innovation_history[-10:])  # last 10
            features.append(np.mean(recent_innov))
            features.append(np.std(recent_innov))
            features.append(np.max(np.abs(recent_innov)))
        else:
            features.extend([0.0, 0.0, 0.0])
            
        # 8. Time of day ( cyclical encoding)
        # Not needed for space, but could for Earth-orbiting
        # features.append(np.sin(2*np.pi*hour/24))
        # features.append(np.cos(2*np.pi*hour/24))
        
        return np.array(features, dtype=np.float32)
    
    def train(self, 
              normal_measurements: List[Tuple[Measurement, Beacon, np.ndarray, List[np.ndarray]]]):
        """
        Train anomaly detector on normal (non-anomalous) measurements.
        
        Args:
            normal_measurements: List of (measurement, beacon, state, innov_history) tuples
        """
        print(f"Training {self.method} anomaly detector on {len(normal_measurements)} samples...")
        
        # Extract features
        X = []
        for meas, beacon, state, hist in normal_measurements:
            feat = self.extract_features(meas, beacon, state, hist)
            X.append(feat)
            
        X = np.array(X)
        
        # Normalize
        self.scaler_mean = X.mean(axis=0)
        self.scaler_std = X.std(axis=0) + 1e-8
        X_norm = (X - self.scaler_mean) / self.scaler_std
        
        # Train model
        if self.method == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
        elif self.method == "svm":
            self.model = OneClassSVM(
                nu=self.contamination,
                kernel="rbf"
            )
        elif self.method == "lof":
            self.model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        self.model.fit(X_norm)
        self.is_trained = True
        print("Training complete.")
        
    def predict(self, 
                measurement: Measurement,
                beacon: Beacon,
                navigator_state: np.ndarray,
                innovation_history: List[np.ndarray]) -> Tuple[bool, float]:
        """
        Predict if measurement is anomalous.
        
        Returns:
            is_anomaly (True if outlier), confidence score [0,1]
        """
        if not self.is_trained:
            return False, 0.0
            
        feat = self.extract_features(measurement, beacon, navigator_state, innovation_history)
        feat_norm = (feat - self.scaler_mean) / self.scaler_std
        
        if self.method == "lof":
            # LOF returns negative outlier factor
            score = self.model.decision_function([feat_norm])[0]
            # Negative is outlier; more negative = more anomalous
            is_anomaly = score < 0
            confidence = 1.0 / (1.0 + np.exp(-score))  # sigmoid
        else:
            # IsolationForest/OneClassSVM: -1 for outlier, 1 for inlier
            pred = self.model.predict([feat_norm])[0]
            is_anomaly = (pred == -1)
            # Get anomaly score (lower = more anomalous)
            score = self.model.decision_function([feat_norm])[0]
            confidence = 1.0 - (score + 1) / 2  # normalize to [0,1], crude
            
        return is_anomaly, confidence
    
    def save(self, filepath: str):
        """Save trained model"""
        import joblib
        joblib.dump({
            'model': self.model,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'method': self.method
        }, filepath)
        
    @classmethod
    def load(cls, filepath: str) -> "MeasurementAnomalyDetector":
        """Load trained model"""
        import joblib
        data = joblib.load(filepath)
        detector = cls(method=data['method'])
        detector.model = data['model']
        detector.scaler_mean = data['scaler_mean']
        detector.scaler_std = data['scaler_std']
        detector.is_trained = True
        return detector

class DeepAnomalyDetector(nn.Module):
    """
    Deep autoencoder for anomaly detection.
    Reconstructs input features; high reconstruction error indicates anomaly.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

def train_anomaly_detector(detector: MeasurementAnomalyDetector,
                          normal_data: List,
                          test_data: Optional[List] = None,
                          epochs: int = 50) -> Dict[str, float]:
    """
    Train anomaly detector (deep version if method='autoencoder').
    
    Returns:
        Training metrics (loss, accuracy)
    """
    if detector.method == 'autoencoder':
        return _train_autoencoder(detector, normal_data, test_data, epochs)
    else:
        # Traditional ML models trained in .train()
        detector.train(normal_data)
        return {"status": "trained"}

def _train_autoencoder(detector: MeasurementAnomalyDetector,
                      normal_data: List,
                      test_data: Optional[List],
                      epochs: int) -> Dict[str, float]:
    """Train deep autoencoder"""
    # Extract features
    X = []
    for meas, beacon, state, hist in normal_data:
        feat = detector.extract_features(meas, beacon, state, hist)
        X.append(feat)
    X = np.array(X)
    
    # Normalize
    detector.scaler_mean = X.mean(axis=0)
    detector.scaler_std = X.std(axis=0) + 1e-8
    X_norm = (X - detector.scaler_mean) / detector.scaler_std
    
    # Convert to PyTorch
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_norm)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Model
    model = DeepAnomalyDetector(input_dim=X_norm.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Train
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            recon, _ = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total_loss/len(loader):.6f}")
            
    detector.model = model
    detector.is_trained = True
    return {"final_loss": total_loss/len(loader)}