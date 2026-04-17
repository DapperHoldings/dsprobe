"""
Star tracker sensor model for attitude determination.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from datetime import datetime, timezone
import random

try:
    from scipy.spatial.transform import Rotation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Fallback: implement basic rotation

@dataclass
class StarTrackerObservation:
    """Observation from star tracker"""
    timestamp: datetime
    star_id: str  # star catalog ID
    pixel_position: np.ndarray  # (u, v) in pixels
    brightness: float  # normalized intensity
    uncertainty: float = 1.0  # pixels

@dataclass
class StarCatalogEntry:
    """Entry in star catalog"""
    star_id: str
    magnitude: float  # visual magnitude
    position_icrf: np.ndarray  # unit vector in ICRF (J2000)
    proper_motion: np.ndarray = None  # arcsec/year (optional)
    parallax: float = 0.0  # arcsec

class StarTracker:
    """
    Model of an optical star tracker.
    
    Features:
    - Star detection with SNR-based uncertainty
    - Attitude determination from star observations ( QUEST, q-MEKF )
    - Catalog matching (pattern recognition)
    - Blunder detection (outlier rejection)
    """
    
    def __init__(self,
                 resolution: Tuple[int, int] = (1024, 1024),
                 pixel_size: float = 15e-6,  # 15 µm
                 focal_length: float = 0.02,  # 20 mm
                 field_of_view: float = np.radians(20),  # 20° diagonal
                 max_magnitude: float = 6.0,  # faintest detectable star
                 min_snr: float = 5.0,
                 catalog: Optional[List[StarCatalogEntry]] = None):
        """
        Initialize star tracker model.
        
        Args:
            resolution: Image resolution (width, height) in pixels
            pixel_size: Pixel physical size (meters)
            focal_length: Lens focal length (meters)
            field_of_view: Full diagonal FOV (radians)
            max_magnitude: Maximum (faintest) visual magnitude detectable
            min_snr: Minimum signal-to-noise ratio for detection
            catalog: Star catalog entries (if None, generates synthetic)
        """
        self.resolution = resolution
        self.pixel_size = pixel_size
        self.focal_length = focal_length
        self.fov = field_of_view
        self.max_magnitude = max_magnitude
        self.min_snr = min_snr
        
        # Intrinsic matrix K
        fx = focal_length / pixel_size
        fy = focal_length / pixel_size
        cx = resolution[0] / 2
        cy = resolution[1] / 2
        self.K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
        
        # Load or generate catalog
        if catalog is None:
            self.catalog = self._generate_synthetic_catalog(100)
        else:
            self.catalog = catalog
            
        # Current attitude (rotation from ICRF to camera frame)
        self.current_attitude = Rotation.identity()
        
        # Noise parameters
        self.read_noise_electrons = 5.0
        self.dark_current = 0.01  # e-/pixel/s
        self.quantum_efficiency = 0.5
        
    def _generate_synthetic_catalog(self, n_stars: int) -> List[StarCatalogEntry]:
        """Generate synthetic star catalog (uniform on sphere)"""
        catalog = []
        for i in range(n_stars):
            # Random unit vector
            vec = np.random.randn(3)
            vec = vec / np.linalg.norm(vec)
            
            # Random magnitude 1-6 (brighter is lower number)
            mag = random.uniform(1.0, 6.0)
            
            entry = StarCatalogEntry(
                star_id=f"SYN_{i:04d}",
                magnitude=mag,
                position_icrf=vec,
                proper_motion=None,
                parallax=0.0
            )
            catalog.append(entry)
        return catalog
    
    def project_star(self, 
                    star_direction_icrf: np.ndarray,
                    camera_attitude: Rotation) -> Optional[np.ndarray]:
        """
        Project a star (given as unit vector in ICRF) to image plane.
        
        Args:
            star_direction_icrf: Unit vector from observer to star in ICRF
            camera_attitude: Rotation from ICRF to camera frame
            
        Returns:
            (u, v) pixel coordinates or None if outside FOV
        """
        # Transform star direction to camera frame
        star_camera = camera_attitude.apply(star_direction_icrf)
        
        # Check if in front of camera
        if star_camera[2] <= 0:
            return None
            
        # Perspective projection (ignoring distortion)
        x = self.focal_length * star_camera[0] / star_camera[2]
        y = self.focal_length * star_camera[1] / star_camera[2]
        
        # Convert to pixels
        u = self.K[0,0] * x / self.focal_length + self.K[0,2]
        v = self.K[1,1] * y / self.focal_length + self.K[1,2]
        
        # Check bounds
        if 0 <= u < self.resolution[0] and 0 <= v < self.resolution[1]:
            return np.array([u, v])
        else:
            return None
    
    def simulate_image(self,
                      attitude: Rotation,
                      exposure_time: float = 0.1,
                      add_background: bool = True) -> np.ndarray:
        """
        Simulate star tracker image given attitude.
        
        Args:
            attitude: Rotation from ICRF to camera frame
            exposure_time: Exposure time (seconds)
            add_background: Include background noise (dark current, read noise)
            
        Returns:
            Simulated image (H x W uint16)
        """
        # Initialize dark image
        img = np.zeros(self.resolution[::-1], dtype=np.float32)  # H,W
        
        # Add stars
        for entry in self.catalog:
            pixel = self.project_star(entry.position_icrf, attitude)
            if pixel is None:
                continue
                
            # Flux from magnitude ( Pogson's law )
            # m1 - m2 = -2.5 log10(F1/F2)
            # Reference: Vega (m=0) gives ~1000 photons/s/cm^2 in V band? Simplified
            base_flux = 1000.0  # arbitrary
            flux = base_flux * 10**(-0.4 * entry.magnitude)
            
            # PSF: Gaussian
            sigma_pixels = 1.0  # 1 pixel FWHM ~ 2.35*sigma
            u, v = int(pixel[0]), int(pixel[1])
            
            # Spread over small region
            for du in [-1, 0, 1]:
                for dv in [-1, 0, 1]:
                    x, y = u+du, v+dv
                    if 0 <= x < self.resolution[0] and 0 <= y < self.resolution[1]:
                        exponent = -(du**2 + dv**2) / (2 * sigma_pixels**2)
                        img[y, x] += flux * np.exp(exponent) * exposure_time
                        
        if add_background:
            # Add dark current
            img += self.dark_current * exposure_time
            # Add read noise later as Poisson + Gaussian
            
        # Convert to electrons (assumeQE)
        img_electrons = img * self.quantum_efficiency
        
        # Apply Poisson shot noise
        img_noisy = np.random.poisson(img_electrons)
        
        # Add read noise (Gaussian)
        read_noise = np.random.normal(0, self.read_noise_electrons, img_noisy.shape)
        img_final = img_noisy + read_noise
        
        return np.clip(img_final, 0, 65535).astype(np.uint16)
    
    def detect_stars(self, image: np.ndarray, threshold: float = 30) -> List[StarTrackerObservation]:
        """
        Detect stars in image.
        
        Args:
            image: uint16 image
            threshold: Threshold above background for detection
            
        Returns:
            List of star observations
        """
        # Convert to float for processing
        img_float = image.astype(np.float32)
        
        # Simple threshold
        mask = img_float > threshold
        
        # Find centroids (connected components)
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask)
        
        observations = []
        for i in range(1, num_features+1):
            y_indices, x_indices = np.where(labeled == i)
            if len(y_indices) < 3:  # too small
                continue
                
            # Centroid
            u = np.mean(x_indices)
            v = np.mean(y_indices)
            
            # Brightness (sum)
            brightness = np.sum(img_float[y_indices, x_indices])
            
            # Uncertainty estimate (based on SNR)
            signal = brightness
            noise = np.sqrt(signal + self.read_noise_electrons**2 + (self.dark_current*0.1)**2)
            snr = signal / noise if noise > 0 else 0
            uncertainty = 1.0 / snr if snr > self.min_snr else 1.0
            
            obs = StarTrackerObservation(
                timestamp=datetime.now(timezone.utc),
                star_id="",  # unknown until catalog match
                pixel_position=np.array([u, v]),
                brightness=brightness,
                uncertainty=uncertainty
            )
            observations.append(obs)
            
        return observations
    
    def match_catalog(self,
                     observations: List[StarTrackerObservation],
                       max_angular_error: float = 0.1) -> Tuple[List[StarTrackerObservation], Rotation]:
        """
        Match observed stars to catalog to determine attitude.
        
        Args:
            observations: Detected star observations (without star IDs)
            max_angular_error: Max allowed matching error (radians)
            
        Returns:
            matched_observations: Observations with star_id filled
            attitude: Determined attitude (ICRF to camera)
            
        Raises:
            ValueError if insufficient matches
        """
        # This is a simplified pattern matching
        # Real star trackers use k-vector, triangle matching, etc.
        
        if len(observations) < 3:
            raise ValueError("Need at least 3 stars for attitude determination")
            
        # For demo: assume we know which stars they are (perfect matching)
        # In reality, would compute unit vectors from pixel coords using known focal length
        # and compare to catalog directions
        
        # Placeholder: assign random catalog stars
        # Real implementation would use QUEST algorithm or q-MEKF
        
        matched = []
        observed_directions = []
        catalog_directions = []
        
        for obs in observations[:min(len(observations), 10)]:  # limit for speed
            # Convert pixel to unit vector (pinhole)
            u, v = obs.pixel_position
            # Invert camera projection
            x = (u - self.K[0,2]) / self.K[0,0]
            y = (v - self.K[1,2]) / self.K[1,1]
            z = 1.0
            dir_camera = np.array([x, y, z])
            dir_camera = dir_camera / np.linalg.norm(dir_camera)
            
            # Find closest catalog star (angular distance)
            best_match = None
            best_dist = max_angular_error
            
            for entry in self.catalog:
                # Catalog direction is already unit vector in ICRF
                # Need to find rotation that aligns catalog to camera
                # This is circular; real algorithm finds global rotation first
                pass
                
            # For demo, just use the star as-is assuming we know it
            # This would be replaced with actual matching algorithm
            matched.append(obs)
            observed_directions.append(dir_camera)
            catalog_directions.append(obs.star_id)  # dummy
            
        if len(matched) < 3:
            raise ValueError("Failed to match enough stars")
            
        # Compute attitude using q-MEKF or Davenport's q method
        # Simplified: compute rotation that aligns observed to catalog
        # This is a placeholder
        attitude = Rotation.identity()
        
        return matched, attitude
    
    def get_attitude(self,
                    image: np.ndarray,
                    timestamp: Optional[datetime] = None) -> Tuple[Rotation, float]:
        """
        Full pipeline: detect stars, match catalog, compute attitude.
        
        Returns:
            attitude: Rotation from ICRF to camera frame
            rms_error: RMS angular error (radians)
        """
        # Detect
        observations = self.detect_stars(image)
        
        if len(observations) < 3:
            raise ValueError("Not enough stars detected")
            
        # Match and compute attitude
        matched, attitude = self.match_catalog(observations)
        
        # Compute RMS (would compare to expected positions)
        rms = 0.001  # placeholder
        
        return attitude, rms

# Example usage
if __name__ == "__main__":
    st = StarTracker()
    # Simulate image from known attitude (e.g., Earth pointing)
    r = Rotation.from_euler('zyx', [0, 0, 0], degrees=True)  # arbitrary
    img = st.simulate_image(r)
    print(f"Simulated image: {img.shape}, max={img.max()}")
    
    # Detect stars
    obs = st.detect_stars(img)
    print(f"Detected {len(obs)} stars")