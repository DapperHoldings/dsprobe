"""
Optical camera model for optical navigation (OpNav).
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timezone
import cv2  # OpenCV for image processing (if available)

@dataclass
class ImageMeasurement:
    """Measurement from optical camera"""
    beacon_id: str
    timestamp: datetime
    pixel_coordinates: np.ndarray  # (u, v) or (x, y)
    image_shape: Tuple[int, int]  # (height, width)
    exposure_time: float  # seconds
    signal_strength: float  # normalized
    uncertainty: float = 1.0  # pixel

class OpticalCamera:
    """
    Pinhole camera model with distortion.
    """
    
    def __init__(self,
                 resolution: Tuple[int, int] = (2048, 2048),
                 pixel_size: float = 5.5e-6,  # meters
                 focal_length: float = 0.1,  # meters (100mm)
                 distortion_coeffs: Optional[np.ndarray] = None,
                 quantum_efficiency: float = 0.5,
                 read_noise: float = 5.0,  # electrons
                 dark_current: float = 0.01,  # e-/pixel/sec
                 prnu: float = 0.01,  # 1% pixel response non-uniformity
                 ):
        """
        Initialize camera model.
        
        Args:
            resolution: (width, height) in pixels
            pixel_size: Size of pixel (meters)
            focal_length: Focal length (meters)
            distortion_coeffs: [k1, k2, p1, p2, k3] radial & tangential
            quantum_efficiency: QE (0-1)
            read_noise: RMS read noise (electrons)
            dark_current: Dark current (e-/s/pixel)
            prnu: Photon response non-uniformity (0-1)
        """
        self.resolution = resolution
        self.pixel_size = pixel_size
        self.focal_length = focal_length
        
        # Intrinsic matrix K
        fx = focal_length / pixel_size
        fy = focal_length / pixel_size
        cx = resolution[0] / 2
        cy = resolution[1] / 2
        self.K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
        
        self.dist_coeffs = distortion_coeffs if distortion_coeffs is not None else np.zeros(5)
        self.QE = quantum_efficiency
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.prnu = prnu
        
        # Calibration status
        self.calibrated = True if distortion_coeffs is not None else False
        
    def project_point(self, 
                     point_camera: np.ndarray) -> np.ndarray:
        """
        Project 3D point in camera frame to 2D pixel coordinates.
        
        Args:
            point_camera: 3D point in camera coordinate frame (meters)
            
        Returns:
            (u, v) pixel coordinates
        """
        # Perspective division
        if point_camera[2] <= 0:
            raise ValueError("Point behind camera")
        x = point_camera[0] / point_camera[2]
        y = point_camera[1] / point_camera[2]
        
        # Apply radial distortion
        r2 = x*x + y*y
        r4 = r2 * r2
        r6 = r2 * r3
        
        k1, k2, p1, p2, k3 = self.dist_coeffs
        radial = 1 + k1*r2 + k2*r4 + k3*r6
        tangential_x = 2*p1*x*y + p2*(r2 + 2*x*x)
        tangential_y = p1*(r2 + 2*y*y) + 2*p2*x*y
        
        x_distorted = x*radial + tangential_x
        y_distorted = y*radial + tangential_y
        
        # Convert to pixels
        pixel = self.K @ np.array([x_distorted, y_distorted, 1.0])
        return pixel[:2]
    
    def backproject_pixel(self,
                         pixel: np.ndarray,
                         depth: float) -> np.ndarray:
        """
        Backproject pixel to 3D ray at given depth.
        
        Args:
            pixel: (u, v) coordinates
            depth: Depth along ray (meters)
            
        Returns:
            3D point in camera frame (meters)
        """
        # Undistort pixel
        x = (pixel[0] - self.K[0,2]) / self.K[0,0]
        y = (pixel[1] - self.K[1,2]) / self.K[1,1]
        
        # Inverse distortion (approximate)
        # For simplicity, skip distortion; in reality, solve polynomial
        x_corr = x
        y_corr = y
        
        # Direction vector
        dir_camera = np.array([x_corr, y_corr, 1.0])
        dir_camera = dir_camera / np.linalg.norm(dir_camera)
        
        return dir_camera * depth
    
    def simulate_image(self,
                      scene_points: np.ndarray,  # Nx3 in camera frame (meters)
                      beacon_ids: List[str],
                      exposure_time: float,
                      background_flux: float = 100.0) -> np.ndarray:
        """
        Simulate raw image with noise.
        
        Returns:
            Simulated image (2D array of electron counts)
        """
        # Initialize dark image
        dark = np.zeros(self.resolution[::-1], dtype=np.float32)  # H,W
        
        # Dark current
        dark += self.dark_current * exposure_time
        
        # Add scene points
        for point, bid in zip(scene_points, beacon_ids):
            try:
                pixel = self.project_point(point)
            except ValueError:
                continue  # behind camera
                
            u, v = int(pixel[0]), int(pixel[1])
            if 0 <= u < self.resolution[0] and 0 <= v < self.resolution[1]:
                # Simple PSF: Gaussian
                flux = 1000.0  # e- per point, scaled by distance/apparent magnitude
                # Add to image (convolution would be more accurate)
                dark[v, u] += flux
                
        # Apply PRNU
        prnu_map = 1.0 + self.prnu * (2*np.random.rand(*dark.shape)-1)
        signal = dark * prnu_map
        
        # Add Poisson shot noise (photon noise)
        signal_poisson = np.random.poisson(signal)
        
        # Add read noise
        read_noise = np.random.normal(0, self.read_noise, dark.shape)
        final = signal_poisson + read_noise + background_flux
        
        return np.clip(final, 0, 65535).astype(np.uint16)  # 16-bit
    
    def detect_features(self, 
                        image: np.ndarray,
                        threshold: int = 100) -> List[np.ndarray]:
        """
        Detect bright features (beacon images) in image.
        
        Returns:
            List of (x, y) pixel coordinates
        """
        # Simple blob detection (could use cv2.SimpleBlobDetector)
        # Threshold
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                centers.append(np.array([cx, cy]))
        return centers
    
    def estimate_uncertainty(self, 
                           pixel_coord: np.ndarray,
                           signal_strength: float) -> float:
        """
        Estimate centroiding uncertainty (pixels) from SNR.
        
        Uses Crámer-Rao bound for Gaussian PSF.
        σ ~ (pixel_size * fwhm) / (SNR * sqrt(N))
        """
        # Simplified: σ = 1 / SNR (pixels)
        snr = signal_strength / self.read_noise
        if snr > 1:
            return 1.0 / snr
        else:
            return 1.0  # pixel