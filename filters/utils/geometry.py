"""
Geometric utility functions for navigation.
"""

import numpy as np
from typing import Tuple

def compute_pdop(position_covariance: np.ndarray) -> float:
    """
    Compute Position Dilution of Precision (PDOP).
    
    Args:
        position_covariance: 3x3 position covariance matrix (km^2)
        
    Returns:
        PDOP = sqrt(trace(position_covariance))
    """
    return np.sqrt(np.trace(position_covariance))

def compute_hdop(position_covariance: np.ndarray) -> float:
    """Horizontal DOP (assuming Z is up)"""
    cov_h = position_covariance[0:2, 0:2]
    return np.sqrt(np.trace(cov_h))

def compute_vdop(position_covariance: np.ndarray) -> float:
    """Vertical DOP"""
    return np.sqrt(position_covariance[2, 2])

def dops_from_covariance(full_cov: np.ndarray) -> Dict[str, float]:
    """Compute all DOPs from full covariance"""
    pos_cov = full_cov[0:3, 0:3]
    return {
        "pdop": compute_pdop(pos_cov),
        "hdop": compute_hdop(pos_cov),
        "vdop": compute_vdop(pos_cov)
    }

def angles_from_direction(direction: np.ndarray) -> Tuple[float, float]:
    """
    Convert unit direction vector to azimuth and elevation.
    
    Args:
        direction: 3D unit vector in NED or ECEF? Assume ECI with Z = ecliptic north
        
    Returns:
        (azimuth, elevation) in radians
    """
    # Assuming direction in (x=vernal equinox, y=90 deg east, z=north ecliptic)
    az = np.arctan2(direction[1], direction[0])
    el = np.arcsin(direction[2] / np.linalg.norm(direction))
    return az, el

def check_visibility(observer_pos: np.ndarray,
                    target_pos: np.ndarray,
                    body_positions: Dict[str, np.ndarray],
                    body_radii: Dict[str, float]) -> bool:
    """
    Check if target is visible from observer (not behind body).
    
    Simple ray-trace: does line-of-sight intersect any body?
    """
    # Vector from observer to target
    d = target_pos - observer_pos
    distance = np.linalg.norm(d)
    if distance < 1e-6:
        return False  # same point
        
    dir_vec = d / distance
    
    for body_name, body_pos in body_positions.items():
        radius = body_radii[body_name]
        # Vector from observer to body center
        oc = body_pos - observer_pos
        # Project oc onto dir_vec to find closest point on ray
        proj = np.dot(oc, dir_vec)
        if proj < 0:
            continue  # body behind observer
        # Closest point on ray to body center
        closest = observer_pos + dir_vec * proj
        dist_to_center = np.linalg.norm(closest - body_pos)
        if dist_to_center < radius:
            # Ray goes through body -> occulted
            return False
            
    return True