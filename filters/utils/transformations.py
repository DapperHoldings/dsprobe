"""
Coordinate frame transformations.
"""

import numpy as np
from datetime import datetime, timezone
from typing import Tuple

def datetime_to_j2000(dt: datetime) -> float:
    """Convert datetime to seconds since J2000.0 (TDB)"""
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return (dt - j2000).total_seconds()

def j2000_to_datetime(t: float) -> datetime:
    """Convert seconds since J2000 to datetime (UTC)"""
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return j2000 + timedelta(seconds=t)

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Euler angles (3-2-1 sequence) to quaternion [w, x, y, z].
    Angles in radians.
    """
    cr = np.cos(roll/2)
    sr = np.sin(roll/2)
    cp = np.cos(pitch/2)
    sp = np.sin(pitch/2)
    cy = np.cos(yaw/2)
    sy = np.sin(yaw/2)
    
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])

def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """Quaternion [w,x,y,z] to Euler angles (roll, pitch, yaw)"""
    w, x, y, z = q
    # roll (x-axis rotation)
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2*(w*y - z*x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi/2, sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def rotation_matrix(axis: str, angle: float) -> np.ndarray:
    """
    Rotation matrix about principal axis.
    
    Args:
        axis: "x", "y", or "z"
        angle: Rotation angle (radians)
    """
    c = np.cos(angle)
    s = np.sin(angle)
    if axis.lower() == "x":
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    elif axis.lower() == "y":
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    elif axis.lower() == "z":
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")