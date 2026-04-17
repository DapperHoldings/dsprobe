"""
SPICE (Spacecraft Planet Instrument C-Matrix Events) integration.
Uses NASA's NAIF SPICE toolkit via spiceypy.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import warnings

# Try to import spiceypy
try:
    import spiceypy as spice
    SPICE_AVAILABLE = True
except ImportError:
    SPICE_AVAILABLE = False
    warnings.warn(
        "spiceypy not installed. SPICE features disabled. "
        "Install with: pip install spiceypy"
    )

@dataclass
class SPICEKernel:
    """Represents a loaded SPICE kernel"""
    path: str
    kernel_type: str  # 'spk', 'ck', 'fk', 'ti', etc.
    body_ids: List[int]  # NAIF IDs covered
    load_time: datetime

class SPICEIntegration:
    """
    Interface to NASA's SPICE toolkit for high-accuracy ephemerides and frames.
    
    Features:
    - Load and manage SPICE kernels (.bsp, .tf, .tf, etc.)
    - Get precise positions/velocities of planets, spacecraft, etc.
    - Frame transformations (ICRF, J2000, body-fixed frames)
    - Light-time correction
    - Coordinate conversions
    
    Requires:
        pip install spiceypy
        Download SPICE kernels from https://naif.jpl.nasa.gov/naif/
    """
    
    # Common NAIF IDs
    NAIF_IDS = {
        'SSB': 0,      # Solar System Barycenter
        'SUN': 10,
        'MERCURY': 199,
        'VENUS': 299,
        'EARTH': 399,
        'MOON': 301,
        'MARS': 499,
        'JUPITER': 599,
        'SATURN': 699,
        'URANUS': 799,
        'NEPTUNE': 899,
        'PLUTO': 999,
    }
    
    def __init__(self, 
                 kernel_paths: Optional[List[str]] = None,
                 auto_load_standard: bool = False):
        """
        Initialize SPICE integration.
        
        Args:
            kernel_paths: List of paths to SPICE kernel files (.bsp, .tpc, .fk, etc.)
            auto_load_standard: If True, attempt to load standard kernels from
                                standard locations (e.g., $SPICE_PATH)
        """
        if not SPICE_AVAILABLE:
            raise ImportError(
                "spiceypy is required for SPICE integration. "
                "Install via: pip install spiceypy"
            )
            
        self.loaded_kernels: List[SPICEKernel] = []
        self.kernel_count = 0
        
        if kernel_paths:
            self.load_kernels(kernel_paths)
        elif auto_load_standard:
            self._load_standard_kernels()
            
    def _load_standard_kernels(self):
        """Attempt to load standard kernels from environment variable"""
        import os
        spice_path = os.environ.get('SPICE_PATH')
        if spice_path:
            # Look for common kernel files
            standard_kernels = [
                'de440s.bsp',  # planetary ephemeris
                'pck00011.tpc',  # planetary constants
                'naif0012.tls',  # leap seconds
                'earth_000101_240914_240914.bsp'  # Earth high precision (example)
            ]
            paths = [os.path.join(spice_path, k) for k in standard_kernels]
            existing = [p for p in paths if os.path.exists(p)]
            if existing:
                self.load_kernels(existing)
                print(f"Loaded {len(existing)} standard SPICE kernels")
                
    def load_kernels(self, paths: List[str]):
        """
        Load SPICE kernel files.
        
        Args:
            paths: List of kernel file paths
        """
        for path in paths:
            try:
                spice.furnsh(path)
                # Determine kernel type from extension
                ext = path.split('.')[-1].lower()
                kernel_type = {
                    'bsp': 'spk',  # ephemeris
                    'bpc': 'spk',
                    'tpc': 'pck',  # planetary constants
                    'fk': 'fk',    # frame kernel
                    'tf': 'tf',    # text kernel
                    'ls': 'lsk',   # leapseconds
                    'tsc': 'sclk', # spacecraft clock
                }.get(ext, 'unknown')
                
                # Try to get body IDs covered - would need to parse header
                # For simplicity, just record the path
                kernel = SPICEKernel(
                    path=path,
                    kernel_type=kernel_type,
                    body_ids=[],  # TODO: parse from kernel
                    load_time=datetime.now(timezone.utc)
                )
                self.loaded_kernels.append(kernel)
                self.kernel_count += 1
            except Exception as e:
                warnings.warn(f"Failed to load kernel {path}: {e}")
                
    def unload_all(self):
        """Unload all kernels"""
        spice.kclear()
        self.loaded_kernels = []
        self.kernel_count = 0
        
    def get_state(self, 
                  target: str,
                  observer: str = 'SSB',
                  epoch: Optional[datetime] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get position and velocity of target relative to observer.
        
        Args:
            target: Target NAIF ID or name (e.g., 'Mars', '499', 'MARS')
            observer: Observer NAIF ID or name
            epoch: Time of state (UTC). If None, uses current time.
            
        Returns:
            position_km: 3D position vector in km
            velocity_km_s: 3D velocity vector in km/s
            light_time: Light time from observer to target (seconds)
        """
        if epoch is None:
            epoch = datetime.now(timezone.utc)
            
        # Convert datetime to SPICE ephemeris time (TDB seconds past J2000)
        et = self.datetime_to_et(epoch)
        
        # Call spkezr (state of target relative to observer)
        try:
            state, lt = spice.spkezr(
                target, et, 'J2000', 'NONE', observer
            )
        except spice.SpiceError as e:
            raise RuntimeError(f"SPICE error: {e}")
            
        position = state[:3].copy()  # km
        velocity = state[3:].copy()  # km/s
        
        return position, velocity, lt
    
    def get_position(self, 
                    target: str,
                    observer: str = 'SSB',
                    epoch: Optional[datetime] = None) -> np.ndarray:
        """Get position only"""
        pos, _, _ = self.get_state(target, observer, epoch)
        return pos
    
    def get_light_time(self,
                      target: str,
                      observer: str = 'SSB',
                      epoch: Optional[datetime] = None) -> float:
        """Get one-way light time"""
        _, _, lt = self.get_state(target, observer, epoch)
        return lt
    
    def get_orientation(self,
                        target: str,
                        frame: str = 'IAU_EARTH',
                        epoch: Optional[datetime] = None) -> np.ndarray:
        """
        Get rotation matrix from target's body-fixed frame to J2000.
        
        Returns:
            3x3 rotation matrix
        """
        if epoch is None:
            epoch = datetime.now(timezone.utc)
        et = self.datetime_to_et(epoch)
        
        # Get rotation matrix
        rot = spice.pxform(frame, 'J2000', et)
        return rot.copy()
    
    def datetime_to_et(self, dt: datetime) -> float:
        """Convert datetime to SPICE ephemeris time (TDB seconds past J2000)"""
        # Convert to TDB (approximate; for high precision need to account for 
        # relativistic corrections)
        # SPICE's str2et handles UTC with leap seconds
        dt_str = dt.strftime('%Y-%m-%dT%H:%M:%S')
        et = spice.str2et(dt_str)
        return et
    
    def et_to_datetime(self, et: float) -> datetime:
        """Convert SPICE ET to datetime (UTC)"""
        dt_str = spice.et2datetime(et)  # returns string
        # Parse string to datetime
        # Format: '2000 JAN 01 12:00:00.000000'
        from datetime import datetime as dt_class
        return dt_class.strptime(dt_str, '%Y %b %d %H:%M:%S.%f')
    
    def frame_transform(self,
                       vector: np.ndarray,
                       from_frame: str,
                       to_frame: str,
                       epoch: Optional[datetime] = None) -> np.ndarray:
        """
        Transform a vector from one frame to another.
        
        Args:
            vector: 3D vector in from_frame
            from_frame: Source frame (e.g., 'IAU_EARTH', 'J2000')
            to_frame: Destination frame
            epoch: Time of transformation (required for rotating frames)
            
        Returns:
            vector in to_frame
        """
        if epoch is None:
            epoch = datetime.now(timezone.utc)
        et = self.datetime_to_et(epoch)
        
        # Get rotation matrix from from_frame to to_frame
        # pxform returns matrix that transforms vectors from from_frame to to_frame
        rot = spice.pxform(from_frame, to_frame, et)
        return rot @ vector
    
    def create_ephemeris_from_spice(self,
                                   target: str,
                                   observer: str = 'SSB') -> callable:
        """
        Create an ephemeris function that uses SPICE under the hood.
        
        Returns:
            Function f(epoch_seconds) -> position (km)
            where epoch_seconds is seconds since J2000.0 (TDB)
        """
        def ephemeris(epoch_seconds: float) -> np.ndarray:
            # Convert epoch seconds to datetime (approximate)
            # J2000 = 2000-01-01 12:00:00 TDB
            j2000_et = self.datetime_to_et(datetime(2000,1,1,12,0,0,tzinfo=timezone.utc))
            et = j2000_et + epoch_seconds
            dt = self.et_to_datetime(et)
            pos = self.get_position(target, observer, dt)
            return pos
        return ephemeris
    
    def compute_shapiro_delay(self,
                             source: str,
                             receiver: str,
                             epoch: datetime,
                             masses: Optional[List[Tuple[str, float]]] = None) -> float:
        """
        Compute Shapiro delay (general relativistic time delay) due to 
        gravitational fields along signal path.
        
        Args:
            source: Signal source (e.g., pulsar)
            receiver: Observer/spacecraft
            epoch: Time of reception (or emission?)
            masses: List of (body_name, GM) tuples; if None, use Sun only
            
        Returns:
            Shapiro delay in seconds
        """
        # Simplified: use positions from SPICE
        if masses is None:
            masses = [('SUN', 1.32712440018e11)]  # Sun GM in km^3/s^2
            
        # Get positions at emission time (approximate)
        # For accurate Shapiro, need to integrate along light path
        # Here: simple formula Δt = (2GM/c^3) * ln(4 r1 r2 / b^2)
        # This is a placeholder
        
        c = 299792.458  # km/s
        G = 6.67430e-20  # km^3 kg^-1 s^-2, but we use GM directly
        
        # Get positions
        pos_source = self.get_position(source, 'SSB', epoch)
        pos_receiver = self.get_position(receiver, 'SSB', epoch)
        
        delay = 0.0
        for body_name, GM in masses:
            pos_body = self.get_position(body_name, 'SSB', epoch)
            
            # Impact parameter (closest approach of light ray to mass)
            # Vector from body to source and receiver
            vec_s = pos_source - pos_body
            vec_r = pos_receiver - pos_body
            # Light ray approx: source -> receiver straight line
            # Closest approach distance:
            d = np.linalg.norm(np.cross(vec_s - vec_r, vec_s)) / np.linalg.norm(vec_r - vec_s)
            b = max(d, 1e3)  # avoid div0
            
            r1 = np.linalg.norm(vec_s)
            r2 = np.linalg.norm(vec_r)
            
            # Shapiro delay
            term = 2 * GM / (c**3) * np.log(4 * r1 * r2 / (b**2))
            delay += term
            
        return delay
    
    def get_kernel_info(self) -> List[Dict[str, Any]]:
        """Get information about loaded kernels"""
        info = []
        for k in self.loaded_kernels:
            info.append({
                'path': k.path,
                'type': k.kernel_type,
                'loaded': k.load_time.isoformat()
            })
        return info

# Test if SPICE is available globally
try:
    SPICE_AVAILABLE = 'spice' in globals()
except:
    SPICE_AVAILABLE = False