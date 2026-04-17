"""
Collision avoidance module for spacecraft safety.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

from core.state import State
from core.beacon import Beacon

@dataclass
class KeepOutZone:
    """Define a keep-out zone (sphere, ellipsoid, etc.)"""
    center: np.ndarray  # km in ECLIPJ2000
    radius: float  # km (sphere)
    name: str = ""
    severity: float = 1.0  # 1=critical, <1=warning
    dynamic: bool = False  # moves with central body?

@dataclass
class AvoidanceManeuver:
    """Suggested maneuver to avoid collision"""
    delta_v: np.ndarray  # km/s
    burn_duration: float  # seconds
    execute_time: datetime  # when to execute
    reason: str = ""
    confidence: float = 1.0

class CollisionAvoidance:
    """
    Monitors trajectory against keep-out zones and suggests maneuvers.
    """
    
    def __init__(self,
                 keep_out_zones: List[KeepOutZone],
                 min_safe_altitude: float = 100.0,  # km above planet surface
                 warning_time: float = 3600.0,  # 1 hour warning
                 prediction_horizon: float = 86400.0,  # 1 day
                 ):
        self.zones = keep_out_zones
        self.min_altitude = min_safe_altitude
        self.warning_time = warning_time
        self.prediction_horizon = prediction_horizon
        
        # planetary radii for altitude checks (simple dict)
        self.planet_radii = {
            "earth": 6371.0,
            "mars": 3389.5,
            "moon": 1737.4,
        }
        
    def check_keep_out_zones(self,
                            trajectory: List[Tuple[datetime, np.ndarray]],
                            current_epoch: float) -> List[Tuple[KeepOutZone, float, float]]:
        """
        Check if predicted trajectory violates any keep-out zones.
        
        Args:
            trajectory: List of (time, position) pairs
            current_epoch: Current time (seconds since J2000)
            
        Returns:
            List of (zone, time_of_closest_approach, miss_distance) for violations
        """
        violations = []
        
        for zone in self.zones:
            min_dist = np.inf
            t_closest = None
            
            for t, pos in trajectory:
                # Simple: assume trajectory is linear between points? Here we have discrete
                dist = np.linalg.norm(pos - zone.center)
                if dist < min_dist:
                    min_dist = dist
                    t_closest = t
                    
            if min_dist < zone.radius:
                # Violation
                time_until = t_closest - current_epoch
                if time_until < self.warning_time:
                    # Imminent violation
                    violations.append((zone, t_closest, min_dist))
                    
        return violations
    
    def check_planetary_altitude(self,
                               position: np.ndarray,
                               body_name: str) -> Tuple[bool, float]:
        """
        Check if altitude above planetary body is safe.
        
        Returns:
            (is_safe, altitude_km)
        """
        radius = self.planet_radii.get(body_name.lower(), None)
        if radius is None:
            return True, np.inf  # unknown body, assume safe
            
        distance = np.linalg.norm(position)
        altitude = distance - radius
        return altitude >= self.min_safe_altitude, altitude
    
    def compute_avoidance_maneuver(self,
                                 current_state: np.ndarray,
                                 violation_zone: KeepOutZone,
                                 violation_time: float,
                                 current_time: float) -> Optional[AvoidanceManeuver]:
        """
        Compute a maneuver to avoid a keep-out zone violation.
        
        Uses simple impulsive maneuver at current time to change closest approach.
        
        Returns:
            AvoidanceManeuver or None if no safe maneuver found
        """
        # Linear planetocentric approximation:
        # We want to change velocity by delta_v such that at time t_ca, 
        # the miss distance >= zone.radius + safety_margin
        
        t_go = violation_time - current_time  # time to closest approach currently
        if t_go <= 0:
            return None  # already past violation
            
        # Current relative position and velocity to zone center
        p_rel = current_state[0:3] - violation_zone.center
        v_rel = current_state[3:6]
        
        # Current closest approach distance without maneuver:
        # point of closest approach occurs at t_ca = - (p·v)/|v|^2
        # but we have absolute times; we need to compute relative to zone
        # Actually simpler: the trajectory is x(t) = p + v*t (relative)
        # Distance squared: d^2 = |p+v t|^2
        # Minimum at t* = - (p·v)/|v|^2
        v2 = np.dot(v_rel, v_rel)
        if v2 < 1e-9:
            return None  #几乎静止，无法避免通过推航天器？也许可以
            
        t_star = -np.dot(p_rel, v_rel) / v2
        
        # If t_star > t_go, then current closest approach is in past? Actually we need to align with violation_time
        # We want at t = t_go, distance = d_min. Our linear model: t_go should equal t_star if that's CA.
        # Actually violation_time is when distance is minimum (according to our prediction). So t_go is t_star.
        # So t_star = t_go.
        
        current_d = np.linalg.norm(p_rel + v_rel * t_go)
        
        if current_d >= violation_zone.radius:
            return None  # no violation? maybe already resolved
            
        # We need to change velocity such that at t_go, new distance >= safe_dist
        safe_dist = violation_zone.radius * 1.1  # 10% margin
        
        # We apply delta_v now (t=0). New velocity: v' = v + delta_v
        # New relative trajectory: p' = p + v' t
        # At t=t_go: p_final = p + (v + dv) t_go = (p + v t_go) + dv t_go = p_ca + dv t_go
        # We need |p_ca + dv t_go| >= safe_dist
        # So we need to move the closest approach point by at least Δ = safe_dist - current_d in radial direction
        # That's a minimum impulse in the direction of (p_ca - center)/|p_ca|
        p_ca = p_rel + v_rel * t_go
        dir_away = p_ca / np.linalg.norm(p_ca)  # unit vector from zone to CA point
        
        delta_d = safe_dist - current_d
        # Since p_ca + dv*t_go = shifted point
        # So we need dv = dir_away * (delta_d / t_go)
        if t_go < 1.0:
            # Too close, need large dv, maybe not feasible
            return None
            
        dv_mag = delta_d / t_go
        dv = dir_away * dv_mag
        
        # Check if this delta_v is within spacecraft capability
        max_dv = 0.1  # 100 m/s typical for small corrections
        if np.linalg.norm(dv) > max_dv:
            # Maybe do multiple burns? For now, fail
            return None
            
        return AvoidanceManeuver(
            delta_v=dv,
            burn_duration=10.0,  # placeholder, depends on thrust
            execute_time=datetime.now(timezone.utc),  # immediate
            reason=f"Avoid {violation_zone.name}",
            confidence=0.9
        )
    
    def generate_trajectory_prediction(self,
                                      current_state: np.ndarray,
                                      dt: float = 10.0,
                                      steps: int = 100) -> List[Tuple[float, np.ndarray]]:
        """
        Predict trajectory for horizon seconds.
        
        Returns:
            List of (time, position) from current_time (0) to horizon
        """
        traj = []
        state = current_state.copy()
        for i in range(steps):
            t = i * dt
            pos = state[0:3] + state[3:6] * t  # constant velocity approx
            traj.append((t, pos))
        return traj