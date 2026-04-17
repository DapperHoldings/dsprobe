"""
Beacon manager handles dynamic beacon visibility and health.
"""

import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timezone

from core.beacon import Beacon
from core.state import State

class BeaconManager:
    """
    Manages beacon lifecycle: visibility, health degradation, addition/removal.
    """
    
    def __init__(self, 
                 beacons: Dict[str, Beacon],
                 config):
        self.beacons = beacons
        self.config = config
        
        # Statistics
        self.visibility_history: Dict[str, List[bool]] = {}
        self.health_history: Dict[str, List[float]] = {}
        
    def update_visibility(self, 
                         observer_pos: np.ndarray,
                         observer_orientation: Optional[np.ndarray] = None,
                         timestamp: Optional[datetime] = None) -> Dict[str, bool]:
        """
        Update visibility status for all beacons.
        
        Returns:
            Dict[beacon_id -> visible]
        """
        visible = {}
        epoch = AdvancedBeaconNavigator._datetime_to_epoch(timestamp) if timestamp else 0.0
        
        for bid, beacon in self.beacons.items():
            # Simple range check
            b_pos = beacon.get_position(epoch)
            r = np.linalg.norm(observer_pos - b_pos)
            
            is_visible = True
            if beacon.max_range is not None and r > beacon.max_range:
                is_visible = False
            if beacon.health < 0.3:  # failed beacon
                is_visible = False
                
            visible[bid] = is_visible
            
            # Record history
            if bid not in self.visibility_history:
                self.visibility_history[bid] = []
            self.visibility_history[bid].append(is_visible)
            
        return visible
    
    def get_visible_beacons(self,
                           observer_pos: np.ndarray,
                           timestamp: Optional[datetime] = None) -> List[Beacon]:
        """Return list of currently visible beacons"""
        visible_ids = self.update_visibility(observer_pos, None, timestamp)
        return [b for bid, b in self.beacons.items() if visible_ids.get(bid, False)]
    
    def update_health(self,
                     beacon_id: str,
                     measurement_success: bool,
                     residual: Optional[float] = None):
        """
        Update beacon health based on measurement success.
        Health decays on failures, recovers slowly on successes.
        """
        if beacon_id not in self.beacons:
            return
            
        beacon = self.beacons[beacon_id]
        
        if measurement_success:
            # Slight recovery
            beacon.health = min(1.0, beacon.health + 0.01)
        else:
            # Decay
            beacon.health = max(0.0, beacon.health - 0.05)
            
        # Record
        if beacon_id not in self.health_history:
            self.health_history[beacon_id] = []
        self.health_history[beacon_id].append(beacon.health)
        
        # Limit history length
        max_len = 1000
        if len(self.health_history[beacon_id]) > max_len:
            self.health_history[beacon_id] = self.health_history[beacon_id][-max_len:]