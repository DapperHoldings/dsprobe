"""
Intelligent beacon selection using various strategies.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from config.settings import NavConfig
from core.beacon import Beacon
from filters.ekf import EKF
from utils.geometry import compute_pdop

@dataclass
class BeaconScore:
    """Score for a candidate beacon"""
    beacon: Beacon
    geometry_score: float  # based on PDOP contribution
    reliability_score: float  # health, success rate
    signal_score: float  # expected SNR
    accessibility_score: float  # visibility, eclipse, etc.
    total_score: float  # weighted sum
    metadata: Dict[str, any] = None

class BeaconSelector:
    """
    Select optimal subset of beacons to track.
    Strategies:
        1. Greedy PDOP minimization
        2. Adaptive (adds reliability weighting)
        3. Reinforcement Learning (RL)
        4. Information-theoretic (mutual information)
        5. Random (baseline)
    """
    
    def __init__(self, config: NavConfig):
        self.config = config
        self.selection_method = config.beacon_selection_method
        
        # Cache for PDOP computations
        self.pdop_cache = {}
        
    def select_beacons(self,
                      candidate_beacons: List[Beacon],
                      navigator: EKF,
                      current_time: float,
                      n_select: int = None) -> List[Beacon]:
        """
        Main selection interface.
        
        Args:
            candidate_beacons: List of available beacons
            navigator: Current navigator instance (to get state)
            current_time: Current epoch
            n_select: Number to select (default from config)
            
        Returns:
            Selected beacons
        """
        if n_select is None:
            n_select = self.config.max_beacons_tracked
            
        if self.selection_method == "greedy":
            return self.greedy_pdop_selection(candidate_beacons, navigator, current_time, n_select)
        elif self.selection_method == "adaptive":
            return self.adaptive_selection(candidate_beacons, navigator, current_time, n_select)
        elif self.selection_method == "rl":
            return self.rl_selection(candidate_beacons, navigator, current_time, n_select)
        elif self.selection_method == "random":
            return np.random.choice(candidate_beacons, min(n_select, len(candidate_beacons)), replace=False).tolist()
        elif self.selection_method == "information":
            return self.information_gain_selection(candidate_beacons, navigator, current_time, n_select)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def greedy_pdop_selection(self,
                             candidates: List[Beacon],
                             navigator: EKF,
                             current_time: float,
                             n_select: int) -> List[Beacon]:
        """
        Greedy selection: add beacon that reduces PDOP the most.
        PDOP = sqrt(trace(position covariance))
        """
        selected = []
        remaining = candidates.copy()
        
        state, _ = navigator.get_state()
        current_pdop = compute_pdop(state[0:3])  # not using beacons? Maybe better: compute H from candidate set
        
        for _ in range(min(n_select, len(remaining))):
            best_beacon = None
            best_pdop_reduction = 0.0
            
            for beacon in remaining:
                # Temporarily add this beacon
                test_selection = selected + [beacon]
                # Compute PDOP with this set
                pdop = self._compute_selection_pdop(test_selection, state)
                reduction = current_pdop - pdop
                
                if reduction > best_pdop_reduction:
                    best_pdop_reduction = reduction
                    best_beacon = beacon
                    
            if best_beacon is not None:
                selected.append(best_beacon)
                remaining.remove(best_beacon)
                current_pdop -= best_pdop_reduction
            else:
                break
                
        return selected
    
    def adaptive_selection(self,
                          candidates: List[Beacon],
                          navigator: EKF,
                          current_time: float,
                          n_select: int) -> List[Beacon]:
        """
        Adaptive: weight PDOP by beacon reliability and health.
        """
        # First get greedy geometry set (maybe larger)
        geometry_candidates = self.greedy_pdop_selection(candidates, navigator, current_time, 
                                                        min(len(candidates), n_select*2))
        
        # Score each candidate
        scores = []
        for beacon in geometry_candidates:
            score = self._compute_beacon_score(beacon, navigator, current_time)
            scores.append(BeaconScore(beacon=beacon, **score))
            
        # Sort by total score descending
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return [s.beacon for s in scores[:n_select]]
    
    def information_gain_selection(self,
                                  candidates: List[Beacon],
                                  navigator: EKF,
                                  current_time: float,
                                  n_select: int) -> List[Beacon]:
        """
        Select beacons that maximize expected information gain (reduce uncertainty most).
        Uses Fisher information: I = H^T R^{-1} H
        """
        state, cov = navigator.get_state()
        selected = []
        remaining = candidates.copy()
        
        for _ in range(min(n_select, len(remaining))):
            best_beacon = None
            best_info_gain = -np.inf
            
            for beacon in remaining:
                # Information matrix contribution from this beacon (assuming both range & direction)
                # H^T R^{-1} H
                R_inv = 1.0 / np.diag(beacon.base_uncertainty[0]**2)  # simplified
                # Jacobian for range+direction
                p = state[0:3]
                b_pos = beacon.get_position(current_time)
                r_vec = p - b_pos
                r = np.linalg.norm(r_vec)
                r_hat = r_vec / r
                
                # H for both: 4x6, but we can compute trace(P_prior - P_posterior) approx
                # Information gain ~ log det(I + H^T R^{-1} H)
                # Approx: gain = trace(P_prior) - trace(P_posterior)
                # For single measurement, P_post_inv = P_prior_inv + H^T R^{-1} H
                # Hard to compute closed form without matrix inversion
                
                # Simplified: use PDOP reduction like greedy but with R weighting
                pdop_reduction = self._quick_pdop_estimate(beacon, state)
                # Weight by reliability
                weight = beacon.reliability * beacon.health
                info_gain = pdop_reduction * weight
                
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_beacon = beacon
                    
            if best_beacon:
                selected.append(best_beacon)
                remaining.remove(best_beacon)
                
        return selected
    
    def rl_selection(self,
                    candidates: List[Beacon],
                    navigator: EKF,
                    current_time: float,
                    n_select: int) -> List[Beacon]:
        """
        Reinforcement learning based selection.
        Uses a pre-trained policy network (not implemented here).
        Placeholder.
        """
        # Would load trained model from ml.beacon_selection_ml
        # For now, fallback to adaptive
        return self.adaptive_selection(candidates, navigator, current_time, n_select)
    
    def _compute_selection_pdop(self,
                               selected_beacons: List[Beacon],
                               state: np.ndarray) -> float:
        """
        Compute PDOP that would result from observing these beacons.
        Uses Fisher information matrix approximation.
        """
        # Build H matrix from all selected beacons
        H = []
        for beacon in selected_beacons:
            b_pos = beacon.get_position(0.0)  # need epoch
            r_vec = state[0:3] - b_pos
            r = np.linalg.norm(r_vec)
            if r < 1e-6:
                continue
            r_hat = r_vec / r
            # Assume both range and direction
            h_range = np.zeros(3)
            h_range = r_hat
            h_dir = (np.eye(3) - np.outer(r_hat, r_hat)) / r
            H.append(h_range)
            H.extend(h_dir)
            
        if len(H) < 3:
            return np.inf
            
        H = np.array(H)
        # Fisher information: I = H^T R^{-1} H, but assume equal weights
        I = H.T @ H
        try:
            cov = np.linalg.inv(I)
            pdop = np.sqrt(np.trace(cov[0:3,0:3]))
            return pdop
        except np.linalg.LinAlgError:
            return np.inf
    
    def _quick_pdop_estimate(self, beacon: Beacon, state: np.ndarray) -> float:
        """Quick PDOP estimate for a single beacon (heuristic)"""
        p = state[0:3]
        b_pos = beacon.get_position(0.0)
        r = np.linalg.norm(p - b_pos)
        # Simple: closer is better, orthogonal direction is better
        # Compute angle between position vector and beacon direction
        if r < 1e-6:
            return 0.0
        # DOP roughly ~ 1/sin(elevation) for single beacon? Not exactly.
        # Use geometric dilution factor
        # For a single beacon, can't determine 3D position; PDOP is infinite.
        # But in combination with others, this beacon's contribution depends on geometry.
        # Simplify: return 1/r as proxy (closer beacons give stronger range measurement)
        return 1.0 / max(r, 1e3)  # avoid div0
    
    def _compute_beacon_score(self, beacon: Beacon, 
                            navigator: EKF,
                            current_time: float) -> Dict[str, float]:
        """Compute multi-factor score for a beacon"""
        state, _ = navigator.get_state()
        
        # 1. Geometry score (inverse PDOP contribution)
        # Faster approximation: beam angle diversity
        # Compute angle between this beacon and already selected (if any)
        # For now, use reliability-weighted distance
        r = np.linalg.norm(state[0:3] - beacon.get_position(current_time))
        geometry_score = 1.0 / max(r, 1e3) if r > 0 else 0.0
        
        # 2. Reliability & health
        reliability_score = beacon.reliability * beacon.health
        
        # 3. Signal strength (SNR)
        # Use inverse of range uncertainty estimate
        _, dir_std = beacon.get_uncertainty(r, state[0:3])
        signal_score = 1.0 / max(dir_std, 1e-6)
        
        # 4. Accessibility (visibility, eclipse)
        accessibility_score = 1.0  # placeholder; would need to check Sun/Moon obstruction
        
        # Weighted sum
        weights = {
            "geometry": 1.0,
            "reliability": 2.0,  # prioritize reliable beacons
            "signal": 0.5,
            "accessibility": 1.0
        }
        total = (weights["geometry"] * geometry_score +
                 weights["reliability"] * reliability_score +
                 weights["signal"] * signal_score +
                 weights["accessibility"] * accessibility_score)
        
        return {
            "geometry_score": geometry_score,
            "reliability_score": reliability_score,
            "signal_score": signal_score,
            "accessibility_score": accessibility_score,
            "total_score": total
        }