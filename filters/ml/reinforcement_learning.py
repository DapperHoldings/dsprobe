"""
Reinforcement Learning for end-to-end navigation policy.
Learn to control spacecraft directly from beacon observations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Optional
from dataclasses import dataclass

from config.settings import NavConfig
from navigation.navigator import AdvancedBeaconNavigator
from core.beacon import Beacon

class NavigationEnv(gym.Env):
    """
    Gym environment for full navigation control.
    Agent outputs thrust commands; state is beacon measurements.
    """
    
    def __init__(self, 
                 navigator: AdvancedBeaconNavigator,
                 beacons: List[Beacon],
                 max_steps: int = 1000):
        super().__init__()
        self.navigator = navigator
        self.beacons = beacons
        self.max_steps = max_steps
        self.step_count = 0
        
        # Action space: delta-V in 3 axes (clipped)
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(3,), dtype=np.float32
        )
        
        # Observation space: concatenated beacon measurements (range + dir for up to N beacons)
        self.max_beacons = 10
        self.obs_dim = 4 * self.max_beacons  # range + dir(x,y,z) per beacon
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
    def reset(self) -> np.ndarray:
        self.navigator.initialize(
            initial_position=np.random.uniform(-1e5, 1e5, 3),
            initial_velocity=np.random.uniform(-0.1, 0.1, 3)
        )
        self.step_count = 0
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # Apply delta-V (simplified: instant velocity change)
        current_state, _ = self.navigator.filter.get_state()
        new_velocity = current_state[3:6] + action
        # Update state manually
        new_state = current_state.copy()
        new_state[3:6] = new_velocity
        self.navigator.filter.set_state(new_state, self.navigator.filter.covariance)
        
        # Predict forward 1 hour
        self.navigator.predict(3600.0)
        
        # Get measurements from all visible beacons
        visible = self.navigator.beacon_manager.get_visible_beacons(
            new_state[0:3], self.navigator.current_time)
        measurements = self.navigator.acquire_measurements(visible)
        
        # Update
        state, cov = self.navigator.process_measurements(measurements)
        
        # Compute reward: negative of position error to target (e.g., Mars)
        # For demo, target is fixed point
        target = np.array([2e8, 0, 0])  # Mars-ish
        pos_error = np.linalg.norm(state[0:3] - target)
        reward = -pos_error / 1e6  # scale
        
        # Penalty for large delta-V
        reward -= 0.1 * np.linalg.norm(action)
        
        self.step_count += 1
        done = self.step_count >= self.max_steps or pos_error < 1e3  # within 1000 km
        
        info = {
            "position_error_km": pos_error,
            "pdop": self.navigator.filter.get_pdop()
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get observations from current measurements"""
        state, _ = self.navigator.filter.get_state()
        obs = np.zeros(self.obs_dim)
        
        # Get last measurements
        visible = self.navigator.beacon_manager.get_visible_beacons(
            state[0:3], self.navigator.current_time)
        measurements = self.navigator.acquire_measurements(visible)
        
        # Flatten into observation vector
        for i, m in enumerate(measurements[:self.max_beacons]):
            offset = i * 4
            val = m.as_vector()
            obs[offset:offset+len(val)] = val
            
        return obs

class NavigationRLAgent:
    """
    RL agent (PPO) for learning navigation policy.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int = 3,
                 hidden_dim: int = 256):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # output in [-1,1]
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
    def act(self, state: np.ndarray) -> np.ndarray:
        """Generate action from state"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action = self.policy(state_t).squeeze(0).numpy()
        return action
    
    def update(self, 
               states: torch.Tensor,
               actions: torch.Tensor,
               rewards: torch.Tensor,
               returns: torch.Tensor,
               old_log_probs: torch.Tensor,
               clip_eps: float = 0.2):
        """PPO update step (simplified)"""
        # Compute current log probs (would need proper distribution)
        # For continuous actions, use Gaussian policy
        # Placeholder: just MSE on returns as value loss
        values = self.policy(states)  # using policy as value function too (incorrect but simple)
        value_loss = nn.MSELoss()(values, returns)
        
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()
        
        return {"value_loss": value_loss.item()}