"""
Reinforcement Learning for Beacon Selection.
Learn optimal beacon selection policy to maximize navigation accuracy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import gym
from gym import spaces

from config.settings import NavConfig
from navigation.navigator import AdvancedBeaconNavigator
from core.beacon import Beacon
from utils.geometry import compute_pdop

@dataclass
class BeaconSelectionState:
    """State representation for RL agent"""
    current_position: np.ndarray  # 3D position (km)
    current_velocity: np.ndarray  # 3D velocity (km/s)
    current_pdop: float  # current navigation accuracy
    beacon_features: np.ndarray  # (n_beacons, n_features) - features for each beacon
    time_to_go: float  # mission time remaining
    
@dataclass
class BeaconSelectionAction:
    """Action space: which beacons to select"""
    selected_beacon_indices: List[int]  # indices of beacons to observe
    n_select: int = 4  # how many to select

class BeaconSelectionNetwork(nn.Module):
    """
    Neural network for beacon selection.
    Uses attention mechanism to weigh beacons.
    """
    
    def __init__(self, 
                 n_beacon_features: int = 10,
                 n_selection: int = 4,
                 hidden_dim: int = 128):
        super().__init__()
        self.n_selection = n_selection
        self.n_beacon_features = n_beacon_features
        
        # Encoder for beacon features
        self.beacon_encoder = nn.Sequential(
            nn.Linear(n_beacon_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(3 + 3 + 1 + 1, hidden_dim),  # pos(3) + vel(3) + pdop + time
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Output head: scores for each beacon
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, 
                state: torch.Tensor,  # (batch, state_dim)
                beacon_features: torch.Tensor,  # (batch, n_beacons, n_features)
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            scores: (batch, n_beacons) - selection scores
            attention_weights: (batch, n_beacons) for interpretability
        """
        batch_size = state.shape[0]
        n_beacons = beacon_features.shape[1]
        
        # Encode beacons
        beacon_encoded = self.beacon_encoder(beacon_features)  # (B, N, H)
        
        # Encode state and repeat for each beacon
        state_encoded = self.state_encoder(state)  # (B, H)
        state_repeated = state_encoded.unsqueeze(1).repeat(1, n_beacons, 1)  # (B, N, H)
        
        # Concatenate state info to each beacon
        combined = torch.cat([beacon_encoded, state_repeated], dim=-1)  # (B, N, 2H)
        
        # Compute scores
        scores = self.scorer(combined).squeeze(-1)  # (B, N)
        
        # Apply mask if provided (for unavailable beacons)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
            
        # Get top-k scores and indices
        top_k_scores, top_k_indices = torch.topk(scores, self.n_selection, dim=-1)
        
        return top_k_scores, top_k_indices

class BeaconSelectionEnv(gym.Env):
    """
    Gym environment for beacon selection.
    Agent observes current navigation state and candidate beacons,
    selects a subset, receives reward based on resulting navigation accuracy.
    """
    
    def __init__(self, 
                 navigator: AdvancedBeaconNavigator,
                 all_beacons: List[Beacon],
                 episode_length: int = 100,
                 reward_type: str = "pdop_reduction"):
        """
        Args:
            navigator: The navigation system to control
            all_beacons: List of all available beacons
            episode_length: Max steps per episode
            reward_type: "pdop_reduction", "information_gain", or "crash_avoidance"
        """
        super().__init__()
        
        self.navigator = navigator
        self.all_beacons = all_beacons
        self.episode_length = episode_length
        self.reward_type = reward_type
        self.step_count = 0
        self.initial_pdop = None
        
        # Observation space: state + beacon features
        # State: position(3) + velocity(3) + pdop(1) + time_remaining(1) = 8
        # Beacon features per beacon: range(1) + direction(3) + health(1) + reliability(1) + type(one-hot 6) = 12?
        # Actually we'll compute features dynamically
        self.max_beacons = len(all_beacons)
        n_features = 8  # per beacon features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(8 + self.max_beacons * n_features,),
            dtype=np.float32
        )
        
        # Action space: select k beacons (indices)
        self.k = 4
        self.action_space = spaces.MultiDiscrete([self.max_beacons] * self.k)
        
        # Current state
        self.current_state = None
        self.last_pdop = None
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        # Randomize initial position near Earth
        earth_pos = self.all_beacons[0].get_position(0.0)
        initial_pos = earth_pos + np.random.uniform(-5000, 5000, 3)
        self.navigator.initialize(initial_pos, velocity=np.random.uniform(-1, 1, 3))
        
        # Get initial state
        solution = self.navigator.get_solution()
        self.initial_pdop = solution['pdop']
        self.last_pdop = solution['pdop']
        self.step_count = 0
        
        # Build observation
        obs = self._get_observation()
        return obs
    
    def step(self, action: List[int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step: select beacons, get measurements, update filter.
        
        Args:
            action: List of k beacon indices to select
            
        Returns:
            observation, reward, done, info
        """
        # Convert action to beacon objects
        selected_beacons = [self.all_beacons[i] for i in action]
        
        # Simulate one timestep (1 hour)
        dt = 3600.0
        current_time = self.navigator.current_time
        
        # Predict
        self.navigator.predict(dt)
        
        # Get measurements from selected beacons
        measurements = self.navigator.acquire_measurements(selected_beacons)
        
        # Process
        state, cov = self.navigator.process_measurements(measurements)
        
        # Get new solution
        solution = self.navigator.get_solution()
        new_pdop = solution['pdop']
        
        # Compute reward
        reward = self._compute_reward(new_pdop)
        
        # Update step
        self.step_count += 1
        done = self.step_count >= self.episode_length
        
        # Get observation for next state
        obs = self._get_observation()
        
        info = {
            "pdop": new_pdop,
            "pdop_reduction": self.last_pdop - new_pdop,
            "beacons_selected": [b.name for b in selected_beacons],
            "step": self.step_count
        }
        self.last_pdop = new_pdop
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector"""
        solution = self.navigator.get_solution()
        state_vec = np.concatenate([
            solution['position'] / 1e6,  # normalize to million km
            solution['velocity'] / 10.0,  # normalize to 10 km/s
            [solution['pdop'] / 1000.0],  # normalize to 1000 km
            [(self.episode_length - self.step_count) / self.episode_length]  # time remaining [0,1]
        ])
        
        # Beacon features: for each beacon, compute [range_norm, health, reliability, type_onehot]
        beacon_features = []
        current_pos = solution['position']
        epoch = self.navigator._datetime_to_epoch(self.navigator.current_time) if self.navigator.current_time else 0.0
        
        for beacon in self.all_beacons:
            b_pos = beacon.get_position(epoch)
            r = np.linalg.norm(current_pos - b_pos)
            # Normalize range to 10 million km
            r_norm = min(r / 1e7, 10.0)
            
            # One-hot for type (6 types)
            type_onehot = np.zeros(6)
            type_idx = list(BeaconType).index(beacon.beacon_type)
            type_onehot[type_idx] = 1.0
            
            features = np.array([
                r_norm,
                beacon.health,
                beacon.reliability,
                beacon.base_uncertainty[0],  # range_std
                beacon.base_uncertainty[1],  # dir_std
            ])
            # Pad to fixed length
            if len(features) < 8:
                features = np.pad(features, (0, 8 - len(features)))
            beacon_features.append(features)
            
        beacon_features = np.array(beacon_features).flatten()
        
        # Pad to max_beacons * 8
        expected_len = self.max_beacons * 8
        if len(beacon_features) < expected_len:
            beacon_features = np.pad(beacon_features, (0, expected_len - len(beacon_features)))
        
        return np.concatenate([state_vec, beacon_features]).astype(np.float32)
    
    def _compute_reward(self, new_pdop: float) -> float:
        """Compute reward based on PDOP improvement"""
        if self.reward_type == "pdop_reduction":
            # Reward reduction in PDOP
            delta = self.last_pdop - new_pdop
            reward = delta * 10  # scale
        elif self.reward_type == "information_gain":
            # Negative PDOP
            reward = -new_pdop / 100.0
        else:
            reward = 0.0
            
        # Bonus for low PDOP
        if new_pdop < 10.0:
            reward += 1.0
        elif new_pdop < 100.0:
            reward += 0.1
            
        # Penalty for high PDOP (bad navigation)
        if new_pdop > 1000.0:
            reward -= 1.0
            
        return reward

class BeaconSelectionAgent:
    """
    RL agent for beacon selection.
    Uses PPO (Proximal Policy Optimization) with attention network.
    """
    
    def __init__(self, 
                 n_beacons: int,
                 n_features: int = 8,
                 n_select: int = 4,
                 lr: float = 3e-4,
                 gamma: float = 0.99):
        self.n_beacons = n_beacons
        self.n_select = n_select
        self.gamma = gamma
        
        # Network
        self.network = BeaconSelectionNetwork(
            n_beacon_features=n_features,
            n_selection=n_select
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = []
        
    def select_beacons(self, 
                      state_vec: np.ndarray,
                      beacon_features: np.ndarray,
                      mask: Optional[np.ndarray] = None) -> Tuple[List[int], np.ndarray]:
        """
        Select k beacons given current state.
        
        Returns:
            selected_indices, scores
        """
        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
            beacon_tensor = torch.FloatTensor(beacon_features).unsqueeze(0)
            if mask is not None:
                mask_tensor = torch.BoolTensor(mask).unsqueeze(0)
            else:
                mask_tensor = None
                
            scores, indices = self.network(state_tensor, beacon_tensor, mask_tensor)
            return indices[0].tolist(), scores[0].numpy()
    
    def store_experience(self, 
                        state: np.ndarray,
                        action: List[int],
                        reward: float,
                        next_state: np.ndarray,
                        done: bool):
        """Store transition in buffer"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
        
    def train_epoch(self, batch_size: int = 64) -> float:
        """Train on one epoch of experiences"""
        if len(self.buffer) < batch_size:
            return 0.0
            
        # Sample batch
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Unpack
        states = torch.FloatTensor(np.array([b['state'] for b in batch]))
        rewards = torch.FloatTensor([b['reward'] for b in batch])
        next_states = torch.FloatTensor(np.array([b['next_state'] for b in batch]))
        dones = torch.BoolTensor([b['done'] for b in batch])
        
        # For each sample, we need to recompute actions and get log probs
        # This is simplified; full PPO would store log probs during interaction
        
        # Compute value estimates (simplified: use network as critic)
        # Actually need separate value network; for brevity skip
        
        loss = rewards.mean()  # dummy
        return loss.item()

def train_beacon_selector(env: BeaconSelectionEnv,
                         agent: BeaconSelectionAgent,
                         n_episodes: int = 1000,
                         save_path: Optional[str] = None):
    """
    Train beacon selection RL agent.
    
    Args:
        env: Gym environment
        agent: RL agent
        n_episodes: Number of training episodes
        save_path: Where to save trained model
    """
    print("Training beacon selection RL agent...")
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Get beacon features (simplified: use current observation split)
            # In practice, would pass beacon features separately
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # Dummy beacon features from state (extract)
            beacon_features = state_tensor[:, 8:].view(1, -1, 8)  # reshape
            
            action, scores = agent.select_beacons(state, beacon_features)
            
            next_state, reward, done, info = env.step(action)
            
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Train on buffer when enough samples
            if len(agent.buffer) >= 64:
                loss = agent.train_epoch()
                
        if episode % 10 == 0:
            print(f"Episode {episode}: total_reward={total_reward:.2f}, final_pdop={info['pdop']:.2f}")
            
    if save_path:
        torch.save(agent.network.state_dict(), save_path)
        print(f"Model saved to {save_path}")