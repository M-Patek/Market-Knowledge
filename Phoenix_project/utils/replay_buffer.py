# utils/replay_buffer.py
import torch
import numpy as np
from typing import Dict, Tuple

class ReplayBuffer:
    """A simple replay buffer for off-policy multi-agent reinforcement learning."""
    def __init__(self, capacity: int, obs_shape: int, action_shapes: Dict[str, int], device: str):
        self.capacity = capacity
        self.device = device
        
        self.global_states = torch.zeros((capacity, obs_shape), dtype=torch.float32)
        self.next_global_states = torch.zeros((capacity, obs_shape), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
        
        self.actions = {
            agent_id: torch.zeros((capacity, shape), dtype=torch.float32)
            for agent_id, shape in action_shapes.items()
        }
        
        self.ptr = 0
        self.size = 0

    def add(self, state, actions, reward, next_state, done):
        self.global_states[self.ptr] = torch.tensor(state, dtype=torch.float32)
        for agent_id, action in actions.items():
            self.actions[agent_id][self.ptr] = torch.tensor(action, dtype=torch.float32)
        self.rewards[self.ptr] = torch.tensor(reward, dtype=torch.float32)
        self.next_global_states[self.ptr] = torch.tensor(next_state, dtype=torch.float32)
        self.dones[self.ptr] = torch.tensor(done, dtype=torch.float32)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.randint(0, self.size, size=batch_size)
        
        sampled_actions = {
            agent_id: actions[indices].to(self.device)
            for agent_id, actions in self.actions.items()
        }
        
        return (
            self.global_states[indices].to(self.device),
            sampled_actions,
            self.rewards[indices].to(self.device),
            self.next_global_states[indices].to(self.device),
            self.dones[indices].to(self.device)
        )
    
    def __len__(self):
        return self.size
