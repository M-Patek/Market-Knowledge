# drl/networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

class ActorNetwork(nn.Module):
    """A simple MLP for the Actor (policy)."""
    def __init__(self, obs_dim: int, action_dim: int, activation_type: Literal['tanh', 'sigmoid', 'none'] = 'tanh'):
        super(ActorNetwork, self).__init__()
        self.layer1 = nn.Linear(obs_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_dim)
        self.activation_type = activation_type

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))
        if self.activation_type == 'tanh':
            # Output in [-1, 1] for AlphaAgent
            return torch.tanh(self.layer3(x))
        elif self.activation_type == 'sigmoid':
            # Output in [0, 1] for RiskAgent
            return torch.sigmoid(self.layer3(x))
        else: # 'none'
            # Output raw logits (for Categorical distribution)
            return self.layer3(x)

class CriticNetwork(nn.Module):
    """A simple MLP for the Centralized Critic (Q-value or V-value)."""
    def __init__(self, critic_input_dim: int):
        super(CriticNetwork, self).__init__()
        self.layer1 = nn.Linear(critic_input_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, global_state: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        # This forward pass is for DDPG-style Q(s, a)
        x = torch.cat([global_state, all_actions], dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x) # Output a single Q-value

    def forward_value(self, global_state: torch.Tensor) -> torch.Tensor:
        # This forward pass is for A2C-style V(s)
        x = F.relu(self.layer1(global_state))
        x = F.relu(self.layer2(x))
        return self.layer3(x) # Output a single V-value
