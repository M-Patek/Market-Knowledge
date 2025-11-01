import torch
import torch.nn as nn
from typing import Dict, Any

from .base_agent import BaseAgent

class AlphaAgent(BaseAgent):
    """
    A DRL agent responsible for generating the primary alpha signal
    (e.g., predicting market direction or relative strength).
    
    This agent's "action" is typically a continuous value (e.g., -1 to 1).
    """

    def __init__(self, observation_space, action_space, network: nn.Module, config: Dict[str, Any]):
        """
        Initializes the AlphaAgent.
        
        Args:
            observation_space: The Gym observation space.
            action_space: The Gym action space.
            network (nn.Module): The policy/value network.
            config: Configuration dictionary.
        """
        super().__init__(
            agent_id="alpha_agent",
            observation_space=observation_space,
            action_space=action_space,
            network=network,
            config=config
        )
        self.agent_type = "alpha"

    def compute_action(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Computes the action (e.g., signal) based on the observation.
        
        Args:
            observation (torch.Tensor): The current state observation.
            
        Returns:
            torch.Tensor: The action to take.
        """
        # Get the policy distribution from the network
        dist = self.network(observation)
        
        # Sample from the distribution (stochastic policy)
        # For a deterministic policy, you might take dist.mean
        action = dist.sample()
        
        # Store the log probability for training
        self.last_log_prob = dist.log_prob(action)
        
        # Clamp or clip the action to be within the valid action space bounds
        # (e.g., -1 to 1)
        action = torch.clamp(action, self.action_space.low[0], self.action_space.high[0])
        
        return action

    def compute_reward(self, state: Any, action: Any, next_state: Any) -> float:
        """
        Defines the reward function for the AlphaAgent.
        This is a critical part of the DRL design.
        
        Example: Reward based on subsequent price movement (Sharpe ratio, PnL).
        
        Args:
            state: The state when the action was taken.
            action: The action (signal) taken.
            next_state: The resulting state.
            
        Returns:
            float: The reward.
        """
        
        # This reward logic is highly strategy-dependent.
        # Example: Simple reward based on PnL
        # This assumes 'state' and 'next_state' are dicts with price info
        
        try:
            price_t = state['price']
            price_t_plus_1 = next_state['price']
            
            # Calculate return
            market_return = (price_t_plus_1 - price_t) / price_t
            
            # Action is the signal (-1 to 1)
            signal = action.item() 
            
            # Reward is the PnL from holding the position
            # (This is a simplified "mark-to-market" reward)
            reward = signal * market_return
            
            # Penalize for transaction costs (e.g., changing position)
            # if 'last_signal' in state:
            #     turnover = abs(signal - state['last_signal'])
            #     reward -= turnover * self.config.get('turnover_penalty', 0.0001)
                
            return reward
            
        except Exception as e:
            # Handle cases where state doesn't have price
            return 0.0
