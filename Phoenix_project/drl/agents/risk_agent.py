import torch
import torch.nn as nn
from typing import Dict, Any

from .base_agent import BaseAgent

class RiskAgent(BaseAgent):
    """
    A DRL agent responsible for managing portfolio risk.
    Its action modifies the capital allocation or exposure
    of the AlphaAgent's signal.
    
    Observation: Portfolio volatility, drawdown, AI uncertainty, VIX.
    Action: Capital modifier (e.g., 0.1 to 1.0).
    """

    def __init__(self, observation_space, action_space, network: nn.Module, config: Dict[str, Any]):
        """
        Initializes the RiskAgent.
        """
        super().__init__(
            agent_id="risk_agent",
            observation_space=observation_space,
            action_space=action_space,
            network=network,
            config=config
        )
        self.agent_type = "risk"

    def compute_action(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Computes the action (e.g., capital modifier) based on the observation.
        
        Args:
            observation (torch.Tensor): The current state observation.
            
        Returns:
            torch.Tensor: The action to take.
        """
        dist = self.network(observation)
        action = dist.sample()
        self.last_log_prob = dist.log_prob(action)
        
        # Action is a modifier (e.g., 0.1 to 1.0)
        action = torch.clamp(action, self.action_space.low[0], self.action_space.high[0])
        
        return action

    def compute_reward(self, state: Any, action: Any, next_state: Any) -> float:
        """
        Defines the reward function for the RiskAgent.
        This is often the most complex reward, as it's portfolio-level.
        
        Example: Reward = Portfolio Sharpe Ratio, Penalize = Max Drawdown
        
        Args:
            state: The state when the action was taken.
            action: The action (capital modifier).
            next_state: The resulting state.
            
        Returns:
            float: The reward.
        """
        
        # RiskAgent reward is often based on the *portfolio* performance,
        # which is a result of (Alpha_Signal * Risk_Modifier)
        
        try:
            # Get portfolio-level metrics from the next state
            portfolio_return = next_state['portfolio_return']
            portfolio_volatility = next_state['portfolio_volatility']
            
            # Simple Sharpe Ratio calculation
            # (Assuming risk-free rate is 0)
            if portfolio_volatility > 0:
                sharpe_ratio = portfolio_return / portfolio_volatility
            else:
                sharpe_ratio = 0.0
                
            reward = sharpe_ratio
            
            # Penalize for high volatility or drawdowns
            if portfolio_volatility > self.config.get('vol_target', 0.02):
                reward -= (portfolio_volatility - self.config.get('vol_target', 0.02)) * \
                          self.config.get('vol_penalty', 1.0)
            
            if next_state['drawdown'] > self.config.get('max_drawdown_target', 0.1):
                reward -= (next_state['drawdown'] - self.config.get('max_drawdown_target', 0.1)) * \
                          self.config.get('drawdown_penalty', 5.0)

            return reward

        except Exception as e:
            return 0.0
