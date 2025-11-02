import torch
import torch.nn as nn
from typing import Dict, Any

from .base_agent import BaseAgent

class ExecutionAgent(BaseAgent):
    """
    A DRL agent responsible for optimizing trade execution.
    Its goal is to minimize slippage and market impact given a
    target order (e.g., from the AlphaAgent or PortfolioConstructor).
    
    Observation: Market micro-structure (LOB, VWAP), remaining order size, time left.
    Action: How much to trade *now* (e.g., % of remaining order).
    """

    def __init__(self, observation_space, action_space, network: nn.Module, config: Dict[str, Any]):
        """
        Initializes the ExecutionAgent.
        """
        super().__init__(
            agent_id="execution_agent",
            observation_space=observation_space,
            action_space=action_space,
            network=network,
            config=config
        )
        self.agent_type = "execution"

    def compute_action(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Computes the action (e.g., % of order to execute) based on the observation.
        
        Args:
            observation (torch.Tensor): The current state observation.
            
        Returns:
            torch.Tensor: The action to take.
        """
        dist = self.network(observation)
        action = dist.sample()
        self.last_log_prob = dist.log_prob(action)
        
        # Action is likely a percentage (0 to 1)
        action = torch.clamp(action, 0.0, 1.0)
        
        return action

    def compute_reward(self, state: Any, action: Any, next_state: Any) -> float:
        """
        Defines the reward function for the ExecutionAgent.
        The goal is to minimize slippage (Implementation Shortfall).
        
        Reward = (Benchmark_Price - Execution_Price) * Shares_Executed
        
        Args:
            state: The state when the action was taken.
            action: The action (e.g., 0.2 -> execute 20% of remaining).
            next_state: The resulting state.
            
        Returns:
            float: The reward.
        """
        
        try:
            # Benchmark price (e.g., price when the order was received)
            benchmark_price = state['benchmark_price']
            
            # Actual execution price from this step
            execution_price = next_state['execution_price']
            
            # Number of shares executed in this step
            shares_executed = next_state['shares_executed']
            
            # Direction of the trade (1 for buy, -1 for sell)
            trade_direction = state['trade_direction'] # 1 or -1
            
            # Calculate slippage (implementation shortfall)
            # For BUY: (Benchmark - Exec_Price) -> we want Exec_Price to be low
            # For SELL: (Exec_Price - Benchmark) -> we want Exec_Price to be high
            # This can be unified:
            slippage_per_share = (benchmark_price - execution_price) * trade_direction
            
            # Reward is total slippage cost/gain
            reward = slippage_per_share * shares_executed
            
            # Penalize for not executing the full order by the end
            if next_state['is_terminal'] and next_state['shares_remaining'] > 0:
                # Heavy penalty for leftover shares
                reward -= next_state['shares_remaining'] * self.config.get('leftover_penalty_factor', 1.0)
                
            return reward

        except Exception as e:
            return 0.0
