import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List

from ..monitor.logging import get_logger
from .trading_env import TradingEnv
from .agents.alpha_agent import AlphaAgent
from .agents.risk_agent import RiskAgent
from .agents.base_agent import BaseAgent
from ..utils.replay_buffer import ReplayBuffer
from .drl_model_registry import DRLModelRegistry

logger = get_logger(__name__)

class MultiAgentTrainer:
    """
    Manages the training loop for a multi-agent DRL setup
    (e.g., Centralized Training, Decentralized Execution - CTDE).
    
    This example assumes a simple PPO (Proximal Policy Optimization)
    or A2C (Advantage Actor-Critic) style update.
    """

    def __init__(self, config: Dict[str, Any], env: TradingEnv, agents: List[BaseAgent], model_registry: DRLModelRegistry):
        """
        Initializes the MultiAgentTrainer.
        
        Args:
            config: Configuration dictionary.
            env (TradingEnv): The main trading environment.
            agents (List[BaseAgent]): List of agents (Alpha, Risk) to be trained.
            model_registry (DRLModelRegistry): For saving trained models.
        """
        self.config = config.get('drl_trainer', {})
        self.env = env
        self.agents = {agent.agent_id: agent for agent in agents}
        self.model_registry = model_registry
        
        # Training parameters
        self.num_episodes = self.config.get('num_episodes', 1000)
        self.max_steps_per_episode = self.config.get('max_steps_per_episode', 1000)
        self.gamma = self.config.get('gamma', 0.99) # Discount factor
        self.learning_rate = self.config.get('learning_rate', 3e-4)
        
        # Experience Replay (for off-policy) or Trajectory Buffer (for on-policy)
        self.replay_buffer = ReplayBuffer(self.config.get('buffer_size', 100000))
        self.batch_size = self.config.get('batch_size', 256)
        
        # Optimizers (one for each agent's network)
        self.optimizers = {
            agent_id: optim.Adam(agent.network.parameters(), lr=self.learning_rate)
            for agent_id, agent in self.agents.items()
        }
        
        logger.info(f"MultiAgentTrainer initialized for agents: {list(self.agents.keys())}")

    def train(self):
        """
        Runs the main training loop.
        """
        logger.info(f"Starting DRL training for {self.num_episodes} episodes...")
        
        episode_rewards = []
        
        for episode in range(self.num_episodes):
            
            # 1. Collect Trajectories
            # Use a 'global' observation from the environment
            global_obs = self.env.reset()
            episode_reward = 0
            
            # Store trajectory (for on-policy updates like PPO)
            trajectory = [] 

            for step in range(self.max_steps_per_episode):
                
                # --- Decentralized Execution ---
                # Each agent observes its part of the state and computes an action
                actions = {}
                agent_states = {}
                
                for agent_id, agent in self.agents.items():
                    # Get the agent-specific observation from the global state
                    agent_obs = self._get_agent_observation(global_obs, agent_id)
                    agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
                    
                    # Compute action
                    action_tensor = agent.compute_action(agent_obs_tensor)
                    actions[agent_id] = action_tensor.squeeze(0).cpu().numpy()
                    
                    # Store agent state (e.g., log_prob for PPO)
                    agent_states[agent_id] = {"log_prob": agent.last_log_prob}

                # --- Environment Step ---
                # Environment takes the *combined* actions
                next_global_obs, rewards, done, info = self.env.step(actions)
                
                # Store experience
                experience = (global_obs, actions, agent_states, rewards, next_global_obs, done)
                trajectory.append(experience)
                
                # Update for next step
                global_obs = next_global_obs
                episode_reward += sum(rewards.values())
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)

            # 2. Perform Training Update (Centralized Training)
            # (This example uses a simple on-policy update at the end of the episode)
            self._update_agents(trajectory)
            
            # 3. Logging
            if episode % self.config.get('log_interval', 10) == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(f"Episode {episode}/{self.num_episodes} | "
                            f"Total Reward: {episode_reward:.2f} | "
                            f"Avg Reward (100-ep): {avg_reward:.2f}")

            # 4. Save Model Checkpoint
            if episode % self.config.get('save_interval', 100) == 0:
                self.save_models(f"episode_{episode}")

        logger.info("DRL Training complete.")
        self.save_models("final")

    def _get_agent_observation(self, global_obs: Dict[str, Any], agent_id: str) -> np.ndarray:
        """
        Extracts the agent-specific view from the global observation.
        """
        # This is a critical part of the MARL design.
        # Example:
        if agent_id == 'alpha_agent':
            # Alpha agent sees market features
            return global_obs['market_features']
        elif agent_id == 'risk_agent':
            # Risk agent sees portfolio and market volatility features
            return global_obs['risk_features']
        
        raise ValueError(f"Unknown agent_id for observation: {agent_id}")

    def _update_agents(self, trajectory: List[tuple]):
        """
        Performs the "Centralized Training" update (e.g., PPO or A2C).
        This example uses a simple REINFORCE-style update (for simplicity).
        
        A real A2C/PPO update is more complex (calculating advantages,
        value loss, policy loss, entropy).
        """
        
        # Calculate discounted returns (G_t)
        # We do this separately for each agent
        
        for agent_id, agent in self.agents.items():
            
            policy_loss = []
            discounted_return = 0
            
            # Iterate trajectory backwards
            for (global_obs, actions, agent_states, rewards, next_global_obs, done) in reversed(trajectory):
                
                # Get the reward specific to this agent
                reward = rewards.get(agent_id, 0)
                
                # G_t = r_t + gamma * G_{t+1}
                discounted_return = reward + self.gamma * discounted_return
                
                # Get the log_prob stored during action computation
                log_prob = agent_states[agent_id].get('log_prob')
                
                if log_prob is not None:
                    # Policy loss = - (G_t * log_prob)
                    # We want to maximize expected return (G_t)
                    # So we minimize the negative
                    policy_loss.append(-discounted_return * log_prob)

            # --- Perform Gradient Update ---
            if policy_loss:
                # Sum loss for the episode
                optimizer = self.optimizers[agent_id]
                optimizer.zero_grad()
                
                loss = torch.cat(policy_loss).sum()
                
                loss.backward()
                optimizer.step()
                
                logger.debug(f"Agent {agent_id} updated. Loss: {loss.item():.4f}")

    def save_models(self, tag: str):
        """Saves the network state for all agents."""
        logger.info(f"Saving models with tag: {tag}")
        for agent_id, agent in self.agents.items():
            try:
                self.model_registry.save_model(
                    agent=agent,
                    tag=tag
                )
            except Exception as e:
                logger.error(f"Failed to save model for {agent_id}: {e}", exc_info=True)
