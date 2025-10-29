# drl/multi_agent_trainer.py

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from drl.trading_env import TradingEnv # This is now the "Strategic" Env
from drl.execution_env import ExecutionEnv # This is the "Execution" Env
from drl.agents.base_agent import BaseAgent
from drl.agents.alpha_agent import AlphaAgent
from drl.agents.risk_agent import RiskAgent
from drl.agents.execution_agent import ExecutionAgent
from .networks import ActorNetwork, CriticNetwork
from utils.replay_buffer import ReplayBuffer

class MultiAgentTrainer:
    """
    Implements the Centralized Training, Decentralized Execution (CTDE) framework
    in a hierarchical manner.
    - Strategic Layer (Alpha, Risk) decides *what* to do.
    - Execution Layer (Exec) decides *how* to do it.
    """

    def __init__(self, 
                 strategic_env: TradingEnv, 
                 execution_env_factory: Any, # A class or function to create ExecutionEnvs
                 agents: Dict[str, BaseAgent], 
                 config: Dict[str, Any]):
        
        self.strategic_env = strategic_env
        self.execution_env_factory = execution_env_factory
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Strategic Layer Hyperparameters ---
        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.005)
        self.batch_size = self.config.get('batch_size', 256)
        self.exploration_noise = self.config.get('exploration_noise', 0.1)
        self.variance_scaling_factor = self.config.get('variance_scaling_factor', 10.0) # Factor to scale variance into a [0, 1] penalty
        self.signal_variance_index = self.config.get('signal_variance_index', 3) # Index of signal_variance in global_state

        # --- Initialize Strategic Layer (Alpha, Risk) ---
        self.strategic_agents = {id: agent for id, agent in agents.items() if id in ['alpha', 'risk']}
        self.strategic_actor_networks = {}
        self.strategic_actor_optimizers = {}
        self.strategic_target_actor_networks = {}
        lr_actor_strat = self.config.get('lr_actor_strat', 1e-4)

        for agent_id, agent in self.strategic_agents.items():
            obs_dim = agent.get_observation_space().shape[0]
            action_dim = agent.get_action_space().shape[0]
            activation = 'tanh' if isinstance(agent, AlphaAgent) else 'sigmoid' # Assumes RiskAgent
            
            actor = ActorNetwork(obs_dim, action_dim, activation_type=activation).to(self.device)
            self.strategic_actor_networks[agent_id] = actor
            self.strategic_actor_optimizers[agent_id] = optim.Adam(actor.parameters(), lr=lr_actor_strat)
            
            target_actor = ActorNetwork(obs_dim, action_dim, activation_type=activation).to(self.device)
            target_actor.load_state_dict(actor.state_dict())
            self.strategic_target_actor_networks[agent_id] = target_actor

        global_state_dim = self.strategic_env.observation_space.shape[0]
        total_strat_action_dim = sum(a.get_action_space().shape[0] for a in self.strategic_agents.values())
        strat_critic_input_dim = global_state_dim + total_strat_action_dim
        
        self.strategic_critic_network = CriticNetwork(strat_critic_input_dim).to(self.device)
        lr_critic_strat = self.config.get('lr_critic_strat', 1e-3)
        self.strategic_critic_optimizer = optim.Adam(self.strategic_critic_network.parameters(), lr=lr_critic_strat)
        self.strategic_target_critic_network = CriticNetwork(strat_critic_input_dim).to(self.device)
        self.strategic_target_critic_network.load_state_dict(self.strategic_critic_network.state_dict())

        strat_action_shapes = {id: agent.get_action_space().shape[0] for id, agent in self.strategic_agents.items()}
        buffer_capacity = self.config.get('buffer_capacity', 1_000_000)
        self.strategic_replay_buffer = ReplayBuffer(buffer_capacity, global_state_dim, strat_action_shapes, self.device)
        
        # --- Initialize Execution Layer ---
        self.execution_agent = agents.get('exec')
        if self.execution_agent:
            exec_obs_dim = self.execution_agent.get_observation_space().shape[0]
            exec_action_dim = self.execution_agent.get_action_space().n # It's Discrete!
            
            # We'll use a simple Actor (policy) network that outputs logits for the Discrete actions
            self.execution_actor_network = ActorNetwork(exec_obs_dim, exec_action_dim, 'none').to(self.device) # 'none' = raw logits
            self.execution_actor_optimizer = optim.Adam(self.execution_actor_network.parameters(), lr=config.get('lr_exec_actor', 1e-4))
            
            # We'll use a Critic (value) network for A2C/PPO style updates (on-policy)
            # Critic input is just state, outputs V(s)
            self.execution_critic_network = CriticNetwork(exec_obs_dim + 1).to(self.device) # Re-using Critic as V-net
            self.execution_critic_optimizer = optim.Adam(self.execution_critic_network.parameters(), lr=config.get('lr_exec_critic', 1e-3))


    def train(self, num_episodes: int):
        """Main training loop."""
        for episode in range(num_episodes):
            global_state, _ = self.strategic_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # train_step now returns the *next_state* as well
                next_global_state, reward, done = self.train_step(global_state)
                episode_reward += reward
                global_state = next_global_state
            
            print(f"Strategic Episode {episode + 1}: Total Reward = {episode_reward}")

    def train_step(self, global_state: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Executes one hierarchical step: 
        1. Strategy Phase (Alpha, Risk)
        2. Execution Phase (ExecutionAgent sub-loop)
        """
        # --- 1. STRATEGY PHASE ---
        with torch.no_grad():
            agent_observations = self._get_strategic_observations(global_state)
            strategic_actions = {}
            for agent_id, obs in agent_observations.items():
                actor = self.strategic_actor_networks[agent_id]
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action = actor(obs_tensor).cpu().numpy()
                action += np.random.normal(0, self.exploration_noise, size=action.shape)
                action = np.clip(action, self.strategic_agents[agent_id].get_action_space().low, self.strategic_agents[agent_id].get_action_space().high)
                strategic_actions[agent_id] = action

        # Fuse actions to create a parent order goal
        fused_pct = self._fuse_actions(strategic_actions, global_state)
        parent_order_goal = self._create_parent_order(fused_pct) # New helper

        # --- 2. EXECUTION PHASE ---
        # Create the high-frequency environment for this specific goal
        # This assumes the factory is a class that can be instantiated
        execution_env = self.execution_env_factory(parent_order_goal) 
        cumulative_reward, exec_info = self._run_execution_loop(execution_env)
        
        # --- 3. ADVANCE STRATEGIC ENV & STORE ---
        # We advance the strategic env by 1 step. 
        # The 'action' it takes is the fused percentage.
        next_global_state, _, done, _, _ = self.strategic_env.step(fused_pct)

        # Store the strategic experience
        self.strategic_replay_buffer.add(global_state, strategic_actions, cumulative_reward, next_global_state, done)

        # Trigger a strategic training update
        if len(self.strategic_replay_buffer) > self.batch_size:
            self._update_strategic_networks()
            
        return next_global_state, cumulative_reward, done

    def _run_execution_loop(self, exec_env: ExecutionEnv) -> Tuple[float, Dict]:
        """
        Runs the high-frequency execution sub-loop and trains the ExecutionAgent.
        (This assumes an on-policy A2C-style update for the Discrete agent)
        """
        hf_obs, _ = exec_env.reset()
        hf_done = False
        cumulative_reward = 0.0
        
        # Storage for the on-policy rollout
        rollout_obs, rollout_actions, rollout_rewards, rollout_log_probs, rollout_values = [], [], [], [], []

        while not hf_done:
            hf_obs_tensor = torch.tensor(hf_obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action_logits = self.execution_actor_network(hf_obs_tensor)
                value = self.execution_critic_network(hf_obs_tensor, torch.tensor([[0.0]], device=self.device)) # Dummy action
                action_dist = torch.distributions.Categorical(logits=action_logits)
                exec_action = action_dist.sample()
                log_prob = action_dist.log_prob(exec_action)
            
            exec_action_int = exec_action.item()
            next_hf_obs, hf_reward, hf_done, _, info = exec_env.step(exec_action_int)
            
            # Store on-policy transition
            rollout_obs.append(hf_obs_tensor)
            rollout_actions.append(exec_action)
            rollout_rewards.append(hf_reward)
            rollout_log_probs.append(log_prob)
            rollout_values.append(value)
            
            cumulative_reward += hf_reward
            hf_obs = next_hf_obs
        
        # Now, perform the on-policy update for the ExecutionAgent
        self._update_execution_networks(rollout_obs, rollout_values, rollout_rewards, rollout_log_probs)
        
        return cumulative_reward, {} # Return cumulative reward for the strategic layer

    def _create_parent_order(self, fused_pct: np.ndarray) -> Dict:
        """Creates the parent order goal from the strategic fused action."""
        # This is a placeholder. Logic depends on portfolio value, etc.
        # Get current portfolio value from the strategic env
        portfolio_value = self.strategic_env.portfolio_value
        target_value = fused_pct[0] * portfolio_value
        
        # Get current position value
        ticker = self.strategic_env.trading_ticker
        current_price = self.strategic_env._get_observation(self.strategic_env.data_iterator.next())[0] # HACK: get current price
        current_size = self.strategic_env.positions.get(ticker, {}).get('size', 0.0)
        current_value = current_size * current_price
        
        value_delta = target_value - current_value
        
        return {
            'side': 'BUY' if value_delta > 0 else 'SELL',
            'size': abs(value_delta) / current_price if current_price > 0 else 0,
            'duration_steps': 60 # e.g., execute over 60 seconds
        }

    def _get_strategic_observations(self, global_state: np.ndarray) -> Dict[str, np.ndarray]:
        """Extracts agent-specific observations from the global state. Placeholder."""
        # This needs a concrete implementation based on the feature order in TradingEnv
        # For now, we assume a simple slicing for demonstration.
        return {
            'alpha': global_state[[2, 0, 1]], # e.g., Signal Mean, Price, Volume
            'risk': global_state[[3, -2, -1]]   # e.g., Signal Variance, CVaR (needs to be added), Risk Budget
        }

    def _fuse_actions(self, actions: Dict[str, np.ndarray], global_state: np.ndarray) -> np.ndarray:
        """
        (Task 2.2) Combines actions using an Uncertainty-Weighted Fusion Mechanism.
        """
        alpha_action = actions.get('alpha', np.array([0.0]))
        risk_action = actions.get('risk', np.array([1.0]))

        # 1. Get cognitive uncertainty (Signal Variance) from the global state
        signal_variance = global_state[self.signal_variance_index]
        
        # 2. Calculate uncertainty penalty and alpha trust weight
        uncertainty_penalty = np.clip(self.variance_scaling_factor * signal_variance, 0.0, 1.0)
        trust_weight = 1.0 - uncertainty_penalty
        
        # 3. Apply cognitive fusion
        # The AlphaAgent's conviction is scaled by our trust, then throttled by the RiskAgent.
        return (alpha_action * trust_weight) * risk_action

    def _update_strategic_networks(self):
        """Performs a single DDPG-style update for the STRATEGIC critic and actor networks."""
        states, actions, rewards, next_states, dones = self.strategic_replay_buffer.sample(self.batch_size)

        # --- Strategic Critic Update ---
        with torch.no_grad():
            next_actions = {}
            next_agent_obs = self._get_strategic_observations(next_states.cpu().numpy()) # Simplified
            for agent_id, obs in next_agent_obs.items():
                 next_actions[agent_id] = self.strategic_target_actor_networks[agent_id](torch.tensor(obs, device=self.device))
            
            all_next_actions_tensor = torch.cat(list(next_actions.values()), dim=1)
            target_q_values = self.strategic_target_critic_network(next_states, all_next_actions_tensor)
            td_target = rewards + self.gamma * (1 - dones) * target_q_values

        all_actions_tensor = torch.cat(list(actions.values()), dim=1)
        current_q_values = self.strategic_critic_network(states, all_actions_tensor)
        critic_loss = F.mse_loss(current_q_values, td_target)

        self.strategic_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.strategic_critic_optimizer.step()

        # --- Strategic Actor Update ---
        agent_obs_for_loss = self._get_strategic_observations(states.cpu().numpy()) # Simplified
        suggested_actions = {}
        for agent_id, obs in agent_obs_for_loss.items():
            suggested_actions[agent_id] = self.strategic_actor_networks[agent_id](torch.tensor(obs, device=self.device))
        
        all_suggested_actions_tensor = torch.cat(list(suggested_actions.values()), dim=1)
        actor_loss = -self.strategic_critic_network(states, all_suggested_actions_tensor).mean()

        for optimizer in self.strategic_actor_optimizers.values():
            optimizer.zero_grad()
        actor_loss.backward()
        for optimizer in self.strategic_actor_optimizers.values():
            optimizer.step()

        # --- Soft Update Strategic Target Networks ---
        self._soft_update_strategic_target_networks()

    def _update_execution_networks(self, obs, values, rewards, log_probs):
        """(Task 2.3) Implement the on-policy (e.g., A2C) update for the ExecutionAgent."""
        # Calculate returns and advantages
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device).squeeze()
        values = torch.cat(values).squeeze()
        advantages = returns - values
        
        # Stack rollout data
        log_probs = torch.cat(log_probs)
        
        # Calculate Actor (policy) loss
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Calculate Critic (value) loss
        critic_loss = F.mse_loss(values, returns)
        
        # Update Actor
        self.execution_actor_optimizer.zero_grad()
        actor_loss.backward()
        self.execution_actor_optimizer.step()
        
        # Update Critic
        self.execution_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.execution_critic_optimizer.step()


    def _soft_update_strategic_target_networks(self):
        """Blends the strategic network weights into the target networks."""
        for target_param, param in zip(self.strategic_target_critic_network.parameters(), self.strategic_critic_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for agent_id in self.strategic_agents.keys():
            target_actor = self.strategic_target_actor_networks[agent_id]
            actor = self.strategic_actor_networks[agent_id]
            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
