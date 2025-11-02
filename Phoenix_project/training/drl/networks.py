# (原: drl/networks.py)
# (无内部导入，无需修复)

import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """
    一个自定义的 CNN 特征提取器 (用于 TradingEnv)。
    :param observation_space: (gym.Space)
    :param features_dim: (int) 神经网络的输出特征数量
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        # observation_space.shape (lookback_window, 5) -> (N, C, H, W)
        # 我们需要将其视为 (N, 1, lookback_window, 5)
        
        n_input_channels = 1 # 把它当作一个单通道图像
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 计算展平后的特征维度
        # 我们需要一个虚拟输入来计算
        with th.no_grad():
            dummy_input = th.as_tensor(observation_space.sample()[None]).unsqueeze(1)
            n_flatten = self.cnn(dummy_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # (N, H, W) -> (N, 1, H, W)
        observations = observations.unsqueeze(1)
        return self.linear(self.cnn(observations))

class CustomMLP(BaseFeaturesExtractor):
    """
    一个自定义的 MLP 特征提取器 (用于 ExecutionEnv)。
    :param observation_space: (gym.Space)
    :param features_dim: (int) 神经网络的输出特征数量
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 32):
        super(CustomMLP, self).__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0] # (2,) -> 2
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.network(observations)
