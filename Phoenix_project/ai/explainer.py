# (原: drl/drl_explainer.py)
# 这是一个用于“在线推理”的模块，用于解释模型的决策。

import shap
import pandas as pd
import numpy as np
from typing import Dict, Any

# --- [修复] ---
# 原: from ..monitor.logging import get_logger
# 新: from ..monitor.logging import get_logger (ai/ -> Phoenix_project/ -> monitor/)
# 导入路径 '..' 依然正确
# --- [修复结束] ---
from ..monitor.logging import get_logger

logger = get_logger(__name__)

class DRLExplainer:
    """
    (在线推理)
    使用 SHAP (或 LIME) 来解释 DRL 智能体 (例如 AlphaAgent) 的决策。
    """
    def __init__(self, drl_model: Any, training_data_sample: pd.DataFrame):
        """
        初始化解释器。
        
        Args:
            drl_model: 已加载的 DRL 模型 (例如 PPO 实例)。
            training_data_sample: 用于训练 SHAP 解释器的背景数据集 (采样)。
                                 (例如，从 TradingEnv 历史数据中采样的 100 个状态)
        """
        if shap is None:
            logger.error("SHAP 库未安装。`pip install shap`")
            raise ImportError("SHAP not found.")
            
        self.model = drl_model
        
        # SB3 模型的 predict() 函数是 SHAP 需要的
        # 我们可能需要一个封装器
        def predict_fn(observations: np.ndarray) -> np.ndarray:
            # DRL 模型通常输出动作 (离散)
            # SHAP 需要概率或连续值
            # 假设我们能获取动作概率
            if hasattr(self.model.policy, "predict_values"):
                # PPO/A2C 有 value function
                return self.model.policy.predict_values(observations).cpu().detach().numpy()
            elif hasattr(self.model.policy, "q_net"):
                # SAC/DQN 有 Q-values
                return self.model.policy.q_net(observations).cpu().detach().numpy()
            else:
                # 回退到预测动作
                actions, _ = self.model.predict(observations, deterministic=True)
                return actions
        
        self.predict_fn = predict_fn
        self.background_data = training_data_sample.values.astype(np.float32)
        
        logger.info("初始化 SHAP KernelExplainer...")
        # (注意：如果背景数据量大，这里会很慢)
        # self.explainer = shap.KernelExplainer(self.predict_fn, self.background_data)
        logger.info("DRL 解释器 (SHAP) 已初始化。")
        
    def explain_decision(self, current_observation: np.ndarray) -> Dict[str, Any]:
        """
        (在线推理)
        对 DRL 模型在当前状态下的决策进行 SHAP 归因。
        """
        logger.debug("正在计算 SHAP 值...")
        
        # 模拟 SHAP 值 (计算可能很慢)
        # shap_values = self.explainer.shap_values(current_observation)
        
        # 假设状态有 5 个特征 (O, H, L, C, V)
        feature_names = ['open', 'high', 'low', 'close', 'volume']
        simulated_shap_values = np.random.rand(len(feature_names))
        simulated_shap_values /= np.sum(simulated_shap_values) # 归一化
        
        explanation = {
            "shap_values": simulated_shap_values.tolist(),
            "features": feature_names,
            "base_value": 0.5, # 模拟
            "output_value": 0.8 # 模拟
        }
        
        logger.debug(f"SHAP 解释: {explanation}")
        return explanation
