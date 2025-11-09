import logging
import asyncio
from typing import List, Any, AsyncGenerator, Dict
import numpy as np

from Phoenix_project.agents.l3.base import L3Agent
from Phoenix_project.core.schemas.task_schema import Task
from Phoenix_project.execution.signal_protocol import Signal, SignalType

logger = logging.getLogger(__name__)

# [任务 4 (Risk): 定制模型输入 - 已解决]
# 根据 execution_env.py (dim=10)
# [TODO] 主人喵：请确认 data_manager.get_latest_features 返回的字典包含这 10 个键！
RISK_MARKET_FEATURE_KEYS = [
    "feature_1", # 示例： "price_norm"
    "feature_2", # 示例： "volume_norm"
    "feature_3", # 示例： "volatility_norm"
    "feature_4", # 示例： "rsi"
    "feature_5", # 示例： "macd"
    "feature_6", # 示例： "sma_50_diff"
    "feature_7", # 示例： "sma_200_diff"
    "feature_8", # 示例： "bb_upper_norm"
    "feature_9", # 示例： "bb_lower_norm"
    "feature_10" # 示例： "atr_norm"
]

# [任务 4 (Risk): 定制模型输入 - 已解决]
# 根据 execution_env.py (dim=2)
# [TODO] 主人喵：请确认 data_manager.get_latest_features 返回 "cash" 和 "asset_value"
RISK_PORTFOLIO_STATE_KEYS = ["cash", "asset_value"] 


class RiskAgent(L3Agent):
    """
    L3 智能体：风险管理 (RiskAgent)
    接收 L3 AlphaAgent 的信号 (Signal)，并使用 DRL/Quant 模型
    来确定资本分配（调整信号的 'weight'）。
    """

    def _prepare_model_features(
        self, 
        task: Task, 
        original_signal: Signal,
        market_data: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        (内部) 将 L3 信号和市场数据打包成风险 DRL 模型的输入特征向量。
        
        [任务 4 (Risk): 定制模型输入 - 已解决]
        根据 execution_env.py，模型期望一个包含
        'alpha_signal_features', 'market_features', 和 'portfolio_state' 的字典。
        """
        logger.debug(f"[{self.agent_id}] Preparing features for Risk DRL model...")
        
        # 1. 准备 "alpha_signal_features" (维度 2)
        # 风险模型需要知道 AlphaAgent 的决定
        # (根据 execution_env.py 的 observation_space)
        signal_map = {SignalType.BUY: 1.0, SignalType.SELL: -1.0, SignalType.HOLD: 0.0}
        alpha_signal_type = signal_map.get(original_signal.signal_type, 0.0)
        # AlphaAgent 产出的 weight (例如 1.0)
        alpha_signal_weight = original_signal.weight 
        
        alpha_signal_features = np.array(
            [alpha_signal_type, alpha_signal_weight], 
            dtype=np.float32
        )
        
        # 2. 准备 "market_features" (维度 10)
        try:
            market_features = np.array(
                [market_data.get(key, 0.0) for key in RISK_MARKET_FEATURE_KEYS], 
                dtype=np.float32
            )
            if len(market_features) != 10:
                logger.warning(f"[{self.agent_id}] Risk market features dim mismatch! Expected 10, got {len(market_features)}")
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to build risk market_features vector: {e}")
            market_features = np.zeros(len(RISK_MARKET_FEATURE_KEYS), dtype=np.float32) # Fallback

        # 3. 准备 "portfolio_state" (维度 2)
        try:
            portfolio_state = np.array(
                [market_data.get(key, 0.0) for key in RISK_PORTFOLIO_STATE_KEYS],
                dtype=np.float32
            )
            if len(portfolio_state) != 2:
                logger.warning(f"[{self.agent_id}] Risk portfolio state dim mismatch! Expected 2, got {len(portfolio_state)}")
        except Exception as e:
             logger.error(f"[{self.agent_id}] Failed to build risk portfolio_state vector: {e}")
             portfolio_state = np.zeros(len(RISK_PORTFOLIO_STATE_KEYS), dtype=np.float32) # Fallback

        
        # 4. 组装成模型期望的最终字典
        # (根据 execution_env.py 的 observation_space)
        features_dict = {
            "alpha_signal_features": alpha_signal_features,
            "market_features": market_features,
            "portfolio_state": portfolio_state
        }
        
        logger.debug(f"[{self.agent_id}] Prepared features dict for risk model.")
        return features_dict
        
    def _parse_model_action(
        self, 
        action: Any, 
        original_signal: Signal
    ) -> Signal:
        """
        (内部) 将风险 DRL 模型的原始输出（例如：一个资本调整系数）
        解析为最终的、经过风险调整的 Signal 对象。
        
        [任务 5 (Risk): 定制模型输出 - 已解决]
        根据 execution_env.py (spaces.Discrete(3))，我们使用离散百分比。
        """
        logger.debug(f"[{self.agent_id}] Parsing risk model action: {action}")

        # [任务 5: 定制模型输出 - 已解决]
        # 示例：离散百分比 (Discrete)
        # 根据 execution_env.py: 0=25%, 1=50%, 2=100%
        try:
            action_int = int(action)
            
            if action_int == 0:
                capital_modifier = 0.25 # 25%
            elif action_int == 1:
                capital_modifier = 0.50 # 50%
            elif action_int == 2:
                capital_modifier = 1.00 # 100%
            else:
                logger.warning(f"[{self.agent_id}] Unknown risk model action {action_int}, defaulting to 0.")
                capital_modifier = 0.0 # 安全回退

            # [核心逻辑] 将 AlphaAgent 的权重 乘以 风险系数
            final_weight = original_signal.weight * capital_modifier
            
            # 创建一个*新的*信号，或者修改原始信号的副本
            final_signal = original_signal.model_copy()
            final_signal.weight = final_weight
            
            # 更新元数据以进行审计
            final_signal.metadata["risk_agent_id"] = self.agent_id
            final_signal.metadata["risk_model_output"] = action_int
            final_signal.metadata["capital_modifier"] = capital_modifier
            final_signal.metadata["original_alpha_weight"] = original_signal.weight
            
            return final_signal

        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to parse Risk DRL model action (Discrete) '{action}'. Error: {e}")
            # [安全回退] 如果风险模型失败，阻止交易（权重设为0）
            final_signal = original_signal.model_copy()
            final_signal.weight = 0.0
            final_signal.metadata["error"] = f"Risk model action parsing failed: {e}"
            return final_signal
        # -------------------------------------------------------------------

    async def run(self, task: Task, dependencies: List[Any]) -> AsyncGenerator[Signal, None]:
        """
        异步运行智能体，将 L3 Alpha 信号 调整为 最终 L3 信号。
        
        参数:
            task (Task): 当前任务。
            dependencies (List[Any]): 来自 L3 (AlphaAgent) 的 Signal 列表。

        收益:
            AsyncGenerator[Signal, None]: 异步生成 *单一* 的、经过风险调整的 Signal 对象。
        """
        logger.info(f"[{self.agent_id}] Running RiskAgent for task: {task.task_id}")

        # 1. 查找 L3 AlphaAgent 的信号
        alpha_signals = [dep for dep in dependencies if isinstance(dep, Signal)]
        
        if not alpha_signals:
            logger.warning(f"[{self.agent_id}] No Signal found in dependencies for task {task.task_id}. Cannot run RiskAgent.")
            return

        # L3 AlphaAgent 保证只产出 *一个* 信号
        original_signal = alpha_signals[0]
        symbol = original_signal.symbol
        
        # 如果 AlphaAgent 决定 HOLD，风险智能体无需工作
        if original_signal.signal_type == SignalType.HOLD or original_signal.weight == 0.0:
            logger.info(f"[{self.agent_id}] AlphaAgent signal is HOLD. Yielding original signal.")
            yield original_signal
            return
        
        logger.info(f"[{self.agent_id}] Received Alpha Signal: {original_signal.signal_type} @ {original_signal.weight}")

        try:
            # 2. [核心逻辑] - 获取 DRL 模型所需的状态
            logger.debug(f"[{self.agent_id}] Fetching market state for {symbol}...")
            # [关键] 确保 get_latest_features 包含风险模型需要的所有键
            market_data = await self.data_manager.get_latest_features(symbol)
            
            # 3. [核心逻辑] - G准备特征向量
            features = self._prepare_model_features(task, original_signal, market_data)

            # 4. [核心逻辑] - 调用 *风险* DRL/Quant 模型
            logger.debug(f"[{self.agent_id}] Calling Risk DRL/Quant model (model_client.predict)...")
            # [关键] 确保 self.model_client 加载的是 *风险模型*，而不是 *Alpha 模型*
            # (这必须在外部的依赖注入中完成 [任务 1 (Risk)])
            action = await asyncio.to_thread(self.model_client.predict, features)
            
            # 5. [核心逻辑] - 解析模型动作并转换为*最终* Signal
            final_signal = self._parse_model_action(action, original_signal)
            
            logger.info(f"[{self.agent_id}] Generated Final Signal for {symbol}: {final_signal.signal_type} @ {final_signal.weight} (Original: {original_signal.weight})")
            
            yield final_signal

        except Exception as e:
            logger.error(f"[{self.agent_id}] An unexpected error occurred during RiskAgent run. Error: {e}", exc_info=True)
            # [安全回退] 发生未知错误时，阻止交易
            yield Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                weight=0.0,
                confidence=0.0,
                metadata={"source_agent": self.agent_id, "error": str(e)}
            )
