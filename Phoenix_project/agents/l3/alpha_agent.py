import logging
import asyncio
from typing import List, Any, AsyncGenerator, Dict
import numpy as np # 用于特征和动作解析

from Phoenix_project.agents.l3.base import L3Agent
from Phoenix_project.core.schemas.task_schema import Task
from Phoenix_project.core.schemas.fusion_result import FusionResult
from Phoenix_project.execution.signal_protocol import Signal, SignalType

logger = logging.getLogger(__name__)

# [任务 4: 定制模型输入 - 已解决]
# 根据 trading_env.py (dim=10) 和 data_manager 的能力
# [TODO] 主人喵：请确认 data_manager.get_latest_features 返回的字典包含这 10 个键！
# 顺序 *必须* 与训练时一致！
MARKET_FEATURE_KEYS = [
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

# [任务 4: 定制模型输入 - 已解决]
# 根据 trading_env.py (dim=2)
# [TODO] 主人喵：请确认 data_manager.get_latest_features 返回 "cash" 和 "asset_value"
PORTFOLIO_STATE_KEYS = ["cash", "asset_value"] 

class AlphaAgent(L3Agent):
    """
    L3 智能体：阿尔法生成 (AlphaAgent)
    接收 L2 的融合决策 (FusionResult)，并使用 DRL/Quant 模型
    将其转换为可执行的交易信号 (Signal)。
    """

    def _prepare_model_features(
        self, 
        task: Task, 
        fusion_result: FusionResult,
        market_data: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        (内部) 将 L2 决策和市场数据打包成 DRL 模型的输入特征向量。
        
        [任务 4: 定制模型输入 - 已解决]
        根据 trading_env.py 和 networks.py，模型期望一个包含
        'llm_features', 'market_features', 和 'portfolio_state' 的字典。
        """
        logger.debug(f"[{self.agent_id}] Preparing features for DRL model...")
        
        # 1. 准备 "llm_features" (维度 2)
        sentiment_map = {"Strong Bullish": 1.0, "Bullish": 0.5, "Neutral": 0.0, "Bearish": -0.5, "Strong Bearish": -1.0}
        llm_sentiment = sentiment_map.get(fusion_result.overall_sentiment, 0.0)
        llm_confidence = fusion_result.confidence
        
        llm_features = np.array(
            [llm_sentiment, llm_confidence], 
            dtype=np.float32
        )
        
        # 2. 准备 "market_features" (维度 10)
        # [关键] 确保 market_data 包含 MARKET_FEATURE_KEYS 中定义的所有键
        try:
            market_features = np.array(
                [market_data.get(key, 0.0) for key in MARKET_FEATURE_KEYS], 
                dtype=np.float32
            )
            if len(market_features) != 10:
                logger.warning(f"[{self.agent_id}] Market features dim mismatch! Expected 10, got {len(market_features)}")
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to build market_features vector: {e}")
            market_features = np.zeros(10, dtype=np.float32) # Fallback

        # 3. 准备 "portfolio_state" (维度 2)
        # [关键] 确保 market_data 包含 PORTFOLIO_STATE_KEYS 中定义的所有键
        try:
            portfolio_state = np.array(
                [market_data.get(key, 0.0) for key in PORTFOLIO_STATE_KEYS],
                dtype=np.float32
            )
            if len(portfolio_state) != 2:
                 logger.warning(f"[{self.agent_id}] Portfolio state dim mismatch! Expected 2, got {len(portfolio_state)}")
        except Exception as e:
             logger.error(f"[{self.agent_id}] Failed to build portfolio_state vector: {e}")
             portfolio_state = np.zeros(2, dtype=np.float32) # Fallback

        
        # 4. 组装成模型期望的最终字典
        features_dict = {
            "llm_features": llm_features,
            "market_features": market_features,
            "portfolio_state": portfolio_state
        }
        
        logger.debug(f"[{self.agent_id}] Prepared features dict with keys: {features_dict.keys()}")
        return features_dict
        
    def _parse_model_action(
        self, 
        action: Any, 
        task: Task
    ) -> Signal:
        """
        (内部) 将 DRL 模型的原始输出（例如：一个动作向量）
        解析为 L4 可以理解的 Signal 对象。
        
        [任务 5: 定制模型输出 - 已解决]
        根据 trading_env.py (spaces.Discrete(3))，我们使用示例 2。
        """
        logger.debug(f"[{self.agent_id}] Parsing model action: {action}")
        symbol = task.symbols[0]

        # -------------------------------------------------------------------
        # 示例 1：回归 (Regression) - (已禁用)
        # (此逻辑与 spaces.Discrete(3) 不匹配)
        # -------------------------------------------------------------------

        # [任务 5: 定制模型输出 - 已解决]
        # 示例 2：离散动作 (Discrete) - (例如：0=Hold, 1=Buy, 2=Sell)
        # (此逻辑匹配 spaces.Discrete(3))
        try:
            action_int = int(action)
            weight = 1.0 # DRL 决定动作，仓位管理可能是固定的
            
            if action_int == 1: # Buy
                signal_type = SignalType.BUY
            elif action_int == 2: # Sell
                signal_type = SignalType.SELL
            else: # 0 = Hold
                signal_type = SignalType.HOLD
                weight = 0.0
                
            return Signal(
                symbol=symbol,
                signal_type=signal_type,
                weight=weight,
                confidence=1.0, # DRL 模型的决策是确定性的
                metadata={"source_agent": self.agent_id, "model_output": action_int}
            )
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to parse DRL model action (Discrete) '{action}'. Error: {e}")
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                weight=0.0,
                confidence=0.0,
                metadata={"source_agent": self.agent_id, "error": f"Model action parsing failed: {e}"}
            )
        # -------------------------------------------------------------------

        # # 示例 3：Softmax 向量 - (已禁用)
        # (此逻辑与 spaces.Discrete(3) 不匹配)
        # -------------------------------------------------------------------


    async def run(self, task: Task, dependencies: List[Any]) -> AsyncGenerator[Signal, None]:
        """
        异步运行智能体，将 L2 决策转换为 L3 信号。
        """
        logger.info(f"[{self.agent_id}] Running AlphaAgent for task: {task.task_id}")

        # 1. 查找 L2 的最终决策
        fusion_results = [dep for dep in dependencies if isinstance(dep, FusionResult)]
        
        if not fusion_results:
            logger.warning(f"[{self.agent_id}] No FusionResult found in dependencies for task {task.task_id}. Cannot generate signal.")
            return

        fusion_result = fusion_results[0]
        symbol = task.symbols[0]
        
        logger.info(f"[{self.agent_id}] Received L2 FusionResult: {fusion_result.overall_sentiment} (Conf: {fusion_result.confidence})")

        try:
            # 2. [核心逻辑] - 获取 DRL 模型所需的状态
            logger.debug(f"[{self.agent_id}] Fetching market state for {symbol}...")
            
            # [任务 2: 激活数据管理器 - 已完成]
            # [关键] 确保 get_latest_features 返回一个包含所有所需键的字典
            # (MARKET_FEATURE_KEYS 和 PORTFOLIO_STATE_KEYS)
            market_data = await self.data_manager.get_latest_features(symbol)
            
            # 3. [核心逻辑] - 准备特征向量
            # (此函数现在返回模型期望的字典结构)
            features = self._prepare_model_features(task, fusion_result, market_data)

            # 4. [核心逻辑] - 调用 DRL/Quant 模型
            logger.debug(f"[{self.agent_id}] Calling DRL/Quant model (model_client.predict)...")
            
            # [任务 3: 激活 DRL/Quant 模型 - 已完成]
            action = await asyncio.to_thread(self.model_client.predict, features)
            
            # 5. [核心逻辑] - 解析模型动作并转换为 Signal
            # (此函数现在使用 Discrete(3) 逻辑)
            signal = self._parse_model_action(action, task)
            
            logger.info(f"[{self.agent_id}] Generated Signal for {symbol}: {signal.signal_type} @ {signal.weight}")
            
            yield signal

        except Exception as e:
            logger.error(f"[{self.agent_id}] An unexpected error occurred during AlphaAgent run. Error: {e}", exc_info=True)
            yield Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                weight=0.0,
                confidence=0.0,
                metadata={"source_agent": self.agent_id, "error": str(e)}
            )
