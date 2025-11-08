import logging
import asyncio
from typing import List, Any, AsyncGenerator, Dict

from Phoenix_project.agents.l3.base import L3Agent
from Phoenix_project.core.schemas.task_schema import Task
from Phoenix_project.core.schemas.fusion_result import FusionResult
from Phoenix_project.execution.signal_protocol import Signal, SignalType

logger = logging.getLogger(__name__)

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
    ) -> Dict[str, Any]:
        """
        (内部) 将 L2 决策和市场数据打包成 DRL 模型的输入特征向量。
        
        [这是一个需要根据您的 DRL 模型输入来定制的关键函数]
        """
        logger.debug(f"[{self.agent_id}] Preparing features for model...")
        
        # [示例 MOCK 特征]
        # 您的 DRL 模型可能需要标准化的数值输入
        sentiment_map = {"Strong Bullish": 1.0, "Bullish": 0.5, "Neutral": 0.0, "Bearish": -0.5, "Strong Bearish": -1.0}
        
        features = {
            "l2_sentiment": sentiment_map.get(fusion_result.overall_sentiment, 0.0),
            "l2_confidence": fusion_result.confidence,
            "market_volatility": market_data.get("volatility", 0.0), # 来自 data_manager
            "current_price": market_data.get("price", 0.0), # 来自 data_manager
            # ... 您模型所需的其他特征 (例如：RSI, MACD, 当前持仓)
        }
        return features
        
    def _parse_model_action(
        self, 
        action: Any, 
        task: Task
    ) -> Signal:
        """
        (内部) 将 DRL 模型的原始输出（例如：一个动作向量）
        解析为 L4 可以理解的 Signal 对象。
        
        [这是一个需要根据您的 DRL 模型输出来定制的关键函数]
        """
        logger.debug(f"[{self.agent_id}] Parsing model action: {action}")
        symbol = task.symbols[0]

        # [示例 MOCK 解析]
        # 假设模型输出一个介于 -1.0 (全仓卖出) 到 +1.0 (全仓买入) 之间的浮点数
        try:
            target_weight = float(action)
            
            if target_weight > 0.01:
                signal_type = SignalType.BUY
                weight = target_weight
            elif target_weight < -0.01:
                signal_type = SignalType.SELL
                weight = abs(target_weight)
            else:
                signal_type = SignalType.HOLD
                weight = 0.0
                
            return Signal(
                symbol=symbol,
                signal_type=signal_type,
                weight=weight,
                confidence=1.0, # DRL 模型的输出是“确定性”的
                metadata={"source_agent": self.agent_id, "model_output": action}
            )

        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to parse DRL model action '{action}'. Error: {e}")
            # 默认返回 HOLD 信号
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                weight=0.0,
                confidence=0.0,
                metadata={"source_agent": self.agent_id, "error": "Model action parsing failed"}
            )


    async def run(self, task: Task, dependencies: List[Any]) -> AsyncGenerator[Signal, None]:
        """
        异步运行智能体，将 L2 决策转换为 L3 信号。
        
        参数:
            task (Task): 当前任务。
            dependencies (List[Any]): 来自 L2 (FusionAgent) 的 FusionResult 列表。

        收益:
            AsyncGenerator[Signal, None]: 异步生成 *单一* 的 Signal 对象。
        """
        logger.info(f"[{self.agent_id}] Running AlphaAgent for task: {task.task_id}")

        # 1. 查找 L2 的最终决策
        fusion_results = [dep for dep in dependencies if isinstance(dep, FusionResult)]
        
        if not fusion_results:
            logger.warning(f"[{self.agent_id}] No FusionResult found in dependencies for task {task.task_id}. Cannot generate signal.")
            return

        # L2 FusionAgent 保证只产出 *一个* 结果
        fusion_result = fusion_results[0]
        symbol = task.symbols[0] # 假设 L3 针对主要标的
        
        logger.info(f"[{self.agent_id}] Received L2 FusionResult: {fusion_result.overall_sentiment} (Conf: {fusion_result.confidence})")

        try:
            # 2. [核心逻辑] - 获取 DRL 模型所需的状态
            # (这可能是一个异步调用)
            logger.debug(f"[{self.agent_id}] Fetching market state for {symbol}...")
            # [MOCK] 替换为真实的数据管理器调用
            # market_data = await self.data_manager.get_latest_features(symbol)
            market_data = {"volatility": 0.2, "price": 150.0} # MOCK DATA
            
            # 3. [核心逻辑] - 准备特征向量
            features = self._prepare_model_features(task, fusion_result, market_data)

            # 4. [核心逻辑] - 调用 DRL/Quant 模型
            # [关键] 我们使用 asyncio.to_thread 来包装 *可能* 同步的 DRL 模型 'predict' 调用
            # 以避免阻塞异步事件循环。
            logger.debug(f"[{self.agent_id}] Calling DRL/Quant model (model_client.predict)...")
            
            # [MOCK] 替换为真实的模型调用
            # action = await asyncio.to_thread(self.model_client.predict, features)
            action = 0.85 # MOCK ACTION (对应 "Strong Bullish", 0.85)
            
            # 5. [核心逻辑] - 解析模型动作并转换为 Signal
            signal = self._parse_model_action(action, task)
            
            logger.info(f"[{self.agent_id}] Generated Signal for {symbol}: {signal.signal_type} @ {signal.weight}")
            
            yield signal

        except Exception as e:
            logger.error(f"[{self.agent_id}] An unexpected error occurred during AlphaAgent run. Error: {e}", exc_info=True)
            # 可以在此处 yield 一个紧急 HOLD 信号
            yield Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                weight=0.0,
                confidence=0.0,
                metadata={"source_agent": self.agent_id, "error": str(e)}
            )
