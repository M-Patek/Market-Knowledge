import logging
import asyncio
from typing import List, Any, AsyncGenerator, Dict
import numpy as np
import uuid

from Phoenix_project.agents.l3.base import L3Agent
from Phoenix_project.core.schemas.task_schema import Task
# [关键] 我们现在导入 Signal (用于输入) 和 Order/OrderType (用于输出)
from Phoenix_project.execution.signal_protocol import Signal, SignalType, Order, OrderType, OrderStatus

logger = logging.getLogger(__name__)

# [任务 4 (Exec): 定制模型输入 - TODO]
# 您的执行 DRL 模型需要哪些特征？
# 它可能需要信号、波动率、以及可能的 *订单簿深度* (LOB)
# [TODO] 主人喵：请确保 data_manager.get_latest_features 提供了这些键！
EXECUTION_MARKET_FEATURE_KEYS = [
    "feature_1", # 示例： "volatility_30m"
    "feature_2", # 示例： "spread_bid_ask"
    "feature_3", # 示例： "lob_depth_level_1"
    "feature_4", # 示例： "lob_depth_level_2"
    # ...
]

# [任务 4 (Exec): 定制模型输入 - TODO]
# [TODO] 主人喵：请确保 data_manager.get_latest_features 提供了这些键！
EXECUTION_PORTFOLIO_STATE_KEYS = ["cash", "asset_value"] 


class ExecutionAgent(L3Agent):
    """
    L3 智能体：订单执行 (ExecutionAgent)
    接收 L3 RiskAgent 的最终信号 (Signal)，并将其转换为
    一个或多个 L4 可执行的订单 (Order)。
    """

    def _prepare_model_features(
        self, 
        task: Task, 
        final_signal: Signal,
        market_data: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        (内部) 将 L3 最终信号和市场数据打包成 *执行* DRL 模型的输入。
        
        [任务 4 (Exec): 定制模型输入 - TODO]
        [这是一个需要根据您的 *执行* DRL 模型输入来定制的关键函数]
        """
        logger.debug(f"[{self.agent_id}] Preparing features for Execution DRL model...")
        
        # 1. 准备 "signal_features" (维度 2)
        signal_map = {SignalType.BUY: 1.0, SignalType.SELL: -1.0, SignalType.HOLD: 0.0}
        signal_type = signal_map.get(final_signal.signal_type, 0.0)
        signal_weight = final_signal.weight # 经过风险调整的权重 (例如 0.5)
        
        signal_features = np.array(
            [signal_type, signal_weight], 
            dtype=np.float32
        )
        
        # 2. 准备 "market_features"
        # [TODO] 主人喵：请确保 data_manager 提供了 EXECUTION_MARKET_FEATURE_KEYS
        try:
            market_features = np.array(
                [market_data.get(key, 0.0) for key in EXECUTION_MARKET_FEATURE_KEYS], 
                dtype=np.float32
            )
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to build execution market_features vector: {e}")
            market_features = np.zeros(len(EXECUTION_MARKET_FEATURE_KEYS), dtype=np.float32)

        # 3. 准备 "portfolio_state"
        # [TODO] 主人喵：请确保 data_manager 提供了 EXECUTION_PORTFOLIO_STATE_KEYS
        try:
            portfolio_state = np.array(
                [market_data.get(key, 0.0) for key in EXECUTION_PORTFOLIO_STATE_KEYS],
                dtype=np.float32
            )
        except Exception as e:
             logger.error(f"[{self.agent_id}] Failed to build execution portfolio_state vector: {e}")
             portfolio_state = np.zeros(len(EXECUTION_PORTFOLIO_STATE_KEYS), dtype=np.float32)

        
        # 4. 组装成模型期望的最终字典
        # [TODO] 检查您的执行 DRL 模型的 observation_space！
        features_dict = {
            "signal_features": signal_features,
            "market_features": market_features,
            "portfolio_state": portfolio_state
        }
        
        logger.debug(f"[{self.agent_id}] Prepared features dict for execution model.")
        return features_dict
        
    def _parse_model_action(
        self, 
        action: Any, 
        final_signal: Signal,
        market_data: Dict[str, Any] # [激活] 传入 market_data 以移除 MOCK
    ) -> List[Order]:
        """
        (内部) 将执行 DRL 模型的原始输出解析为 L4 的 Order 列表。
        
        [任务 5 (Exec): 定制模型输出 - TODO]
        [这是一个需要根据您的 *执行* DRL 模型输出来定制的关键函数]
        """
        logger.debug(f"[{self.agent_id}] Parsing execution model action: {action}")

        # [任务 5: 定制模型输出 - TODO]
        # 执行 DRL 模型的 action_space 是什么？
        # MOCK 示例：Discrete(3) 
        # 0 = 立即执行 (1个市价单)
        # 1 = TWAP 30 分钟 (拆分为 10 个限价单)
        # 2 = VWAP 60 分钟 (拆分为 20 个限价单)
        
        orders_to_execute = []
        
        # [MOCK 已移除] 从 market_data (或 portfolio_state) 获取真实数据
        # [TODO] 确保 data_manager.get_latest_features 提供了 "cash" 和 "current_price"
        total_capital = market_data.get("cash", 0.0) # 应该来自 portfolio_state
        current_price = market_data.get("current_price", 0.0) # 应该来自 market_features
        
        if current_price == 0.0 or total_capital == 0.0:
            logger.error(f"[{self.agent_id}] Current price ({current_price}) or total capital ({total_capital}) is 0. Cannot calculate quantity.")
            return []
        
        # (例如: (1M * 0.5) / 150) = 3333 股
        total_quantity_to_trade = (total_capital * final_signal.weight) / current_price
        
        try:
            action_int = int(action)
            
            if action_int == 0:
                # 策略：立即执行 (市价单)
                logger.info(f"[{self.agent_id}] Execution Strategy: Immediate (Market Order)")
                orders_to_execute.append(
                    Order(
                        symbol=final_signal.symbol,
                        order_type=OrderType.MARKET,
                        quantity=total_quantity_to_trade,
                        signal_id=final_signal.signal_id, # 追踪溯源
                        metadata={"exec_strategy": "Immediate (MOCK)"}
                    )
                )
            
            elif action_int == 1:
                # 策略：TWAP 30 分钟 (拆分为 10 单)
                 logger.info(f"[{self.agent_id}] Execution Strategy: TWAP 30min (10 child orders) (MOCK)")
                 child_quantity = total_quantity_to_trade / 10
                 for i in range(10):
                     orders_to_execute.append(
                         Order(
                             symbol=final_signal.symbol,
                             order_type=OrderType.LIMIT, # TWAP/VWAP 通常使用限价单
                             quantity=child_quantity,
                             price=current_price - (0.01 * (i % 2)), # MOCK 限价
                             signal_id=final_signal.signal_id,
                             metadata={"exec_strategy": "TWAP_30m (MOCK)", "child_id": f"{i+1}_of_10"}
                         )
                     )
            
            else: # action_int == 2 或其他
                # 策略：VWAP 60 分钟 (拆分为 20 单)
                logger.info(f"[{self.agent_id}] Execution Strategy: VWAP 60min (20 child orders) (MOCK)")
                child_quantity = total_quantity_to_trade / 20
                for i in range(20):
                    orders_to_execute.append(
                        Order(
                             symbol=final_signal.symbol,
                             order_type=OrderType.LIMIT,
                             quantity=child_quantity,
                             price=current_price - (0.01 * (i % 3)), # MOCK 限价
                             signal_id=final_signal.signal_id,
                             metadata={"exec_strategy": "VWAP_60m (MOCK)", "child_id": f"{i+1}_of_20"}
                         )
                    )
            
            return orders_to_execute

        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to parse Execution DRL model action '{action}'. Error: {e}")
            return [] # 失败则不产生任何订单

    async def run(self, task: Task, dependencies: List[Any]) -> AsyncGenerator[Order, None]:
        """
        异步运行智能体，将 L3 最终信号 转换为 L4 订单。
        
        参数:
            task (Task): 当前任务。
            dependencies (List[Any]): 来自 L3 (RiskAgent) 的 Signal 列表。

        收益:
            AsyncGenerator[Order, None]: 异步生成一个或多个 Order 对象。
        """
        logger.info(f"[{self.agent_id}] Running ExecutionAgent for task: {task.task_id}")

        # 1. 查找 L3 RiskAgent 的最终信号
        final_signals = [dep for dep in dependencies if isinstance(dep, Signal)]
        
        if not final_signals:
            logger.warning(f"[{self.agent_id}] No final Signal found in dependencies for task {task.task_id}. Cannot run ExecutionAgent.")
            return

        final_signal = final_signals[0]
        symbol = final_signal.symbol
        
        # 如果最终信号是 HOLD，则无需执行
        if final_signal.signal_type == SignalType.HOLD or final_signal.weight == 0.0:
            logger.info(f"[{self.agent_id}] Final signal is HOLD. No orders to execute.")
            return
        
        logger.info(f"[{self.agent_id}] Received Final Signal: {final_signal.signal_type} @ {final_signal.weight}")

        try:
            # 2. [核心逻辑] - 获取 DRL 模型所需的状态
            logger.debug(f"[{self.agent_id}] Fetching market state for execution (e.g., LOB)...")
            # [关键] 确保 get_latest_features 包含 "cash", "current_price" 和所有 EXECUTION_* 键
            market_data = await self.data_manager.get_latest_features(symbol)
            
            # 3. [核心逻辑] - 准备特征向量
            features = self._prepare_model_features(task, final_signal, market_data)

            # 4. [核心逻辑] - 调用 *执行* DRL/Quant 模型
            logger.debug(f"[{self.agent_id}] Calling Execution DRL/Quant model (model_client.predict)...")
            # [关键] 确保 self.model_client 加载的是 *执行模型*
            action = await asyncio.to_thread(self.model_client.predict, features)
            
            # 5. [核心逻辑] - 解析模型动作并转换为*订单列表*
            # [激活] 传入 market_data 以移除 MOCK
            orders_to_execute = self._parse_model_action(action, final_signal, market_data)
            
            logger.info(f"[{self.agent_id}] Generated {len(orders_to_execute)} orders for L4.")
            
            # 6. Yield 所有订单
            for order in orders_to_execute:
                yield order

        except Exception as e:
            logger.error(f"[{self.agent_id}] An unexpected error occurred during ExecutionAgent run. Error: {e}", exc_info=True)
            # 发生错误时，不应产生任何订单
            return
