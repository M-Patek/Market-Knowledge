"""
L3 Agent: Execution Agent
Refactored from training/drl/agents/execution_agent.py.
Responsible for "Order Execution."
"""
from typing import Any, List, Optional, Dict  # <-- 导入 Dict
import uuid

from Phoenix_project.agents.l3.base import BaseL3Agent
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.fusion_result import FusionResult
from Phoenix_project.core.schemas.data_schema import Signal, Order, OrderStatus
from Phoenix_project.core.schemas.risk_schema import RiskAdjustment

class ExecutionAgent(BaseL3Agent):
    """
    Implements the L3 Execution agent.
    Inherits from BaseL3Agent and implements the run method
    to convert a Signal and RiskAdjustment into Orders.
    """
    
    # 更改 'run' 签名以接收 'dependencies'
    def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> List[Order]:
        """
        Splits large orders and optimizes the execution path.
        This agent consumes the outputs of its L3 peers (Alpha and Risk)
        by retrieving them from the dependencies dictionary.
        
        Args:
            state (PipelineState): The current state.
            dependencies (Dict[str, Any]): The outputs from peer L3 agents 
                                         (expected to contain List[Signal] and List[RiskAdjustment]).
            
        Returns:
            List[Order]: A list of Order objects to be executed.
        """
        
        # 从 'dependencies' 检索输出，而不是 'state'
        signals: List[Signal] = []
        adjustments: List[RiskAdjustment] = []

        # 解析依赖项，获取 Signals 和 RiskAdjustments
        # L3 AlphaAgent 和 RiskAgent 可能返回列表
        for result in dependencies.values():
            if isinstance(result, list):
                if result and isinstance(result[0], Signal):
                    signals.extend(result)
                elif result and isinstance(result[0], RiskAdjustment):
                    adjustments.extend(result)
            elif isinstance(result, Signal):
                # 以防万一它们没有返回列表
                signals.append(result)
            elif isinstance(result, RiskAdjustment):
                # 以防万一它们没有返回列表
                adjustments.append(result)
        
        orders = []
        
        # TODO: Implement actual DRL/Quant model logic for execution.
        # This logic would use self.model_client to split orders (TWAP, VWAP, etc.)
        
        # This is a mock execution logic.
        # It just converts each signal into a single MARKET order,
        # applying the risk adjustment.
        
        for signal in signals:
            # Find the matching risk adjustment
            adjustment = next((adj for adj in adjustments if adj.target_symbol == signal.symbol), None)
            capital_modifier = adjustment.capital_modifier if adjustment else 1.0
            
            # Mock total capital per trade = 100,000
            total_capital = 100000.0
            trade_value = total_capital * signal.strength * capital_modifier
            
            # We can't know price, so we'll create a simple Market Order
            # In a real system, we'd need a price oracle to get quantity.
            # For this mock, we'll assume a placeholder quantity.
            mock_quantity = 100.0 * (1.0 if signal.signal_type == "BUY" else -1.0)

            orders.append(Order(
                id=str(uuid.uuid4()),
                symbol=signal.symbol,
                quantity=mock_quantity, # Placeholder quantity
                order_type="MARKET",
                status=OrderStatus.NEW,
                metadata={"source_agent": self.agent_id, "signal_id": signal.id}
            ))
            
        return orders

    def __repr__(self) -> str:
        return f"<ExecutionAgent(id='{self.agent_id}')>"
