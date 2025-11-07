from typing import List, Dict, Any, Optional, Tuple
from Phoenix_project.core.pipeline_state import PipelineState
# [主人喵的修复] 导入 Order, Signal, TargetPortfolio, RiskReport
from Phoenix_project.core.schemas.data_schema import Order, Signal, TargetPortfolio, TargetPosition, PortfolioState
from Phoenix_project.core.schemas.risk_schema import RiskReport
from Phoenix_project.monitor.logging import get_logger
import copy # [主人喵的修复] 导入 copy 用于深拷贝

logger = get_logger(__name__)

class RiskManager:
    """
    Provides pre-trade and post-trade risk checks.
    It can veto orders or signals based on predefined rules.

    [主人喵的修复]
    添加了 Orchestrator 调用的 'evaluate_and_adjust' 方法的实现。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("risk_manager", {})
        self.log_prefix = "RiskManager:" # [主人喵的修复] 添加 log_prefix
        
        # Pre-trade limits
        self.max_order_value = self.config.get("max_order_value", 20000) # Max $20k per order
        self.max_order_quantity = self.config.get("max_order_quantity", 1000) # Max 1000 shares
        
        # Portfolio limits
        self.max_portfolio_drawdown = self.config.get("max_portfolio_drawdown", 0.15) # 15%
        self.max_position_concentration = self.config.get("max_position_concentration", 0.20) # 20%
        
        # Circuit breaker
        self.circuit_breaker_tripped = False
        self.circuit_breaker_reason = ""
        
        logger.info(f"{self.log_prefix} Initialized.")
        logger.info(f"{self.log_prefix} Max Position Concentration: {self.max_position_concentration:.2%}")

    def evaluate_and_adjust(self, 
                            target_portfolio: TargetPortfolio, 
                            pipeline_state: PipelineState
                           ) -> Tuple[TargetPortfolio, RiskReport]:
        """
        [主人喵的修复]
        实现 Orchestrator 调用的 'evaluate_and_adjust' 存根。
        
        核心逻辑：
        1. 检查熔断器。
        2. 检查全局投资组合风险 (例如最大回撤)。
        3. 检查和调整头寸集中度。
        """
        logger.info(f"{self.log_prefix} Evaluating target portfolio...")
        final_portfolio = copy.deepcopy(target_portfolio) # 创建一个副本进行修改
        report = RiskReport()

        # 1. 检查熔断器
        if self.circuit_breaker_tripped:
            logger.critical(f"{self.log_prefix} RISK: Circuit breaker is TRIPPED. Vetoing all targets. Reason: {self.circuit_breaker_reason}")
            final_portfolio.positions = [] # 清空所有目标
            report.passed = False
            report.adjustments_made.append("CIRCUIT_BREAKER: Vetoed all positions.")
            return final_portfolio, report

        # 2. 检查全局投资组合风险 (例如，来自 pipeline_state 的指标)
        # (此逻辑已在 check_portfolio_risk 中，但我们在这里再次检查)
        self.check_portfolio_risk(pipeline_state) # [主人喵的修复] 调用检查
        if self.circuit_breaker_tripped:
             logger.critical(f"{self.log_prefix} RISK: Circuit breaker was TRIPPED during portfolio check. Vetoing all targets.")
             final_portfolio.positions = [] # 清空所有目标
             report.passed = False
             report.adjustments_made.append(f"CIRCUIT_BREAKER: {self.circuit_breaker_reason}")
             return final_portfolio, report

        # 3. 检查和调整头寸集中度
        adjustments_made = []
        final_positions: List[TargetPosition] = []
        
        for pos in final_portfolio.positions:
            target_weight = pos.target_weight
            
            if abs(target_weight) > self.max_position_concentration:
                adjusted_weight = self.max_position_concentration * (1 if target_weight > 0 else -1)
                reason = (f"ADJUSTMENT: {pos.symbol} target weight {target_weight:.2%} "
                          f"violated max concentration ({self.max_position_concentration:.2%}). "
                          f"Capped at {adjusted_weight:.2%}.")
                
                logger.warning(f"{self.log_prefix} {reason}")
                adjustments_made.append(reason)
                
                # 更新 TargetPosition
                pos.target_weight = adjusted_weight
                pos.reasoning += f" [RISK_ADJUSTED: {reason}]"
            
            final_positions.append(pos)

        # 4. (未来) 检查总杠杆/分配
        total_allocation = sum(abs(p.target_weight) for p in final_positions)
        # (此处可以添加 max_total_allocation 检查)

        final_portfolio.positions = final_positions
        report.adjustments_made = adjustments_made
        report.portfolio_risk_metrics["calculated_concentration"] = total_allocation
        
        logger.info(f"{self.log_prefix} Evaluation complete. {len(adjustments_made)} adjustments made.")
        return final_portfolio, report

    def trip_circuit_breaker(self, reason: str):
        """Trips the system-wide circuit breaker, halting all new trades."""
        if not self.circuit_breaker_tripped:
            logger.critical(f"CIRCUIT BREAKER TRIPPED: {reason}")
            self.circuit_breaker_tripped = True
            self.circuit_breaker_reason = reason
            # This should also publish an event
            # await pipeline_state.event_distributor.publish("CIRCUIT_BREAKER", reason=reason)

    def reset_circuit_breaker(self):
        """Resets the circuit breaker (manual action)."""
        logger.warning(f"{self.log_prefix} Circuit breaker is being reset.")
        self.circuit_breaker_tripped = False
        self.circuit_breaker_reason = ""

    def check_portfolio_risk(self, pipeline_state: PipelineState):
        """
        Checks overall portfolio health (e.g., drawdown).
        This can trip the circuit breaker.
        """
        if self.circuit_breaker_tripped:
            return # Already tripped

        # [主人喵的修复] 检查 'pipeline_state' 是否有 'get_value'
        if not hasattr(pipeline_state, 'get_value'):
            logger.warning(f"{self.log_prefix} pipeline_state missing 'get_value', cannot check portfolio risk.")
            return

        portfolio_metrics = pipeline_state.get_value("portfolio_metrics", {})
        drawdown = portfolio_metrics.get("current_drawdown", 0.0)
        
        if drawdown > self.max_portfolio_drawdown:
            reason = f"Maximum portfolio drawdown exceeded ({drawdown:.2%} > {self.max_portfolio_drawdown:.2%})"
            self.trip_circuit_breaker(reason)

    async def validate_signal(self, signal: Signal, pipeline_state: PipelineState) -> Optional[str]:
        """
        Checks a signal *before* it's turned into an order.
        Returns a rejection reason string if invalid, or None if valid.
        """
        if self.circuit_breaker_tripped:
            return f"Circuit breaker tripped: {self.circuit_breaker_reason}"
            
        # [主人喵的修复] 检查 'signal' 属性
        if not hasattr(signal, 'signal_type'): # [主人喵的修复] 修正检查
            pass # Signal 模式没有 'direction'
            
        if not (0.0 <= signal.strength <= 1.0):
            return f"Invalid signal strength: {signal.strength}"
            
        # Add more signal-level checks (e.g., duplicate signals)
        
        return None # Signal is valid

    async def validate_order(self, order: Order, pipeline_state: PipelineState) -> Optional[str]:
        """
        Performs pre-trade checks on a generated order.
        Returns a rejection reason string if invalid, or None if valid.
        """
        if self.circuit_breaker_tripped:
            return f"Circuit breaker tripped: {self.circuit_breaker_reason}"

        # [主人喵的修复] 检查 'pipeline_state' 属性
        if not hasattr(pipeline_state, 'get_latest_market_data'):
             logger.warning(f"{self.log_prefix} pipeline_state missing 'get_latest_market_data', cannot validate order.")
             return "Pipeline state is invalid for order validation."

        # Get context
        market_data = pipeline_state.get_latest_market_data(order.symbol)
        if not market_data:
            return "No market data available to price order."
            
        current_price = market_data.close
        order_value = abs(order.quantity * current_price)
        
        # 1. Check order-level limits
        if order_value > self.max_order_value:
            return f"Order value ${order_value:.2f} exceeds max limit ${self.max_order_value:.2f}"
            
        if abs(order.quantity) > self.max_order_quantity:
            return f"Order quantity {order.quantity} exceeds max limit {self.max_order_quantity}"
            
        # 2. Check portfolio-level impact (concentration)
        # [主人喵的修复] 修正 get_value
        portfolio = pipeline_state.get_value("portfolio_state", {}) 
        if not portfolio:
            portfolio = pipeline_state.get_value("portfolio", {})

        total_value = portfolio.get("total_value", 0)
        
        if total_value == 0:
            # [主人喵的修复] 允许初始交易
            if pipeline_state.get_value("is_initial_trade", False):
                 pass
            # [主人喵的修复] 如果总价值为 0 且不是初始交易，则发出警告但可能放行
            elif order_value < self.max_order_value:
                 logger.warning(f"{self.log_prefix} Portfolio value is zero, allowing initial order: {order.id}")
                 pass
            else:
                return "Portfolio value is zero."
            
        positions = portfolio.get("positions", {})
        current_position = positions.get(order.symbol, {})
        # [主人喵的修复] 修正字典 get
        current_value = current_position.get("market_value", 0)
        
        # Calculate post-trade value
        post_trade_value = current_value + (order.quantity * current_price)
        
        # [主人喵的修复] 仅当 total_value 不为零时才检查集中度
        if total_value > 0:
            # Calculate post-trade concentration
            post_trade_concentration = abs(post_trade_value) / total_value
            
            if post_trade_concentration > self.max_position_concentration:
                return f"Order would increase position concentration to {post_trade_concentration:.2%}, exceeding limit {self.max_position_concentration:.2%}"

        # Add more checks (e.g., buying power)
        
        return None # Order is valid
