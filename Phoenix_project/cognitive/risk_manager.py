import pandas as pd
from typing import Dict, Any, Optional

from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.execution.signal_protocol import Signal
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.core.schemas.risk_schema import RiskDecision

logger = get_logger(__name__)

class RiskManager:
    """
    Applies pre-trade risk controls and circuit breakers.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the RiskManager.
        
        Args:
            config: The strategy configuration.
        """
        self.config = {}
        self.set_config(config) # Apply initial config
        
        # 跟踪已触发的断路器
        self.circuit_breakers_tripped: Dict[str, bool] = {}
        
        # [✅ 优化] 存储用于最大回撤的状态
        self._peak_portfolio_value: float = -1.0
        self._current_portfolio_value: float = 0.0
        
        logger.info("RiskManager initialized.")

    def set_config(self, config: Dict[str, Any]):
        """
        Dynamically updates the component's configuration.
        """
        self.config = config.get('risk_manager', {})
        logger.info(f"RiskManager config set: {self.config}")

    def _get_current_portfolio_value(self, state: 'CurrentState') -> float:
        """ 辅助方法：计算当前总权益 """
        balance = state.get_balance()
        all_holdings = state.get_all_holdings()
        market_data = state.get_all_market_data()
        
        total_equity = balance
        
        for sym, qty in all_holdings.items():
            price = market_data.get(sym, {}).get('close')
            if price is not None and price > 0:
                position_value = qty * price
                total_equity += position_value
        
        return total_equity

    def evaluate_and_adjust(
        self, 
        signal: Signal, 
        pipeline_state: PipelineState
    ) -> Signal:
        """
        Evaluates a signal against risk rules and adjusts it (e.g., veto).
        
        [任务 C.3] TODO: Implement checks for total leverage/allocation.
        """
        
        # 如果任何断路器被触发，否决所有新信号
        if any(self.circuit_breakers_tripped.values()):
            logger.critical(f"CIRCUIT BREAKER TRIPPED. Rejecting all new signals. Breakers: {self.circuit_breakers_tripped}")
            signal.quantity = 0.0
            signal.metadata['risk_veto'] = "CircuitBreakerTripped"
            return signal

        symbol = signal.symbol
        
        try:
            current_state = pipeline_state.get_current_state()
            all_holdings = current_state.get_all_holdings()
            balance = current_state.get_balance()
            market_data = current_state.get_all_market_data()
            
            # (计算总权益)
            self._current_portfolio_value = self._get_current_portfolio_value(current_state)
            total_equity = self._current_portfolio_value
            if total_equity <= 0:
                logger.critical(f"Total equity is {total_equity:.2f}. Rejecting signal.")
                signal.quantity = 0.0
                signal.metadata['risk_veto'] = "ZeroOrNegativeEquity"
                return signal
                
            # (计算总毛曝险和杠杆)
            total_gross_exposure = 0.0
            for sym, qty in all_holdings.items():
                price = market_data.get(sym, {}).get('close')
                if price is not None and price > 0:
                    total_gross_exposure += abs(qty * price)

            current_leverage = total_gross_exposure / total_equity
            max_leverage = self.config.get('max_total_leverage', 3.0)

            # 1. [任务 C.3] 检查总杠杆
            if current_leverage > max_leverage:
                logger.warning(
                    f"Risk VETO check: Current leverage ({current_leverage:.2f}x) exceeds limit ({max_leverage}x)."
                )
                
                # 杠杆已超限。否决任何 *增加* 风险的交易
                current_qty = all_holdings.get(signal.symbol, 0.0)
                current_price = market_data.get(signal.symbol, {}).get('close', 0)
                new_qty = current_qty + signal.quantity
                
                # 检查信号是否在增加绝对仓位价值
                if abs(new_qty * current_price) > abs(current_qty * current_price):
                    logger.warning(
                        f"Rejecting risk-increasing signal for {signal.symbol} due to max leverage breach."
                    )
                    signal.quantity = 0.0 # 否决
                    signal.metadata['risk_veto'] = "MaxLeverageBreached"
                    return signal # 提前返回

        except Exception as e:
            logger.error(f"Error during C.3 leverage check: {e}", exc_info=True)
            # 安全起见，如果风险检查失败，否决交易
            signal.quantity = 0.0
            signal.metadata['risk_veto'] = "RiskCheckFailed"
            return signal

        # 2. [✅ 优化] 检查止损 (Stop-Loss)
        # FIXME: 未实现。
        # 要实现基于价格的止损，PipelineState 需要跟踪每个仓位的
        # 平均入场价格 (average entry price)。
        # 目前 `get_all_holdings()` 只有一个 `qty` 字典。
        logger.warning("FIXME: Stop-Loss check not implemented (requires position entry price in PipelineState).")
        
        # 3. [✅ 优化] 检查最大仓位限制
        try:
            max_position_pct = self.config.get('max_position_pct', 0.10) # 占总权益的 10%
            current_price = market_data.get(symbol, {}).get('close')
            
            if current_price is not None and current_price > 0:
                current_qty = all_holdings.get(symbol, 0.0)
                new_qty = current_qty + signal.quantity
                new_position_value = abs(new_qty * current_price)
                
                position_limit_value = total_equity * max_position_pct
                
                if new_position_value > position_limit_value:
                    logger.warning(
                        f"Risk VETO check: New position value for {symbol} ({new_position_value:.2f}) "
                        f"exceeds limit ({position_limit_value:.2f} / {max_position_pct*100}% of equity)."
                    )
                    
                    # 调整信号大小以匹配限制，而不是完全否决
                    allowed_qty_abs = position_limit_value / current_price
                    # 保持信号方向
                    allowed_qty = allowed_qty_abs if new_qty > 0 else -allowed_qty_abs
                    
                    # 计算调整后的信号数量
                    adjusted_signal_qty = allowed_qty - current_qty
                    
                    # 确保我们不会因为调整而翻转方向
                    if (signal.quantity > 0 and adjusted_signal_qty < 0) or \
                       (signal.quantity < 0 and adjusted_signal_qty > 0):
                        signal.quantity = 0.0 # 不允许增加仓位
                    else:
                        signal.quantity = adjusted_signal_qty
                        
                    signal.metadata['risk_veto'] = "MaxPositionSizeBreached"
                    logger.info(f"Signal for {symbol} quantity adjusted to {signal.quantity:.4f} to respect max position size.")
                    
            else:
                logger.warning(f"Cannot perform max position check for {symbol}: No price data.")

        except Exception as e:
            logger.error(f"Error during Max Position check: {e}", exc_info=True)
            signal.quantity = 0.0
            signal.metadata['risk_veto'] = "RiskCheckFailed"
            return signal


        logger.debug(f"Signal for {symbol} (Qty: {signal.quantity}) passed risk evaluation.")
        return signal

    def check_circuit_breakers(self, pipeline_state: PipelineState) -> Optional[RiskDecision]:
        """
        Checks for portfolio-level circuit breakers (e.g., max drawdown).
        """
        
        # [✅ 优化] 检查最大回撤 (Max Drawdown)
        # FIXME: 这是一个简化的实现。
        # 'pipeline_state' 应该在每个周期结束时调用此方法来更新 PnL。
        # 峰值权益应该被持久化 (e.g., in Redis or DB) 以便在重启后幸存。
        
        try:
            # 1. 更新当前权益 (已在 evaluate_and_adjust 中计算, 但这里再次计算以保持独立)
            current_state = pipeline_state.get_current_state()
            self._current_portfolio_value = self._get_current_portfolio_value(current_state)
            
            # 2. 更新峰值权益
            if self._current_portfolio_value > self._peak_portfolio_value:
                self._peak_portfolio_value = self._current_portfolio_value
                logger.info(f"New peak portfolio value reached: {self._peak_portfolio_value:.2f}")

            # 3. 检查回撤
            max_drawdown_limit = self.config.get('max_drawdown_pct', 0.20) # 20%
            
            if self._peak_portfolio_value > 0: # 避免在开始时除以零
                current_drawdown = (self._peak_portfolio_value - self._current_portfolio_value) / self._peak_portfolio_value
                
                if current_drawdown > max_drawdown_limit:
                    logger.critical(
                        f"CIRCUIT BREAKER: Max drawdown {current_drawdown*100:.2f}% breached limit {max_drawdown_limit*100:.2f}%. "
                        f"(Peak: {self._peak_portfolio_value:.2f}, Current: {self._current_portfolio_value:.2f})"
                    )
                    self.circuit_breakers_tripped['max_drawdown'] = True
                    return RiskDecision(
                        decision="HALT_TRADING",
                        reason=f"Max drawdown breached: {current_drawdown*100:.2f}%"
                    )
            
        except Exception as e:
            logger.error(f"Error during Max Drawdown check: {e}", exc_info=True)
            # 安全起见，触发断路器
            self.circuit_breakers_tripped['drawdown_check_failed'] = True
            return RiskDecision(
                decision="HALT_TRADING",
                reason=f"Failed to check max drawdown: {e}"
            )
            
        return None

    async def trip_system_circuit_breaker(self, reason: str):
        """
        [✅ 新增] 由 ErrorHandler 调用的外部触发器，用于停止系统。
        """
        if not self.circuit_breakers_tripped.get('system_error'):
            logger.critical(f"SYSTEM CIRCUIT BREAKER TRIPPED by ErrorHandler: {reason}")
            self.circuit_breakers_tripped['system_error'] = True
            # 在真实的系统中，这会：
            # 1. 向 ContextBus 广播 "HALT_TRADING" 事件
            # 2. 触发 Execution (OrderManager) 清算所有仓位
            # 3. 停止 LoopManager
            # 目前，我们只设置标志并记录。
