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
        
        # TBD: Add state for circuit breakers
        self.circuit_breakers_tripped: Dict[str, bool] = {}
        
        logger.info("RiskManager initialized.")

    def set_config(self, config: Dict[str, Any]):
        """
        Dynamically updates the component's configuration.
        """
        self.config = config.get('risk_manager', {})
        logger.info(f"RiskManager config set: {self.config}")

    def evaluate_and_adjust(
        self, 
        signal: Signal, 
        pipeline_state: PipelineState
    ) -> Signal:
        """
        Evaluates a signal against risk rules and adjusts it (e.g., veto).
        
        [任务 C.3] TODO: Implement checks for total leverage/allocation.
        """
        
        symbol = signal.symbol
        
        # 1. [任务 C.3] 检查总杠杆
        try:
            current_state = pipeline_state.get_current_state()
            all_holdings = current_state.get_all_holdings()
            balance = current_state.get_balance()
            market_data = current_state.get_all_market_data()
            
            total_equity = balance
            total_gross_exposure = 0.0

            for sym, qty in all_holdings.items():
                price = market_data.get(sym, {}).get('close')
                if price is not None and price > 0:
                    position_value = qty * price
                    total_equity += position_value
                    total_gross_exposure += abs(position_value)
            
            # (处理总权益为 0 或负的情况)
            if total_equity <= 1.0: # 避免除以零
                logger.critical(f"Total equity is {total_equity:.2f}. Triggering leverage circuit breaker.")
                current_leverage = float('inf')
            else:
                current_leverage = total_gross_exposure / total_equity
            
            max_leverage = self.config.get('max_total_leverage', 3.0)

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

        # 2. TBD: 检查止损 (Stop-Loss)
        # ...
        
        # 3. TBD: 检查最大仓位限制
        # ...

        logger.debug(f"Signal for {symbol} (Qty: {signal.quantity}) passed risk evaluation.")
        return signal

    def check_circuit_breakers(self, pipeline_state: PipelineState) -> Optional[RiskDecision]:
        """
        Checks for portfolio-level circuit breakers (e.g., max drawdown).
        """
        # TBD: Implement logic to check portfolio PnL against max drawdown limits
        
        # max_drawdown_limit = self.config.get('max_drawdown_pct', 0.20)
        # current_drawdown = ... (get from pipeline_state/portfolio)
        
        # if current_drawdown > max_drawdown_limit:
        #     logger.critical(f"CIRCUIT BREAKER: Max drawdown {current_drawdown} breached limit {max_drawdown_limit}.")
        #     self.circuit_breakers_tripped['max_drawdown'] = True
        #     return RiskDecision(
        #         decision="HALT_TRADING",
        #         reason=f"Max drawdown breached: {current_drawdown}"
        #     )
            
        return None
