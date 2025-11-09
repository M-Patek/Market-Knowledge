from typing import List, Dict, Any, Optional
from phoenix_project.cognitive.engine import CognitiveEngine
from phoenix_project.core.schemas.risk_schema import RiskSignal, RiskSignalType
from phoenix_project.monitor.logging import get_logger

log = get_logger("RiskManager")


class RiskManager:
    """
    认知风险管理器。
    负责评估系统状态、投资组合和市场数据，以识别和量化风险。
    """

    def __init__(self, cognitive_engine: "CognitiveEngine"):
        self.cognitive_engine = cognitive_engine
        self.config = cognitive_engine.config.get("risk_manager", {})
        
        # [✅ 优化] 从配置加载阈值
        self.stop_loss_threshold = self.config.get("stop_loss_threshold", 0.05)  # 5% 止损
        self.max_drawdown_threshold = self.config.get("max_drawdown_threshold", 0.15) # 15% 最大回撤
        self.volatility_threshold = self.config.get("volatility_threshold", 0.03) # 3% 日波动率
        self.concentration_threshold = self.config.get("concentration_threshold", 0.25) # 25% 集中度

        # [✅ 优化] 持久化峰值权益
        # 
        # 
        try:
            initial_capital = self.cognitive_engine.pipeline_state.portfolio.initial_capital
        except AttributeError:
            log.warning("Could not find initial_capital in pipeline_state. Defaulting peak_equity to 0.")
            initial_capital = 0.0
            
        self.peak_equity = initial_capital
        
        log.info(f"RiskManager initialized. StopLoss: {self.stop_loss_threshold}, "
                   f"MaxDrawdown: {self.max_drawdown_threshold}, Initial Peak Equity: {self.peak_equity}")

    def assess_risk(self) -> List[RiskSignal]:
        """
        运行所有风险检查并返回发现的风险信号。
        """
        log.debug("Running risk assessment...")
        signals = []
        
        try:
            portfolio = self.cognitive_engine.pipeline_state.portfolio
            market_data = self.cognitive_engine.pipeline_state.latest_market_data
            
            if not portfolio or not market_data:
                log.warning("Portfolio or Market Data not available. Skipping risk assessment.")
                return []

            # 
            current_equity = portfolio.total_equity

            signals.extend(self.check_stop_loss(portfolio, market_data))
            signals.extend(self.check_max_drawdown(current_equity))
            signals.extend(self.check_concentration(portfolio))
            signals.extend(self.check_volatility(market_data))
            
            if signals:
                log.warning(f"Risk assessment generated {len(signals)} signals.")
            else:
                log.debug("Risk assessment complete. No immediate risks detected.")

        except Exception as e:
            log.error(f"Error during risk assessment: {e}", exc_info=True)
            signals.append(
                RiskSignal(
                    type=RiskSignalType.SYSTEM_ERROR,
                    message=f"Risk assessment failed: {e}",
                    level=10,
                )
            )
            
        return signals

    def check_stop_loss(self, portfolio: Any, market_data: Dict[str, Any]) -> List[RiskSignal]:
        """
        [✅ 优化] 检查止损 (Stop-Loss)。
        遍历所有持仓，检查是否达到了止损阈值。
        """
        signals = []
        if not hasattr(portfolio, 'positions') or not portfolio.positions:
            return signals

        log.debug(f"Checking stop-loss for {len(portfolio.positions)} positions...")
        
        for position in portfolio.positions:
            try:
                symbol = position.symbol
                
                # 
                if not hasattr(position, 'entry_price') or position.entry_price <= 0:
                    log.warning(f"Position {symbol} missing valid entry_price. Skipping stop-loss check.")
                    continue
                    
                if symbol not in market_data or 'price' not in market_data[symbol]:
                    log.warning(f"No current market price for {symbol}. Skipping stop-loss check.")
                    continue

                current_price = market_data[symbol]['price']
                entry_price = position.entry_price
                
                # 
                loss_percent = (entry_price - current_price) / entry_price
                
                # 
                if loss_percent > self.stop_loss_threshold:
                    log.warning(f"STOP-LOSS triggered for {symbol}. Loss: {loss_percent:.2%}")
                    signals.append(
                        RiskSignal(
                            type=RiskSignalType.STOP_LOSS,
                            message=f"Stop-loss triggered for {symbol}. "
                                    f"Current Loss: {loss_percent:.2%} "
                                    f"(Entry: {entry_price}, Current: {current_price})",
                            level=8,
                            affected_symbols=[symbol],
                        )
                    )
            except Exception as e:
                log.error(f"Error checking stop-loss for position {getattr(position, 'symbol', 'UNKNOWN')}: {e}")

        return signals


    def check_max_drawdown(self, current_equity: float) -> List[RiskSignal]:
        """
        [✅ 优化] 检查最大回撤 (Max Drawdown)。
        使用持久化的 peak_equity。
        """
        signals = []
        
        # 
        self.peak_equity = max(self.peak_equity, current_equity)
        
        if self.peak_equity == 0:
             log.debug("Peak equity is 0, skipping drawdown check.")
             return signals # 

        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        log.debug(f"Drawdown check: Current={current_equity}, Peak={self.peak_equity}, Drawdown={drawdown:.2%}")

        if drawdown > self.max_drawdown_threshold:
            log.critical(f"MAX DRAWDOWN breached. Drawdown: {drawdown:.2%}")
            signals.append(
                RiskSignal(
                    type=RiskSignalType.MAX_DRAWDOWN,
                    message=f"Max drawdown threshold breached. "
                            f"Current Drawdown: {drawdown:.2%} "
                            f"(Peak Equity: {self.peak_equity}, Current Equity: {current_equity})",
                    level=10, # 
                )
            )
        return signals

    def check_concentration(self, portfolio: Any) -> List[RiskSignal]:
        """
        检查投资组合集中度。
        """
        signals = []
        if not hasattr(portfolio, 'positions_value') or not portfolio.positions_value:
            return signals

        total_value = portfolio.total_equity
        if total_value == 0:
            return signals

        for symbol, value in portfolio.positions_value.items():
            concentration = value / total_value
            if concentration > self.concentration_threshold:
                log.warning(f"High concentration risk for {symbol}. Concentration: {concentration:.2%}")
                signals.append(
                    RiskSignal(
                        type=RiskSignalType.CONCENTRATION,
                        message=f"High portfolio concentration in {symbol}. "
                                f"Concentration: {concentration:.2%}",
                        level=6,
                        affected_symbols=[symbol],
                    )
                )
        return signals

    def check_volatility(self, market_data: Dict[str, Any]) -> List[RiskSignal]:
        """
        检查市场波动性 (简化)。
        """
        # 
        # 
        signals = []
        for symbol, data in market_data.items():
            if "change_pct" in data and abs(data["change_pct"]) > self.volatility_threshold:
                log.warning(f"High volatility detected for {symbol}. Change: {data['change_pct']:.2%}")
                signals.append(
                    RiskSignal(
                        type=RiskSignalType.VOLATILITY,
                        message=f"High volatility detected in {symbol}. "
                                f"Daily Change: {data['change_pct']:.2%}",
                        level=5,
                        affected_symbols=[symbol],
                    )
                )
        return signals
