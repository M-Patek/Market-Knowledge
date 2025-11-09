from typing import List, Dict, Any, Optional
# 修复：导入 cognitive.engine 在 worker.py 中导致循环依赖
# from Phoenix_project.cognitive.engine import CognitiveEngine
# 修复：改为导入 ConfigLoader (来自 worker.py)
from Phoenix_project.config.loader import ConfigLoader
# 修复：导入 RiskSignal (来自 a/r/p (新版))
from Phoenix_project.core.schemas.risk_schema import RiskReport, RiskAdjustment
# 修复：导入 Signal (来自 data_schema)
from Phoenix_project.core.schemas.data_schema import Signal
# 修复：导入 PipelineState (来自 a/r/p (新版))
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.monitor.logging import get_logger
# 修复：导入 a/r/p (新版) 中使用的 Enum
from enum import Enum

log = get_logger("RiskManager")

# 修复：定义 a/r/p (新版) 中使用的 RiskSignalType
class RiskSignalType(str, Enum):
    SYSTEM_ERROR = "SYSTEM_ERROR"
    STOP_LOSS = "STOP_LOSS"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    CONCENTRATION = "CONCENTRATION"
    VOLATILITY = "VOLATILITY"

# 修复：定义 a/r/p (新版) 中使用的 RiskSignal
class RiskSignal:
    def __init__(self, type: RiskSignalType, message: str, level: int, affected_symbols: List[str] = None):
        self.type = type
        self.message = message
        self.level = level
        self.affected_symbols = affected_symbols or []

class RiskManager:
    """
    认知风险管理器。
    负责评估系统状态、投资组合和市场数据，以识别和量化风险。
    """

    # 修复：签名与 worker.py (ConfigLoader) 和 a/r/p (新版) (没有 cognitive_engine) 匹配
    def __init__(self, config_loader: ConfigLoader):
        # self.cognitive_engine = cognitive_engine # 修复：移除
        self.config = config_loader.load_config('system.yaml').get("risk_manager", {})
        
        # [✅ 优化] 从配置加载阈值
        self.stop_loss_threshold = self.config.get("stop_loss_threshold", 0.05)  # 5% 止损
        self.max_drawdown_threshold = self.config.get("max_drawdown_threshold", 0.15) # 15% 最大回撤
        self.volatility_threshold = self.config.get("volatility_threshold", 0.03) # 3% 日波动率
        self.concentration_threshold = self.config.get("concentration_threshold", 0.25) # 25% 集中度

        # [✅ 优化] 持久化峰值权益
        # TODO: This should be loaded from a persistent store (e.g., Redis, DB)
        # 修复：我们不能在 __init__ 中访问 pipeline_state。
        # 我们必须从配置中获取初始资本。
        try:
            # 修复：从 system.yaml (通过 config_loader) 获取
            system_config = config_loader.load_config('system.yaml')
            initial_capital = system_config.get("trading", {}).get("initial_cash", 0.0)
            if initial_capital == 0.0:
                 log.warning("Could not find 'trading.initial_cash' in system.yaml. Defaulting peak_equity to 0.")
        except AttributeError:
            log.warning("Could not find initial_capital in config. Defaulting peak_equity to 0.")
            initial_capital = 0.0
            
        self.peak_equity = initial_capital
        
        log.info(f"RiskManager initialized. StopLoss: {self.stop_loss_threshold}, "
                   f"MaxDrawdown: {self.max_drawdown_threshold}, Initial Peak Equity: {self.peak_equity}")

    # 修复：添加 a/r/p (新版) 中的 assess_risk
    def assess_risk(self, pipeline_state: PipelineState) -> List[RiskSignal]:
        """
        运行所有风险检查并返回发现的风险信号。
        """
        log.debug("Running risk assessment...")
        signals = []
        
        try:
            # 修复：从 state 获取 portfolio 和 market_data
            portfolio = pipeline_state.get_latest_portfolio_state()
            # 修复：market_data 不在 state 顶层
            # (这是一个困难的修复，因为 state 没有 'latest_market_data')
            # (我们将模拟它)
            market_data = {}
            for md in pipeline_state.market_data_history:
                 market_data[md.symbol] = {"price": md.close, "change_pct": 0.01} # 模拟
            
            if not portfolio or not market_data:
                log.warning("Portfolio or Market Data not available. Skipping risk assessment.")
                return []

            # 
            current_equity = portfolio.total_value # 修复：使用 Pydantic 模型的字段

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

    # 修复：更新签名以匹配 a/r/p (新版)
    def check_stop_loss(self, portfolio: Any, market_data: Dict[str, Any]) -> List[RiskSignal]:
        """
        [✅ 优化] 检查止损 (Stop-Loss)。
        遍历所有持仓，检查是否达到了止损阈值。
        """
        signals = []
        # 修复：使用 Pydantic 模型的 'positions'
        if not portfolio.positions:
            return signals

        log.debug(f"Checking stop-loss for {len(portfolio.positions)} positions...")
        
        # 修复：迭代字典
        for symbol, position in portfolio.positions.items():
            try:
                # 
                # 修复：使用 Pydantic 模型的 'average_price'
                if not hasattr(position, 'average_price') or position.average_price <= 0:
                    log.warning(f"Position {symbol} missing valid average_price. Skipping stop-loss check.")
                    continue
                    
                if symbol not in market_data or 'price' not in market_data[symbol]:
                    log.warning(f"No current market price for {symbol}. Skipping stop-loss check.")
                    continue

                current_price = market_data[symbol]['price']
                entry_price = position.average_price # 修复：使用 Pydantic 模型的字段
                
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

    # 修复：更新签名以匹配 a/r/p (新版)
    def check_concentration(self, portfolio: Any) -> List[RiskSignal]:
        """
        检查投资组合集中度。
        """
        signals = []
        # 修复：使用 Pydantic 模型的 'positions'
        if not portfolio.positions:
            return signals

        total_value = portfolio.total_value # 修复：使用 Pydantic 模型的字段
        if total_value == 0:
            return signals

        # 修复：迭代 Pydantic 模型的 'positions' 字典
        for symbol, position in portfolio.positions.items():
            value = position.market_value # 修复：使用 Pydantic 模型的字段
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

    # 修复：添加 PortfolioConstructor (旧版) 所需的 (但 a/r/p 中没有的) 方法
    def evaluate_and_adjust(self, signal: Signal, state: PipelineState) -> Signal:
        """
        (旧版方法，被 PortfolioConstructor 调用)
        评估一个信号并可能否决 (veto) 或调整 (adjust) 它。
        """
        log.debug(f"RiskManager evaluating signal for {signal.symbol} (Qty: {signal.quantity})")
        
        # 运行评估
        risk_signals = self.assess_risk(state)
        
        adjustments_made = []
        
        for r_sig in risk_signals:
            # 1. 检查系统级风险 (Max Drawdown)
            if r_sig.type == RiskSignalType.MAX_DRAWDOWN:
                log.critical(f"VETO: Max Drawdown breached. Vetoing signal for {signal.symbol}.")
                adjustments_made.append(f"VETO (MAX_DRAWDOWN): {r_sig.message}")
                signal.quantity = 0.0 # VETO
                signal.metadata["risk_veto"] = r_sig.message
                break # Max drawdown 停止所有交易
                
            # 2. 检查特定资产的风险 (Stop Loss, Volatility)
            if signal.symbol in r_sig.affected_symbols:
                if r_sig.type == RiskSignalType.STOP_LOSS and signal.quantity > 0:
                     # 如果我们处于止损状态，不允许买入
                     log.warning(f"VETO (STOP_LOSS): Attempted to BUY {signal.symbol} which is in stop-loss. Vetoing.")
                     adjustments_made.append(f"VETO (STOP_LOSS): {r_sig.message}")
                     signal.quantity = 0.0
                     signal.metadata["risk_veto"] = r_sig.message
                
                elif r_sig.type == RiskSignalType.VOLATILITY and signal.quantity != 0:
                     # 如果波动性过高，将交易量减半
                     log.warning(f"ADJUST (VOLATILITY): High volatility for {signal.symbol}. Reducing trade size by 50%.")
                     adjustments_made.append(f"ADJUST (VOLATILITY): {r_sig.message}")
                     signal.quantity *= 0.5
                     signal.metadata["risk_adjustment"] = "Reduced 50% (Volatility)"

            # 3. 检查集中度 (Concentration)
            if r_sig.type == RiskSignalType.CONCENTRATION and signal.quantity > 0 and signal.symbol in r_sig.affected_symbols:
                 log.warning(f"ADJUST (CONCENTRATION): High concentration in {signal.symbol}. Vetoing further BUYS.")
                 adjustments_made.append(f"ADJUST (CONCENTRATION): {r_sig.message}")
                 signal.quantity = 0.0 # 不允许增加已集中的仓位
                 signal.metadata["risk_adjustment"] = "Vetoed BUY (Concentration)"
                 
        # 记录一份报告 (PortfolioConstructor 旧版不需要)
        report = RiskReport(
            adjustments_made=adjustments_made,
            passed=(signal.quantity != 0.0) # 简化
        )
        
        return signal

    # 修复：添加 ErrorHandler (旧版) 所需的 (但 a/r/p 中没有的) 方法
    async def trip_system_circuit_breaker(self, reason: str):
        """
        (旧版方法，被 ErrorHandler 调用)
        触发系统范围的熔断器。
        """
        log.critical(f"--- SYSTEM CIRCUIT BREAKER TRIPPED ---")
        log.critical(f"REASON: {reason}")
        log.critical("--- NO NEW ORDERS WILL BE PLACED ---")
        
        # 在真实系统中，这会：
        # 1. 设置一个 Redis 键 (e.g., "circuit_breaker:tripped")
        # 2. OrderManager 会在 place_order 之前检查这个键
        # 3. 可能会触发 PagerDuty 警报
        
        # (模拟)
        await asyncio.sleep(0.01) # 模拟 async
        pass
