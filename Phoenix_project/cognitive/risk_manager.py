"""
Risk Manager for the Phoenix cognitive engine.

Monitors portfolio state, market conditions, and operational metrics
to enforce risk controls and trigger circuit breakers.
[Task 5] Persistence and Warm-up
"""

import logging
from typing import Any, Dict, Optional, List
from collections import deque
from datetime import datetime, timedelta
import asyncio
import numpy as np 
import redis  # type: ignore

from ..core.schemas.data_schema import MarketData, Position, Portfolio
from ..core.schemas.risk_schema import (
    RiskSignal,
    SignalType,
    RiskParameter,
    VolatilitySignal,
    DrawdownSignal,
    ConcentrationSignal,
)
from ..core.exceptions import RiskViolationError, CircuitBreakerError

logger = logging.getLogger(__name__)

# Forward declaration or import
from ..data_manager import DataManager


class RiskManager:
    """
    Manages risk for the trading system.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        redis_client: redis.Redis,
        data_manager: Optional[DataManager] = None, # [Task 5] Inject DataManager
        initial_capital: float = 100000.0,
    ):
        """
        Initializes the RiskManager.

        Args:
            config: Configuration dictionary for risk parameters.
            redis_client: Client for persistent state (e.g., peak_equity).
            data_manager: DataManager instance for warm-up.
            initial_capital: The starting capital for the portfolio.
        """
        self.config = config.get("risk_manager", {})
        self.redis_client = redis_client
        self.data_manager = data_manager
        self.initial_capital = initial_capital
        
        # 风险参数
        self.max_drawdown_pct = self.config.get("max_drawdown_pct", 0.15)
        self.max_position_concentration_pct = self.config.get(
            "max_position_concentration_pct", 0.20
        )
        self.volatility_threshold = self.config.get("volatility_threshold", 0.05)
        self.volatility_window = self.config.get("volatility_window", 30) # 用于计算标准差的窗口大小

        # 内部状态
        self.current_equity: float = initial_capital
        # [Task 5] Load persistent circuit breaker state
        self.circuit_breaker_tripped: bool = self._load_circuit_breaker_state()
        self.active_signals: List[RiskSignal] = []

        # OPTIMIZED: 从 Redis 加载持久化的峰值权益 (peak_equity)
        # 替换原有的 TODO
        self.peak_equity: float = self._load_peak_equity()

        # OPTIMIZED: 为波动性计算保留价格历史
        # 使用 deque 实现一个高效的滚动窗口
        self.price_history: Dict[str, deque] = {} # Key: symbol, Value: deque of prices

        logger.info("RiskManager initialized.")
        logger.info(f"Loaded Peak Equity: {self.peak_equity}")
        logger.info(f"Circuit Breaker State: {'TRIPPED' if self.circuit_breaker_tripped else 'OK'}")
        logger.info(f"Max Drawdown: {self.max_drawdown_pct * 100}%")
        logger.info(f"Max Concentration: {self.max_position_concentration_pct * 100}%")

    def _load_circuit_breaker_state(self) -> bool:
        """[Task 5] Loads circuit breaker state from Redis."""
        try:
            state = self.redis_client.get("phoenix:risk:halted")
            return bool(int(state)) if state else False
        except Exception as e:
            logger.error(f"Failed to load circuit breaker state: {e}")
            return False

    async def initialize(self, symbols: List[str]):
        """
        [Task 5] Cold-start Warm-up: Backfill price history for volatility checks.
        """
        if not self.data_manager:
            logger.warning("DataManager not provided. Skipping RiskManager warm-up.")
            return

        logger.info(f"Warming up RiskManager for {len(symbols)} symbols...")
        end_time = self.data_manager.get_current_time()
        # Fetch extra buffer to ensure we have enough data points (skipping holidays/weekends)
        start_time = end_time - timedelta(days=int(self.volatility_window * 2) + 5)

        async def _fetch_and_fill(sym):
            try:
                # Use standard historical fetch (respects simulation time)
                df = await self.data_manager.get_market_data_history(sym, start_time, end_time)
                if df is not None and not df.empty:
                    # Take the last N close prices
                    closes = df['close'].values[-self.volatility_window:].tolist()
                    
                    if sym not in self.price_history:
                        self.price_history[sym] = deque(maxlen=self.volatility_window)
                    
                    self.price_history[sym].extend(closes)
            except Exception as e:
                logger.error(f"Failed to warm up risk data for {sym}: {e}")

        await asyncio.gather(*[_fetch_and_fill(s) for s in symbols])
        logger.info(f"RiskManager warm-up complete.")

    def _load_peak_equity(self) -> float:
        """
        OPTIMIZED: Loads the peak equity from persistent storage (Redis).
        """
        try:
            peak_equity_raw = self.redis_client.get("phoenix:risk:peak_equity")
            if peak_equity_raw:
                peak_equity = float(peak_equity_raw)
                logger.info(f"Loaded peak_equity from Redis: {peak_equity}")
                return max(peak_equity, self.initial_capital)
            else:
                logger.info("No peak_equity found in Redis. Using initial_capital.")
                return self.initial_capital
        except Exception as e:
            logger.error(f"Failed to load peak_equity from Redis: {e}. Defaulting.")
            return self.initial_capital

    def _save_peak_equity(self):
        """
        OPTIMIZED: Saves the current peak equity to persistent storage (Redis).
        """
        try:
            self.redis_client.set("phoenix:risk:peak_equity", self.peak_equity)
        except Exception as e:
            logger.error(f"Failed to save peak_equity to Redis: {e}")

    def check_pre_trade(
        self, proposed_position: Position, portfolio: Portfolio
    ) -> List[RiskSignal]:
        """
        Checks risk before a new trade is executed.
        """
        if self.circuit_breaker_tripped:
            logger.error("CIRCUIT BREAKER TRIPPED. Pre-trade check failed.")
            raise CircuitBreakerError(
                "Risk circuit breaker is active. No new trades allowed."
            )

        signals = []

        # 1. 检查集中度
        conc_signal = self.check_concentration(proposed_position, portfolio)
        if conc_signal:
            signals.append(conc_signal)

        # 可以在此处添加其他预交易检查 (例如流动性、杠杆等)

        if signals:
            logger.warning(
                f"Pre-trade check failed for {proposed_position.symbol}: {signals}"
            )
            # 抛出异常以阻止交易
            raise RiskViolationError(
                f"Pre-trade risk violation: {signals[0].description}",
                signals
            )

        logger.debug(f"Pre-trade check passed for {proposed_position.symbol}")
        return signals

    def check_post_trade(self, portfolio: Portfolio) -> List[RiskSignal]:
        """
        Checks risk after trades have been executed and the portfolio updated.
        """
        if self.circuit_breaker_tripped:
            return self.active_signals

        self.active_signals = []

        # 1. 更新权益并检查回撤
        self.update_portfolio_value(portfolio.total_value)
        drawdown_signal = self.check_drawdown()
        if drawdown_signal:
            self.active_signals.append(drawdown_signal)
        
        if self.active_signals:
            logger.critical(
                f"Post-trade risk violations detected: {self.active_signals}"
            )
            if any(s.triggers_circuit_breaker for s in self.active_signals):
                self.trip_circuit_breaker(
                    f"Violation: {self.active_signals[0].description}"
                )

        return self.active_signals

    def on_market_data(self, market_data: MarketData) -> Optional[RiskSignal]:
        """
        Processes real-time market data to check for dynamic risks.
        """
        if self.circuit_breaker_tripped:
            return None
        
        # 检查波动性
        vol_signal = self.check_volatility(market_data)
        if vol_signal:
            logger.warning(f"High volatility detected: {vol_signal.description}")
            # 可以在此处决定是否将波动性信号添加到 active_signals
            # 或触发熔断
            if vol_signal.triggers_circuit_breaker:
                self.trip_circuit_breaker(vol_signal.description)
            return vol_signal
        
        return None

    def update_portfolio_value(self, new_equity: float):
        """
        Updates the current equity and peak equity.
        """
        self.current_equity = new_equity
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
            # OPTIMIZED: 持久化更新后的 peak_equity
            self._save_peak_equity()
        
        logger.debug(
            f"Equity updated: Current={self.current_equity}, Peak={self.peak_equity}"
        )

    def check_drawdown(self) -> Optional[DrawdownSignal]:
        """
        Checks for a violation of the maximum drawdown limit.
        """
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        
        if drawdown > self.max_drawdown_pct:
            desc = (
                f"Maximum drawdown exceeded: {drawdown*100:.2f}% "
                f"(Limit: {self.max_drawdown_pct*100:.2f}%)"
            )
            logger.critical(desc)
            return DrawdownSignal(
                description=desc,
                current_drawdown=drawdown,
                max_drawdown=self.max_drawdown_pct,
                triggers_circuit_breaker=True,
            )
        return None

    def check_concentration(
        self, proposed_position: Position, portfolio: Portfolio
    ) -> Optional[ConcentrationSignal]:
        """
        Checks if a new position would violate concentration limits.
        """
        symbol = proposed_position.symbol
        # 假设 proposed_position.market_value 是新头寸的名义价值
        new_position_value = proposed_position.market_value
        
        # 查找现有头寸 (如果存在)
        existing_value = 0.0
        if symbol in portfolio.positions:
            existing_value = portfolio.positions[symbol].market_value

        # 计算新头寸之后的总价值
        # (注意: 假设 'market_value' 为正)
        final_position_value = existing_value + new_position_value
        
        # 假设投资组合总价值 *不会* 因为这个新头寸而立即改变
        total_portfolio_value = portfolio.total_value

        if total_portfolio_value == 0:
            return None # 避免除以零

        concentration = final_position_value / total_portfolio_value
        
        if concentration > self.max_position_concentration_pct:
            desc = (
                f"Position concentration limit exceeded for {symbol}: "
                f"{concentration*100:.2f}% "
                f"(Limit: {self.max_position_concentration_pct*100:.2f}%)"
            )
            logger.warning(desc)
            return ConcentrationSignal(
                description=desc,
                symbol=symbol,
                current_concentration=concentration,
                max_concentration=self.max_position_concentration_pct,
            )
        return None

    def check_volatility(self, market_data: MarketData) -> Optional[VolatilitySignal]:
        """
        OPTIMIZED: Checks for excessive market volatility using standard deviation
        of returns, replacing the simplified implementation.
        """
        symbol = market_data.symbol
        price = market_data.price

        # 1. 获取该 symbol 的价格历史
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.volatility_window)
        
        history = self.price_history[symbol]
        history.append(price)

        # 2. 检查是否有足够的数据
        if len(history) < history.maxlen:
            # 数据不足以计算有意义的标准差
            return None

        try:
            # 3. 计算收益率的标准差
            prices = np.array(history)
            
            # 检查价格是否为零或负数
            if np.any(prices <= 0):
                logger.warning(f"Invalid prices in history for {symbol}, cannot calc returns.")
                history.clear() # 清除坏数据
                return None

            # (prices[1:] - prices[:-1]) / prices[:-1]
            log_returns = np.log(prices[1:] / prices[:-1])
            
            # 年化波动率 (假设数据是每日的)
            current_volatility = np.std(log_returns)

            # 4. 与阈值比较
            if current_volatility > self.volatility_threshold:
                desc = (
                    f"High volatility detected for {symbol}: "
                    f"StdDev({self.volatility_window} periods) = {current_volatility:.4f} "
                    f"(Limit: {self.volatility_threshold:.4f})"
                )
                logger.warning(desc)
                return VolatilitySignal(
                    description=desc,
                    symbol=symbol,
                    current_volatility=current_volatility,
                    volatility_threshold=self.volatility_threshold,
                    triggers_circuit_breaker=self.config.get(
                        "volatility_triggers_breaker", False
                    ),
                )
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
        
        return None

    def trip_circuit_breaker(self, reason: str):
        """
        Trips the system-wide circuit breaker, halting new trades.
        """
        self.circuit_breaker_tripped = True
        # [Task 5] Persist Halt
        self.redis_client.set("phoenix:risk:halted", "1")
        self.active_signals.append(
            RiskSignal(
                type=SignalType.CIRCUIT_BREAKER,
                description=f"CIRCUIT BREAKER TRIPPED: {reason}",
                triggers_circuit_breaker=True,
            )
        )
        logger.critical(f"CIRCUIT BREAKER TRIPPED. Reason: {reason}")
        # 可以在此处添加通知逻辑 (例如, 发送警报)

    def reset_circuit_breaker(self):
        """
        Resets the circuit breaker (manual intervention usually required).
        """
        self.circuit_breaker_tripped = False
        # [Task 5] Clear Halt
        self.redis_client.delete("phoenix:risk:halted")
        self.active_signals = [
            s for s in self.active_signals if s.type != SignalType.CIRCUIT_BREAKER
        ]
        logger.info("RiskManager circuit breaker has been reset.")

    def get_status(self) -> Dict[str, Any]:
        """
        Returns the current status of the RiskManager.
        """
        return {
            "circuit_breaker_tripped": self.circuit_breaker_tripped,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "current_drawdown": (self.peak_equity - self.current_equity)
            / self.peak_equity
            if self.peak_equity > 0
            else 0,
            "active_signals": [s.model_dump() for s in self.active_signals],
        }
