"""
Phoenix_project/cognitive/risk_manager.py
[Phase 3 Task 5] Fix RiskManager Silent Failure.
Replace silent 'return None' in anomaly paths with 'raise RiskDataError' 
to enforce Fail-Closed logic.
"""

import logging
from typing import Any, Dict, Optional, List, Set
from collections import deque
from datetime import datetime, timedelta
import asyncio
import numpy as np 
import redis.asyncio as redis

from ..core.schemas.data_schema import MarketData, Position, PortfolioState
from ..core.schemas.risk_schema import (
    RiskSignal, SignalType, VolatilitySignal, DrawdownSignal, ConcentrationSignal, RiskReport
)
from ..core.exceptions import RiskViolationError, CircuitBreakerError

# Define specific error for data failures
class RiskDataError(Exception):
    pass

logger = logging.getLogger(__name__)

# Forward declaration
from ..data_manager import DataManager


class RiskManager:
    """
    Manages risk for the trading system.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        redis_client: redis.Redis,
        data_manager: Optional[DataManager] = None, 
        initial_capital: float = 100000.0,
    ):
        self.config = config.get("risk_manager", {})
        self.redis_client = redis_client
        self.data_manager = data_manager
        self.initial_capital = initial_capital
        
        self.halted_symbols: Set[str] = set()
        
        self.max_drawdown_pct = self.config.get("max_drawdown_pct", 0.15)
        self.max_position_concentration_pct = self.config.get("max_position_concentration_pct", 0.20)
        self.volatility_threshold = self.config.get("volatility_threshold", 0.05)
        self.volatility_window = self.config.get("volatility_window", 30)
        
        if "frequency_factor" in self.config:
            self.frequency_factor = self.config["frequency_factor"]
        else:
            trading_config = config.get("trading", {})
            data_freq = trading_config.get("data_frequency", "daily")
            asset_class = self.config.get("asset_class", "stock").lower()
            
            if asset_class == "crypto":
                self.frequency_factor = 525600 if data_freq == "minute" else 365
            elif data_freq == "minute":
                self.frequency_factor = 252 * 390 
            else:
                self.frequency_factor = 252

        self.current_equity: float = initial_capital 
        
        self.circuit_breaker_tripped: bool = False 
        self.active_signals: List[RiskSignal] = []
        self.peak_equity: float = initial_capital 

        self.price_history: Dict[str, deque] = {} 

        logger.info("RiskManager initialized.")

    async def _load_circuit_breaker_state(self) -> bool:
        try:
            if not self.redis_client: 
                logger.warning("RiskManager: No Redis client. Defaulting to TRIPPED (Safe Mode).")
                return True
            state = await self.redis_client.get("phoenix:risk:halted")
            return bool(int(state)) if state else False
        except Exception as e:
            logger.error(f"Failed to load circuit breaker state: {e}. Defaulting to TRIPPED.")
            return True

    async def initialize(self, symbols: List[str]):
        self.circuit_breaker_tripped = await self._load_circuit_breaker_state()
        self.peak_equity = await self._load_peak_equity()
        self.current_equity = await self._load_current_equity()
        
        logger.info(f"Loaded Peak Equity: {self.peak_equity}")
        logger.info(f"Loaded Current Equity: {self.current_equity}")
        logger.info(f"Circuit Breaker State: {'TRIPPED' if self.circuit_breaker_tripped else 'OK'}")

        if not self.data_manager:
            return

        logger.info(f"Warming up RiskManager for {len(symbols)} symbols...")
        end_time = await self.data_manager.get_current_time()
        start_time = end_time - timedelta(days=int(self.volatility_window * 2) + 5)

        async def _fetch_and_fill(sym):
            try:
                df = await self.data_manager.get_market_data_history(sym, start_time, end_time)
                if df is not None and not df.empty:
                    closes = df['close'].values[-self.volatility_window:].tolist()
                    if sym not in self.price_history:
                        self.price_history[sym] = deque(maxlen=self.volatility_window)
                    self.price_history[sym].extend(closes)
            except Exception as e:
                logger.error(f"Failed to warm up risk data for {sym}: {e}")

        await asyncio.gather(*[_fetch_and_fill(s) for s in symbols])
        asyncio.create_task(self._monitor_circuit_breaker())
        logger.info(f"RiskManager warm-up complete.")

    async def _load_peak_equity(self) -> float:
        try:
            if not self.redis_client: return self.initial_capital
            peak_equity_raw = await self.redis_client.get("phoenix:risk:peak_equity")
            if peak_equity_raw:
                val = float(peak_equity_raw)
                return max(val, self.initial_capital)
            return self.initial_capital
        except Exception as e:
            logger.error(f"Failed to load peak_equity: {e}")
            return self.initial_capital

    async def _save_peak_equity(self):
        try:
            if self.redis_client:
                await self.redis_client.set("phoenix:risk:peak_equity", self.peak_equity)
        except Exception as e:
            logger.error(f"Failed to save peak_equity: {e}")

    async def _load_current_equity(self) -> float:
        try:
            if not self.redis_client: return self.initial_capital
            val = await self.redis_client.get("phoenix:risk:current_equity")
            return float(val) if val else self.initial_capital
        except Exception as e:
            logger.error(f"Failed to load current_equity: {e}")
            return self.initial_capital

    async def _save_current_equity(self):
        try:
            if self.redis_client:
                await self.redis_client.set("phoenix:risk:current_equity", self.current_equity)
        except Exception as e:
            logger.error(f"Failed to save current_equity: {e}")

    async def validate_allocations(
        self, target_weights: Dict[str, float], current_portfolio: Dict[str, Any], market_data: Any
    ) -> RiskReport:
        if self.circuit_breaker_tripped:
            return RiskReport(passed=False, adjustments_made="Circuit Breaker Tripped")

        violations = []
        adjustments = []
        pf_value = current_portfolio.get("cash", 0.0)
        
        for sym, weight in target_weights.items():
            if abs(weight) > self.max_position_concentration_pct:
                violations.append(f"Concentration {weight:.2f} > {self.max_position_concentration_pct}")
                adjustments.append(f"Capped {sym} at {self.max_position_concentration_pct}")
                
        if violations:
            logger.warning(f"Allocation validation failed: {violations}")
            return RiskReport(passed=False, adjustments_made="; ".join(adjustments))
            
        return RiskReport(passed=True, adjustments_made="None")

    def get_clamped_weights(self, target_weights: Dict[str, float], current_portfolio: Dict[str, Any]) -> Dict[str, float]:
        clamped = {}
        for sym, w in target_weights.items():
            if abs(w) > self.max_position_concentration_pct:
                sign = 1.0 if w > 0 else -1.0
                clamped[sym] = sign * self.max_position_concentration_pct
            else:
                clamped[sym] = w
        return clamped

    def check_pre_trade(
        self, proposed_position: Position, portfolio: PortfolioState
    ) -> List[RiskSignal]:
        if self.circuit_breaker_tripped:
            symbol = proposed_position.symbol
            existing_pos = portfolio.positions.get(symbol)
            existing_qty = float(existing_pos.quantity) if existing_pos else 0.0
            trade_qty = float(proposed_position.quantity)

            is_reducing = (existing_qty != 0) and (existing_qty * trade_qty < 0) and (abs(trade_qty) <= abs(existing_qty) + 1e-9)
            
            if is_reducing:
                logger.warning(f"Circuit Breaker Override: Allowing risk-reducing trade for {symbol} (Qty: {trade_qty}).")
            else:
                raise CircuitBreakerError("Risk circuit breaker is active. No new/increasing trades allowed.")

        if proposed_position.symbol in self.halted_symbols:
            raise RiskViolationError(f"Trading halted for {proposed_position.symbol} due to volatility", [])

        signals = []
        conc_signal = self.check_concentration(proposed_position, portfolio)
        if conc_signal:
            signals.append(conc_signal)

        if signals:
            logger.warning(f"Pre-trade check failed for {proposed_position.symbol}: {signals}")
            raise RiskViolationError(f"Pre-trade risk violation: {signals[0].description}", signals)

        return signals

    async def check_post_trade(self, portfolio: PortfolioState) -> List[RiskSignal]:
        if self.circuit_breaker_tripped:
            return self.active_signals

        self.active_signals = []
        await self.update_portfolio_value(float(portfolio.total_value))
        drawdown_signal = self.check_drawdown()
        if drawdown_signal:
            self.active_signals.append(drawdown_signal)
        
        if self.active_signals:
            if any(s.triggers_circuit_breaker for s in self.active_signals):
                await self.trip_circuit_breaker(f"Violation: {self.active_signals[0].description}")

        return self.active_signals

    async def on_market_data(self, market_data: MarketData) -> Optional[RiskSignal]:
        if self.circuit_breaker_tripped: return None
        
        try:
            vol_signal = self.check_volatility(market_data)
            if vol_signal:
                if vol_signal.triggers_circuit_breaker:
                    await self.trip_circuit_breaker(vol_signal.description)
                return vol_signal
        except RiskDataError as e:
            # [Task 5] Catch data errors and Trip Breaker (Fail-Closed)
            logger.critical(f"Critical Risk Data Failure: {e}. Tripping Circuit Breaker.")
            await self.trip_circuit_breaker(f"Data Failure: {e}")
            
        return None

    async def update_portfolio_value(self, new_equity: float):
        self.current_equity = new_equity
        await self._save_current_equity()
        
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
            await self._save_peak_equity()

    def check_drawdown(self) -> Optional[DrawdownSignal]:
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        if drawdown > self.max_drawdown_pct:
            desc = f"Maximum drawdown exceeded: {drawdown*100:.2f}%"
            return DrawdownSignal(description=desc, current_drawdown=drawdown, max_drawdown=self.max_drawdown_pct, triggers_circuit_breaker=True)
        return None

    def check_concentration(
        self, proposed_position: Position, portfolio: PortfolioState
    ) -> Optional[ConcentrationSignal]:
        symbol = proposed_position.symbol
        
        existing_pos = portfolio.positions.get(symbol)
        existing_qty = float(existing_pos.quantity) if existing_pos else 0.0
        trade_qty = float(proposed_position.quantity)
        
        final_qty = existing_qty + trade_qty
        
        same_sign = (existing_qty * final_qty) >= 0
        if existing_qty != 0 and abs(final_qty) < abs(existing_qty) - 1e-9 and same_sign:
            logger.info(f"Risk reduction detected for {symbol}. Allowing trade.")
            return None

        price = 0.0
        if existing_pos and float(existing_pos.market_value) > 0 and abs(existing_qty) > 0:
            price = float(existing_pos.market_value) / abs(existing_qty)
        elif float(proposed_position.market_value) > 0 and abs(trade_qty) > 0:
            price = float(proposed_position.market_value) / abs(trade_qty)
            
        final_position_value = abs(final_qty) * price
        
        total_portfolio_value = float(portfolio.total_value)
        if total_portfolio_value == 0: return None

        concentration = final_position_value / total_portfolio_value
        
        if concentration > self.max_position_concentration_pct:
            desc = f"Position concentration limit exceeded for {symbol}: {concentration*100:.2f}%"
            return ConcentrationSignal(description=desc, symbol=symbol, current_concentration=concentration, max_concentration=self.max_position_concentration_pct)
        return None

    def check_volatility(self, market_data: MarketData) -> Optional[VolatilitySignal]:
        symbol = market_data.symbol
        
        # [Task 5 Fix] Validate Market Data Input
        if market_data.close is None or market_data.close <= 0:
             raise RiskDataError(f"Invalid market data for {symbol}: close={market_data.close}")

        price = float(market_data.close) 

        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.volatility_window)
        
        self.price_history[symbol].append(price)

        if len(self.price_history[symbol]) < self.volatility_window:
            return VolatilitySignal(
                description="Circuit Breaker Tripped due to Cold Start (Insufficient Data).",
                symbol=symbol,
                current_volatility=999.0,
                volatility_threshold=self.volatility_threshold,
                triggers_circuit_breaker=True
            )

        try:
            prices = np.array(self.price_history[symbol])
            prices = prices[prices > 0] 
            
            # [Task 5 Fix] Fail-Closed on Calculation Error
            if len(prices) < 2: 
                raise RiskDataError(f"Insufficient valid prices for {symbol} volatility calc.")

            log_returns = np.log(prices[1:] / prices[:-1])
            
            # Check for NaN/Inf
            if not np.all(np.isfinite(log_returns)):
                 raise RiskDataError(f"Volatility calc resulted in NaN/Inf for {symbol}")

            current_volatility = np.std(log_returns) * np.sqrt(self.frequency_factor)

            if current_volatility > self.volatility_threshold:
                desc = f"High volatility detected for {symbol}: {current_volatility:.4f}"
                self.halted_symbols.add(symbol)
                logger.warning(f"Symbol {symbol} HALTED due to volatility.")
                return VolatilitySignal(description=desc, symbol=symbol, current_volatility=current_volatility, volatility_threshold=self.volatility_threshold, triggers_circuit_breaker=False)
                
        except RiskDataError:
            raise # Re-raise known errors
        except Exception as e:
            # Catch unexpected numpy errors
            raise RiskDataError(f"Volatility calculation crashed: {e}")
            
        return None

    async def trip_circuit_breaker(self, reason: str):
        self.circuit_breaker_tripped = True
        if self.redis_client:
            try:
                await self.redis_client.set("phoenix:risk:halted", "1")
                await self.redis_client.publish("phoenix:risk:halted", "TRIP")
            except Exception as e:
                logger.error(f"Failed to persist/publish halt state: {e}")
        logger.critical(f"CIRCUIT BREAKER TRIPPED. Reason: {reason}")

    async def reset_circuit_breaker(self):
        self.circuit_breaker_tripped = False
        if self.redis_client:
            await self.redis_client.delete("phoenix:risk:halted")
            await self.redis_client.publish("phoenix:risk:halted", "RESET")
        logger.info("RiskManager circuit breaker has been reset.")

    async def _monitor_circuit_breaker(self):
        if not self.redis_client: return
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe("phoenix:risk:halted")
            logger.info("Listening for distributed circuit breaker events on 'phoenix:risk:halted'...")

            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = message["data"].decode("utf-8")
                    if data in ("1", "TRIP"):
                        if not self.circuit_breaker_tripped:
                            self.circuit_breaker_tripped = True
                            logger.critical("DISTRIBUTED CIRCUIT BREAKER TRIPPED via Pub/Sub!")
                    elif data in ("0", "RESET"):
                        if self.circuit_breaker_tripped:
                            self.circuit_breaker_tripped = False
                            logger.info("Circuit breaker RESET via Pub/Sub.")
        except Exception as e:
            logger.error(f"Circuit breaker monitor failed: {e}")
