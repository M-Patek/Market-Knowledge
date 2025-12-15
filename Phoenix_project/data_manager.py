"""
Phoenix_project/data_manager.py
[Task 2.1] TimeProvider Integration & Centralized Clock.
Redirects all time-sensitive operations to the injected TimeProvider.
"""
import json
import logging
import asyncio
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta, timezone
import pandas as pd
import redis.asyncio as redis

from Phoenix_project.config.loader import ConfigLoader
from Phoenix_project.core.schemas.data_schema import MarketData
from Phoenix_project.config.constants import REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE
# [Task 2.1] Import TimeProvider Interface
from Phoenix_project.core.time_provider import TimeProvider

logger = logging.getLogger(__name__)

class DataManager:
    """
    数据管理器 (DataManager)
    负责所有数据的访问、缓存和持久化。
    充当系统的 "海马体"，连接 Redis (短期记忆) 和 Temporal/Tabular DB (长期记忆)。
    
    [Task 4] Integrated Failover & Read-Repair
    [Task 2.1] Centralized Time Management via TimeProvider
    """

    def __init__(self, config: Any, time_provider: TimeProvider, redis_client: Optional[redis.Redis] = None, tabular_db: Any = None, temporal_db: Any = None):
        # [Refactor] Support both ConfigLoader (legacy) and direct DictConfig (Registry injected)
        if hasattr(config, 'load_config'):
            self.config = config.load_config('system.yaml')
        else:
            self.config = config

        # [Task 2.1] Mandatory TimeProvider Injection
        if time_provider is None:
            raise ValueError("TimeProvider must be injected into DataManager.")
        self.time_provider = time_provider
        
        self.redis_client = redis_client
        
        # [Fix Task 1] Dependency Injection to prevent Double Connection Pool
        if tabular_db is not None:
            self.tabular_db = tabular_db
        else:
            from Phoenix_project.ai.tabular_db_client import TabularDBClient
            self.tabular_db = TabularDBClient(self.config.get("data_manager", {}).get("tabular_db", {}))

        if temporal_db is not None:
            self.temporal_db = temporal_db
        else:
            from Phoenix_project.ai.temporal_db_client import TemporalDBClient
            self.temporal_db = TemporalDBClient(self.config.get("data_manager", {}).get("temporal_db", {}))
        
        # [Config] Allow configuring stale threshold
        self.stale_threshold_minutes = self.config.get("data_manager", {}).get("stale_threshold_minutes", 5)
        
        # [Task 2.1] Removed internal _simulation_time state
        
        logger.info("DataManager initialized with TimeProvider.")

    # [Task 2.1] Removed set_simulation_time / clear_simulation_time (Delegated to TimeProvider)

    async def get_latest_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        获取最新的市场数据。
        [Hot Path] 优先读取 Redis。
        [Cold Path] 失败时降级到 TemporalDB (Failover)，并触发读修复 (Read-Repair)。
        [Fix] 严禁返回过期数据 (Zombie Data)。
        [Task 2.1] Uses TimeProvider for current reference time.
        """
        current_time = self.get_current_time()
        
        # [Time Machine Check] In backtest/simulation, we MUST query DB for specific time point.
        # Redis (Live Cache) is only valid for REAL-TIME mode.
        if self.time_provider.is_simulation_mode():
            return await self._get_market_data_at_time(symbol, current_time)

        key = REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE.format(symbol=symbol)
        
        # 1. Try Redis (Hot Path - LIVE ONLY)
        if self.redis_client:
            try:
                data = await self.redis_client.get(key)
                if data:
                    return MarketData.model_validate_json(data)
            except Exception as e:
                logger.error(f"Redis read failed for {symbol}: {e}")

        # 2. Failover to TemporalDB (Cold Path)
        logger.warning(f"Cache miss for {symbol}, attempting DB failover...")
        if self.temporal_db:
            try:
                # [Task 2.1] Use TimeProvider for reference time
                end = current_time
                start = end - timedelta(hours=72)
                df = await self.temporal_db.query_market_data(symbol, start, end)
                
                if not df.empty:
                    latest_record = df.iloc[-1].to_dict()
                    market_data = MarketData(**latest_record)
                    
                    # [Fix Phase 2 Task 4] Strict Stale Check (Fail-Closed)
                    data_age = end - market_data.timestamp
                    is_stale = data_age > timedelta(minutes=self.stale_threshold_minutes)
                    
                    if is_stale:
                        logger.error(f"CRITICAL: Stale data fetched for {symbol}. FAIL-SAFE TRIGGERED.")
                        return None
                    
                    # 3. Read-Repair (Only in Live Mode)
                    if self.redis_client and not self.time_provider.is_simulation_mode():
                        try:
                            await self.redis_client.setex(key, 60, market_data.model_dump_json())
                            logger.info(f"Read-Repair: Restored {symbol} to Redis cache.")
                        except Exception:
                            pass 
                            
                    logger.info(f"Restored valid market data for {symbol} from TemporalDB.")
                    return market_data
                    
            except Exception as e:
                logger.error(f"Failover failed for {symbol}: {e}")
                
        return None

    async def _get_market_data_at_time(self, symbol: str, timestamp: datetime) -> Optional[MarketData]:
        """Helper to fetch data exactly at or before simulation time."""
        if not self.temporal_db: return None
        try:
            # Query range slightly before sim time
            start = timestamp - timedelta(hours=24)
            df = await self.temporal_db.query_market_data(symbol, start, timestamp)
            if not df.empty:
                latest_record = df.iloc[-1].to_dict()
                return MarketData(**latest_record)
        except Exception:
            pass
        return None

    async def get_market_data_history(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        从 TemporalDB 获取历史行情数据。
        """
        if not self.temporal_db:
            logger.warning("TemporalDB not configured.")
            return pd.DataFrame()
            
        return await self.temporal_db.query_market_data(symbol, start_time, end_time)

    async def get_fundamental_data(self, symbol: str, as_of_date: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """
        获取基本面数据 (Financials, Ratios)。
        [Time Machine] Supports as_of_date filtering to prevent future data leakage.
        """
        if not self.tabular_db:
            logger.warning("TabularDB not configured.")
            return None
            
        try:
            # [Time Machine] Apply date filter if provided, or default to current system time
            target_date = as_of_date or self.get_current_time()
            
            # Ensure simple date comparison
            date_str = target_date.strftime("%Y-%m-%d")

            # [Task P0-004] SQL Injection Fix: Use parameter binding via execute_sql
            # We strictly avoid f-string interpolation for the symbol.
            sql = "SELECT * FROM fundamentals WHERE symbol = :symbol AND date <= :target_date ORDER BY date DESC LIMIT 1"
            params = {"symbol": symbol, "target_date": date_str}

            # Use execute_sql for direct, safe execution if available
            if hasattr(self.tabular_db, 'execute_sql'):
                results = await self.tabular_db.execute_sql(sql, params=params)
                if results:
                    return results[0]
            else:
                # Fail securely if parameterized execution is not supported
                logger.error("TabularDBClient.execute_sql not available. Aborting get_fundamental_data to prevent SQL Injection.")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch fundamental data for {symbol}: {e}")
            
        return None

    async def fetch_news_data(self, start_time: datetime, end_time: datetime, limit: int = 100) -> List[Dict[str, Any]]:
        """
        [Restored] 获取新闻/事件数据。
        """
        if not self.temporal_db:
            logger.warning("TemporalDB not configured. Cannot fetch news.")
            return []
            
        try:
            return await self.temporal_db.query_events(
                event_type="news", 
                start_time=start_time, 
                end_time=end_time,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Failed to fetch news data: {e}")
            return []

    async def get_current_portfolio(self) -> Optional[Dict[str, Any]]:
        """
        [Task 18 Dependency] 获取当前投资组合状态 (从持久化存储或 Ledger)。
        """
        if self.tabular_db:
            try:
                balance_res = await self.tabular_db.query("SELECT cash, realized_pnl FROM ledger_balance ORDER BY id DESC LIMIT 1")
                positions_res = await self.tabular_db.query("SELECT * FROM ledger_positions")
                
                portfolio = {
                    "cash": 100000.0,
                    "positions": {},
                    "realized_pnl": 0.0
                }
                
                if balance_res and "results" in balance_res and balance_res["results"]:
                    row = balance_res["results"][0]
                    portfolio["cash"] = float(row["cash"])
                    portfolio["realized_pnl"] = float(row["realized_pnl"])
                    
                if positions_res and "results" in positions_res:
                    for row in positions_res["results"]:
                        sym = row["symbol"]
                        portfolio["positions"][sym] = {
                            "symbol": sym,
                            "quantity": float(row["quantity"]),
                            "average_price": float(row["average_price"]),
                            "market_value": float(row["market_value"]),
                            "unrealized_pnl": float(row["unrealized_pnl"])
                        }
                return portfolio
            except Exception as e:
                logger.error(f"Failed to fetch portfolio from Ledger: {e}")
        
        return None

    def get_current_time(self) -> datetime:
        """
        获取当前系统时间 (UTC)。
        [Task 2.1] Delegated to TimeProvider.
        """
        return self.time_provider.get_current_time()

    async def save_state(self, state: Dict[str, Any]):
        """
        保存一般系统状态 (Snapshot)。
        """
        pass

    async def close(self):
        # [Fix Task 3] Resource Cleanup
        if self.tabular_db and hasattr(self.tabular_db, 'close'):
            # Check if async
            if asyncio.iscoroutinefunction(self.tabular_db.close):
                await self.tabular_db.close()
            else:
                self.tabular_db.close()
        
        if self.temporal_db and hasattr(self.temporal_db, 'close'):
            if asyncio.iscoroutinefunction(self.temporal_db.close):
                await self.temporal_db.close()
            else:
                self.temporal_db.close()
