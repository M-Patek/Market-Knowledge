"""
Phoenix_project/data_manager.py
[Phase 2 Task 4] Fix Zombie Data Resurrection.
Implement Strict Stale Data Check in Failover path.
Fail-Closed: Return None if DB data is older than 5 minutes.
[Phase 4 Task 2] Time Machine Support.
"""
import json
import logging
import asyncio
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta, timezone
import pandas as pd
import redis.asyncio as redis

from Phoenix_project.ai.tabular_db_client import TabularDBClient
from Phoenix_project.ai.temporal_db_client import TemporalDBClient
from Phoenix_project.config.loader import ConfigLoader
from Phoenix_project.core.schemas.data_schema import MarketData
from Phoenix_project.config.constants import REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE

logger = logging.getLogger(__name__)

class DataManager:
    """
    数据管理器 (DataManager)
    负责所有数据的访问、缓存和持久化。
    充当系统的 "海马体"，连接 Redis (短期记忆) 和 Temporal/Tabular DB (长期记忆)。
    
    [Task 4] Integrated Failover & Read-Repair
    [Task 4.2] Time Machine Support
    """

    def __init__(self, config_loader: ConfigLoader, redis_client: Optional[redis.Redis] = None):
        self.config = config_loader.load_config('system.yaml')
        self.redis_client = redis_client
        
        # Initialize DB Clients
        self.tabular_db = TabularDBClient(self.config.get("data_manager", {}).get("tabular_db", {}))
        self.temporal_db = TemporalDBClient(self.config.get("data_manager", {}).get("temporal_db", {}))
        
        # [Config] Allow configuring stale threshold
        self.stale_threshold_minutes = self.config.get("data_manager", {}).get("stale_threshold_minutes", 5)
        
        # [Phase 4 Task 2] Time Machine State
        self._simulation_time: Optional[datetime] = None
        
        logger.info("DataManager initialized.")

    def set_simulation_time(self, sim_time: datetime):
        """
        [Phase 4 Task 2] Activate Time Machine.
        Sets the internal clock to a historical timestamp.
        """
        if sim_time.tzinfo is None:
            sim_time = sim_time.replace(tzinfo=timezone.utc)
        self._simulation_time = sim_time
        # logger.debug(f"DataManager Time Machine set to: {self._simulation_time}")

    def clear_simulation_time(self):
        """Deactivate Time Machine (Return to Live)."""
        self._simulation_time = None

    async def get_latest_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        获取最新的市场数据。
        [Hot Path] 优先读取 Redis。
        [Cold Path] 失败时降级到 TemporalDB (Failover)，并触发读修复 (Read-Repair)。
        [Fix] 严禁返回过期数据 (Zombie Data)。
        """
        # [Time Machine] In backtest, query DB for specific time point.
        if self._simulation_time:
            return await self._get_market_data_at_time(symbol, self._simulation_time)

        key = REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE.format(symbol=symbol)
        
        # 1. Try Redis (Hot Path)
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
                # Use aware UTC time for failover query
                end = datetime.now(timezone.utc)
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
                    
                    # 3. Read-Repair
                    if self.redis_client:
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

    async def get_current_time(self) -> datetime:
        """
        获取当前系统时间 (UTC)。
        [Phase 4 Task 2] Time Machine: Returns simulation time if active.
        """
        if self._simulation_time:
            return self._simulation_time
        return datetime.now(timezone.utc)

    async def save_state(self, state: Dict[str, Any]):
        """
        保存一般系统状态 (Snapshot)。
        """
        pass
