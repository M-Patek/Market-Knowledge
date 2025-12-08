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
    [Restored] fetch_news_data for L1 Agents
    """

    def __init__(self, config_loader: ConfigLoader, redis_client: Optional[redis.Redis] = None):
        self.config = config_loader.load_config('system.yaml')
        self.redis_client = redis_client
        
        # Initialize DB Clients
        self.tabular_db = TabularDBClient(self.config.get("data_manager", {}).get("tabular_db", {}))
        self.temporal_db = TemporalDBClient(self.config.get("data_manager", {}).get("temporal_db", {}))
        
        logger.info("DataManager initialized.")

    async def get_latest_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        获取最新的市场数据。
        [Hot Path] 优先读取 Redis，失败时降级到 TemporalDB (Failover)，并触发读修复 (Read-Repair)。
        """
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
                # Fetch recent data (last 72h to ensure relevance and cover weekends)
                # [Task 0.3 Fix] Use aware UTC time for failover query
                end = datetime.now(timezone.utc)
                # [Fix] Weekend Amnesia: Extend lookback to 72 hours
                start = end - timedelta(hours=72)
                df = await self.temporal_db.query_market_data(symbol, start, end)
                
                if not df.empty:
                    # Get latest record
                    latest_record = df.iloc[-1].to_dict()
                    market_data = MarketData(**latest_record)
                    
                    # 3. Read-Repair: Write back to Redis to restore Hot Path
                    # [Fix] Zombie Data Poisoning Check: Do not poison cache with stale DB data
                    # Check if data is older than 5 minutes
                    is_stale = (end - market_data.timestamp) > timedelta(minutes=5)
                    
                    if self.redis_client and not is_stale:
                        try:
                            # Use 60s TTL as per Task 003
                            await self.redis_client.setex(key, 60, market_data.model_dump_json())
                            logger.info(f"Read-Repair: Restored {symbol} to Redis cache.")
                        except Exception:
                            pass # Ignore write-back errors during failover, return data is priority
                    elif is_stale:
                        logger.warning(f"Stale data fetched for {symbol} (Age > 5m). Skipping Read-Repair.")
                            
                    logger.info(f"Restored market data for {symbol} from TemporalDB.")
                    return market_data
            except Exception as e:
                logger.error(f"Failover failed for {symbol}: {e}")
                
        return None

    async def get_market_data_history(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        从 TemporalDB 获取历史行情数据。
        用于回测、训练或 RiskManager 预热。
        """
        if not self.temporal_db:
            logger.warning("TemporalDB not configured.")
            return pd.DataFrame()
            
        return await self.temporal_db.query_market_data(symbol, start_time, end_time)

    async def fetch_news_data(self, start_time: datetime, end_time: datetime, limit: int = 100) -> List[Dict[str, Any]]:
        """
        [Restored] 获取新闻/事件数据。
        L1 舆情分析师 (Innovation/Geopolitical) 依赖此接口。
        """
        if not self.temporal_db:
            logger.warning("TemporalDB not configured. Cannot fetch news.")
            return []
            
        try:
            # 假设 TemporalDBClient 有 query_events 或类似方法
            # 这里映射到通用的查询接口
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
        用于 PortfolioConstructor 的异步初始化。
        """
        # 优先从 Ledger (TabularDB) 获取
        if self.tabular_db:
            try:
                # 查询最新的余额
                balance_res = await self.tabular_db.query("SELECT cash, realized_pnl FROM ledger_balance ORDER BY id DESC LIMIT 1")
                # 查询最新的持仓 (snapshot)
                positions_res = await self.tabular_db.query("SELECT * FROM ledger_positions")
                
                portfolio = {
                    "cash": 100000.0, # Default if empty
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
                        # 简单的聚合或最新值逻辑由 SQL 保证 (Task 012 Read Logic)
                        # 这里假设 query 返回的是去重后的列表
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
        获取当前系统时间。
        在回测模式下，这应该连接到 TimeMachine/Clock 服务。
        """
        # [Task 0.3 Fix] Return timezone-aware UTC datetime to prevent TypeError in PipelineState
        return datetime.now(timezone.utc)

    async def save_state(self, state: Dict[str, Any]):
        """
        保存一般系统状态 (Snapshot)。
        """
        # 实现状态快照保存逻辑 (e.g., to Redis or S3)
        pass
