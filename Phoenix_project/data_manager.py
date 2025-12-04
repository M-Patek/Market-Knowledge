"""
Data Manager for Phoenix (Restored & Optimized).

Unified Data Access Layer that acts as a Facade for:
1. Hot Data (Redis) - Real-time market data
2. Cold Data (TemporalDB) - Historical market data
3. Asset Data (TabularDB) - Portfolio & Fundamentals
4. External Data (HTTP API) - News & Alternative Data

[Features]
- Smart Caching (Read-Through)
- Distributed Locking (Thundering Herd Protection)
- Connection Pooling (httpx)
- Atomic Snapshots (SQL)
"""

import logging
import json
import asyncio
import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

import pandas as pd
import httpx
import redis.asyncio as redis
from redis.exceptions import LockError

# 使用相对导入适配项目结构
from .core.schemas.data_schema import MarketData, NewsData
from .ai.tabular_db_client import TabularDBClient
from .ai.temporal_db_client import TemporalDBClient
from .config.constants import REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, config: Dict[str, Any], redis_client: redis.Redis, tabular_db=None, temporal_db=None):
        self.config = config.get("data_manager", {})
        self.trading_config = config.get("trading", {}) # 获取交易配置
        self.api_keys = config.get("api_keys", {})
        self.redis_client = redis_client
        self.tabular_db = tabular_db
        self.temporal_db = temporal_db
        
        # [Config]
        self.run_mode = config.get("run_mode", "DEV").lower()
        self.cache_ttl = self.config.get("cache_ttl_sec", 300)
        self.news_api_url = self.config.get("news_api_url", "https://newsapi.org/v2/everything")
        self.news_api_key = self.api_keys.get("news_api") or os.environ.get("NEWS_API_KEY")
        
        # 获取初始资金配置，默认为 100k
        self.initial_capital = self.trading_config.get("initial_capital", 100000.0)

        # [Task 5.1] Persistent HTTP Client
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50)
        )
        logger.info(f"DataManager initialized (Mode: {self.run_mode}).")

    # --- Core: Caching Utilities ---

    def _get_cache_key(self, prefix: str, identifier: str) -> str:
        """Namespace isolated cache key."""
        return f"phx:{self.run_mode}:cache:{prefix}:{identifier}"

    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Fail-safe cache retrieval."""
        try:
            if not self.redis_client: return None
            data = await self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning(f"Cache read failed for {key}: {e}")
            return None

    async def _set_to_cache(self, key: str, data: Any, ttl: int = 300):
        """Fail-safe cache write."""
        try:
            if not self.redis_client: return
            
            # Handle Pydantic/JSON serialization
            if hasattr(data, 'model_dump_json'):
                val = data.model_dump_json()
            elif isinstance(data, list) and data and hasattr(data[0], 'model_dump'):
                # Ensure datetime objects in list are serialized
                val = json.dumps([d.model_dump() for d in data], default=str)
            else:
                val = json.dumps(data, default=str)
            
            await self.redis_client.setex(key, ttl, val)
        except Exception as e:
            logger.warning(f"Cache write failed for {key}: {e}")

    # --- Domain: Market Data ---

    async def get_latest_market_data(self, symbol: str) -> Optional[MarketData]:
        """[Hot Path] Direct Redis Access."""
        if not self.redis_client: return None
        key = REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE.format(symbol=symbol)
        try:
            data = await self.redis_client.get(key)
            if data:
                return MarketData.model_validate_json(data)
        except Exception as e:
            logger.error(f"Error fetching live data for {symbol}: {e}")
        return None

    async def get_market_data_history(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """[Cold Path] Cache -> TemporalDB."""
        # 1. Check Cache
        # [Optimization] Use timestamp for precision instead of date() to allow intraday caching
        cache_key = self._get_cache_key("history", f"{symbol}_{int(start.timestamp())}_{int(end.timestamp())}")
        
        cached = await self._get_from_cache(cache_key)
        if cached:
            return pd.DataFrame(cached)

        if not self.temporal_db:
            return pd.DataFrame()

        # 2. Query DB
        df = await self.temporal_db.query_market_data(symbol, start, end)
        
        # 3. Write Cache (if data exists)
        if not df.empty:
            await self._set_to_cache(cache_key, df.to_dict(orient="records"))
            
        return df

    # --- Domain: News (Restored) ---

    async def fetch_news_data(self, query: str, limit: int = 5) -> List[NewsData]:
        """
        [Restored] Fetch news with Distributed Locking & Caching.
        Prevents Thundering Herd on API.
        """
        if not self.news_api_key:
            logger.warning("News API Key missing. Returning empty list.")
            return []

        cache_key = self._get_cache_key("news", f"{query}_{limit}")
        
        # 1. Fast Path: Check Cache
        cached = await self._get_from_cache(cache_key)
        if cached:
            # Rehydrate Pydantic models from cache dicts
            return [NewsData(**item) for item in cached]

        # 2. Slow Path: Distributed Lock
        lock_key = f"phx:lock:news:{query}"
        try:
            if not self.redis_client: raise LockError("No Redis")
            
            # Wait up to 5s for lock, hold for 60s
            async with self.redis_client.lock(lock_key, timeout=60, blocking_timeout=5):
                # 3. Double-Check Cache (Optimization)
                cached = await self._get_from_cache(cache_key)
                if cached:
                    return [NewsData(**item) for item in cached]

                # 4. Call External API
                logger.info(f"Fetching live news for: {query}")
                params = {
                    "q": query,
                    "apiKey": self.news_api_key,
                    "pageSize": limit,
                    "sortBy": "publishedAt",
                    "language": "en"
                }
                
                response = await self.http_client.get(self.news_api_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                articles = []
                for item in data.get("articles", []):
                    # Minimal mapping
                    articles.append(NewsData(
                        headline=item.get("title"),
                        url=item.get("url"),
                        timestamp=datetime.now(), # Simplified for demo
                        summary=item.get("description"),
                        source=item.get("source", {}).get("name")
                    ))

                # 5. Update Cache
                if articles:
                    await self._set_to_cache(cache_key, articles, ttl=600) # Cache for 10 mins
                
                return articles

        except (LockError, AttributeError):
            logger.warning(f"Could not acquire lock or Redis missing for news: {query}. Skipping.")
            return []
        except Exception as e:
            logger.error(f"News fetch failed: {e}")
            return []

    # --- Domain: Portfolio (Preserved) ---

    async def get_current_portfolio(self) -> Dict[str, Any]:
        """[Task 1.3] Atomic Snapshot via Single Query."""
        if not self.tabular_db: return {}
        
        try:
            # [Fix] Atomic LEFT JOIN ensures Balance and Positions are from the same snapshot
            query = """
                SELECT b.cash, p.symbol, p.quantity 
                FROM ledger_balance b 
                LEFT JOIN ledger_positions p ON 1=1 
                WHERE b.id = 1
            """
            res = await self.tabular_db.query(query)
            
            # [Optimization] Use configured initial capital instead of hardcoded 100k
            cash = self.initial_capital
            positions = {}
            
            if res and "results" in res and res["results"]:
                first = res["results"][0]
                if first.get("cash") is not None:
                    cash = float(first["cash"])
                
                for row in res["results"]:
                    if row.get("symbol"):
                        positions[row["symbol"]] = float(row["quantity"])
                        
            return {"cash": cash, "positions": positions}
            
        except Exception as e:
            logger.error(f"Portfolio snapshot failed: {e}")
            return {"cash": self.initial_capital, "positions": {}}

    async def close(self):
        if self.http_client:
            await self.http_client.aclose()
        logger.info("DataManager closed.")
