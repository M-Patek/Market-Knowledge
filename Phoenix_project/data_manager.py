"""
DataManager for the Phoenix project.

Handles loading, caching, and refreshing of all data sources required
by the cognitive engine and agents (e.g., market data, news, fundamentals).
[Phase I Fix] Environment Isolation & Time Travel Prevention
[Phase II Fix] Cache Isolation & Session ID
[Phase IV Fix] Async Locking (Thundering Herd)
[Phase V Fix] Distributed Redis Lock & Connection Pooling
"""

import logging
import json
import requests
import asyncio
import httpx # [Task 3.1] Import httpx for async I/O
import os # [Task 2] 导入 os
import redis.exceptions # [Task 4.2] For LockError
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd  # type: ignore
import redis.asyncio as redis  # type: ignore

from Phoenix_project.core.schemas.data_schema import MarketData, NewsData, FundamentalData
from Phoenix_project.ai.tabular_db_client import TabularDBClient
from Phoenix_project.ai.temporal_db_client import TemporalDBClient
from Phoenix_project.config.constants import REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE

logger = logging.getLogger(__name__)


class DataManager:
    """
    Orchestrates data retrieval from various sources like Redis, SQL DBs,
    Temporal DBs, and external APIs.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        redis_client: redis.Redis,
        tabular_db: Optional[TabularDBClient] = None,
        temporal_db: Optional[TemporalDBClient] = None,
        session_id: Optional[str] = None, # [Task 2.3] Session ID for cache isolation
    ):
        """
        Initializes the DataManager.

        Args:
            config: System configuration dictionary.
            redis_client: Client for Redis cache.
            tabular_db: (Optional) Client for tabular (SQL) data.
            temporal_db: (Optional) Client for time-series data.
            session_id: (Optional) Unique ID for the current run session.
        """
        # [Phase I Fix] Extract run_mode for key isolation
        self.run_mode = config.get("run_mode", "DEV").lower()
        self.config = config.get("data_manager", {})
        self.api_keys = config.get("api_keys", {})
        self.redis_client = redis_client
        self.tabular_db = tabular_db
        self.temporal_db = temporal_db
        self.session_id = session_id
        self._simulation_time: Optional[datetime] = None
        
        # [Task 4.2 Fix] Removed local _request_locks in favor of Distributed Redis Lock

        # 加载数据目录 (Data Catalog)
        self.data_catalog = self._load_data_catalog(
            self.config.get("catalog_path", "data_catalog.json")
        )
        
        self.cache_ttl_sec = self.config.get("cache_ttl_sec", 300) # 5 分钟
        
        # API URL 配置
        self.news_api_url = self.config.get("news_api_url", "https://newsapi.org/v2/everything")
        self.market_data_api_url = self.config.get("market_data_api_url", "https://api.polygon.io/v2/aggs/ticker")
        
        # [Task 5.1] Persistent HTTP Client for Connection Pooling
        # Enables Keep-Alive to avoid TCP/SSL handshake overhead on every request
        self.http_client = httpx.AsyncClient(timeout=10.0, limits=httpx.Limits(max_keepalive_connections=20, max_connections=100))
        
        logger.info(f"DataManager initialized (Run Mode: {self.run_mode}).")

    def set_simulation_time(self, sim_time: Optional[datetime]):
        """
        [Time Machine] 设置仿真时间。
        如果设置为非 None，系统将进入回测模式，所有数据获取都将相对于此时刻。
        """
        self._simulation_time = sim_time
        mode = f"BACKTEST ({sim_time})" if sim_time else "LIVE"
        logger.info(f"DataManager time mode switched to: {mode}")

    def get_current_time(self) -> datetime:
        """
        获取当前系统时间 (如果是回测模式，则返回仿真时间)。
        """
        return self._simulation_time if self._simulation_time else datetime.now()

    def _load_data_catalog(self, catalog_path: str) -> Dict[str, Any]:
        """Loads the data catalog JSON file."""
        try:
            with open(catalog_path, "r") as f:
                catalog = json.load(f)
                logger.info(f"Successfully loaded data catalog from {catalog_path}")
                return catalog.get("sources", {})
        except FileNotFoundError:
            logger.error(f"Data catalog file not found at {catalog_path}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {catalog_path}")
        except Exception as e:
            logger.error(f"Error loading data catalog: {e}")
        
        return {} # 返回空目录作为回退

    def _get_cache_key(self, prefix: str, identifier: str) -> str:
        """Generates a standardized Redis cache key."""
        # [Phase I Fix] Apply namespace isolation
        # [Task 2.3] Add session_id to key if in backtest mode to prevent collisions
        if self.run_mode == "backtest" and self.session_id:
            return f"phx:{self.run_mode}:{self.session_id}:cache:{prefix}:{identifier}"
            
        return f"phx:{self.run_mode}:cache:{prefix}:{identifier}"

    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Retrieves and deserializes data from Redis cache."""
        try:
            cached_data = await self.redis_client.get(key) # [Fix] await for async client
            if cached_data:
                logger.debug(f"Cache HIT for key: {key}")
                return json.loads(cached_data)
            logger.debug(f"Cache MISS for key: {key}")
            return None
        except redis.RedisError as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Cache JSON decode error for key {key}: {e}")
            return None # 缓存数据已损坏

    async def _set_to_cache(self, key: str, data: Any):
        """Serializes and stores data in Redis cache with a TTL."""
        try:
            # 确保 data 可以被 JSON 序列化
            if hasattr(data, 'model_dump_json'):
                data_json = data.model_dump_json()
            elif isinstance(data, list) and data and hasattr(data[0], 'model_dump'):
                 data_json = json.dumps([d.model_dump() for d in data])
            else:
                data_json = json.dumps(data, default=str) # [Task 2] 添加 default=str
                
            await self.redis_client.setex(
                key, self.cache_ttl_sec, data_json
            ) # [Fix] await for async client
            logger.debug(f"Cache SET for key: {key}")
        except redis.RedisError as e:
            logger.error(f"Redis SET error for key {key}: {e}")
        except TypeError as e:
            logger.error(f"Cache JSON serialization error for key {key}: {e}")

    async def get_market_data_history(
        self, symbol: str, start: datetime, end: datetime, allow_future_lookup: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Retrieves historical market data, checking cache first,
        then falling back to the temporal DB.

        Args:
            allow_future_lookup: If True, bypasses the simulation time check (used for backtest preload).
        """
        if not self.temporal_db:
            logger.warning("TemporalDB client not configured. Cannot fetch history.")
            return None
        
        # [Time Machine] 防止窥探未来 (除非显式特权访问)
        if not allow_future_lookup and self._simulation_time and end > self._simulation_time:
            logger.debug(f"Clamping history request end time from {end} to {self._simulation_time}")
            end = self._simulation_time

        cache_key = self._get_cache_key(
            "market_history", f"{symbol}_{start.isoformat()}_{end.isoformat()}"
        )
        
        cached = await self._get_from_cache(cache_key)
        if cached:
            return pd.DataFrame(cached)

        # 从时序数据库 (Temporal DB) 获取
        logger.info(f"Fetching market history for {symbol} from TemporalDB.")
        try:
            # [Contract] 期望 TemporalDBClient 实现 query_range
            df = await self.temporal_db.query_range(
                measurement="market_data",
                symbol=symbol,
                start=start,
                end=end,
                columns=["open", "high", "low", "close", "volume"],
            )
            
            if df is not None and not df.empty:
                # 缓存结果 (DataFrame -> dict -> JSON)
                await self._set_to_cache(cache_key, df.to_dict(orient="records"))
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to fetch history from TemporalDB for {symbol}: {e}")
            return None

    async def _get_historical_latest(self, symbol: str, point_in_time: datetime) -> Optional[MarketData]:
        """
        [Time Machine] 从历史数据库中获取指定时间点之前最新的数据。
        """
        if not self.temporal_db:
            return None
        # [Contract] 期望 TemporalDBClient 实现 get_latest_data_point
        data = await self.temporal_db.get_latest_data_point(symbol, point_in_time)
        if data:
            return MarketData(**data)
        return None

    async def get_latest_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        OPTIMIZED: Retrieves the latest market data for a symbol
        from the 'production' (live) Redis stream/cache.
        This replaces the mock object.
        """
        # [Time Machine] 路由逻辑
        if self._simulation_time:
            return await self._get_historical_latest(symbol, self._simulation_time)

        # --- Live Mode (Redis) ---
        # [Phase I Fix] Apply namespace isolation
        live_key = f"phx:{self.run_mode}:market_data:live:{symbol}"
        
        try:
            data_json = await self.redis_client.get(live_key) # [Fix] await
            
            if data_json:
                # 使用 model_validate_json 自动处理 datetime 和类型转换
                return MarketData.model_validate_json(data_json)
            
            return None
                
        except redis.RedisError as e:
            logger.error(f"Redis error getting live market data for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error validating market data for {symbol}: {e}")
            
        return None

    async def get_market_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """
        [Task 2.1] 批量获取市场数据。使用 asyncio.gather 并发拉取。
        [Task 3.1] Fix: Relax suicidal validation. Return partial results.
        """
        tasks = [self.get_latest_market_data(sym) for sym in symbols]
        results = await asyncio.gather(*tasks)
        
        # [Task 3.1] Return partial results (Soft Fail)
        # Prevent partial failure from crashing the entire pipeline
        failed_symbols = [sym for sym, res in zip(symbols, results) if res is None]
        if failed_symbols:
            logger.warning(f"Partial data failure. Missing market data for symbols: {failed_symbols}")

        return {sym: res for sym, res in zip(symbols, results) if res is not None}

    async def get_news_data(self, query: str = "", limit: int = 10, start_date: datetime = None, end_date: datetime = None) -> List[NewsData]:
        """
        [Task 2.1] DataIterator 适配器接口。
        """
        return await self.fetch_news_data(query=query, limit=limit, start_date=start_date, end_date=end_date)

    async def get_fundamental_data(
        self, symbol: str
    ) -> Optional[FundamentalData]:
        """
        Retrieves fundamental data, checking cache, then tabular DB.
        """
        if not self.tabular_db:
            logger.warning("TabularDB client not configured. Cannot fetch fundamentals.")
            return None
        
        cache_key = self._get_cache_key("fundamentals", symbol)
        
        cached = await self._get_from_cache(cache_key)
        if cached:
            data = FundamentalData(**cached)
            # [Task 2.1] Time Barrier
            if self._simulation_time and data.timestamp > self._simulation_time:
                logger.debug(f"Cache invalidated for {symbol}: Future data detected.")
            else:
                return data

        # 从表格数据库 (Tabular DB) 获取
        logger.info(f"Fetching fundamentals for {symbol} from TabularDB.")
        try:
            # [Phase I Fix] Explicit PIT Retrieval
            fund_data = await self._get_historical_fundamentals(symbol, self.get_current_time())
            
            if fund_data:
                await self._set_to_cache(cache_key, fund_data)
                return fund_data
            else:
                logger.warning(f"No fundamental data found in TabularDB for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch fundamentals from TabularDB for {symbol}: {e}")
            return None

    async def _get_historical_fundamentals(self, symbol: str, pit_date: datetime) -> Optional[FundamentalData]:
        """
        Helper: Executes Point-in-Time (PIT) query against TabularDB.
        """
        query = """
            SELECT * FROM fundamentals 
            WHERE symbol = :symbol 
            AND timestamp <= :pit_date 
            ORDER BY timestamp DESC 
            LIMIT 1
        """
        params = {"symbol": symbol, "pit_date": pit_date}
        
        rows = await self.tabular_db.execute_sql(query, params)
        
        if rows:
            fund_data_dict = rows[0]
            return FundamentalData(
                symbol=symbol,
                timestamp=fund_data_dict.get("timestamp") or pit_date,
                market_cap=fund_data_dict.get("market_cap"),
                pe_ratio=fund_data_dict.get("pe_ratio"),
                sector=fund_data_dict.get("sector"),
                industry=fund_data_dict.get("industry"),
                dividend_yield=fund_data_dict.get("dividend_yield"),
                eps=fund_data_dict.get("eps"),
                beta=fund_data_dict.get("beta")
            )
        return None

    async def fetch_news_data(
        self, query: str, limit: int = 10, start_date: datetime = None, end_date: datetime = None, force_refresh: bool = False
    ) -> List[NewsData]:
        """
        OPTIMIZED: Fetches news data from an external API, using caching.
        [Task 4.2] Uses Distributed Redis Lock to prevent Thundering Herd in distributed env.
        [Task 5.1] Uses Persistent HTTP Client for connection reuse.
        """
        cache_key = self._get_cache_key("news", query)
        
        if start_date or end_date:
            cache_key = self._get_cache_key("news", f"{query}_{start_date}_{end_date}")
        
        # 1. First Check (Fast Path)
        if not force_refresh:
            cached = await self._get_from_cache(cache_key)
            if cached:
                return [NewsData(**item) for item in cached]
        
        # 2. Acquire Distributed Lock (Slow Path)
        lock_key = f"phx:{self.run_mode}:lock:news_fetch:{query}"
        
        try:
            # [Task 4.2] Distributed Lock with Timeout
            async with self.redis_client.lock(lock_key, timeout=60, blocking_timeout=10):
                # 3. Second Check (Double-Checked Locking)
                if not force_refresh:
                    cached = await self._get_from_cache(cache_key)
                    if cached:
                        logger.debug(f"Cache hit on second check for {query} (Request Coalesced)")
                        return [NewsData(**item) for item in cached]

                # Fetch from API
                logger.info(f"Fetching news from API for query: {query}")
                news_api_key = self.api_keys.get("news_api")
                if not news_api_key:
                    logger.error("News API key not configured.")
                    return []
                    
                params = {
                    "q": query,
                    "apiKey": news_api_key,
                    "pageSize": limit,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "from": start_date.isoformat() if start_date else None,
                    "to": end_date.isoformat() if end_date else None,
                }
                params = {k: v for k, v in params.items() if v is not None}
                
                try:
                    # [Task 5.1] Use persistent self.http_client
                    response = await self.http_client.get(self.news_api_url, params=params)
                    response.raise_for_status()
                    
                    articles = response.json().get("articles", [])
                    news_list = []
                    
                    for article in articles:
                        news_item = NewsData(
                            id=article.get("url"),
                            symbol=query,
                            headline=article.get("title"),
                            summary=article.get("description"),
                            content=article.get("content"),
                            url=article.get("url"),
                            source=article.get("source", {}).get("name"),
                            timestamp=datetime.fromisoformat(article.get("publishedAt").replace("Z", "+00:00")),
                        )
                        news_list.append(news_item)
                    
                    if news_list:
                        await self._set_to_cache(cache_key, news_list)
                    
                    return news_list

                except (requests.exceptions.RequestException, httpx.HTTPError) as e:
                    logger.error(f"Failed to fetch news from API: {e}")
                    return []
                    
        except redis.exceptions.LockError:
            # [Task 4.2] Failed to acquire lock (timeout)
            logger.warning(f"Failed to acquire distributed lock for {query}. Skipping news fetch.")
            return []
            
        except Exception as e:
            logger.error(f"Error processing news data: {e}")
            return []

    # --- 内部辅助方法 (用于 refresh_data_sources) ---

    async def _fetch_and_store_market_data(self, config: Dict[str, Any]): # [Task 3.1] async def
        """
        Helper: Fetches market data from external API and stores
        it in persistent storage (e.g., TemporalDB).
        """
        symbol = config.get("symbol")
        if not symbol or not self.temporal_db:
            return

        logger.info(f"Refreshing market data for {symbol}...")
        
        market_api_key = self.api_keys.get("polygon_api")
        if not market_api_key:
            logger.error("Market data API key not configured.")
            return

        yesterday = (self.get_current_time() - timedelta(days=1)).strftime('%Y-%m-%d')
        url = f"{self.market_data_api_url}/{symbol}/range/1/day/{yesterday}/{yesterday}"
        params = {"apiKey": market_api_key}

        try:
            # [Task 5.1] Use persistent http_client
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if "results" in data:
                # 2. 准备数据
                df = pd.DataFrame(data["results"])
                df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "time"})
                df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
                df = df.set_index("time")
                
                # 3. 写入 TemporalDB (历史)
                await self.temporal_db.write_dataframe(
                    measurement="market_data",
                    df=df,
                    tags={"symbol": symbol}
                )
                
                # [Task 3.1 Fix] Sync to Redis Cache (Live Data)
                # The TemporalDB write is historical; we must also update the hot cache 
                # so get_latest_market_data() can find it immediately.
                try:
                    latest = df.iloc[-1]
                    md = MarketData(
                        symbol=symbol,
                        timestamp=latest.name, # Index is time
                        open=latest['open'],
                        high=latest['high'],
                        low=latest['low'],
                        close=latest['close'],
                        volume=latest['volume']
                    )
                    live_key = f"phx:{self.run_mode}:market_data:live:{symbol}"
                    await self.redis_client.set(live_key, md.model_dump_json())
                    logger.debug(f"Synced latest market data to Redis for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to sync Redis cache for {symbol}: {e}")

                logger.info(f"Successfully refreshed and stored market data for {symbol}")

        except (requests.exceptions.RequestException, httpx.HTTPError) as e:
            logger.error(f"Failed to fetch market data from API for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error processing/storing market data for {symbol}: {e}")


    async def _fetch_and_store_fundamentals(self, config: Dict[str, Any]):
        """
        Helper: Fetches fundamental data from Alpha Vantage and stores it in TabularDB.
        """
        if self.run_mode == "backtest":
            logger.warning("Skipping fundamental fetch: Alpha Vantage API is snapshot-only and cannot be used in BACKTEST mode.")
            return

        symbol = config.get("symbol")
        if not symbol or not self.tabular_db:
            return

        logger.info(f"Refreshing fundamental data for {symbol}...")
        
        api_key = self.api_keys.get("alpha_vantage") or os.environ.get("ALPHA_VANTAGE_API_KEY")

        if not api_key:
            logger.error("Alpha Vantage API key not found.")
            return
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": api_key
        }

        try:
            # [Task 5.1] Use persistent http_client
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            api_data = response.json()
            
            if not api_data or "Symbol" not in api_data:
                logger.warning(f"Alpha Vantage returned no data for {symbol}: {api_data.get('Information')}")
                return

            data_to_store = {
                "symbol": api_data.get("Symbol"),
                "timestamp": self.get_current_time(),
                "market_cap": float(api_data.get("MarketCapitalization", 0)),
                "pe_ratio": float(api_data.get("PERatio", 0) if api_data.get("PERatio") != "None" else 0),
                "sector": api_data.get("Sector"),
                "industry": api_data.get("Industry"),
                "dividend_yield": float(api_data.get("DividendYield", 0) if api_data.get("DividendYield") != "None" else 0),
                "eps": float(api_data.get("EPS", 0) if api_data.get("EPS") != "None" else 0),
                "beta": float(api_data.get("Beta", 0) if api_data.get("Beta") != "None" else 0)
            }

            success = await self.tabular_db.upsert_data(
                table_name="fundamentals",
                data=data_to_store,
                unique_key="symbol"
            )
            
            if success:
                logger.info(f"Successfully refreshed and stored fundamentals for {symbol}")
            else:
                logger.error(f"Failed to store fundamentals for {symbol} in TabularDB.")

        except (requests.exceptions.RequestException, httpx.HTTPError) as e:
            logger.error(f"Failed to fetch fundamentals from Alpha Vantage for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error processing/storing fundamentals for {symbol}: {e}", exc_info=True)


    async def refresh_data_sources(self):
        """
        OPTIMIZED: Iterates through the data catalog and triggers
        updates for sources marked for refresh.
        """
        logger.info("Starting data source refresh...")
        
        for source_name, config in self.data_catalog.items():
            source_type = config.get("type")
            try:
                if source_type == "market_data_api":
                    await self._fetch_and_store_market_data(config)
                elif source_type == "news_api":
                    query = config.get("query")
                    if query:
                        logger.info(f"Refreshing news for query: {query}")
                        await self.fetch_news_data(query, force_refresh=True)
                elif source_type == "fundamental_api":
                    await self._fetch_and_store_fundamentals(config)
                else:
                    logger.debug(f"Skipping refresh for source type: {source_type}")
            except Exception as e:
                logger.error(f"Failed to refresh data source '{source_name}': {e}")

        logger.info("Data source refresh finished.")

    async def close(self):
        """[Task 5.1] Gracefully close the persistent HTTP client."""
        await self.http_client.aclose()
        logger.info("DataManager HTTP client closed.")
