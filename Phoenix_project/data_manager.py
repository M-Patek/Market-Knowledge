"""
DataManager for the Phoenix project.

Handles loading, caching, and refreshing of all data sources required
by the cognitive engine and agents (e.g., market data, news, fundamentals).
[Phase I Fix] Environment Isolation & Time Travel Prevention
"""

import logging
import json
import requests
import asyncio
import httpx # [Task 3.1] Import httpx for async I/O
import os # [Task 2] 导入 os
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd  # type: ignore
import redis  # type: ignore

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
    ):
        """
        Initializes the DataManager.

        Args:
            config: System configuration dictionary.
            redis_client: Client for Redis cache.
            tabular_db: (Optional) Client for tabular (SQL) data.
            temporal_db: (Optional) Client for time-series data.
        """
        # [Phase I Fix] Extract run_mode for key isolation
        self.run_mode = config.get("run_mode", "DEV").lower()
        self.config = config.get("data_manager", {})
        self.api_keys = config.get("api_keys", {})
        self.redis_client = redis_client
        self.tabular_db = tabular_db
        self.temporal_db = temporal_db
        self._simulation_time: Optional[datetime] = None

        # 加载数据目录 (Data Catalog)
        self.data_catalog = self._load_data_catalog(
            self.config.get("catalog_path", "data_catalog.json")
        )
        
        self.cache_ttl_sec = self.config.get("cache_ttl_sec", 300) # 5 分钟
        
        # API URL 配置
        self.news_api_url = self.config.get("news_api_url", "https://newsapi.org/v2/everything")
        self.market_data_api_url = self.config.get("market_data_api_url", "https://api.polygon.io/v2/aggs/ticker")
        
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
        return f"phx:{self.run_mode}:cache:{prefix}:{identifier}"

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Retrieves and deserializes data from Redis cache."""
        try:
            cached_data = self.redis_client.get(key)
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

    def _set_to_cache(self, key: str, data: Any):
        """Serializes and stores data in Redis cache with a TTL."""
        try:
            # 确保 data 可以被 JSON 序列化
            # (例如, Pydantic 模型需要 .model_dump())
            if hasattr(data, 'model_dump_json'):
                data_json = data.model_dump_json()
            elif isinstance(data, list) and data and hasattr(data[0], 'model_dump'):
                 data_json = json.dumps([d.model_dump() for d in data])
            else:
                data_json = json.dumps(data, default=str) # [Task 2] 添加 default=str
                
            self.redis_client.setex(
                key, self.cache_ttl_sec, data_json
            )
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

        # 缓存键可以基于 symbol 和时间范围
        cache_key = self._get_cache_key(
            "market_history", f"{symbol}_{start.isoformat()}_{end.isoformat()}"
        )
        
        cached = self._get_from_cache(cache_key)
        if cached:
            return pd.DataFrame(cached)

        # 从时序数据库 (Temporal DB) 获取
        logger.info(f"Fetching market history for {symbol} from TemporalDB.")
        try:
            # [Contract] 期望 TemporalDBClient 实现 query_range
            # 这是一个 DataFrame 返回接口
            df = await self.temporal_db.query_range(
                measurement="market_data",
                symbol=symbol,
                start=start,
                end=end,
                columns=["open", "high", "low", "close", "volume"],
            )
            
            if df is not None and not df.empty:
                # 缓存结果 (DataFrame -> dict -> JSON)
                self._set_to_cache(cache_key, df.to_dict(orient="records"))
            
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
            data_json = self.redis_client.get(live_key)
            
            if data_json:
                # 使用 model_validate_json 自动处理 datetime 和类型转换
                return MarketData.model_validate_json(data_json)
            
            # Data not found is expected for illiquid assets, verify logic before warning
            # logger.debug(f"No live market data found in Redis for {symbol}")
            return None
                
        except redis.RedisError as e:
            logger.error(f"Redis error getting live market data for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error validating market data for {symbol}: {e}")
            
        return None

    async def get_market_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """
        [Task 2.1] 批量获取市场数据。使用 asyncio.gather 并发拉取。
        """
        tasks = [self.get_latest_market_data(sym) for sym in symbols]
        results = await asyncio.gather(*tasks)
        
        # [Task 2] Strict Data Integrity Check (All-or-Nothing)
        # Prevent silent failures that could lead to naked exposure in pair trading
        failed_symbols = [sym for sym, res in zip(symbols, results) if res is None]
        
        if failed_symbols:
            raise ValueError(f"Data integrity check failed. Missing market data for symbols: {failed_symbols}")
            
        # Only return if ALL data is present
        return {sym: res for sym, res in zip(symbols, results)}

    async def get_news_data(self, query: str = "", limit: int = 10, start_date: datetime = None, end_date: datetime = None) -> List[NewsData]:
        """
        [Task 2.1] DataIterator 适配器接口。
        [Fix] 支持时间范围过滤，解决回测中的“新闻黑洞”问题。
        """
        # 简单的包装现有的 fetch_news_data
        # 注意：这里未来可以扩展为从 Redis 流 (REDIS_KEY_NEWS_STREAM) 读取实时新闻
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
        
        cached = self._get_from_cache(cache_key)
        if cached:
            data = FundamentalData(**cached)
            # [Task 2.1] Time Barrier: Invalidate cache if it contains future data relative to sim time
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
                self._set_to_cache(cache_key, fund_data)
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
        [Phase I Fix] Encapsulated PIT logic.
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
                # Use actual data timestamp, fallback to pit_date only if missing
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
        OPTIMIZED: Fetches news data from an external API,
        using caching. Replaces the mock.
        [Fix] Supports time range filtering for accurate backtesting.
        """
        cache_key = self._get_cache_key("news", query)
        
        # 如果提供了时间范围，将时间加入缓存键，防止不同时间段的查询混淆
        if start_date or end_date:
            cache_key = self._get_cache_key("news", f"{query}_{start_date}_{end_date}")
        
        if not force_refresh:
            cached = self._get_from_cache(cache_key)
            if cached:
                return [NewsData(**item) for item in cached]
        
        # 从 API 获取
        logger.info(f"Fetching news from API for query: {query}")
        
        # 检查 API 密钥
        news_api_key = self.api_keys.get("news_api")
        if not news_api_key:
            logger.error("News API key (api_keys.news_api) not configured.")
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
        # 移除值为 None 的参数
        params = {k: v for k, v in params.items() if v is not None}
        
        try:
            async with httpx.AsyncClient() as client: # [Task 3.1] Use httpx
                response = await client.get(self.news_api_url, params=params, timeout=10.0)
            response.raise_for_status() # 如果是 4xx/5xx 则抛出异常
            
            articles = response.json().get("articles", [])
            news_list = []
            
            for article in articles:
                news_item = NewsData(
                    id=article.get("url"), # 使用 URL 作为唯一 ID
                    symbol=query, # 粗略地将查询词作为 symbol
                    headline=article.get("title"),
                    summary=article.get("description"),
                    content=article.get("content"),
                    url=article.get("url"),
                    source=article.get("source", {}).get("name"),
                    timestamp=datetime.fromisoformat(article.get("publishedAt").replace("Z", "+00:00")),
                )
                news_list.append(news_item)
            
            if news_list:
                # 缓存结果
                self._set_to_cache(cache_key, news_list)
            
            return news_list

        except (requests.exceptions.RequestException, httpx.HTTPError) as e:
            logger.error(f"Failed to fetch news from API: {e}")
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
        
        # 1. 从 API 获取数据 (例如 Polygon.io)
        market_api_key = self.api_keys.get("polygon_api")
        if not market_api_key:
            logger.error("Market data API key (api_keys.polygon_api) not configured.")
            return

        # 示例: 获取过去一天的数据
        yesterday = (self.get_current_time() - timedelta(days=1)).strftime('%Y-%m-%d')
        url = f"{self.market_data_api_url}/{symbol}/range/1/day/{yesterday}/{yesterday}"
        params = {"apiKey": market_api_key}

        try:
            async with httpx.AsyncClient() as client: # [Task 3.1] Use httpx
                response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            if "results" in data:
                # 2. 准备数据
                df = pd.DataFrame(data["results"])
                df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "time"})
                # 将纳秒时间戳转换为 datetime
                df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
                df = df.set_index("time")
                
                # 3. 写入 TemporalDB
                await self.temporal_db.write_dataframe(
                    measurement="market_data",
                    df=df,
                    tags={"symbol": symbol}
                )
                logger.info(f"Successfully refreshed and stored market data for {symbol}")

        except (requests.exceptions.RequestException, httpx.HTTPError) as e:
            logger.error(f"Failed to fetch market data from API for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error processing/storing market data for {symbol}: {e}")


    async def _fetch_and_store_fundamentals(self, config: Dict[str, Any]):
        """
        [Task 2] 已实现
        Helper: Fetches fundamental data from Alpha Vantage and stores it in TabularDB.
        """
        # [Phase I Fix] Time Travel Prevention
        # Snapshot APIs cannot be used in Backtest mode
        if self.run_mode == "backtest":
            logger.warning("Skipping fundamental fetch: Alpha Vantage API is snapshot-only and cannot be used in BACKTEST mode.")
            return

        symbol = config.get("symbol")
        if not symbol or not self.tabular_db:
            logger.warning(f"Skipping fundamentals: Symbol ({symbol}) or TabularDB ({self.tabular_db}) missing.")
            return

        logger.info(f"Refreshing fundamental data for {symbol}...")
        
        # 1. 从 API 获取数据 (Alpha Vantage)
        api_key = self.api_keys.get("alpha_vantage") # 假设 'api_keys' 是从 config 加载的
        if not api_key:
            # [Task 2] 尝试从 env.example 定义的环境变量回退
            api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")

        if not api_key:
            logger.error("Alpha Vantage API key not found in config (api_keys.alpha_vantage) or env (ALPHA_VANTAGE_API_KEY).")
            return
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": api_key
        }

        try:
            async with httpx.AsyncClient() as client: # [Task 3.2] Use httpx
                response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            api_data = response.json()
            
            if not api_data or "Symbol" not in api_data or api_data.get("Symbol") is None:
                logger.warning(f"Alpha Vantage returned no data or invalid data for {symbol}: {api_data.get('Information')}")
                return

            # 2. 准备要写入的数据
            # (我们将 Alpha Vantage 键 映射到 我们的数据库列)
            data_to_store = {
                "symbol": api_data.get("Symbol"),
                "timestamp": self.get_current_time(), # [Time Machine] 使用统一时间
                "market_cap": float(api_data.get("MarketCapitalization", 0)),
                "pe_ratio": float(api_data.get("PERatio", 0) if api_data.get("PERatio") != "None" else 0),
                "sector": api_data.get("Sector"),
                "industry": api_data.get("Industry"),
                "dividend_yield": float(api_data.get("DividendYield", 0) if api_data.get("DividendYield") != "None" else 0),
                "eps": float(api_data.get("EPS", 0) if api_data.get("EPS") != "None" else 0),
                "beta": float(api_data.get("Beta", 0) if api_data.get("Beta") != "None" else 0)
            }

            # 3. 存入 TabularDB
            # (我们假设表名为 'fundamentals'，主键为 'symbol')
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
        updates for sources marked for refresh. Replaces placeholder.
        """
        logger.info("Starting data source refresh...")
        
        for source_name, config in self.data_catalog.items():
            source_type = config.get("type")
            
            try:
                if source_type == "market_data_api":
                    # [Task 3.1] Now natively async
                    await self._fetch_and_store_market_data(config)
                    
                elif source_type == "news_api":
                    # (异步)
                    query = config.get("query")
                    if query:
                        logger.info(f"Refreshing news for query: {query}")
                        await self.fetch_news_data(query, force_refresh=True)
                        
                elif source_type == "fundamental_api":
                    # (异步)
                    await self._fetch_and_store_fundamentals(config)
                    
                else:
                    logger.debug(f"Skipping refresh for source type: {source_type}")
                    
            except Exception as e:
                logger.error(f"Failed to refresh data source '{source_name}': {e}")

        logger.info("Data source refresh finished.")
