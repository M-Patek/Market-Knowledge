import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Union
import redis.asyncio as redis
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from Phoenix_project.core.schemas.data_schema import MarketData, PortfolioState, Order, Fill
from Phoenix_project.config.constants import REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE, REDIS_KEY_PORTFOLIO_LATEST
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

# --- Time Provider Abstraction ---
class TimeProvider(ABC):
    @abstractmethod
    def now(self) -> datetime:
        pass

class RealTimeProvider(TimeProvider):
    def now(self) -> datetime:
        return datetime.utcnow()

class SimulatedTimeProvider(TimeProvider):
    def __init__(self, start_time: datetime):
        self._current_time = start_time

    def now(self) -> datetime:
        return self._current_time
    
    def advance(self, delta: timedelta):
        self._current_time += delta
        
    def set_time(self, new_time: datetime):
        self._current_time = new_time

# --- Data Manager ---

class DataManager:
    """
    Centralized Data Manager for Phoenix Project.
    Handles data retrieval/storage across Redis (Hot), TimescaleDB (Warm), and S3/Tabular (Cold).
    """

    def __init__(
        self, 
        config: Dict[str, Any],
        redis_client: redis.Redis,
        temporal_db: Any = None,
        tabular_db: Any = None,
        s3_client: Any = None,
        time_provider: Optional[TimeProvider] = None 
    ):
        self.config = config
        self.redis_client = redis_client
        self.temporal_db = temporal_db
        self.tabular_db = tabular_db
        self.s3_client = s3_client
        
        # [Feature 1] Time Provider Integration
        self.time_provider = time_provider or RealTimeProvider()
        
        # Configurable staleness threshold (e.g., 5 minutes)
        self.max_data_age_seconds = self.config.get("max_data_age_seconds", 300) 
        
        logger.info(f"DataManager initialized. Mode: {type(self.time_provider).__name__}")

    async def get_current_time(self) -> datetime:
        """Returns current system time via the abstracted provider."""
        return self.time_provider.now()

    async def get_latest_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        Retrieves the single latest market data point for a symbol.
        Strategy: Redis Cache -> [Staleness Check] -> Temporal DB -> None
        """
        if not symbol: return None
        
        current_time = await self.get_current_time()
        
        # 1. Try Redis
        if self.redis_client:
            key = REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE.format(symbol=symbol)
            try:
                data = await self.redis_client.get(key)
                if data:
                    if isinstance(data, bytes): data = data.decode('utf-8')
                    market_data = MarketData.model_validate_json(data)
                    
                    # [Feature 2] Zombie Data Check & Read Repair
                    # Check if data is too old relative to current (simulated or real) time
                    data_age = (current_time - market_data.timestamp).total_seconds()
                    
                    if data_age > self.max_data_age_seconds:
                        logger.warning(f"Stale data detected for {symbol} in Redis (Age: {data_age:.1f}s). Triggering Read Repair.")
                        # Fall through to DB for fresh data
                    else:
                        return market_data
                        
            except Exception as e:
                logger.warning(f"Redis fetch/validate failed for {symbol}: {e}")

        # 2. Try Temporal DB (Fallback / Read Repair Source)
        if self.temporal_db:
            try:
                # Assuming temporal_db has a method for latest
                db_data = await self.temporal_db.get_latest_market_data(symbol)
                
                if db_data:
                    # [Feature 2] Read Repair: Write back to Redis if DB has fresher data
                    if self.redis_client:
                        try:
                            # Verify DB data isn't also stale (double zombie check)
                            db_age = (current_time - db_data.timestamp).total_seconds()
                            if db_age <= self.max_data_age_seconds:
                                key = REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE.format(symbol=symbol)
                                await self.redis_client.setex(key, 60, db_data.model_dump_json())
                                logger.info(f"Read Repair successful for {symbol}.")
                            else:
                                logger.warning(f"DB data for {symbol} is also stale (Age: {db_age:.1f}s).")
                        except Exception as e:
                            logger.error(f"Read Repair write-back failed: {e}")
                            
                    return db_data
                    
            except Exception as e:
                logger.error(f"TemporalDB fetch failed for {symbol}: {e}")
        
        return None

    # [Task P1-DATA-01] Batch Market Data Interface
    async def get_latest_market_data_batch(self, symbols: List[str]) -> Dict[str, MarketData]:
        """
        Retrieves the latest market data for a list of symbols efficiently.
        Prioritizes Redis MGET to minimize latency.
        Fallback to concurrent existing fetch method for misses.
        """
        if not symbols:
            return {}

        results: Dict[str, MarketData] = {}
        unique_symbols = list(set(symbols))
        missing_symbols = []

        # 1. Try Redis MGET (Fast Path)
        if self.redis_client:
            try:
                # Use standard key template
                keys = [REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE.format(symbol=sym) for sym in unique_symbols]
                
                # Async MGET
                values = await self.redis_client.mget(keys)
                
                for sym, val in zip(unique_symbols, values):
                    if val:
                        try:
                            # Parse JSON
                            if isinstance(val, bytes):
                                val = val.decode('utf-8')
                            data_dict = json.loads(val)
                            
                            # Reconstruct MarketData object
                            try:
                                md_obj = MarketData.model_validate(data_dict)
                            except AttributeError:
                                # Fallback if not Pydantic v2
                                md_obj = MarketData(**data_dict)
                            
                            # [Feature 2] Batch Zombie Check
                            # We can do a quick check here if we trust the timestamp in JSON
                            # Otherwise, delegate to the single fetch which has robust logic
                            current_time = await self.get_current_time()
                            # Ensure timestamp is datetime
                            ts = md_obj.timestamp
                            if isinstance(ts, str):
                                ts = datetime.fromisoformat(ts)
                                
                            if (current_time - ts).total_seconds() > self.max_data_age_seconds:
                                missing_symbols.append(sym) # Treat as miss to trigger repair
                            else:
                                results[sym] = md_obj
                                
                        except (json.JSONDecodeError, TypeError, ValueError, Exception) as e:
                            logger.warning(f"DataManager: Corrupt/Stale Redis data for {sym}: {e}")
                            missing_symbols.append(sym)
                    else:
                        missing_symbols.append(sym)
            except Exception as e:
                logger.error(f"DataManager: Redis batch fetch failed: {e}")
                missing_symbols = unique_symbols # Fail safe: try all in DB
        else:
            missing_symbols = unique_symbols

        # 2. Fallback to Concurrent Single Fetch (Slow Path)
        # Using existing get_latest_market_data (Redundant Redis check accepted for compatibility)
        if missing_symbols:
            # Limit concurrency to prevent overwhelming the DB/App
            semaphore = asyncio.Semaphore(20) 
            
            async def fetch_safe(sym):
                async with semaphore:
                    try:
                        # [Fix] Removed 'use_cache=False' to match existing signature
                        return await self.get_latest_market_data(sym)
                    except Exception as e:
                        logger.error(f"DataManager: Error fetching {sym}: {e}")
                        return None

            tasks = [fetch_safe(sym) for sym in missing_symbols]
            
            if tasks:
                db_items = await asyncio.gather(*tasks, return_exceptions=False)
                
                for sym, item in zip(missing_symbols, db_items):
                    if item and isinstance(item, MarketData):
                        results[sym] = item

        logger.debug(f"Batch data fetch: {len(results)}/{len(unique_symbols)} symbols retrieved.")
        return results

    async def get_market_data_history(self, symbol: str, start: datetime, end: datetime) -> Any:
        """Fetches historical data from TemporalDB."""
        if self.temporal_db:
            return await self.temporal_db.get_market_data_range(symbol, start, end)
        return None
    
    # [Feature 3] Placeholder for News/Event Data
    async def get_news_events(self, symbol: str, lookback_days: int = 7) -> List[Dict]:
        """
        Retrieves news events for a symbol.
        Currently a placeholder; implementation would query S3 or a news API.
        """
        start_date = (await self.get_current_time()) - timedelta(days=lookback_days)
        # TODO: Implement actual news retrieval logic
        logger.info(f"Fetching news for {symbol} since {start_date}")
        return []

    async def get_current_portfolio(self) -> Optional[PortfolioState]:
        """Retrieves the latest portfolio state."""
        if self.redis_client:
            try:
                data = await self.redis_client.get(REDIS_KEY_PORTFOLIO_LATEST)
                if data:
                    return PortfolioState.model_validate_json(data)
            except Exception as e:
                logger.error(f"Failed to fetch portfolio from Redis: {e}")
        
        # Fallback to DB
        if self.tabular_db:
            # Implement DB fetch logic here
            pass
            
        return None

    async def close(self):
        """Closes all database connections."""
        if self.redis_client:
            await self.redis_client.close()
        if self.temporal_db and hasattr(self.temporal_db, 'close'):
            await self.temporal_db.close()
        if self.tabular_db and hasattr(self.tabular_db, 'close'):
            await self.tabular_db.close()
        logger.info("DataManager closed.")
