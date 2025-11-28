"""
数据迭代器
负责按时间顺序模拟数据流，用于回测。
[Phase II Fix] Data Structure Mismatch (Dict -> List)
[Phase VII Fix] News Black Hole (Time-Slicing)
[Phase VII Fix] Zombie Data Misguidance (Staleness Check)
"""
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional # 修复：导入 Optional

# FIX (E1): 导入统一后的核心模式
# 修正：将 'core.schemas...' 转换为 'Phoenix_project.core.schemas...'
from Phoenix_project.core.schemas.data_schema import MarketData, NewsData
# 修复：导入 DataManager
from Phoenix_project.data_manager import DataManager

class DataIterator:
    """
    一个生成器，用于模拟历史数据的时间流逝。
    它按时间戳顺序 'yield' 数据点或数据批次。
    
    [修复] __init__ 已被修改，以匹配 phoenix_project.py 的用法。
    它现在接收 DataManager，而不是预先加载的 DFs。
    """
    
    def __init__(
        self,
        config: Dict[str, Any], # 修复：添加 config
        data_manager: DataManager # 修复：添加 data_manager
    ):
        """
        [修复] 初始化迭代器。
        它现在只存储依赖项。实际数据将在 setup() 中加载。
        """
        self.config = config.get('backtesting', {}) # 修复：获取回测配置
        self.data_manager = data_manager
        
        self.step = pd.Timedelta(self.config.get('step_size', '1d')) # 修复：从配置获取步长
        
        # [Memory Opt] Chunking configuration
        self.chunk_size = pd.Timedelta(self.config.get('chunk_days', 30), unit='d')
        self.lookback_window = pd.Timedelta(self.config.get('lookback_days', 10), unit='d')
        
        # [Task 7.3] Staleness Threshold: Data older than this is "Zombie" and untradeable.
        # Default 4 days covers Friday Close -> Monday Open (3 days) + Buffer.
        self.max_staleness = pd.Timedelta(days=self.config.get('max_staleness_days', 4))
        
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.current_time: Optional[datetime] = None
        self.current_chunk_end: Optional[datetime] = None
        self.symbols: List[str] = []
        
        self.market_data_iters: Dict[str, pd.DataFrame] = {} # 修复：更改为存储 DF
        self.news_data: Optional[pd.DataFrame] = None # 修复：更改为存储 DF

        print(f"DataIterator initialized (unconfigured). Step size: {self.step}")

    async def setup(self, start_date: datetime, end_date: datetime, symbols: List[str]):
        """
        [修复] 新增方法，用于配置 DataIterator 并加载数据。
        (由 BacktestingEngine 调用)
        """
        self.start_date = pd.to_datetime(start_date, utc=True)
        self.end_date = pd.to_datetime(end_date, utc=True)
        self.current_time = self.start_date
        self.symbols = symbols
        
        # 1. [Memory Opt] Initial Chunk Load
        # Load just enough to get started, instead of everything.
        print(f"DataIterator: Preloading initial chunk for {symbols}...")
        
        initial_chunk_end = min(self.start_date + self.chunk_size, self.end_date)
        await self._load_data_chunk(self.start_date, initial_chunk_end)
        
        # 2. (修复) 重新采样 (Resample) 到统一的时间步长 (self.step)
        # 这确保了即使数据稀疏，我们也能按固定的间隔 (例如每天) "tick"
        
        # 创建回测期间的日期范围
        self.date_range = pd.date_range(start=self.start_date, end=self.end_date, freq=self.step, tz='UTC')
        
        print(f"DataIterator setup complete. {len(self.date_range)} steps to iterate.")

    async def _load_data_chunk(self, start_cursor: datetime, end_cursor: datetime):
        """
        [Memory Opt] Helper to load a specific time chunk with lookback overlap.
        """
        # 1. Calculate effective fetch range (including overlap for indicators)
        fetch_start = start_cursor - self.lookback_window
        fetch_end = end_cursor
        
        print(f"DataIterator: Loading chunk {fetch_start} -> {fetch_end} (Lookback: {self.lookback_window})")

        # 2. Clear old memory (force GC if needed, though dict reassignment helps)
        self.market_data_iters.clear()
        self.news_data = None

        # 3. Fetch Market Data with Privileged Access (allow_future_lookup=True)
        # [Task 2 Part 1] Key change: explicitly allow future lookup here
        tasks = [
            self.data_manager.get_market_data_history(
                sym, fetch_start, fetch_end, allow_future_lookup=True
            )
            for sym in self.symbols
        ]
        results = await asyncio.gather(*tasks)
        self.market_data_iters = {sym: df for sym, df in zip(self.symbols, results) if df is not None}

        # 4. Fetch News Data (Time-Sliced to avoid Black Hole)
        # Iterate day-by-day to ensure we don't hit the limit=1000 cap for the whole chunk
        all_news = []
        current_day = fetch_start
        day_step = pd.Timedelta(days=1)
        
        # Limit concurrency to avoid rate limits, or sequential is fine for robustness
        while current_day < fetch_end:
            next_day = min(current_day + day_step, fetch_end)
            try:
                daily_news = await self.data_manager.get_news_data(
                    limit=1000, # Per day limit is safer
                    start_date=current_day,
                    end_date=next_day
                )
                if daily_news:
                    all_news.extend(daily_news)
            except Exception as e:
                # Log but don't crash backtest on partial news failure
                print(f"DataIterator warning: Failed to fetch news for {current_day}: {e}")
            
            current_day = next_day

        # [Fix] Convert List[NewsData] to DataFrame for internal usage
        if all_news:
            # Dedup based on ID if needed, but set_index handles it mostly
            df = pd.DataFrame([n.model_dump() for n in all_news])
            # Ensure unique index if IDs overlap across boundaries
            df = df.drop_duplicates(subset=['id'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            self.news_data = df.set_index('timestamp').sort_index()
        else:
            self.news_data = None

        self.current_chunk_end = end_cursor

    def __aiter__(self):
        """ 
        异步迭代器协议入口 (Fix 1.2: Standard Sync def) 
        Returns self as the async iterator.
        """
        if self.current_time is None or self.date_range is None:
             raise RuntimeError("DataIterator must be configured with setup() before iteration.")
             
        # 重置
        self.current_time = self.start_date
        self._internal_date_iterator = iter(self.date_range)
        
        return self

    async def __anext__(self):
        """ 
        异步 next - [Recommended] 
        Natively handles async chunk loading without blocking.
        """
        try:
            # 1. Advance internal iterator manually to avoid blocking logic inside __next__
            # Note: We need to be careful not to double-consume if mixing sync/async, 
            # but standard usage is one or the other.
            current_tick = next(self._internal_date_iterator)
            self.current_time = current_tick

            # 2. Check Chunk Boundary
            if self.current_time > self.current_chunk_end:
                new_end = min(self.current_time + self.chunk_size, self.end_date)
                # Fully async load
                await self._load_data_chunk(self.current_time, new_end)
            
            # 3. Prepare batch (Logic duplicated from __next__ for thread-safety and performance)
            # [Task 1.3] Refactored to prevent Time Travel.
            # We strictly separate T-1 (Facts) from T (Opportunity).
            snapshot = {
                "timestamp": self.current_time,
                "market_data": [], # [Task 2.2] Standardized List[MarketData]
                "news_data": []
            }
            
            window_start = self.current_time - self.step
            
            for symbol, df in self.market_data_iters.items():
                # A. Past Candle (Strictly < T) -> The "Fact"
                past_slice = df.loc[:self.current_time - pd.Timedelta(seconds=1)]
                
                if not past_slice.empty:
                    row_prev = past_slice.iloc[-1]
                    
                    # [Task 7.3] Check for Zombie Data
                    data_age = self.current_time - row_prev.name
                    if data_age > self.max_staleness:
                        # Skip stale data (simulating suspension/delisting or lack of liquidity)
                        # Agents will see no data for this symbol and likely HOLD.
                        continue

                    # [Task 2.2] Construct MarketData object
                    market_data_obj = MarketData(
                        symbol=symbol,
                        timestamp=row_prev.name,
                        open=row_prev['open'],
                        high=row_prev['high'],
                        low=row_prev['low'],
                        close=row_prev['close'],
                        volume=row_prev['volume']
                    )
                    snapshot["market_data"].append(market_data_obj)

            if self.news_data is not None:
                news_slice = self.news_data.loc[window_start:self.current_time]
                if not news_slice.empty:
                    for ts, row in news_slice.iterrows():
                        snapshot["news_data"].append(
                            NewsData(
                                id=row.get('id', str(ts)),
                                source=row.get('source'),
                                timestamp=ts,
                                symbols=row.get('symbols', []),
                                content=row.get('content'),
                                headline=row.get('headline')
                            )
                        )
            
            return snapshot

        except StopIteration:
            raise StopAsyncIteration
