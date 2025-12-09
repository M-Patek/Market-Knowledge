"""
Phoenix_project/data/data_iterator.py
[Phase 4 Task 4] Clean Zombie Ticks.
Implement Trading Calendar filtering to skip weekends and holidays.
"""
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas_market_calendars as mcal # [Task 4] New dependency

from Phoenix_project.core.schemas.data_schema import MarketData, NewsData
from Phoenix_project.data_manager import DataManager

class DataIterator:
    """
    一个生成器，用于模拟历史数据的时间流逝。
    [Fix] 引入交易日历过滤，跳过非交易时段。
    """
    
    def __init__(
        self,
        config: Dict[str, Any], 
        data_manager: DataManager 
    ):
        self.config = config.get('backtesting', {}) 
        self.data_manager = data_manager
        
        self.step = pd.Timedelta(self.config.get('step_size', '1d')) 
        self.chunk_size = pd.Timedelta(self.config.get('chunk_days', 30), unit='d')
        self.lookback_window = pd.Timedelta(self.config.get('lookback_days', 10), unit='d')
        self.max_staleness = pd.Timedelta(days=self.config.get('max_staleness_days', 4))
        
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.current_time: Optional[datetime] = None
        self.current_chunk_end: Optional[datetime] = None
        self.symbols: List[str] = []
        
        self.market_data_iters: Dict[str, pd.DataFrame] = {} 
        self.news_data: Optional[pd.DataFrame] = None 
        
        # [Task 4] Calendar config
        self.exchange = self.config.get('exchange', 'NYSE') 

        print(f"DataIterator initialized. Exchange: {self.exchange}, Step: {self.step}")

    async def setup(self, start_date: datetime, end_date: datetime, symbols: List[str]):
        """
        [Fix] Configure DataIterator with Calendar-aware schedule.
        """
        self.start_date = pd.to_datetime(start_date, utc=True)
        self.end_date = pd.to_datetime(end_date, utc=True)
        self.current_time = self.start_date
        self.symbols = symbols
        
        print(f"DataIterator: Preloading initial chunk for {symbols}...")
        initial_chunk_end = min(self.start_date + self.chunk_size, self.end_date)
        await self._load_data_chunk(self.start_date, initial_chunk_end)
        
        # [Task 4] Generate Valid Trading Schedule
        try:
            nyse = mcal.get_calendar(self.exchange)
            schedule = nyse.schedule(start_date=self.start_date, end_date=self.end_date)
            
            # Create a valid date range intersection
            
            if self.step == pd.Timedelta('1d'):
                # For daily, use schedule dates directly (market close time)
                # schedule.index is DatetimeIndex of days
                # We usually want the 'market_close' time for daily bars
                self.date_range = mcal.date_range(schedule, frequency='1D').tz_convert('UTC')
            else:
                # For intraday, we need to generate range and filter
                # mcal date_range handles breaks automatically if frequency is supported
                self.date_range = mcal.date_range(schedule, frequency=self.step).tz_convert('UTC')
                
            print(f"Calendar schedule generated. {len(self.date_range)} valid trading steps.")
            
        except Exception as e:
            print(f"Warning: Failed to load market calendar ({e}). Fallback to raw date range (including weekends).")
            self.date_range = pd.date_range(start=self.start_date, end=self.end_date, freq=self.step, tz='UTC')

    async def _load_data_chunk(self, start_cursor: datetime, end_cursor: datetime):
        """
        [Memory Opt] Helper to load a specific time chunk with lookback overlap.
        """
        fetch_start = start_cursor - self.lookback_window
        fetch_end = end_cursor
        
        print(f"DataIterator: Loading chunk {fetch_start} -> {fetch_end}")

        self.market_data_iters.clear()
        self.news_data = None

        tasks = [
            self.data_manager.get_market_data_history(
                sym, fetch_start, fetch_end
            )
            for sym in self.symbols
        ]
        results = await asyncio.gather(*tasks)
        self.market_data_iters = {sym: df for sym, df in zip(self.symbols, results) if df is not None}

        all_news = []
        current_day = fetch_start
        day_step = pd.Timedelta(days=1)
        
        while current_day < fetch_end:
            next_day = min(current_day + day_step, fetch_end)
            try:
                daily_news = await self.data_manager.fetch_news_data(
                    limit=1000, 
                    start_time=current_day,
                    end_time=next_day
                )
                if daily_news:
                    all_news.extend(daily_news)
            except Exception as e:
                print(f"DataIterator warning: Failed to fetch news for {current_day}: {e}")
            current_day = next_day

        if all_news:
            # Normalize News Dict -> DataFrame
            # DataManager.fetch_news_data returns List[Dict]
            df = pd.DataFrame(all_news)
            df.drop_duplicates(subset=['id'], inplace=True) # Ensure unique
            
            # Ensure timestamp column is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                self.news_data = df.set_index('timestamp').sort_index()
            else:
                self.news_data = None
        else:
            self.news_data = None

        self.current_chunk_end = end_cursor

    def __aiter__(self):
        if self.current_time is None or self.date_range is None:
             raise RuntimeError("DataIterator must be configured with setup() before iteration.")
        self.current_time = self.start_date
        self._internal_date_iterator = iter(self.date_range)
        return self

    async def __anext__(self):
        try:
            current_tick = next(self._internal_date_iterator)
            self.current_time = current_tick

            # Check Chunk Boundary
            if self.current_time > self.current_chunk_end:
                new_end = min(self.current_time + self.chunk_size, self.end_date)
                await self._load_data_chunk(self.current_time, new_end)
            
            snapshot = {
                "timestamp": self.current_time,
                "market_data": [],
                "news_data": []
            }
            
            # window_start is inclusive of the step duration backward
            window_start = self.current_time - self.step
            
            for symbol, df in self.market_data_iters.items():
                # Past Candle (Strictly < T)
                past_slice = df.loc[:self.current_time - pd.Timedelta(seconds=1)]
                
                if not past_slice.empty:
                    row_prev = past_slice.iloc[-1]
                    
                    data_age = self.current_time - row_prev.name
                    if data_age > self.max_staleness:
                        continue

                    market_data_obj = MarketData(
                        symbol=symbol,
                        timestamp=row_prev.name, # Use actual candle time
                        open=row_prev['open'],
                        high=row_prev['high'],
                        low=row_prev['low'],
                        close=row_prev['close'],
                        volume=row_prev['volume']
                    )
                    snapshot["market_data"].append(market_data_obj)

            if self.news_data is not None:
                # Slicing news within the step window
                news_slice = self.news_data.loc[window_start:self.current_time]
                
                if not news_slice.empty:
                    for ts, row in news_slice.iterrows():
                        snapshot["news_data"].append(
                            NewsData(
                                id=str(row.get('id', ts)),
                                source=str(row.get('source', 'unknown')),
                                timestamp=ts,
                                symbols=row.get('symbols', []), # Ensure list
                                content=str(row.get('content', '')),
                                headline=str(row.get('headline', ''))
                            )
                        )
            
            # Return as list of dicts/objects to be consistent with Orchestrator injection
            flat_events = []
            for md in snapshot["market_data"]:
                flat_events.append(md.model_dump())
            for nd in snapshot["news_data"]:
                flat_events.append(nd.model_dump())
                
            return flat_events

        except StopIteration:
            raise StopAsyncIteration
