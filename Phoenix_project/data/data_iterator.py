"""
数据迭代器
负责按时间顺序模拟数据流，用于回测。
"""
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Generator, Optional # 修复：导入 Optional

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
        # start_date: datetime, # 修复：移除
        # end_date: datetime, # 修复：移除
        # market_data_dfs: Dict[str, pd.DataFrame], # 修复：移除
        # news_data_df: Optional[pd.DataFrame] = None, # 修复：移除
        # step: timedelta = timedelta(days=1) # 修复：移除
    ):
        """
        [修复] 初始化迭代器。
        它现在只存储依赖项。实际数据将在 setup() 中加载。
        """
        self.config = config.get('backtesting', {}) # 修复：获取回测配置
        self.data_manager = data_manager
        
        self.step = pd.Timedelta(self.config.get('step_size', '1d')) # 修复：从配置获取步长
        
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.current_time: Optional[datetime] = None
        
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
        
        # 1. 从 DataManager 加载所有需要的数据
        print(f"DataIterator: Preloading data for {symbols} from {start_date} to {end_date}...")
        
        # (我们加载一个稍大的窗口，以确保有足够的数据用于回溯 (lookbacks))
        preload_start = self.start_date - pd.Timedelta(days=90) 
        
        # [主人喵 Phase 3 修复] 并发加载历史数据 (DataFrame)
        tasks = [
            self.data_manager.get_market_data_history(sym, preload_start, self.end_date)
            for sym in symbols
        ]
        results = await asyncio.gather(*tasks)
        self.market_data_iters = {sym: df for sym, df in zip(symbols, results) if df is not None}
        
        # 异步加载新闻
        self.news_data = await self.data_manager.get_news_data(
            preload_start, self.end_date
        )
        
        # 2. (修复) 重新采样 (Resample) 到统一的时间步长 (self.step)
        # 这确保了即使数据稀疏，我们也能按固定的间隔 (例如每天) "tick"
        
        # 创建回测期间的日期范围
        self.date_range = pd.date_range(start=self.start_date, end=self.end_date, freq=self.step, tz='UTC')
        
        print(f"DataIterator setup complete. {len(self.date_range)} steps to iterate.")

    def __iter__(self):
        """
        [修复] __iter__ 现在返回 self，因为 __next__ 中有迭代逻辑。
        """
        if self.current_time is None or self.date_range is None:
             raise RuntimeError("DataIterator must be configured with setup() before iteration.")
             
        # 重置迭代器
        self.current_time = self.start_date
        self._internal_date_iterator = iter(self.date_range)
        return self

    def __next__(self) -> Generator[Dict[str, List[Any]], None, None]:
        """
        [修复] __next__ 现在包含迭代逻辑。
        它在 date_range 上迭代，并为该时间戳拉取所有数据。
        """
        
        try:
            # 1. 获取下一个时间戳 "tick"
            current_tick = next(self._internal_date_iterator)
            self.current_time = current_tick
        except StopIteration:
            raise StopIteration # 迭代结束

        # 2. 准备该 "tick" 的数据批次
        batch_data = {
            "timestamp": self.current_time, # [修复] 包含当前时间戳
            "market_data": [],
            "news_data": []
        }
        
        # 3. 检索市场数据
        # (我们查找在 [current_tick - step, current_tick] 之间的数据)
        window_start = self.current_time - self.step
        
        for symbol, df in self.market_data_iters.items():
            # 修复：使用 .loc 进行基于时间戳的切片
            data_slice = df.loc[window_start:self.current_time]
            if not data_slice.empty:
                # (我们只取这个窗口中的最后一条记录，以模拟 "tick")
                # (一个更复杂的迭代器可能会 yield 窗口内的所有数据)
                latest_row = data_slice.iloc[-1]
                
                batch_data["market_data"].append(
                    MarketData(
                        symbol=symbol,
                        timestamp=latest_row.name, # 索引是时间戳
                        open=latest_row['open'],
                        high=latest_row['high'],
                        low=latest_row['low'],
                        close=latest_row['close'],
                        volume=latest_row['volume']
                    )
                )

        # 4. 检索新闻数据
        if self.news_data is not None:
            # 修复：使用 .loc 进行基于时间戳的切片
            news_slice = self.news_data.loc[window_start:self.current_time]
            if not news_slice.empty:
                for ts, row in news_slice.iterrows():
                    batch_data["news_data"].append(
                        NewsData(
                            id=row.get('id', str(ts)),
                            source=row.get('source'),
                            timestamp=ts,
                            symbols=row.get('symbols', []),
                            content=row.get('content'),
                            headline=row.get('headline')
                        )
                    )
        
        # 5. 始终 yield，即使批次为空 (模拟时间的流逝)
        return batch_data

    # 修复：移除旧的 __iter__ 逻辑，因为它已被移到 __next__
    # def __iter__(self) -> Generator[Dict[str, List[Any]], None, None]: ...
    
    # 修复：添加在 ai/retriever.py (旧版) 中使用的 (但 data/data_iterator.py 中没有的) 方法
    async def __aiter__(self):
        """ 异步迭代器 (包装同步迭代器) """
        if self.current_time is None or self.date_range is None:
             raise RuntimeError("DataIterator must be configured with setup() before iteration.")
             
        # 重置
        self.current_time = self.start_date
        self._internal_date_iterator = iter(self.date_range)
        
        return self

    async def __anext__(self):
        """ 异步 next """
        try:
            # 在线程中运行同步的 __next__
            batch = await asyncio.to_thread(self.__next__)
            return batch
        except StopIteration:
            raise StopAsyncIteration
