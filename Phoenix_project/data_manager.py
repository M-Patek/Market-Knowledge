"""
数据管理器 (DataManager)
负责从各种来源（CSV, Parquet, 数据库）加载、缓存和提供数据。
"""
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime
import os

# FIX (E1): 导入统一后的核心模式
from core.schemas.data_schema import MarketData, NewsData, EconomicIndicator
from config.loader import ConfigLoader

class DataManager:
    """
    集中管理所有数据的加载和访问。
    在真实系统中，这将连接到数据库或数据仓库。
    在当前版本中，它主要从文件（如 Parquet 或 CSV）加载数据。
    """
    
    # FIX (E5): 构造函数需要 ConfigLoader，而不是 dict
    def __init__(self, config_loader: ConfigLoader, data_catalog: Dict[str, Any]):
        self.config_loader = config_loader
        self.data_catalog = data_catalog
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        # data_base_path 可能在 system.yaml 中定义
        try:
            self.data_base_path = self.config_loader.get_system_config()["data_store"]["local_base_path"]
        except KeyError:
            print("Warning: 'data_store.local_base_path' not in system config. Using relative path.")
            self.data_base_path = "." # 回退到相对路径
            
        print(f"DataManager initialized. Base data path: {self.data_base_path}")

    def _load_data(self, data_id: str) -> pd.DataFrame:
        """
        内部辅助函数：根据 data_catalog 中的定义加载数据。
        """
        if data_id not in self.data_catalog:
            raise ValueError(f"Data ID '{data_id}' not found in data_catalog.json")
            
        if data_id in self.data_cache:
            return self.data_cache[data_id]

        config = self.data_catalog[data_id]
        file_path = os.path.join(self.data_base_path, config["path"])
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at: {file_path}")

        print(f"Loading data '{data_id}' from {file_path}...")
        
        try:
            if config["format"] == "parquet":
                df = pd.read_parquet(file_path)
            elif config["format"] == "csv":
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported data format: {config['format']}")
                
            # 确保时间戳列被正确解析并设为索引
            if "timestamp_col" in config:
                ts_col = config["timestamp_col"]
                df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
                df = df.set_index(ts_col).sort_index()
            
            self.data_cache[data_id] = df
            return df
        
        except Exception as e:
            print(f"Error loading data '{data_id}': {e}")
            raise

    def get_market_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        获取一个或多个资产的市场数据 (OHLCV)。
        """
        result = {}
        for symbol in symbols:
            # 假设 data_catalog 中的 ID 对应于 "market_data_{SYMBOL}"
            data_id = f"market_data_{symbol.upper()}"
            try:
                df = self._load_data(data_id)
                result[symbol] = df[ (df.index >= start_date) & (df.index <= end_date) ]
            except (ValueError, FileNotFoundError) as e:
                print(f"Warning: Could not load market data for {symbol}. {e}")
                
        return result

    def get_news_data(self, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        获取所有来源的新闻/事件数据。
        """
        try:
            # 假设 data_catalog 中有一个ID叫 "news_events"
            df = self._load_data("news_events")
            return df[ (df.index >= start_date) & (df.index <= end_date) ]
        except (ValueError, FileNotFoundError) as e:
            print(f"Warning: Could not load news data. {e}")
            return None

    def get_economic_indicators(self, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        获取宏观经济指标。
        """
        try:
            # 假设 data_catalog 中有一个ID叫 "economic_indicators"
            df = self._load_data("economic_indicators")
            return df[ (df.index >= start_date) & (df.index <= end_date) ]
        except (ValueError, FileNotFoundError) as e:
            print(f"Warning: Could not load economic indicators. {e}")
            return None

    def fetch_data_for_batch(self, start_time: datetime, end_time: datetime, symbols: List[str]) -> Dict[str, List[Any]]:
        """
        为 Orchestrator 的一个周期获取所需的所有数据，并转换为 Pydantic 模式。
        """
        # 1. 获取 MarketData (DataFrame)
        market_dfs = self.get_market_data(symbols, start_time, end_time)
        
        # 2. 获取 NewsData (DataFrame)
        news_df = self.get_news_data(start_time, end_time)
        
        # 3. 获取 EconomicIndicators (DataFrame)
        econ_df = self.get_economic_indicators(start_time, end_time)
        
        # --- 转换为 Pydantic 模式 ---
        
        batch_result = {
            "market_data": [],
            "news_data": [],
            "economic_indicators": []
        }

        # FIX (E1): 转换为 MarketData
        for symbol, df in market_dfs.items():
            for ts, row in df.iterrows():
                batch_result["market_data"].append(
                    MarketData(
                        symbol=symbol,
                        timestamp=ts.to_pydatetime(), # 转换 pandas Timestamp
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume']
                    )
                )
        
        # FIX (E1): 转换为 NewsData
        if news_df is not None:
            for ts, row in news_df.iterrows():
                 batch_result["news_data"].append(
                    NewsData(
                        id=row['id'],
                        source=row['source'],
                        timestamp=ts.to_pydatetime(),
                        symbols=row.get('symbols', []),
                        content=row['content'],
                        headline=row.get('headline')
                    )
                )

        # FIX (E1): 转换为 EconomicIndicator
        if econ_df is not None:
            for ts, row in econ_df.iterrows():
                 batch_result["economic_indicators"].append(
                    EconomicIndicator(
                        id=row['id'],
                        name=row['name'],
                        timestamp=ts.to_pydatetime(),
                        value=row['value'],
                        expected=row.get('expected'),
                        previous=row.get('previous')
                    )
                )
                
        return batch_result
