"""
数据管理器 (DataManager)
负责从各种来源（CSV, Parquet, 数据库）加载、缓存和提供数据。

[阶段 3 更新]
- 注入 ConfigLoader 和 Redis 客户端。
- 添加 `get_latest_market_data` 方法。
- `get_latest_market_data` 实现了双模式逻辑：
  - "production": 从 Redis ('latest_prices') 读取。
  - "development": 从文件缓存读取最后一行。
"""
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime
import os
import redis # <-- [阶段 3] 添加
from Phoenix_project.monitor.logging import get_logger # <-- [阶段 3] 添加
import json # 修复：导入 json

# FIX (E1): 导入统一后的核心模式
from Phoenix_project.core.schemas.data_schema import MarketData, NewsData, EconomicIndicator
from Phoenix_project.config.loader import ConfigLoader

logger = get_logger(__name__) # <-- [阶段 3] 添加

class DataManager:
    """
    集中管理所有数据的加载和访问。
    在真实系统中，这将连接到数据库或数据仓库。
    在当前版本中，它主要从文件（如 Parquet 或 CSV）加载数据。
    """
    
    # FIX (E5): 构造函数需要 ConfigLoader，而不是 dict
    # [阶段 3] 更改: 构造函数
    def __init__(self, config_loader: ConfigLoader, data_catalog: Dict[str, Any]):
        self.config_loader = config_loader
        self.data_catalog = data_catalog
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        # --- [阶段 3] 新增 ---
        try:
            # 1. 加载系统配置
            self.system_config = self.config_loader.load_config('system.yaml')
            if not self.system_config:
                 raise FileNotFoundError("system.yaml not loaded")
                 
            # 2. 设置环境
            self.environment = self.system_config.get("system", {}).get("environment", "development")
            
            # 3. 初始化 Redis 客户端
            self.redis_client = redis.StrictRedis(
                host=os.environ.get('REDIS_HOST', 'redis'),
                port=int(os.environ.get('REDIS_PORT', 6379)),
                db=0,
                decode_responses=True # <-- 重要: hget 返回 str 而不是 bytes
            )
            self.redis_client.ping()
            logger.info(f"DataManager 已连接到 Redis。")
            
        except Exception as e:
            logger.error(f"DataManager 初始化失败 (Redis 或 Config): {e}", exc_info=True)
            # 根据需要决定是否在启动时失败
            # 修复：在测试期间 (例如 run_training.py) Redis 可能不可用
            logger.warning("DataManager 无法连接到 Redis。在非生产环境中继续。")
            self.redis_client = None # 修复：设置为 None
            # raise # 修复：移除 raise
            
        # 4. 设置 data_base_path
        # data_base_path 可能在 system.yaml 中定义
        try:
            # (使用已加载的 self.system_config)
            # 修复：路径在 system.yaml 中是 'data_store.local_base_path'
            self.data_base_path = self.system_config["data_store"]["local_base_path"]
        except KeyError:
            logger.warning("Warning: 'data_store.local_base_path' not in system config. Using relative path '.'")
            self.data_base_path = "." # 回退到相对路径
        # --- [阶段 3 结束] ---
            
        logger.info(f"DataManager initialized in '{self.environment}' mode. Base data path: {self.data_base_path}")

    def _load_data(self, data_id: str) -> pd.DataFrame:
        """
        内部辅助函数：根据 data_catalog 中的定义加载数据。
        """
        if data_id not in self.data_catalog:
            # 修复：data_catalog 可能没有 'data_assets' 键
            assets = self.data_catalog.get("data_assets", [])
            config = next((item for item in assets if item.get("asset_id") == data_id), None)
            if config is None:
                 raise ValueError(f"Data ID '{data_id}' not found in data_catalog.json")
        else:
            # 修复：旧版 catalog 格式
            config = self.data_catalog[data_id]

        if data_id in self.data_cache:
            return self.data_cache[data_id]
            
        # 修复：路径在 data_catalog (旧版) 中是 'path'，
        # 但在 data_catalog.json (新版) 中是 'storage_format' / 'asset_id'
        # 我们需要一个更稳健的路径查找
        
        # 尝试从 asset_id (新版) 构建路径
        # "asset_id": "asset_universe_daily_bars"
        # "description": "... Stored in data_cache/asset_data_{ticker}_{hash}.parquet."
        # 这个逻辑太复杂了，无法仅根据 catalog 实现。
        
        # 修复：回退到 data_manager.py (旧版) 的简单逻辑
        # (假设 data_id 是文件名，例如 "market_data_AAPL.parquet")
        
        file_path = os.path.join(self.data_base_path, f"{data_id}.parquet")
        if not os.path.exists(file_path):
             file_path_csv = os.path.join(self.data_base_path, f"{data_id}.csv")
             if os.path.exists(file_path_csv):
                 file_path = file_path_csv
             else:
                raise FileNotFoundError(f"Data file not found at: {file_path} (or .csv)")

        logger.info(f"Loading data '{data_id}' from {file_path}...")
        
        try:
            if file_path.endswith(".parquet"):
                df = pd.read_parquet(file_path)
            elif file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported data format: {file_path}")
                
            # 确保时间戳列被正确解析并设为索引
            # 修复：检查 catalog (新版) 的 schema
            schema_ref = config.get("schema", {}).get("$ref")
            ts_col = "timestamp" # 默认
            if schema_ref:
                 schema_name = schema_ref.split('/')[-1]
                 schema_def = self.data_catalog.get("definitions", {}).get(schema_name, {})
                 # 寻找 "format": "date-time"
                 for prop, props in schema_def.get("properties", {}).items():
                     if props.get("format") == "date-time":
                         ts_col = prop # (例如 "available_at")
                         break
            
            # 修复：检查列名 (例如 'available_at', 'Date', 'timestamp')
            if ts_col in df.columns:
                df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
                df = df.set_index(ts_col).sort_index()
            elif 'timestamp' in df.columns:
                 df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                 df = df.set_index('timestamp').sort_index()
            elif 'Date' in df.columns:
                 df['Date'] = pd.to_datetime(df['Date'], utc=True)
                 df = df.set_index('Date').sort_index()
            else:
                 logger.warning(f"No obvious timestamp column found in {data_id}. Using row index.")
            
            # 修复：重命名 (例如 'Open' -> 'open')
            df.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
            }, inplace=True)

            self.data_cache[data_id] = df
            return df
        
        except Exception as e:
            logger.error(f"Error loading data '{data_id}': {e}")
            raise

    # --- [阶段 3] 新增方法 ---
    def get_latest_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        获取单个资产的最新市场数据。
        在 "production" 模式下，从 Redis 实时读取。
        在 "development" 模式下，从文件缓存读取最后一条记录。
        """
        
        if self.environment == "production":
            # 生产模式：从 Redis HSET 'latest_prices' 读取
            if not self.redis_client:
                 logger.error("[PROD] Cannot get latest market data, Redis client not connected.")
                 return None
            try:
                price_str = self.redis_client.hget("latest_prices", symbol)
                
                if price_str is None:
                    logger.warning(f"[PROD] Redis 中没有 {symbol} 的实时价格。")
                    return None
                    
                price = float(price_str)
                
                # 模拟 MarketData 对象，因为我们只有价格
                return MarketData(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=0 # 实时流中没有 OHLCV，只有 price
                )
            except Exception as e:
                logger.error(f"[PROD] 从 Redis 获取 {symbol} 价格失败: {e}", exc_info=True)
                return None
        
        else:
            # 开发 (回测) 模式：从文件缓存读取
            logger.debug(f"[DEV] 正在从文件缓存中获取 {symbol} 的最新市场数据。")
            data_id = f"market_data_{symbol.upper()}"
            try:
                # 修复：确保 _load_data 知道如何加载 (例如 market_data_AAPL.parquet)
                df = self._load_data(data_id)
                if df.empty:
                    logger.warning(f"[DEV] {symbol} 的数据文件为空。")
                    return None
                
                latest_row = df.iloc[-1]
                
                # 确保索引是 pd.Timestamp
                ts = latest_row.name
                if not isinstance(ts, pd.Timestamp):
                     ts = pd.to_datetime(ts, utc=True)

                return MarketData(
                    symbol=symbol,
                    timestamp=ts.to_pydatetime(), # .name 是索引 (timestamp)
                    open=latest_row['open'],
                    high=latest_row['high'],
                    low=latest_row['low'],
                    close=latest_row['close'],
                    volume=latest_row['volume']
                )
            except (ValueError, FileNotFoundError, IndexError) as e:
                logger.warning(f"[DEV] 无法为 {symbol} 加载最新市场数据 (ID: {data_id}): {e}")
                return None

    # --- 现有方法 (保持不变，用于回测) ---

    def get_market_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        获取一个或多个资产的市场数据 (OHLCV)。
        (此方法用于历史回测)
        """
        result = {}
        for symbol in symbols:
            # 假设 data_catalog 中的 ID 对应于 "market_data_{SYMBOL}"
            data_id = f"market_data_{symbol.upper()}"
            try:
                df = self._load_data(data_id)
                # 修复：确保 start_date 和 end_date 是 tz-aware
                start_date = pd.to_datetime(start_date, utc=True)
                end_date = pd.to_datetime(end_date, utc=True)
                
                result[symbol] = df[ (df.index >= start_date) & (df.index <= end_date) ]
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"Warning: Could not load market data for {symbol} (ID: {data_id}). {e}")
                
        return result

    def get_news_data(self, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        获取所有来源的新闻/事件数据。
        (此方法用于历史回测)
        """
        try:
            # 假设 data_catalog 中有一个ID叫 "news_events"
            data_id = "news_events" # 修复：硬编码
            df = self._load_data(data_id)
            # 修复：确保 start_date 和 end_date 是 tz-aware
            start_date = pd.to_datetime(start_date, utc=True)
            end_date = pd.to_datetime(end_date, utc=True)
            return df[ (df.index >= start_date) & (df.index <= end_date) ]
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"Warning: Could not load news data (ID: {data_id}). {e}")
            return None

    def get_economic_indicators(self, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        获取宏观经济指标。
        (此方法用于历史回测)
        """
        try:
            # 假设 data_catalog 中有一个ID叫 "economic_indicators"
            data_id = "economic_indicators" # 修复：硬编码
            df = self._load_data(data_id)
            # 修复：确保 start_date 和 end_date 是 tz-aware
            start_date = pd.to_datetime(start_date, utc=True)
            end_date = pd.to_datetime(end_date, utc=True)
            return df[ (df.index >= start_date) & (df.index <= end_date) ]
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"Warning: Could not load economic indicators (ID: {data_id}). {e}")
            return None

    def fetch_data_for_batch(self, start_time: datetime, end_time: datetime, symbols: List[str]) -> Dict[str, List[Any]]:
        """
        为 Orchestrator 的一个周期获取所需的所有数据，并转换为 Pydantic 模式。
        (此方法用于历史回测)
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
                        id=row.get('id', str(ts)), # 回退 id
                        source=row.get('source'),
                        timestamp=ts.to_pydatetime(),
                        symbols=row.get('symbols', []),
                        content=row.get('content'),
                        headline=row.get('headline')
                    )
                )

        # FIX (E1): 转换为 EconomicIndicator
        if econ_df is not None:
            for ts, row in econ_df.iterrows():
                 batch_result["economic_indicators"].append(
                    EconomicIndicator(
                        id=row.get('id', str(ts)), # 回退 id
                        name=row.get('name'), # 修复：应为 'name' (根据 schema)
                        timestamp=ts.to_pydatetime(),
                        value=row.get('value'), # 修复：应为 'value' (根据 schema)
                        expected=row.get('expected'), # 修复：应为 'expected' (根据 schema)
                        previous=row.get('previous')
                    )
                )
                
        return batch_result
        
    # 修复：添加 worker.py 中使用的 (但 data_manager.py 中没有的) 方法
    async def refresh_data_sources(self):
        """
        (被 Orchestrator system tick 调用)
        (这是一个占位符，用于触发后台数据刷新)
        """
        logger.debug("Mock refresh_data_sources() called.")
        await asyncio.sleep(0.01) # 模拟 async
        pass

    async def fetch_market_data(self, symbols: List[str]) -> List[MarketData]:
        """
        (被 Orchestrator process_task 调用)
        获取最新的市场数据。
        """
        logger.debug(f"Async fetch_market_data for {symbols}")
        results = []
        for symbol in symbols:
             # (这是一个同步调用)
            data = self.get_latest_market_data(symbol)
            if data:
                results.append(data)
        return results

    async def fetch_news_data(self, symbols: List[str]) -> List[NewsData]:
        """
        (被 Orchestrator process_task 调用)
        获取最新的新闻数据。 (模拟)
        """
        logger.debug(f"Async fetch_news_data for {symbols} (mocked)")
        # (在真实系统中，这会查询 ES/TemporalDB)
        if self.news_data is None:
             try:
                self.get_news_data(datetime.now() - timedelta(days=90), datetime.now())
             except Exception:
                 pass # 加载失败
                 
        if self.news_data is not None and not self.news_data.empty:
             latest_news_row = self.news_data.iloc[-1]
             ts = latest_news_row.name
             return [
                 NewsData(
                    id=latest_news_row.get('id', str(ts)),
                    source=latest_news_row.get('source'),
                    timestamp=ts.to_pydatetime(),
                    symbols=latest_news_row.get('symbols', symbols),
                    content=latest_news_row.get('content'),
                    headline=latest_news_row.get('headline')
                 )
             ]
        return []
