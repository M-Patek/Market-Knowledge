# data_manager.py
import logging
import os
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, Any, List, Optional

from storage.s3_client import S3Client
from features.store import FeatureStore
from ai.vector_db_client import VectorDBClient
from ai.tabular_db_client import TabularDBClient
from ai.temporal_db_client import TemporalDBClient


class BaseAltDataClient(ABC):
    """
    替代数据客户端的抽象基类 (Task 2.1)。
    确保用于获取外部数据的标准化接口。
    """
    @abstractmethod
    def fetch_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """获取数据并将其作为时间点正确的 DataFrame 返回。"""
        pass


class DataManager:
    """
    作为所有数据加载、缓存和检索操作的中心枢纽。
    """

    def __init__(self,
                         s3_client: S3Client,
                         feature_store: FeatureStore,
                         vector_db_client: VectorDBClient,
                         tabular_db_client: TabularDBClient,
                         temporal_db_client: TemporalDBClient,
                         alt_data_clients: List[BaseAltDataClient] = None,
                         feature_cache_dir: str = "data_cache/ai_features/"):
        """
        初始化 DataManager。

        Args:
            s3_client: S3 存储的客户端。
            feature_store: 用于特征工程的客户端。
            vector_db_client: 向量数据库的客户端。
            tabular_db_client: 表格数据库的客户端。
            temporal_db_client: 时间序列数据库的客户端。
            alt_data_clients: (Task 2.1) 替代数据客户端的列表。
            feature_cache_dir: (Task 3.4) AI 特征的持久缓存目录。
        """
        self.logger = logging.getLogger("PhoenixProject.DataManager")
        self.s3_client = s3_client
        self.feature_store = feature_store
        self.vector_db_client = vector_db_client
        self.tabular_db_client = tabular_db_client
        self.temporal_db_client = temporal_db_client
        self.alt_data_clients = alt_data_clients or []
        self.feature_cache_dir = feature_cache_dir

        self.market_data_cache = {}
        self.feature_data_cache = {} # 内存运行时缓存
        os.makedirs(self.feature_cache_dir, exist_ok=True) # 确保磁盘缓存目录存在

        self.logger.info(f"DataManager initialized. Feature cache directory: {self.feature_cache_dir}")

    def load_market_data(self, source: str, tickers: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        从指定来源（例如 S3）加载市场数据。
        """
        cache_key = f"{source}_{'_'.join(tickers)}_{start_date}_{end_date}"
        if cache_key in self.market_data_cache:
            self.logger.debug(f"Loading market data from in-memory cache for key: {cache_key}")
            return self.market_data_cache[cache_key]

        self.logger.info(f"Fetching market data for {tickers} from {start_date} to {end_date}.")
        
        # 这是一个占位符。一个真实的实现会
        # 1. 检查本地 Parquet 缓存 (Task 3.4 逻辑)
        # 2. 如果未命中，从 S3 (self.s3_client) 或数据库获取
        # 3. 将数据存入缓存 (self.market_data_cache 和 Parquet)
        
        # 模拟数据加载
        data = {
            "date": pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='B')),
            "price": np.random.rand(len(pd.date_range(start=start_date, end=end_date, freq='B'))) * 100 + 500
        }
        df = pd.DataFrame(data)
        
        if df.empty:
            self.logger.warning(f"No market data found for key: {cache_key}")
            return None
            
        self.market_data_cache[cache_key] = df
        return df

    def get_features_for_date(self, dt: date) -> Dict[str, Any]:
        """
        检索、生成或缓存给定日期的 AI 特征。
        (由 Task 3.4 的缓存逻辑支持)
        """
        cache_key = f"features_{dt.isoformat()}"
        
        # 1. 尝试从内存缓存加载
        if cache_key in self.feature_data_cache:
            self.logger.debug(f"Loading features from in-memory cache for date: {dt}")
            return self.feature_data_cache[cache_key]

        # 2. 尝试从持久化 Parquet 缓存加载 (Task 3.4)
        cached_df = self.load_features_from_parquet(cache_key)
        if cached_df is not None:
            # 假设特征是 DataFrame 的第一行
            features = cached_df.iloc[0].to_dict()
            self.feature_data_cache[cache_key] = features # 存入内存缓存
            return features

        # 3. 缓存未命中：生成新特征
        self.logger.info(f"Cache miss for features on {dt}. Generating new features.")
        
        # ... (这里是调用 self.feature_store.generate_features(...) 的复杂逻辑) ...
        # 假设我们为日期生成了特征
        new_features = {"feature_a": 0.5, "feature_b": 0.7, "date": dt}
        new_features_df = pd.DataFrame([new_features])
        
        # 4. 保存到持久化缓存 (Task 3.4)
        self.save_features_to_parquet(new_features_df, cache_key)
        
        # 5. 保存到内存缓存
        self.feature_data_cache[cache_key] = new_features
        
        return new_features

    def load_and_process_alternative_data(self, start_date: date, end_date: date):
        """
        迭代所有已注册的替代数据客户端，获取它们的数据，
        并将其传递给 FeatureStore 进行处理。
        (Task 2.1 - Data Injection)
        """
        if not self.alt_data_clients:
            self.logger.info("No alternative data clients configured. Skipping.")
            return

        self.logger.info(f"Loading data from {len(self.alt_data_clients)} alternative data clients.")
        for client in self.alt_data_clients:
            alt_data_df = client.fetch_data(start_date, end_date)
            # 将原始替代数据传递给特征存储进行转换
            self.feature_store.generate_features_from_alt_data(alt_data_df)

    def save_features_to_parquet(self, features_df: pd.DataFrame, cache_key: str):
        """
        将 AI 特征的 DataFrame 保存到持久化的 Parquet 文件。
        (Task 3.4 - AI Feature Caching)
        """
        cache_path = os.path.join(self.feature_cache_dir, f"{cache_key}.parquet")
        try:
            features_df.to_parquet(cache_path, index=False)
            self.logger.info(f"Successfully cached features to {cache_path}")
        except Exception as e:
            self.logger.error(f"Failed to save features to Parquet cache {cache_path}: {e}")

    def load_features_from_parquet(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        如果 Parquet 缓存文件存在，则从中加载 AI 特征。
        (Task 3.4 - AI Feature Caching)
        """
        cache_path = os.path.join(self.feature_cache_dir, f"{cache_key}.parquet")
        if os.path.exists(cache_path):
            try:
                features_df = pd.read_parquet(cache_path)
                self.logger.info(f"Loaded features from cache: {cache_path}")
                return features_df
            except Exception as e:
                self.logger.error(f"Failed to load features from Parquet cache {cache_path}: {e}")
        
        self.logger.info(f"Feature cache miss for key: {cache_key}")
        return None
