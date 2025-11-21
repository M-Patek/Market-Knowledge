"""
特征存储 (Feature Store)
用于存储、检索和管理用于模型训练和推理的特征。
"""
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import os # [Task 10] Import os for file operations

# FIX (E8): 导入 IFeatureStore 接口 (原为 FeatureBase)
from .base import IFeatureStore

class FeatureStore(IFeatureStore):
    """
    特征存储的实现。
    (Task 10: Implemented Parquet file-based storage)
    """
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.cache: Dict[str, pd.DataFrame] = {}
        
        # [Task 10] Ensure directory exists
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
            
        print(f"FeatureStore initialized at {base_path}")

    def get_features(
        self,
        feature_set_name: str,
        entity_ids: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> Optional[pd.DataFrame]:
        """
        检索一个特征集。支持按时间和实体(Symbol)切片。
        """
        # 构建文件路径
        file_path = os.path.join(self.base_path, f"{feature_set_name}.parquet")
        
        if not os.path.exists(file_path):
            print(f"Warning: Feature set {feature_set_name} not found at {file_path}")
            return None
            
        try:
            # 1. Load dataset
            # (Optimization Note: For huge files, use pyarrow.parquet.read_table with filters)
            df = pd.read_parquet(file_path)
            
            # 2. Filter by Time
            # Standardize index to datetime for filtering
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.sort_index()
                # Flexible slicing (inclusive)
                df = df[start_time:end_time]
            
            # 3. Filter by Entity (Symbol)
            if entity_ids and 'symbol' in df.columns:
                df = df[df['symbol'].isin(entity_ids)]
                
            print(f"Retrieved {len(df)} rows for {feature_set_name}")
            return df
            
        except Exception as e:
            print(f"Error reading feature set {feature_set_name}: {e}")
            return None
        

    def register_feature_set(self, feature_set_name: str, schema: Dict[str, Any]):
        """
        注册一个新的特征集（例如，创建表或目录）。
        """
        print(f"Registering feature set: {feature_set_name} with schema {schema}")
        pass

    def ingest_features(self, feature_set_name: str, data: pd.DataFrame):
        """
        将新的特征数据写入存储 (增量更新/Upsert)。
        """
        file_path = os.path.join(self.base_path, f"{feature_set_name}.parquet")
        print(f"Ingesting {len(data)} rows into {feature_set_name}...")
        
        try:
            final_df = data
            
            # [Task 10] Incremental Upsert Logic
            if os.path.exists(file_path):
                existing_df = pd.read_parquet(file_path)
                final_df = pd.concat([existing_df, data])
                
                # Deduplicate based on index and symbol to allow re-runs
                if 'symbol' in final_df.columns:
                    # Reset index to use timestamp in duplicate check
                    if isinstance(final_df.index, pd.DatetimeIndex):
                         final_df = final_df.reset_index()
                         final_df = final_df.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
                         final_df = final_df.set_index('timestamp')
                    else:
                         # Assuming 'timestamp' is a column if not index
                         if 'timestamp' in final_df.columns:
                            final_df = final_df.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
                else:
                     # Basic deduplication
                     final_df = final_df.drop_duplicates(keep='last')
            
            final_df.to_parquet(file_path)
            print(f"Successfully ingested features into {file_path}. Total rows: {len(final_df)}")
            
        except Exception as e:
             print(f"Error ingesting features: {e}")
