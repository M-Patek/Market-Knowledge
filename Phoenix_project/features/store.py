import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os

# 修复：
# 1. 导入 'Feature' 而不是 'FeatureDefinition'
# 2. 导入路径从 'schemas.feature_schema' 改为 '..core.schemas.feature_schema'
from ..core.schemas.feature_schema import Feature
from ..monitor.logging import get_logger

logger = get_logger(__name__)

class FeatureStore:
    """
    Manages the lifecycle of features: definition, computation, and storage.
    
    (Note: This is a simplified in-memory/simple file store for demonstration.
     A real implementation would use a dedicated DB like Feast, Redis, or a Parquet store.)
    """
    
    def __init__(self, 
                 config: Dict[str, Any], 
                 data_manager: "DataManager"): # Fwd reference
        """
        Initialize the feature store.
        
        Args:
            config (Dict[str, Any]): Configuration, e.g., {'store_path': '...'}.
            data_manager (DataManager): Used to fetch raw data for computation.
        """
        self.config = config
        self.store_path = config.get('store_path', './feature_store_data')
        self.data_manager = data_manager
        # 修复：使用 'Feature'
        self.feature_definitions: Dict[str, Feature] = {}
        
        os.makedirs(self.store_path, exist_ok=True)
        self._load_definitions()
        logger.info(f"FeatureStore initialized at path: {self.store_path}")

    def _load_definitions(self):
        """Loads feature definitions from a central JSON file."""
        def_path = os.path.join(self.store_path, '_definitions.json')
        if not os.path.exists(def_path):
            return

        try:
            with open(def_path, 'r') as f:
                defs_json = json.load(f)
                for name, data in defs_json.items():
                    # 修复：使用 'Feature'
                    self.feature_definitions[name] = Feature(**data)
            logger.info(f"Loaded {len(self.feature_definitions)} feature definitions.")
        except Exception as e:
            logger.error(f"Error loading feature definitions: {e}")

    def _save_definitions(self):
        """Saves all current feature definitions to the central JSON file."""
        def_path = os.path.join(self.store_path, '_definitions.json')
        try:
            with open(def_path, 'w') as f:
                json.dump(
                    {name: fdef.model_dump() for name, fdef in self.feature_definitions.items()},
                    f,
                    indent=2,
                    default=str # Handle datetimes
                )
        except Exception as e:
            logger.error(f"Error saving feature definitions: {e}")

    # 修复：使用 'Feature'
    def register_feature(self, feature_def: Feature):
        """
        Registers a new feature definition.
        """
        if feature_def.name in self.feature_definitions:
            logger.warning(f"Feature '{feature_def.name}' is already registered. Overwriting.")
        
        self.feature_definitions[feature_def.name] = feature_def
        self._save_definitions()
        logger.info(f"Registered feature: {feature_def.name}")

    async def compute_feature(self, feature_name: str, symbol: str, end_date: datetime):
        """
        Computes a feature value for a given symbol and date.
        (This is a placeholder for a complex computation engine)
        """
        if feature_name not in self.feature_definitions:
            raise ValueError(f"Feature '{feature_name}' not defined.")
            
        fdef = self.feature_definitions[feature_name]
        
        # 1. Get required raw data
        # (This is simplified; real version needs date ranges, joins, etc.)
        try:
            # 修复：假设 fdef 有 'lookback_days' 和 'computation_logic'
            # 'Feature' schema 比较简单，我们从 fdef.metadata 中获取
            lookback_days = fdef.metadata.get('lookback_days', 30) 
            
            raw_data = await self.data_manager.get_historical_data(
                symbol, 
                start_date=end_date - pd.Timedelta(days=lookback_days + 5), # Add buffer
                end_date=end_date
            )
            if raw_data.empty:
                logger.warning(f"No raw data found for {symbol} to compute {feature_name}")
                return None
        except Exception as e:
            logger.error(f"Failed to get raw data for feature computation: {e}")
            return None

        # 2. Apply computation logic (Example: SMA)
        value = None
        try:
            computation_logic = fdef.metadata.get('computation_logic', 'SMA') # 示例
            
            if computation_logic == 'SMA':
                window = fdef.metadata.get('window', 20)
                if len(raw_data) >= window:
                    value = raw_data['close'].rolling(window=window).mean().iloc[-1]
            
            elif computation_logic == 'RSI':
                # Requires pandas_ta
                pass # Placeholder
            
            else:
                logger.warning(f"Computation logic '{computation_logic}' not implemented.")

        except Exception as e:
            logger.error(f"Error during feature computation for {feature_name}: {e}")
            return None
        
        if value is not None and not pd.isna(value):
            # 3. Store the computed value
            await self.save_feature_value(symbol, feature_name, end_date, value)
            return value
        
        return None


    async def save_feature_value(self, 
                                 symbol: str, 
                                 feature_name: str, 
                                 timestamp: datetime, 
                                 value: Any):
        """
        Saves a single computed feature value to its store.
        (Stores as daily CSVs for simplicity)
        """
        date_str = timestamp.strftime('%Y-%m-%d')
        feature_file = os.path.join(self.store_path, f"{feature_name}.csv")
        
        try:
            # This is highly inefficient for production, but simple
            if os.path.exists(feature_file):
                df = pd.read_csv(feature_file, index_col='timestamp', parse_dates=True)
            else:
                df = pd.DataFrame(columns=['symbol', 'value'])
                df.index.name = 'timestamp'
            
            # Add or update value
            # 修正：确保索引是 Timestamp
            df.loc[pd.Timestamp(timestamp), 'symbol'] = symbol
            df.loc[pd.Timestamp(timestamp), 'value'] = value
            df = df[~df.index.duplicated(keep='last')] # Keep last value for timestamp
            
            df.to_csv(feature_file)
            
        except Exception as e:
            logger.error(f"Failed to save feature value for {feature_name}: {e}")


    async def get_feature_values(self, 
                                 feature_names: List[str], 
                                 symbol: str, 
                                 start_date: datetime, 
                                 end_date: datetime) -> pd.DataFrame:
        """
        Retrieves one or more features for a symbol over a date range.
        """
        all_features_df = pd.DataFrame()
        
        for feature_name in feature_names:
            feature_file = os.path.join(self.store_path, f"{feature_name}.csv")
            
            if not os.path.exists(feature_file):
                logger.warning(f"No data file found for feature: {feature_name}")
                continue
                
            try:
                df = pd.read_csv(feature_file, index_col='timestamp', parse_dates=True)
                df = df.sort_index()
                
                # Filter for symbol and date range
                symbol_df = df[
                    (df['symbol'] == symbol) & 
                    (df.index >= pd.Timestamp(start_date)) & 
                    (df.index <= pd.Timestamp(end_date))
                ]
                
                if not symbol_df.empty:
                    # Rename 'value' column to feature name for joining
                    symbol_df = symbol_df.rename(columns={'value': feature_name})
                    
                    if all_features_df.empty:
                        all_features_df = symbol_df[[feature_name]]
                    else:
                        all_features_df = all_features_df.join(symbol_df[[feature_name]], how='outer')
                        
            except Exception as e:
                logger.error(f"Failed to load feature {feature_name}: {e}")
                
        # Forward-fill missing values
        all_features_df = all_features_df.fillna(method='ffill')
        
        return all_features_df

