from typing import Dict, Any, Optional, List
import pandas as pd
from core.schemas.feature_schema import Feature
from features.base import FeatureBase
from monitor.logging import get_logger

logger = get_logger(__name__)

class FeatureStore:
    """
    Manages the lifecycle of features:
    - Registration of feature calculation logic.
    - Calculation/computation of features.
    - Storage and retrieval of feature values.
    
    This is simplified. A real feature store (e.g., Feast, Tecton)
    is a major piece of infrastructure.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._feature_registry: Dict[str, FeatureBase] = {}
        
        # In-memory storage for feature values (e.g., {feature_name: DataFrame})
        self.feature_storage: Dict[str, pd.DataFrame] = {} 
        logger.info("FeatureStore initialized.")

    def register_feature(self, feature_instance: FeatureBase):
        """Registers a feature calculation class."""
        name = feature_instance.name
        if name in self._feature_registry:
            logger.warning(f"Feature '{name}' is already registered. Overwriting.")
        self._feature_registry[name] = feature_instance
        logger.info(f"Feature registered: {name}")

    def load_features_from_config(self, feature_configs: List[Dict[str, Any]]):
        """
        Loads and registers features from a config list.
        Requires dynamic import.
        """
        # Placeholder for dynamic import
        # Example:
        # for config in feature_configs:
        #   module = importlib.import_module(config['module'])
        #   class_ = getattr(module, config['class'])
        #   instance = class_(**config.get('params', {}))
        #   self.register_feature(instance)
        logger.warning("FeatureStore.load_features_from_config is not implemented.")

    async def calculate_features(
        self,
        data_context: Dict[str, pd.DataFrame],
        features_to_run: Optional[List[str]] = None
    ) -> Dict[str, pd.Series]:
        """
        Calculates one or more features based on the provided data.
        
        Args:
            data_context (Dict[str, pd.DataFrame]): All available raw data,
                                                  e.g., {"market_data_AAPL": df, ...}
            features_to_run (Optional[List[str]]): List of feature names to run.
                                                   If None, runs all registered.
                                                   
        Returns:
            A dictionary of {feature_name: pd.Series}
        """
        if features_to_run is None:
            features_to_run = list(self._feature_registry.keys())
            
        logger.info(f"Calculating {len(features_to_run)} features...")
        
        results: Dict[str, pd.Series] = {}
        
        for name in features_to_run:
            if name not in self._feature_registry:
                logger.warning(f"Cannot calculate feature '{name}': Not registered.")
                continue
                
            feature_obj = self._feature_registry[name]
            
            try:
                # Check if dependencies are met
                required_data_key = feature_obj.dependencies[0] # Simplified
                if required_data_key not in data_context:
                    logger.error(f"Cannot calculate feature '{name}': Missing dependency '{required_data_key}'.")
                    continue
                    
                input_df = data_context[required_data_key]
                
                # Run calculation
                feature_series = await feature_obj.calculate(input_df)
                results[name] = feature_series
                
                # Store/cache the result
                # This simple cache just stores the latest calculation
                self.feature_storage[name] = feature_series.to_frame(name)
                
                logger.debug(f"Feature '{name}' calculated successfully.")
                
            except Exception as e:
                logger.error(f"Failed to calculate feature '{name}': {e}", exc_info=True)
                
        return results

    def get_feature(self, feature_name: str) -> Optional[pd.DataFrame]:
        """Retrieves the last calculated values for a feature."""
        return self.feature_storage.get(feature_name)

    def get_latest_feature_value(self, feature_name: str) -> Any:
        """Gets the most recent value of a feature."""
        df = self.get_feature(feature_name)
        if df is not None and not df.empty:
            return df.iloc[-1][feature_name]
        return None
