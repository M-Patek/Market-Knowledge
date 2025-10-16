# features/base.py
from typing import Protocol, Dict, Any
import pandas as pd

class IFeatureStore(Protocol):
    """
    Defines the interface for a feature store.
    Ensures that both training and serving environments get features
    from a single, consistent source.
    """

    def get_features(self, ticker: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates or retrieves all features for a given ticker and its historical data.
        """
        ...

