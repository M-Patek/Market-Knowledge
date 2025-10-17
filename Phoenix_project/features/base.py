# features/base.py
from typing import Protocol, Dict, Any
import pandas as pd
from datetime import date

class IFeatureStore(Protocol):
    """
    Defines the interface for a feature store.
    Ensures that both training and serving environments get features
    from a single, consistent source.
    """
    def get_features(self, ticker: str, as_of_date: date, full_history: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates or retrieves all features for a given ticker as of a specific date,
        ensuring point-in-time correctness.
        Note: The data used for calculation will include all historical data strictly *before* the as_of_date.
        """
        ...
