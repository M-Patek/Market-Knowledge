from pydantic import BaseModel
from datetime import datetime
from typing import Any, Dict

class DataSchema(BaseModel):
    """
    (L1) Standardized data schema for all incoming data points.
    """
    timestamp: datetime
    source: str
    symbol: str
    value: Any
    metadata: Dict[str, Any] = {}
