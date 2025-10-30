from pydantic import BaseModel
from typing import List, Any

class FeatureSchema(BaseModel):
    """
    (L1) Standardized schema for defining features.
    """
    name: str
    dependencies: List[str]
    data_type: Any
    calc_fn: str  # (L1) The registered name of the calculation function
