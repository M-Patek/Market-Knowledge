# Phoenix_project/schemas/feature_schema.py
from typing import List, Dict, Any
from pydantic import BaseModel, ConfigDict

class Feature(BaseModel):
    model_config = ConfigDict(extra='ignore')
    name: str
    value: Any
    timestamp: str
    metadata: Dict[str, Any] = {}
