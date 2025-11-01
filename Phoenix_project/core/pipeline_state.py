import json
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List

@dataclass
class PipelineState:
    """
    (L0) A centralized data structure to hold and manage the state 
    of the entire analysis pipeline.
    """
    ticker: str
    evidence_pool: List[Dict[str, Any]] = field(default_factory=list)
    fusion_result: Dict[str, Any] = field(default_factory=dict)
    context_info: Dict[str, Any] = field(default_factory=dict)
    uncertainty_score: float = 0.0
    logs: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Serializes the state object to a JSON string."""
        return json.dumps(self.__dict__, indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> 'PipelineState':
        """Deserializes a JSON string into a PipelineState object."""
        data = json.loads(json_str)
        # (L0) Filter data to only include keys that are class attributes
        class_field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in class_field_names}
        return cls(**filtered_data)
