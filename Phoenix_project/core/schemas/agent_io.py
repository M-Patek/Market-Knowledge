from enum import Enum
from typing import List
from pydantic import BaseModel

class SignalType(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HALT = "HALT"

class L1AnalysisResult(BaseModel):
    content: str
    confidence: float
    references: List[str]

class L2CritiqueResult(BaseModel):
    is_valid: bool
    score: float
    feedback: str

class L3Action(BaseModel):
    signal_type: SignalType
    weight: float
    reasoning: str
