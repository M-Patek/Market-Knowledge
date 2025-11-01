# Phoenix_project/schemas/data_schema.py

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, field_validator, ConfigDict

class Source(BaseModel):
    model_config = ConfigDict(extra='ignore')
    id: str
    name: str
    url: Optional[str] = None

class Document(BaseModel):
    model_config = ConfigDict(extra='ignore')
    doc_id: str
    text: str
    source: Source
    timestamp: datetime
    metadata: Dict[str, Any] = {}

class MarketEvent(BaseModel):
    model_config = ConfigDict(extra='ignore', from_attributes=True)
    event_id: str
    event_type: str
    timestamp: datetime
    assets: List[str]
    description: str
    source_doc: Document
    impact_score: Optional[float] = None

class AnalystOpinion(BaseModel):
    model_config = ConfigDict(extra='ignore')
    opinion_id: str
    timestamp: datetime
    asset: str
    analyst: str
    rating: str  # e.g., "BUY", "SELL", "HOLD"
    price_target: Optional[float] = None
    source_doc: Document

class FusedAnalysis(BaseModel):
    model_config = ConfigDict(extra='ignore')
    ticker: str
    timestamp: datetime
    summary: str
    market_events: List[MarketEvent]
    analyst_opinions: List[AnalystOpinion]
    final_sentiment: str
    confidence_score: float

    @field_validator('confidence_score')
    @classmethod
    def score_must_be_in_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence score must be between 0 and 1')
        return v
