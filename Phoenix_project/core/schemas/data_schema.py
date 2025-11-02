"""
Pydantic schemas for data validation and standardization across the pipeline.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

class MarketEvent(BaseModel):
    """
    Schema for a market-related event (e.g., news, press release).
    """
    event_id: str = Field(..., description="Unique identifier for the event")
    timestamp: datetime = Field(..., description="Timestamp of when the event occurred or was published")
    source: str = Field(..., description="Source of the event (e.g., 'Reuters', 'Bloomberg')")
    headline: str = Field(..., description="Event headline or title")
    content: str = Field(..., description="Full content of the event")
    symbols: List[str] = Field(default_factory=list, description="List of ticker symbols mentioned or related")
    event_type: str = Field(default="news", description="Type of market event (e.g., 'news', 'earnings', 'sec_filing')")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata (e.g., sentiment scores, entity tags)")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class EconomicEvent(BaseModel):
    """
    Schema for a macroeconomic event (e.g.,
    """
    event_id: str = Field(..., description="Unique identifier for the event")
    timestamp: datetime = Field(..., description="Timestamp of the event (e.g., release time)")
    event_name: str = Field(..., description="Name of the economic indicator (e.g., 'CPI', 'Non-Farm Payrolls')")
    region: str = Field(..., description="Geographical region (e.g., 'USA', 'Eurozone')")
    actual: Optional[float] = Field(None, description="Actual reported value")
    forecast: Optional[float] = Field(None, description="Forecasted value")
    previous: Optional[float] = Field(None, description="Previous value")
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# --- 新增的 TickerData Schema ---
class TickerData(BaseModel):
    """
    Schema for standardized OHLCV market data (ticker data).
    """
    symbol: str = Field(..., description="Ticker symbol")
    timestamp: datetime = Field(..., description="Timestamp of the data point (e.g., bar start time)")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: float = Field(..., description="Volume")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            # pandas Timestamps are common, ensure they are converted
            pd.Timestamp: lambda v: v.to_pydatetime().isoformat(),
            datetime: lambda v: v.isoformat()
        }

# --- 知识图谱 (Knowledge Graph) Schemas ---

class KGNode(BaseModel):
    """
    Represents a node in the Knowledge Graph.
    """
    node_id: str = Field(..., description="Unique identifier for the node (e.g., 'AAPL', 'Tim Cook')")
    node_type: str = Field(..., description="Type of the node (e.g., 'COMPANY', 'PERSON', 'PRODUCT')")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Properties of the node")

class KGRelation(BaseModel):
    """
    Represents a directed edge (relationship) in the Knowledge Graph.
    """
    relation_id: str = Field(..., description="Unique identifier for the relationship")
    source_node_id: str = Field(..., description="ID of the source node")
    target_node_id: str = Field(..., description="ID of the target node")
    relation_type: str = Field(..., description="Type of relationship (e.g., 'IS_CEO_OF', 'MANUFACTURES')")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Properties of the relationship (e.g., 'start_date', 'confidence')")

class KnowledgeGraph(BaseModel):
    """
    Represents a subgraph or a complete Knowledge Graph.
    """
    nodes: List[KGNode] = Field(default_factory=list)
    relations: List[KGRelation] = Field(default_factory=list)

    def add_node(self, node: KGNode):
        self.nodes.append(node)

    def add_relation(self, relation: KGRelation):
        self.relations.append(relation)
