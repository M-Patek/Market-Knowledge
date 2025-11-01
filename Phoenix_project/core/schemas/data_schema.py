from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

class MarketData(BaseModel):
    """
    Schema for market data (OHLCV, trades, quotes).
    """
    symbol: str
    timestamp: datetime
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            pd.Timestamp: lambda v: v.to_pydatetime()
        }

class NewsData(BaseModel):
    """
    Schema for news articles and social media posts.
    """
    source_id: str = Field(..., unique=True)
    timestamp: datetime
    source: str  # e.g., 'Reuters', 'Twitter'
    headline: str
    summary: Optional[str] = None
    content: str
    symbols: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True

class AlternativeData(BaseModel):
    """
    Schema for alternative datasets (e.g., satellite, credit card).
    """
    data_type: str
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    
    class Config:
        arbitrary_types_allowed = True

class DerivedData(BaseModel):
    """
    Schema for features and derived insights.
    """
    feature_name: str
    timestamp: datetime
    value: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

class Node(BaseModel):
    """
    Node in the knowledge graph.
    """
    id: str = Field(..., description="Unique identifier for the node")
    label: str = Field(..., description="Node type (e.g., 'Company', 'Person', 'Event')")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node attributes")

class Relation(BaseModel):
    """
    Relation (edge) in the knowledge graph.
    """
    id: str = Field(..., description="Unique identifier for the relation")
    type: str = Field(..., description="Type of relationship (e.g., 'CEO_OF', 'COMPETES_WITH')")
    start_node_id: str = Field(..., description="ID of the starting node")
    end_node_id: str = Field(..., description="ID of the ending node")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relation attributes (e.g., 'since', 'weight')")


# --- 新增：补全 ai/graph_encoder.py 缺失的 KnoledgeGraph schema ---
class KnowledgeGraph(BaseModel):
    """
    Pydantic 模型，用于表示知识图谱的结构。
    供 ai/graph_encoder.py 使用。
    """
    nodes: List[Node] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
        
# --- 新增：补全 ai/retriever.py 缺失的 QueryResult schema ---
class QueryResult(BaseModel):
    """
    Pydantic 模型，用于统一 RAG 检索器的返回格式。
    供 ai/retriever.py 使用。
    """
    source: str = Field(..., description="数据来源 (e.g., 'vector', 'temporal', 'tabular')")
    content: str = Field(..., description="检索到的文本内容或数据片段")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="相关的元数据 (e.g., source_id, timestamp)")
    score: Optional[float] = Field(None, description="检索相关性得分")

    class Config:
        arbitrary_types_allowed = True


class MarketEvent(BaseModel):
    """
    一个通用的市场事件 schema，用于事件流。
    (从 strategy_handler.py 推断)
    """
    event_id: str = Field(..., description="事件的唯一 ID")
    event_type: str = Field(..., description="事件类型 (e.g., 'PRICE', 'NEWS', 'URGENT_NEWS')")
    timestamp: datetime = Field(..., description="事件发生时间")
    symbols: List[str] = Field(default_factory=list, description="与此事件相关的 Tickers")
    content: Optional[str] = Field(None, description="事件的文本内容 (例如新闻正文)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="其他事件数据")

    class Config:
        arbitrary_types_allowed = True

