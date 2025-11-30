"""
凤凰计划的核心数据模式 (Authoritative Data Schemas)。
此文件是系统中所有数据结构（Pydantic模型）的唯一真实来源。
所有模块必须从此文件导入其所需的数据模型。

FIX (E1, E2, E4):
1.  重命名 TickerData -> MarketData
2.  重命名 MarketEvent -> NewsData
3.  重命名 EconomicEvent -> EconomicIndicator
4.  从 execution/interfaces.py 移入 Order, Fill, OrderStatus, Position 的定义。
5.  添加了 Signal, PortfolioState 的定义。

[主人喵的修复]
6. 添加 TargetPosition 和 TargetPortfolio，供 PortfolioConstructor 和 RiskManager 使用。
[Phase I Fix] Decimal Refactoring for Financial Precision.
[Task 2.1] Implemented Enums, Physics Constraints, and Precision Guards.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any, List, Union
from decimal import Decimal
from datetime import datetime
from uuid import UUID
from enum import Enum

# --- 市场与事件数据 (Market & Event Data) ---

class MarketData(BaseModel):
    """
    统一的市场行情数据（原 TickerData）。
    代表特定时间戳的OHLCV数据。
    """
    symbol: str = Field(..., description="资产代码")
    timestamp: datetime = Field(..., description="数据点的时间戳 (UTC)")
    open: Decimal = Field(..., description="开盘价")
    high: Decimal = Field(..., description="最高价")
    low: Decimal = Field(..., description="最低价")
    close: Decimal = Field(..., description="收盘价")
    volume: Decimal = Field(..., description="成交量")
    ingestion_batch_id: Optional[UUID] = Field(None, description="[Task 3] Atomic batch ID for ingestion tracking.")
    
    # [Task 2.1] Precision Guard: Force Float -> Str -> Decimal conversion
    @field_validator('open', 'high', 'low', 'close', 'volume', mode='before')
    @classmethod
    def float_to_decimal(cls, v: Any) -> Decimal:
        if isinstance(v, float):
            return Decimal(str(v))
        return v

    # [Task 2.1] Enforce Physics
    @model_validator(mode='after')
    def check_physics(self) -> 'MarketData':
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) cannot be less than Low ({self.low})")
        if self.volume < 0:
            raise ValueError(f"Volume ({self.volume}) cannot be negative")
        return self

    class Config:
        frozen = True

class NewsData(BaseModel):
    """
    统一的非结构化事件数据（原 MarketEvent）。
    代表新闻、SEC文件、社交媒体帖子等。
    """
    id: str = Field(..., description="事件的唯一ID")
    source: str = Field(..., description="数据来源 (e.g., 'Reuters', 'SEC')")
    timestamp: datetime = Field(..., description="事件发布时间 (UTC)")
    symbols: List[str] = Field(default_factory=list, description="与此事件相关的资产代码")
    content: str = Field(..., description="事件的文本内容")
    headline: Optional[str] = Field(None, description="事件标题")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="其他元数据")
    ingestion_batch_id: Optional[UUID] = Field(None, description="[Task 3] Atomic batch ID for ingestion tracking.")

    class Config:
        frozen = True

class EconomicIndicator(BaseModel):
    """
    统一的经济指标数据（原 EconomicEvent）。
    代表如CPI, 非农就业人数等宏观经济数据。
    """
    id: str = Field(..., description="指标的唯一ID (e.g., 'CPI_YOY')")
    name: str = Field(..., description="指标名称 (e.g., 'Consumer Price Index YOY')")
    timestamp: datetime = Field(..., description="指标发布时间 (UTC)")
    value: Decimal = Field(..., description="指标的实际值")
    expected: Optional[Decimal] = Field(None, description="市场预期值")
    previous: Optional[Decimal] = Field(None, description="前值")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="其他元数据 (e.g., 'country')")
    ingestion_batch_id: Optional[UUID] = Field(None, description="[Task 3] Atomic batch ID for ingestion tracking.")

    @field_validator('value', 'expected', 'previous', mode='before')
    @classmethod
    def float_to_decimal(cls, v: Any) -> Optional[Decimal]:
        if v is None: return None
        if isinstance(v, float):
            return Decimal(str(v))
        return v

    class Config:
        frozen = True

# --- 知识图谱 (Knowledge Graph) ---

class KGNode(BaseModel):
    """
    知识图谱节点。
    """
    id: str = Field(..., description="节点ID")
    type: str = Field(..., description="节点类型 (e.g., 'Company', 'Sector')")
    properties: Dict[str, Any] = Field(default_factory=dict, description="节点属性")

# --- 交易执行 (Trading & Execution) ---
# FIX (E2, E4): 从 execution/interfaces.py 移入并标准化

class SignalType(str, Enum):
    """[Task 2.1] Strict Enum for Signal Types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class Signal(BaseModel):
    """
    交易信号。由认知层生成，由投资组合构造器消费。
    """
    symbol: str = Field(..., description="资产代码")
    timestamp: datetime = Field(..., description="信号生成时间 (UTC)")
    # [Task 2.1] Use Enum
    signal_type: SignalType = Field(..., description="信号类型 (BUY, SELL, HOLD)")
    strength: Decimal = Field(..., description="信号强度 (e.g., 0.0 to 1.0)", ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="信号来源、原因等元数据")

    @field_validator('strength', mode='before')
    @classmethod
    def float_to_decimal(cls, v: Any) -> Decimal:
        if isinstance(v, float):
            return Decimal(str(v))
        return v

class OrderStatus(str, Enum):
    """
    订单状态枚举。
    """
    NEW = "NEW"                 # 订单已创建，等待发送
    PENDING = "PENDING"         # 订单已发送至券商，等待确认
    ACCEPTED = "ACCEPTED"       # 订单已被券商接受
    PARTIALLY_FILLED = "PARTIALLY_FILLED" # 订单部分成交
    FILLED = "FILLED"           # 订单完全成交
    CANCELLED = "CANCELLED"     # 订单已取消
    REJECTED = "REJECTED"       # 订单被拒绝
    PENDING_SUBMISSION = "PENDING_SUBMISSION" # [Task 3.2] Added intermediate state

class Order(BaseModel):
    """
    交易订单。
    """
    id: str = Field(..., description="唯一的订单ID")
    client_order_id: Optional[str] = Field(None, description="客户端自定义订单ID")
    symbol: str = Field(..., description="资产代码")
    quantity: Decimal = Field(..., description="订单数量 (正数为买入, 负数为卖出)")
    order_type: str = Field(..., description="订单类型 (e.g., 'MARKET', 'LIMIT')")
    limit_price: Optional[Decimal] = Field(None, description="限价单价格")
    time_in_force: str = Field("GTC", description="订单时效 (e.g., 'GTC', 'IOC', 'FOK')")
    status: OrderStatus = Field(OrderStatus.NEW, description="订单当前状态")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="订单创建时间 (UTC)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="其他元数据")

    @field_validator('quantity', 'limit_price', mode='before')
    @classmethod
    def float_to_decimal(cls, v: Any) -> Optional[Decimal]:
        if v is None: return None
        if isinstance(v, float):
            return Decimal(str(v))
        return v

class Fill(BaseModel):
    """
    成交回报 (原 Execution)。
    """
    id: str = Field(..., description="唯一的成交ID")
    order_id: str = Field(..., description="关联的订单ID")
    symbol: str = Field(..., description="资产代码")
    timestamp: datetime = Field(..., description="成交时间 (UTC)")
    quantity: Decimal = Field(..., description="成交数量 (正数为买入, 负数为卖出)")
    price: Decimal = Field(..., description="成交价格")
    commission: Decimal = Field(Decimal("0.0"), description="手续费")

    @field_validator('quantity', 'price', 'commission', mode='before')
    @classmethod
    def float_to_decimal(cls, v: Any) -> Decimal:
        if isinstance(v, float):
            return Decimal(str(v))
        return v

# --- 投资组合状态 (Portfolio State) ---

class Position(BaseModel):
    """
    单个资产的持仓。
    """
    symbol: str = Field(..., description="资产代码")
    quantity: Decimal = Field(..., description="持仓数量 (正数为多头, 负数为实现空头)")
    average_price: Decimal = Field(..., description="平均持仓成本")
    market_value: Decimal = Field(..., description="当前市值")
    unrealized_pnl: Decimal = Field(..., description="未实现盈亏")

    @field_validator('quantity', 'average_price', 'market_value', 'unrealized_pnl', mode='before')
    @classmethod
    def float_to_decimal(cls, v: Any) -> Decimal:
        if isinstance(v, float):
            return Decimal(str(v))
        return v

class PortfolioState(BaseModel):
    """
    整个投资组合在特定时间点的快照。
    """
    timestamp: datetime = Field(..., description="快照时间 (UTC)")
    cash: Decimal = Field(..., description="可用现金")
    total_value: Decimal = Field(..., description="投资组合总价值 (cash + positions market_value)")
    positions: Dict[str, Position] = Field(default_factory=dict, description="当前持仓")
    realized_pnl: Decimal = Field(Decimal("0.0"), description="已实现盈亏")

    @field_validator('cash', 'total_value', 'realized_pnl', mode='before')
    @classmethod
    def float_to_decimal(cls, v: Any) -> Decimal:
        if isinstance(v, float):
            return Decimal(str(v))
        return v

# --- [主人喵的修复] 新增：目标投资组合模式 ---

class TargetPosition(BaseModel):
    """
    由 PortfolioConstructor 和 RiskManager 定义的单个资产的目标分配。
    """
    symbol: str = Field(..., description="资产代码")
    target_weight: Decimal = Field(..., description="目标投资组合权重 (e.g., 0.1 for 10%, -0.05 for -5%)")
    reasoning: str = Field("N/A", description="做出此分配的理由")

    @field_validator('target_weight', mode='before')
    @classmethod
    def float_to_decimal(cls, v: Any) -> Decimal:
        if isinstance(v, float):
            return Decimal(str(v))
        return v

class TargetPortfolio(BaseModel):
    """
    认知层和风险层输出的完整目标投资组合。
    这是 OrderManager 的输入。
    """
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="目标生成时间 (UTC)")
    positions: List[TargetPosition] = Field(default_factory=list, description="目标持仓列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据 (e.g., strategy_id)")
