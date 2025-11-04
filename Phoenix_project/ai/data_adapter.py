"""
AI数据适配器
将来自 DataManager 的原始数据转换为 AI 模型（特别是 RAG）可以理解的格式。
"""
from typing import List, Dict, Any, Union

# FIX (E1): 导入统一后的核心模式
from Phoenix_project.core.schemas.data_schema import MarketData, NewsData, EconomicIndicator

class AIDataAdapter:
    """
    将 Pydantic 数据模式转换为适用于 RAG 检索的文本块或结构化元数据。
    """

    def __init__(self):
        # 可以在此处初始化模板引擎或格式化工具
        pass

    def format_market_data(self, data_points: List[MarketData]) -> List[Dict[str, Any]]:
        """
        将 MarketData 列表转换为 RAG 文档（文本 + 元数据）。
        """
        documents = []
        if not data_points:
            return documents
        
        # 示例：为RAG创建文档
        # 实际实现可能更复杂，例如计算TA指标并将其转为文本
        for dp in data_points:
            text_content = (
                f"On {dp.timestamp.date()} for {dp.symbol}, the market data was: "
                f"Open={dp.open}, High={dp.high}, Low={dp.low}, Close={dp.close}, Volume={dp.volume}."
            )
            metadata = {
                "source": "market_data",
                "symbol": dp.symbol,
                "timestamp": dp.timestamp.isoformat(),
                "doc_type": "MarketData"
            }
            # 在RAG中，ID通常是唯一的
            doc_id = f"market_{dp.symbol}_{dp.timestamp.isoformat()}"
            documents.append({"id": doc_id, "text": text_content, "metadata": metadata})
            
        return documents

    def format_news_data(self, data_points: List[NewsData]) -> List[Dict[str, Any]]:
        """
        将 NewsData 列表转换为 RAG 文档（文本 + 元数据）。
        这是最直接的 RAG 应用。
        """
        documents = []
        for dp in data_points:
            text_content = f"Headline: {dp.headline}\n\n{dp.content}"
            metadata = {
                "source": dp.source,
                "symbols": ",".join(dp.symbols), # 元数据通常不支持列表
                "timestamp": dp.timestamp.isoformat(),
                "doc_type": "NewsData"
            }
            documents.append({"id": dp.id, "text": text_content, "metadata": metadata})
        return documents

    def format_economic_data(self, data_points: List[EconomicIndicator]) -> List[Dict[str, Any]]:
        """
        将 EconomicIndicator 列表转换为 RAG 文档（文本 + 元数据）。
        """
        documents = []
        for dp in data_points:
            text_content = (
                f"Economic Indicator release on {dp.timestamp.date()}: {dp.name}. "
                f"Actual: {dp.value}. "
                f"Expected: {dp.expected if dp.expected is not None else 'N/A'}. "
                f"Previous: {dp.previous if dp.previous is not None else 'N/A'}."
            )
            metadata = {
                "source": "economic_data",
                "indicator_id": dp.id,
                "timestamp": dp.timestamp.isoformat(),
                "doc_type": "EconomicIndicator"
            }
            doc_id = f"econ_{dp.id}_{dp.timestamp.isoformat()}"
            documents.append({"id": doc_id, "text": text_content, "metadata": metadata})
        return documents

    def adapt_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        处理来自 DataIterator 的整个数据批次。
        """
        adapted_batch = {}
        
        # FIX (E1): 使用正确的键名 (基于 data_manager.py 的 fetch_data)
        if "market_data" in batch:
            adapted_batch["market_data"] = self.format_market_data(batch["market_data"])
            
        if "news_data" in batch:
            adapted_batch["news_data"] = self.format_news_data(batch["news_data"])
            
        if "economic_indicators" in batch:
            adapted_batch["economic_indicators"] = self.format_economic_data(batch["economic_indicators"])
            
        return adapted_batch
