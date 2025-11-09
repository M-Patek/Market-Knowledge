import logging
from typing import List, Dict, Any, Union
import pandas as pd
from Phoenix_project.core.schemas.data_schema import MarketData, EventData, Document

logger = logging.getLogger(__name__)

class DataAdapter:
    """
    (AI/RAG) 适配器，用于将结构化/非结构化数据转换为 LLM 和 RAG 系统可以使用的统一格式。
    
    [RAG 架构]：
    此适配器位于 RAG 管道的 "Data Ingestion & Transformation" (数据摄入与转换) 阶段。
    它负责将来自 DataManager 的各种数据类型 (如 MarketData, EventData) 
    转换为 RAG `Document` 对象 (或文本块)，以便后续进行嵌入和索引。
    """
    
    def __init__(self):
        """
        初始化数据适配器。
        """
        logger.info("DataAdapter initialized.")

    def format_market_data(self, data: MarketData) -> List[Document]:
        """
        将 MarketData (通常是 DataFrame) 转换为 RAG Document。
        
        [RAG 架构]：
        这是结构化数据 (K 线) 到文本的转换。
        目标是创建有意义的文本块 (Documents)，捕捉市场随时间变化的状态。
        """
        documents = []
        df = data.data
        
        if df is None or df.empty:
            return documents
        
        # 示例：按天 (或特定时间窗口) 聚合数据并创建文档
        # (这只是一个基本示例；实际应用中可能会按周、按月或按事件窗口聚合)
        
        # (假设 df 是一个以时间为索引的 DataFrame)
        # (为简单起见，我们假设 data.data 是一个包含多行的时间序列)
        
        summary = (
            f"Market data summary for {data.symbol} "
            f"from {df.index.min()} to {df.index.max()}:\n"
            f"Open: {df['open'].iloc[0]:.2f}\n"
            f"High: {df['high'].max():.2f}\n"
            f"Low: {df['low'].min():.2f}\n"
            f"Close: {df['close'].iloc[-1]:.2f}\n"
            f"Volume: {df['volume'].sum():.0f}"
        )
        
        doc = Document(
            doc_id=f"market_{data.symbol}_{df.index.min()}_{df.index.max()}",
            content=summary,
            metadata={
                "symbol": data.symbol,
                "start_date": df.index.min().isoformat(),
                "end_date": df.index.max().isoformat(),
                "data_type": "market_summary"
            }
        )
        documents.append(doc)
        
        return documents

    def format_event_data(self, data: EventData) -> Document:
        """
        将 EventData (新闻、财报等) 转换为 RAG Document。
        
        [RAG 架构]：
        这是非结构化数据 (文本) 的规范化。
        """
        content = f"Headline: {data.headline}\n\n{data.description}"
        
        doc = Document(
            doc_id=data.event_id,
            content=content,
            metadata={
                "symbol": data.symbol,
                "source": data.source,
                "timestamp": data.timestamp,
                "data_type": "event",
                "tags": data.tags
            }
        )
        return doc

    def format_tabular_data(self, data: pd.DataFrame, context: str) -> List[Document]:
        """
        将来自 Tabular DB (例如财务报表) 的任意表格数据转换为 Document。
        
        [RAG 架构]：
        这是结构化数据 (表格) 到文本的转换，通常需要 "text-to-SQL" 或
        "table-to-text" (表格到文本) 的能力。
        
        Args:
            data (pd.DataFrame): 从数据库查询返回的表格数据。
            context (str): 关于此数据的描述 (例如 "AAPL 2023 Q4 收益")。
        
        Returns:
            List[Document]: RAG 文档列表。
        """
        documents = []
        
        # 策略 1: 将整个表转换为 Markdown 或文本
        # (适用于小型表格)
        try:
            table_str = data.to_markdown(index=False)
            content = f"Context: {context}\n\n{table_str}"
            
            doc_id = f"tabular_{context.replace(' ', '_')}_{hash(table_str)}"
            
            doc = Document(
                doc_id=doc_id,
                content=content,
                metadata={
                    "data_type": "tabular",
                    "context": context,
                    "rows": len(data)
                }
            )
            documents.append(doc)

        except Exception as e:
            logger.error(f"Failed to convert tabular data to markdown: {e}")

        # 策略 2: (更高级) 将每一行转换为一个 Document
        # (适用于大型表格，其中每一行都是一个独立的实体)
        
        return documents

    def format_graph_data(self, data_points: List[Dict[str, Any]]) -> List[Document]:
        """
        将来自 Graph DB (知识图谱) 的节点/关系转换为 Document。
        
        [RAG 架构]：
        这是 Graph-RAG 的一部分，将图结构 "扁平化" 为 LLM 可消费的文本。
        
        Args:
            data_points (List[Dict[str, Any]]): 
                例如 [{'node': 'AAPL', 'relation': 'SUPPLIES', 'neighbor': 'TSMC'}, ...]
        
        Returns:
            List[Document]: RAG 文档列表。
        """
        documents = []
        
        if not data_points:
            return documents
        
        # 示例：为RAG创建文档
        for dp in data_points:
            text_content = (
                f"Entity: {dp.get('node', 'N/A')}\n"
                f"Relation: {dp.get('relation', 'N/A')}\n"
                f"Neighbor: {dp.get('neighbor', 'N/A')}"
            )
            
            doc_id = f"graph_{dp.get('node', 'N/A')}_{dp.get('relation', 'N/A')}_{dp.get('neighbor', 'N/A')}"
            
            doc = Document(
                doc_id=doc_id,
                content=text_content,
                metadata={
                    "data_type": "graph_relation",
                    **dp 
                }
            )
            documents.append(doc)
            
        return documents

    def adapt(self, data: Any) -> List[Document]:
        """
        (RAG) 统一的适配器入口点。
        将任何支持的数据类型转换为 RAG Document 列表。
        """
        documents = []
        try:
            if isinstance(data, MarketData):
                documents.extend(self.format_market_data(data))
            
            elif isinstance(data, EventData):
                documents.append(self.format_event_data(data))
            
            elif isinstance(data, list):
                # (假设是来自 Graph DB 的数据点列表)
                if data and isinstance(data[0], dict) and ('node' in data[0] or 'relation' in data[0]):
                    documents.extend(self.format_graph_data(data))
                else:
                    logger.warning(f"Adapting list, but content type is unrecognized: {data[0] if data else 'Empty'}")
            
            elif isinstance(data, pd.DataFrame):
                # (需要上下文来格式化 DataFrame)
                logger.warning("Adapting DataFrame requires 'context'. Using default.")
                documents.extend(self.format_tabular_data(data, context="General Data"))

            else:
                logger.warning(f"DataAdapter received unsupported data type: {type(data)}")

        except Exception as e:
            logger.error(f"Error during data adaptation: {e}", exc_info=True)
            
        return documents
