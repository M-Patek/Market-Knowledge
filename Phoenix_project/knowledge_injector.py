from typing import Dict, Any, List, Tuple
from uuid import uuid4
import pandas as pd

from Phoenix_project.ai.graph_db_client import GraphDBClient
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.core.schemas.fusion_result import FusionResult

logger = get_logger(__name__)

class KnowledgeInjector:
    """
    Injects insights (L1 analysis, L2 fusions) into the Knowledge Graph.
    """

    def __init__(self, graph_db_client: GraphDBClient):
        """
        Initializes the KnowledgeInjector.
        
        Args:
            graph_db_client: The client for interacting with the graph DB (e.g., Neo4j).
        """
        self.db_client = graph_db_client
        logger.info("KnowledgeInjector initialized.")

    async def inject_l1_analysis(self, analysis_result: Dict[str, Any], event: Dict[str, Any]):
        """
        Formats and injects an L1 agent's analysis result into the KG.
        [MODIFIED] 现在传递 analysis_result 'id' 给 _format_for_kg。
        """
        logger.debug(f"Formatting L1 analysis for KG injection...")
        
        try:
            # [MODIFIED] 从 analysis_result (EvidenceItem dict) 获取 ID
            # 假设 analysis_result 是一个 EvidenceItem.model_dump()
            analysis_item_id = analysis_result.get("id", str(uuid4()))
            
            triples = self._format_for_kg(analysis_result, event, analysis_item_id)
            
            if not triples:
                logger.warning("No KG triples were generated from the L1 analysis.")
                return

            logger.info(f"Injecting {len(triples)} triples into Knowledge Graph (L1)...")
            
            # (此处的实现取决于 GraphDBClient 的接口)
            # 假设有一个 'add_triples' 方法
            await self.db_client.add_triples(triples)
            
            logger.info("L1 analysis successfully injected into KG.")

        except Exception as e:
            logger.error(f"Failed to inject L1 analysis into KG: {e}", exc_info=True)
            
    async def inject_l2_fusion(self, fusion_result: FusionResult):
        """
        [Task 1] 已实现
        将 L2（融合层）的最终决策注入知识图谱。
        """
        logger.info(f"Formatting L2 fusion result for KG injection (ID: {fusion_result.id})...")
        
        try:
            triples = []
            # [Task 1] 节点 ID 使用 'FusionDecision:' 前缀
            fusion_id = f"FusionDecision:{fusion_result.id}"
            symbol = fusion_result.target_symbol
            
            # 1. 创建 FusionDecision 节点并链接到 Symbol
            # (我们假设 'targetsSymbol' 在 add_triples 中被处理为关系)
            triples.append((fusion_id, "targetsSymbol", symbol))
            
            # 2. 存储属性
            triples.append((fusion_id, "decision", fusion_result.decision))
            triples.append((fusion_id, "confidence", float(fusion_result.confidence)))
            triples.append((fusion_id, "reasoning", fusion_result.reasoning))
            triples.append((fusion_id, "uncertainty", float(fusion_result.uncertainty)))
            triples.append((fusion_id, "timestamp", str(fusion_result.timestamp)))

            # 3. 链接到它所基于的 L1 Analysis 节点
            l1_ids = set(fusion_result.supporting_evidence_ids) | set(fusion_result.conflicting_evidence_ids)
            
            for l1_id in l1_ids:
                # [Task 1] 链接到 L1 节点 (假设 L1 节点 ID 是 'Analysis:{l1_id}')
                l1_node_id = f"Analysis:{l1_id}"
                # (我们假设 'basedOnAnalysis' 在 add_triples 中被处理为关系)
                triples.append((fusion_id, "basedOnAnalysis", l1_node_id))

            if not triples:
                logger.warning("No triples generated for L2 fusion.")
                return
                
            logger.info(f"Injecting {len(triples)} triples into Knowledge Graph (L2)...")
            await self.db_client.add_triples(triples)
            logger.info("L2 fusion result successfully injected into KG.")

        except Exception as e:
            logger.error(f"Failed to inject L2 fusion result into KG: {e}", exc_info=True)

    def _format_for_kg(
        self, 
        analysis_result: Dict[str, Any], 
        event: Dict[str, Any],
        analysis_item_id: str  # <--- [MODIFIED]
    ) -> List[Tuple[str, str, Any]]:
        """
        [任务 B.4] Implemented.
        [MODIFIED] 使用传入的 analysis_item_id 作为节点 ID (例如 'Analysis:{uuid}')
        Converts an L1 agent's analysis result into a list of triples 
        (subject, predicate, object) for the Knowledge Graph.
        
        Args:
            analysis_result: The output from an L1 agent (EvidenceItem dict).
            event: The source event.
            analysis_item_id: The ID from the EvidenceItem.
            
        Returns:
            A list of (subject, predicate, object) tuples.
        """
        
        triples = []
        
        # 1. 确定核心实体 (Subject)
        symbol = event.get('symbol') or analysis_result.get('symbol')
        if not symbol:
            # L1 'l1_macro_strategist' 可能会分析 'GLOBAL_MACRO' 而不是特定 symbol
            # 但 L2 fusion 应该有 target_symbol。
            # L1 分析 (EvidenceItem) 必须有 symbols 列表。
            symbols_list = analysis_result.get('symbols', [])
            if symbols_list:
                symbol = symbols_list[0] # 只取第一个作为主要链接
            else:
                 logger.warning("Cannot format for KG: 'symbols' is missing in analysis_result.")
                 return []
        
        # 2. 创建一个代表本次分析本身的唯一节点 (Subject)
        # [MODIFIED] 使用 L1 EvidenceItem ID 作为节点 ID
        analysis_id = f"Analysis:{analysis_item_id}"
        
        # 3. 链接元数据到分析节点
        timestamp = analysis_result.get('timestamp', event.get('timestamp') or pd.Timestamp.now(tz='UTC').isoformat())
        source = event.get('source')
        agent_name = analysis_result.get('agent_id', 'UnknownL1Agent') # agent_id 来自 EvidenceItem

        # (我们假设 'isAnalysisOf' 在 add_triples 中被处理为关系)
        triples.append((analysis_id, "isAnalysisOf", symbol)) # (Analysis:uuid) -> (isAnalysisOf) -> (Symbol:AAPL)
        triples.append((analysis_id, "generatedBy", agent_name)) # (Analysis:uuid) -> (generatedBy) -> "l1_technical_analyst"
        triples.append((analysis_id, "generatedAt", str(timestamp))) # (Analysis:uuid) -> (generatedAt) -> "2025-..."
        
        if source:
            triples.append((analysis_id, "basedOnSource", source)) # (Analysis:uuid) -> (basedOnSource) -> "Reuters"
        
        # 4. 提取 L1 智能体的核心洞察
        
        content = analysis_result.get('content', 'N/A')
        triples.append((analysis_id, "content", content))

        confidence = analysis_result.get('confidence')
        if confidence is not None:
            try:
                triples.append((analysis_id, "hasConfidence", round(float(confidence), 4)))
            except (ValueError, TypeError):
                logger.warning(f"Could not parse confidence '{confidence}' as float.")

        evidence_type = analysis_result.get('evidence_type', 'Generic')
        triples.append((analysis_id, "evidenceType", evidence_type))
        
        data_horizon = analysis_result.get('data_horizon', 'Unknown')
        triples.append((analysis_id, "dataHorizon", data_horizon))

        # 5. 提取 Metadata (来自 L1 prompts)
        metadata = analysis_result.get('metadata', {})
        if metadata.get('key_metric'):
            triples.append((analysis_id, "keyMetric", metadata['key_metric']))
        if metadata.get('metric_value'):
            triples.append((analysis_id, "metricValue", str(metadata['metric_value']))) # 确保是字符串

        logger.debug(f"Generated {len(triples)} triples for {symbol} analysis ({analysis_id}).")
        return triples
