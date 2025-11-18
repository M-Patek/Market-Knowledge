import uuid
from typing import Dict, Any, List, Tuple, Union
from uuid import uuid4, UUID
import pandas as pd

from Phoenix_project.ai.graph_db_client import GraphDBClient
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.core.schemas.fusion_result import FusionResult

# [Task 3] Import all necessary components for orchestration
from Phoenix_project.core.schemas.data_schema import (
    NewsData, 
    MarketData, 
    EconomicIndicator
)
from Phoenix_project.ai.embedding_client import EmbeddingClient
from Phoenix_project.ai.relation_extractor import RelationExtractor
from Phoenix_project.memory.vector_store import VectorStore, Document
from Phoenix_project.core.exceptions import CognitiveError

logger = get_logger(__name__)

class KnowledgeInjector:
    """
    Injects insights (L1 analysis, L2 fusions) into the Knowledge Graph.
    Acts as the Orchestrator for the entire Ingestion Pipeline (Task 3).
    """

    def __init__(
        self, 
        graph_db_client: GraphDBClient,
        # [Task 3B] Inject all clients needed for the ingestion orchestra
        embedding_client: EmbeddingClient,
        relation_extractor: RelationExtractor,
        vector_store: VectorStore
    ):
        """
        Initializes the KnowledgeInjector.
        
        Args:
            graph_db_client: The client for interacting with the graph DB (e.g., Neo4j).
            embedding_client: The client for generating text embeddings.
            relation_extractor: The client for extracting KG triples.
            vector_store: The client for storing vector embeddings.
        """
        self.db_client = graph_db_client
        self.embedding_client = embedding_client
        self.relation_extractor = relation_extractor
        self.vector_store = vector_store
        logger.info("KnowledgeInjector initialized.")

    async def process_batch(
        self, 
        batch: List[Union[NewsData, MarketData, EconomicIndicator]]
    ):
        """
        [Task 3B] The main orchestration method for the ingestion flow.
        
        This method:
        1. Generates a single atomic batch ID.
        2. Stamps this ID onto every item in the batch.
        3. Passes the batch to all clients (embedding, relation, DBs) for processing.
        4. Performs final consistency validation.
        """
        if not batch:
            logger.info("KnowledgeInjector: process_batch called with empty batch.")
            return

        # [Task 3B] 1. Generate an Atomic ID at the start (The "Transaction ID")
        ingestion_batch_id = uuid.uuid4()
        logger.info(f"Starting ingestion process for batch_id: {ingestion_batch_id}")

        # [Task 3B] 2. Propagate and Mutate: Stamp the ID onto all items.
        # Also prepare text/documents for embedding and storage.
        texts_to_embed = []
        documents_for_store = []
        
        for item in batch:
            item.ingestion_batch_id = ingestion_batch_id
            
            # Handle different data types safely
            if hasattr(item, 'content'):
                text_content = item.content
            else:
                # Fallback for MarketData/EconomicIndicator: stringify the model
                text_content = str(item.model_dump())
            
            texts_to_embed.append(text_content)
            
            # Create Document object for VectorStore (carrying the batch_id in metadata)
            doc_metadata = item.model_dump(mode='json')
            documents_for_store.append(Document(page_content=text_content, metadata=doc_metadata))

        try:
            # --- Step A: Embedding & Vector Store ---
            logger.debug(f"[{ingestion_batch_id}] Generating embeddings...")
            embeddings = await self.embedding_client.get_embeddings(texts_to_embed)
            
            logger.debug(f"[{ingestion_batch_id}] Adding to VectorStore (Namespace: {ingestion_batch_id})...")
            # [Task 3C Phase 1] Use the new aadd_batch method
            await self.vector_store.aadd_batch(documents_for_store, embeddings, ingestion_batch_id)

            # --- Step B: Relation Extraction & Graph DB ---
            logger.debug(f"[{ingestion_batch_id}] Extracting relations and updating GraphDB...")
            all_triples = []
            for item in batch:
                # Use content if available, else skip relation extraction for numeric data
                if hasattr(item, 'content'):
                    # Extract using LLM
                    graph_data = await self.relation_extractor.extract_relations(item.content, metadata=item.model_dump(mode='json'))
                    # Convert edges to triples (source, relation, target)
                    # Note: This is a simplification. Real logic might need node mapping.
                    for edge in graph_data.get("edges", []):
                        # Assuming edge is dict with 'source', 'relation', 'target'
                        if 'source' in edge and 'relation' in edge and 'target' in edge:
                            all_triples.append((edge['source'], edge['relation'], edge['target']))
            
            if all_triples:
                # [Task 3C Phase 2] Use the new add_triples with batch_id stamping
                await self.db_client.add_triples(all_triples, batch_id=ingestion_batch_id)
        
            # --- Step C: Final Validation (Task 3C) ---
            logger.info(f"[{ingestion_batch_id}] Performing consistency checks...")
            
            # 1. Vector Store Check (Strict)
            vector_count = await self.vector_store.count_by_batch_id(ingestion_batch_id)
            expected_count = len(batch)
            
            if vector_count != expected_count:
                error_msg = (f"[Data Inconsistency] Ingestion batch {ingestion_batch_id} failed vector check. "
                             f"Expected {expected_count}, found {vector_count}.")
                logger.critical(error_msg)
                raise CognitiveError(error_msg)
            
            # 2. Graph DB Check (Sanity)
            graph_count = await self.db_client.count_by_batch_id(ingestion_batch_id)
            if expected_count > 0 and graph_count == 0:
                # Just a warning, as relation extraction is probabilistic/generative
                logger.warning(f"[{ingestion_batch_id}] GraphDB count is 0 for non-empty batch. "
                               "This might be normal if no relations were found, but worth investigating.")
            
            logger.info(f"[{ingestion_batch_id}] Ingestion complete. Vectors: {vector_count}, Graph Nodes: {graph_count}.")

        except Exception as e:
            logger.error(f"[{ingestion_batch_id}] Critical failure during ingestion orchestration: {e}", exc_info=True)
            raise CognitiveError(f"Ingestion failed for batch {ingestion_batch_id}: {e}")

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
