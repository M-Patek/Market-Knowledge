from typing import Dict, Any, List, Tuple
from uuid import uuid4 # [任务 B.4] 导入 uuid
import pandas as pd # [任务 B.4] 导入 pandas

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
        
        Args:
            analysis_result: The output from an L1 agent.
            event: The source event that triggered the analysis.
        """
        logger.debug(f"Formatting L1 analysis for KG injection...")
        
        try:
            # [任务 B.4] 调用已实现的格式化方法
            triples = self._format_for_kg(analysis_result, event)
            
            if not triples:
                logger.warning("No KG triples were generated from the L1 analysis.")
                return

            logger.info(f"Injecting {len(triples)} triples into Knowledge Graph...")
            
            # (此处的实现取决于 GraphDBClient 的接口)
            # 假设有一个 'add_triples' 方法
            await self.db_client.add_triples(triples)
            
            logger.info("L1 analysis successfully injected into KG.")

        except Exception as e:
            logger.error(f"Failed to inject L1 analysis into KG: {e}", exc_info=True)
            
    async def inject_l2_fusion(self, fusion_result: FusionResult):
        """
        Formats and injects an L2 (fused) result into the KG.
        """
        # TBD: Implement L2 injection logic
        logger.info(f"KG injection for L2 Fusion (Symbol: {fusion_result.symbol}) is not yet implemented.")
        pass

    def _format_for_kg(
        self, 
        analysis_result: Dict[str, Any], 
        event: Dict[str, Any]
    ) -> List[Tuple[str, str, Any]]:
        """
        [任务 B.4] Implemented.
        Converts an L1 agent's analysis result into a list of triples 
        (subject, predicate, object) for the Knowledge Graph.
        
        Args:
            analysis_result: The output from an L1 agent.
            event: The source event.
            
        Returns:
            A list of (subject, predicate, object) tuples.
        """
        
        triples = []
        
        # 1. 确定核心实体 (Subject)
        symbol = event.get('symbol') or analysis_result.get('symbol')
        if not symbol:
            logger.warning("Cannot format for KG: 'symbol' is missing in both event and analysis.")
            return []
        
        # 2. 创建一个代表本次分析本身的唯一节点 (Subject)
        # 这允许我们将情绪、信等附加到“分析事件”上，而不是直接附加到“公司”上
        analysis_id = f"Analysis:{uuid4()}"
        
        # 3. 链接元数据到分析节点
        timestamp = event.get('timestamp') or pd.Timestamp.now(tz='UTC')
        source = event.get('source')
        agent_name = analysis_result.get('agent_name', 'UnknownL1Agent')

        triples.append((analysis_id, "isAnalysisOf", symbol)) # (Analysis:uuid) -> (isAnalysisOf) -> (Symbol:AAPL)
        triples.append((analysis_id, "generatedBy", agent_name)) # (Analysis:uuid) -> (generatedBy) -> (Agent:TechnicalAnalyst)
        triples.append((analysis_id, "generatedAt", str(timestamp))) # (Analysis:uuid) -> (generatedAt) -> "2025-..."
        
        if source:
            triples.append((analysis_id, "basedOnSource", source)) # (Analysis:uuid) -> (basedOnSource) -> "Reuters"
        
        # 4. 提取 L1 智能体的核心洞察
        
        # 情绪/观点
        sentiment = analysis_result.get('sentiment_label') or analysis_result.get('sentiment')
        if sentiment:
            triples.append((analysis_id, "hasSentiment", sentiment.upper()))

        # 置信度/分数
        confidence = analysis_result.get('sentiment_score') or analysis_result.get('confidence')
        if confidence is not None:
            try:
                triples.append((analysis_id, "hasConfidence", round(float(confidence), 4)))
            except (ValueError, TypeError):
                logger.warning(f"Could not parse confidence '{confidence}' as float.")

        # 价格目标
        price_target = analysis_result.get('price_target')
        if price_target:
            try:
                triples.append((analysis_id, "hasPriceTarget", float(price_target)))
            except (ValueError, TypeError):
                logger.warning(f"Could not parse price_target '{price_target}' as float.")

        # 关键驱动因素 (List)
        key_drivers = analysis_result.get('key_drivers', [])
        for driver in key_drivers:
            if driver:
                triples.append((analysis_id, "hasDriver", str(driver))) # (Analysis:uuid) -> (hasDriver) -> "iPhone Sales"

        # 关键风险 (List)
        key_risks = analysis_result.get('key_risks', [])
        for risk in key_risks:
            if risk:
                triples.append((analysis_id, "hasRisk", str(risk))) # (Analysis:uuid) -> (hasRisk) -> "China Demand"

        logger.debug(f"Generated {len(triples)} triples for {symbol} analysis.")
        return triples
