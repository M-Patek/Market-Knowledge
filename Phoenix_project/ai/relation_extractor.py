import logging
from typing import Dict, Any, List

# 修正：不再导入 'APIGateway'，它不存在。
# 改为导入 'GeminiPoolManager'，这是实际的 LLM 客户端。
from api.gemini_pool_manager import GeminiPoolManager
from core.schemas.data_schema import KnowledgeGraph

logger = logging.getLogger(__name__)

class RelationExtractor:
    """
    一个使用 LLM (通过 GeminiPoolManager) 从非结构化文本中
    提取实体和关系的组件。
    (Task 7 - 关系提取器)
    """

    def __init__(self, gemini_pool: GeminiPoolManager):
        """
        初始化关系提取器。
        
        Args:
            gemini_pool (GeminiPoolManager): 用于调用 LLM API 的 API 池。
        """
        self.gemini_pool = gemini_pool
        self.prompt_template = self._load_prompt_template()
        logger.info("RelationExtractor initialized.")

    def _load_prompt_template(self) -> str:
        """
        加载用于关系提取的提示模板。
        
        FIXME: 提示应该从 ai/prompt_manager.py 或
               prompts/ 目录加载，而不是硬编码。
        """
        # 这是一个简化的示例提示
        return """
        You are a financial analyst. Extract all entities (Companies, People, Products) 
        and relationships (e.g., 'CEO_OF', 'COMPETES_WITH', 'PARTNERS_WITH') 
        from the text below.
        
        Return the answer ONLY as a JSON object with two keys: "nodes" and "relations".
        
        - "nodes" should be a list of objects, e.g.: 
          [{"id": "NVDA", "label": "Company", "properties": {"name": "NVIDIA"}}]
        - "relations" should be a list of objects, e.g.: 
          [{"id": "r1", "type": "CEO_OF", "start_node_id": "JENSEN_HUANG", "end_node_id": "NVDA"}]

        Text to analyze:
        ---
        {text_content}
        ---
        """

    async def extract_kg_from_text(self, text_content: str, source_id: str) -> KnowledgeGraph:
        """
        从单个文本块中提取知识图谱片段。
        
        Args:
            text_content (str): 要分析的（例如）新闻文章。
            source_id (str): 文本来源的 ID，用于日志记录。
            
        Returns:
            KnowledgeGraph: 一个包含提取的节点和关系的 Pydantic 对象。
        """
        logger.debug(f"Extracting relations from source_id: {source_id}")
        
        prompt = self.prompt_template.format(text_content=text_content)
        
        try:
            # 修正：调用 gemini_pool 上的方法
            result_json = await self.gemini_pool.generate_json_response(
                prompt=prompt
            )
            
            if result_json:
                # 验证并解析为 Pydantic 模型
                kg = KnowledgeGraph(**result_json)
                logger.info(f"Extracted {len(kg.nodes)} nodes and {len(kg.relations)} relations from {source_id}.")
                return kg
            else:
                logger.warning(f"Relation extraction returned empty JSON for {source_id}")
                return KnowledgeGraph(nodes=[], relations=[])

        except Exception as e:
            logger.error(f"Failed to extract relations from {source_id}: {e}", exc_info=True)
            return KnowledgeGraph(nodes=[], relations=[])
