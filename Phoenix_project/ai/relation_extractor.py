# Phoenix_project/ai/relation_extractor.py
import logging
from typing import List, Dict, Any
import google.generativeai as genai
from pydantic import BaseModel, Field

# --- 用于结构化知识图谱输出的Pydantic模型 ---
class Entity(BaseModel):
    """代表文本中的单个命名实体。"""
    id: int = Field(..., description="实体的唯一整数ID。")
    name: str = Field(..., description="实体名称 (例如, '分析师A', '报告B')。")
    type: str = Field(..., description="实体类型 (例如, '人物', '组织', '报告')。")

class Relation(BaseModel):
    """代表两个实体间的有向关系。"""
    source_id: int = Field(..., description="源实体的ID。")
    target_id: int = Field(..., description="目标实体的ID。")
    label: str = Field(..., description="描述关系的标签 (例如, '推荐', '基于', '反驳')。")

class KnowledgeGraph(BaseModel):
    """从证据中提取的完整知识图谱。"""
    entities: List[Entity]
    relations: List[Relation]

class RelationExtractor:
    """
    使用生成式模型从一系列文本证据中提取实体和关系，
    以构建知识图谱。
    """
    def __init__(self, config: Dict[str, Any] = {}):
        self.logger = logging.getLogger("PhoenixProject.RelationExtractor")
        try
            # 使用KnowledgeGraph模型初始化一个专用于工具调用的模型
            self.extractor_model = genai.GenerativeModel(
                "gemini-1.5-flash-latest", tools=[KnowledgeGraph]
            )
        except Exception as e:
            self.logger.error(f"初始化提取器模型失败: {e}")
            self.extractor_model = None

    async def extract_graph(self, evidence_texts: List[str]) -> KnowledgeGraph:
        """
        从一系列非结构化文本文档中提取知识图谱。
        """
        if not self.extractor_model or not evidence_texts:
            self.logger.warning("提取器模型不可用或未提供证据文本。返回空图。")
            return KnowledgeGraph(entities=[], relations=[])

        # 将所有证据合并为单个上下文
        full_context = "\n---\n".join(evidence_texts)

        prompt = f"""
        从以下证据集中，提取所有命名实体及其之间的关系。
        识别人物、组织、报告和金融概念等实体。
        识别如'推荐'、'基于'、'反驳'、'支持'、'提及'等关系。
        构建一个完整的知识图谱。

        上下文:
        {full_context}
        """
        try:
            response = await self.extractor_model.generate_content_async(
                prompt,
                tool_config={'function_calling_config': 'ANY'}
            )
            
            # 从模型函数调用中提取结构化的KnowledgeGraph
            fc = response.candidates[0].content.parts[0].function_call
            graph_args = {key: value for key, value in fc.args.items()}
            
            return KnowledgeGraph(**graph_args)

        except Exception as e:
            self.logger.error(f"使用LLM提取知识图谱失败: {e}")
            return KnowledgeGraph(entities=[], relations=[])
