import logging
import json
from typing import Dict, Any, List, Optional, Union
import asyncio
from pydantic import BaseModel, ValidationError, Field

# FIX (E1, E2, E3): 导入统一的模式
from Phoenix_project.core.schemas.data_schema import MarketData
# 假设的 FactCheckResult 模式，用于结构化输出解析
# 实际应用中，此模式应从 core/schemas/ 导入
class FactCheckResult(BaseModel):
    claim: str = Field(description="The original claim that was checked.")
    is_verified: bool = Field(description="True if the claim is verified, False if refuted or uncertain.")
    confidence: float = Field(description="Confidence score (0.0 to 1.0) in the verification result.")
    verification_reasoning: str = Field(description="Concise reasoning for the verification outcome.")
    sources: List[Dict[str, str]] = Field(description="List of external sources used for verification (e.g., {'uri': '...', 'title': '...'}).")

logger = logging.getLogger(__name__)

class FactChecker:
    """
    负责验证 L1 Agent 生成的 Evidence 中的关键“声明”(Claims) 的事实准确性。
    使用 Gemini Search Tool 进行外部验证。
    """
    
    def __init__(self, config: Dict[str, Any], llm_client: Any, prompt_manager: Any, prompt_renderer: Any, search_adapter: Any):
        self.config = config.get("fact_checker", {})
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.prompt_renderer = prompt_renderer
        self.search_adapter = search_adapter
        
        self.prompt_template_name = self.config.get("prompt_template", "l2_critic") 
        self.response_schema = self._get_response_schema() # 结构化输出模式
        # 注意：system_instruction 不再在此处静态生成，而是在运行时动态生成以包含时间约束
        
        logger.info(f"FactChecker initialized. Prompt: {self.prompt_template_name}")

    def _get_system_instruction(self, current_time_str: Optional[str] = None) -> str:
        """
        [TASK-P2-001 Fix] 核心 LLM 约束注入：强制使用搜索工具、输出格式以及时间约束。
        """
        instruction = (
            "You are a critical fact-checking agent. Your ONLY goal is to verify or refute "
            "a list of claims using the provided search tool. You MUST use the search tool for external verification. "
            "Do NOT use prior knowledge. Your final output MUST be a JSON object conforming to the FactCheckResult array schema."
        )
        
        # [防幻觉关键修正] 注入时间约束
        if current_time_str:
            instruction += (
                f"\n\nTEMPORAL CONSTRAINT: The Current Simulation Time is {current_time_str}. "
                "You MUST NOT access or verify using any information available only AFTER this time. "
                "Treat any event happening after this timestamp as unknown future."
            )
            
        return instruction

    def _get_response_schema(self) -> Dict[str, Any]:
        """
        [TASK-P2-001 Fix] 定义强制结构化输出的 JSON Schema。
        """
        # 定义一个包含 FactCheckResult 数组的顶级模式
        return {
            "type": "ARRAY",
            "items": FactCheckResult.model_json_schema()
        }

    def _parse_response_to_results(self, raw_response: Dict[str, Any]) -> List[FactCheckResult]:
        """
        [TASK-P2-001 Fix] 解析 LLM 的原始响应，提取结构化数据。
        """
        if not raw_response or not raw_response.get('candidates'):
            logger.error("LLM response is empty or invalid.")
            return []
            
        try:
            # 假设结构化输出在第一个 part 的 text 字段中
            json_text = raw_response['candidates'][0]['content']['parts'][0]['text']
            
            # LLM通常会返回JSON字符串，需要反序列化
            raw_results = json.loads(json_text)
            
            validated_results = []
            for item in raw_results:
                try:
                    # 使用 Pydantic 验证和构建对象
                    validated_results.append(FactCheckResult(**item))
                except ValidationError as e:
                    logger.warning(f"Failed to validate FactCheckResult item: {e}. Skipping item.")
            
            return validated_results
            
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            logger.error(f"Failed to parse structured response from LLM: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during result parsing: {e}")
            return []


    async def check_facts(self, claims: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证事实并返回验证报告。
        [TASK-P2-001 Fix] 将输入从 evidence: List[Dict[str, Any]] 改为 claims: List[str]。
        """
        if not claims:
            return {"status": "skipped", "message": "No claims to check."}

        logger.info(f"Starting fact check for {len(claims)} claims.")
        
        # 1. 渲染 Prompt
        # [TASK-P2-001 Fix 1] Use the stored template name string and the new claims list.
        prompt_context = {
            "claims": "\n".join([f"- {c}" for c in claims]),
            "context": str(context)
        }
        prompt = self.prompt_renderer.render(self.prompt_template_name, prompt_context)

        # 2. 定义搜索工具
        search_tool = self.search_adapter.get_tool_function()
        
        # 3. 动态构建系统指令 (包含时间约束)
        # 尝试从 context 中获取时间，通常 context 会包含 current_time
        current_time_val = context.get("current_time") or context.get("timestamp")
        current_time_str = str(current_time_val) if current_time_val else "Unknown"
        
        dynamic_system_instruction = self._get_system_instruction(current_time_str)

        # 4. 调用 LLM
        try:
            raw_response = await self.llm_client.generate_text_with_tools(
                prompt=prompt,
                tools=[search_tool],
                temperature=self.config.get("temperature", 0.0),
                system_instruction=dynamic_system_instruction, # 注入包含时间约束的指令
                response_schema=self.response_schema      # 注入结构化模式
            )

            # 5. 解析结构化输出
            fact_check_results = self._parse_response_to_results(raw_response)
            
            if not fact_check_results:
                return {"status": "failed", "message": "Fact check failed to produce structured results."}

            return {
                "status": "success", 
                "report": [r.model_dump() for r in fact_check_results], # 返回字典列表
                "results_count": len(fact_check_results)
            }

        except Exception as e:
            logger.error(f"Fact checking failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
