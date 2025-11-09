import asyncio
from typing import List, Dict, Any

from ..ai.ensemble_client import EnsembleClient
from ..ai.prompt_manager import PromptManager
from ..ai.prompt_renderer import PromptRenderer
from ..core.schemas.evidence_schema import Evidence, FactCheckResult
from ..monitor.logging import get_logger

logger = get_logger(__name__)

class FactChecker:
    """
    事实核查器 (FactChecker) 负责验证由 L1 智能体
    生成的分析中的具体、可核查的声明。
    
    它使用搜索工具来查找支持或反驳的证据。
    """

    def __init__(self, 
                 llm_client: EnsembleClient, 
                 prompt_manager: PromptManager, 
                 prompt_renderer: PromptRenderer):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.prompt_renderer = prompt_renderer
        self.model_name = "gemini-1.5-flash-latest" # Flash 适用于事实核查
        
        # 定义搜索工具
        # 注意：工具的实现在 EnsembleClient/GeminiPoolManager 级别处理
        self.search_tool = {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Searches for relevant documents, news, and reports based on a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        
        self.prompt = self.prompt_manager.get_prompt("fact_checker")
        
        # --- 优化：移除 HACK，在初始化时强化系统提示 ---
        # 我们在初始化时修改加载的系统提示，
        # 明确指示模型必须使用搜索工具。
        # 这比在运行时动态修改用户提示更清晰、更健壮。
        
        search_instruction = (
            "\n\n---
"
            "MANDATORY INSTRUCTION: You MUST use the 'search_documents' tool "
            "to find evidence for every claim presented. Do not rely on "
            "internal knowledge. Your task is to verify claims using external search."
            "\n---
"
        )
        
        if "system" in self.prompt:
            self.prompt["system"] = self.prompt["system"] + search_instruction
        else:
            self.prompt["system"] = search_instruction.strip()
            
        logger.info("FactChecker initialized and system prompt modified to enforce search tool usage.")
        # --- 结束优化 ---

    async def check_facts(self, claims: List[str]) -> List[FactCheckResult]:
        """
        核查一系列声明。

        Args:
            claims: 需要核查的字符串声明列表。

        Returns:
            一个 FactCheckResult 列表。
        """
        if not claims:
            return []
            
        logger.info(f"Fact-checking {len(claims)} claims...")
        
        claims_str = "\n".join([f"- {claim}" for claim in claims])
        
        try:
            prompt = self.prompt_renderer.render(
                self.prompt, claims=claims_str
            )
            
            # --- 优化：移除了 HACK ---
            # 之前的 HACK 逻辑（在运行时修改用户提示）已被移除，
            # 因为我们在 __init__ 中强化了系统提示。
            # --- 结束优化 ---

            # 我们希望LLM返回结构化数据
            # 假设 run_chain_structured 能够处理工具调用和返回JSON
            response_json = await self.llm_client.run_chain_structured(
                prompt,
                tools=[self.search_tool],
                model_name=self.model_name
                # 注意：如果 EnsembleClient 支持，我们甚至可以强制调用工具：
                # tool_config={"tool_choice": "search_documents"}
            )
            
            # 假设 LLM 返回一个 FactCheckResult 列表
            # e.g., [{"claim": "...", "verified": true, "evidence": "...", "source": "..."}]
            if not isinstance(response_json, list):
                logger.error(f"Fact-checker LLM returned non-list response: {response_json}")
                raise ValueError("Fact-checker response is not a list of results.")

            results = []
            for item in response_json:
                if isinstance(item, dict):
                    # 尝试解析为 FactCheckResult
                    # 注意：Pydantic 验证可以在这里完成
                    evidence = Evidence(
                        source=item.get('source_url', 'Unknown'),
                        snippet=item.get('evidence_snippet', 'N/A')
                    )
                    results.append(FactCheckResult(
                        claim=item.get('claim', 'N/A'),
                        verified=bool(item.get('verified', False)),
                        evidence=evidence,
                        confidence=float(item.get('confidence', 0.0))
                    ))
                else:
                    logger.warning(f"Skipping invalid item in fact-checker response: {item}")
                    
            logger.info(f"Fact-checking complete. Found {len(results)} results.")
            return results

        except Exception as e:
            logger.error(f"Error during fact-checking: {e}", exc_info=True)
            # 如果核查失败，为所有声明返回 "unverified"
            return [
                FactCheckResult(
                    claim=claim,
                    verified=False,
                    evidence=Evidence(source="Fact-Check Error", snippet=str(e)),
                    confidence=0.0
                ) for claim in claims
            ]
