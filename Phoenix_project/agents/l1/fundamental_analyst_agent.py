# Phoenix_project/agents/l1/fundamental_analyst_agent.py
# [主人喵的修复 11.11] 实现了 C.2 (上下文压缩)

import logging
import asyncio
from agents.l1.base import L1BaseAgent
from ai.prompt_manager import PromptManager
from api.gateway import APIGateway
from ai.retriever import Retriever
from reasoning.compressor import ContextCompressor # [新] 假设压缩器在这里

logger = logging.getLogger(__name__)

class FundamentalAnalystAgent(L1BaseAgent):
    """
    L1 智能体：基本面分析师
    - 专注于分析 SEC 文件、财报、收益电话会议记录。
    - 评估公司的财务健康、增长前景、估值和管理效率。
    - 输出结构化的基本面见解。
    """

    def __init__(self, agent_name: str, config: dict, prompt_manager: PromptManager, api_gateway: APIGateway, retriever: Retriever, context_compressor: ContextCompressor):
        super().__init__(agent_name, config, prompt_manager, api_gateway, retriever, context_compressor)
        self.prompt_template_name = "l1_fundamental_analyst"

    async def _prepare_context(self, task_content: dict) -> str:
        """
        [已实现] 准备基本面分析所需的上下文。
        - 检索相关的财务文件 (10-K, 10-Q) 和收益记录。
        - [任务 C.2] TODO: Optimize context compression
        """
        logger.debug(f"{self.agent_name}: Preparing context for task...")
        
        # [任务 C.2 已实现]: 优化上下文压缩
        
        # 1. 检索
        try:
            evidence_list = await self.retriever.retrieve_relevant_context(
                query=task_content.get("query"),
                tickers=task_content.get("tickers", []),
                # 专门针对基本面分析的数据源
                source_types=["10-K", "10-Q", "EarningsCall", "AnalystReport"],
                top_k=self.config.get("retriever_top_k", 10)
            )
        except Exception as e:
            logger.error(f"{self.agent_name}: Failed to retrieve context: {e}", exc_info=True)
            return "Context retrieval failed."

        if not evidence_list:
             logger.warning(f"{self.agent_name}: No evidence found for query, skipping compression.")
             return "No relevant context found."

        # [新] 2. 压缩
        # (ContextCompressor 在 L1BaseAgent 中被注入)
        try:
            # 假设 compressor 接受证据列表并返回一个新的、经过压缩/过滤的证据列表
            compressed_evidence_list = await self.context_compressor.compress(
                evidence_list=evidence_list,
                query=task_content.get("query"),
                max_tokens=self.config.get("compression_max_tokens", 4096) # 添加一个配置
            )
            logger.debug(f"{self.agent_name}: Context compressed successfully. {len(evidence_list)} -> {len(compressed_evidence_list)} items.")
            
            # 3. 格式化
            formatted_context = self.retriever.format_context_for_prompt(compressed_evidence_list)
            return formatted_context

        except Exception as e:
            logger.error(f"{self.agent_name}: Failed to compress context, falling back to uncompressed: {e}", exc_info=True)
            # [回退] 如果压缩失败，使用未压缩的上下文
            formatted_context = self.retriever.format_context_for_prompt(evidence_list)
            return formatted_context

    async def _perform_analysis(self, context: str, task_content: dict) -> dict:
        """
        [已实现] 执行基本面分析 LLM 调用。
        """
        logger.debug(f"{self.agent_name}: Performing analysis...")
        
        prompt_inputs = {
            "query": task_content.get("query"),
            "company_name": task_content.get("company_name", "the target company"),
            "context": context
        }
        
        # 使用 L1BaseAgent 中的 LLM 调用逻辑
        analysis_result = await self.run_llm_analysis(
            prompt_template_name=self.prompt_template_name,
            prompt_inputs=prompt_inputs,
            output_schema=self.config.get("output_schema", {})
        )
        
        return analysis_result
