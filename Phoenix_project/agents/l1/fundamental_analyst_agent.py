import json
import logging
from typing import List, Any, Generator, AsyncGenerator
import time

from Phoenix_project.agents.l1.base import L1Agent
from Phoenix_project.core.schemas.task_schema import Task
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem

# 获取日志记录器
logger = logging.getLogger(__name__)

class FundamentalAnalystAgent(L1Agent):
    """
    L1 智能体：基本面分析师
    评估目标资产的内在价值、财务健康状况和盈利增长。
    """
    def __init__(
        self,
        agent_id: str,
        llm_client: Any,
        data_manager: Any,
    ):
        """
        初始化 FundamentalAnalystAgent。
        
        参数:
            agent_id (str): 智能体的唯一标识符。
            llm_client (Any): 用于与 LLM API 交互的客户端 (例如 EnsembleClient)。
            data_manager (Any): 用于检索数据的 DataManager。
        """
        super().__init__(
            agent_id=agent_id,
            llm_client=llm_client,
            data_manager=data_manager
        )
        logger.info(f"[{self.agent_id}] FundamentalAnalystAgent initialized.")

    async def run(self, task: Task, context: List[Any]) -> AsyncGenerator[EvidenceItem, None]:
        """
        异步运行智能体以执行基本面分析。
        
        参数:
            task (Task): 分配给智能体的任务。
            context (List[Any]): 智能体运行所需的附加上下文（例如，原始文档）。

        收益:
            AsyncGenerator[EvidenceItem, None]: 异步生成一个或多个 EvidenceItem。
        """
        logger.info(f"[{self.agent_id}] Running fundamental analysis for task: {task.task_id} on symbols: {task.symbols}")

        if not task.symbols:
            logger.warning(f"[{self.agent_id}] Task {task.task_id} has no symbols. Skipping.")
            return

        # 1. 准备 Prompt 的上下文
        # 我们假设任务只针对一个主要符号，或者我们将为第一个符号执行分析
        symbol = task.symbols[0]
        
        # 将上下文（可能是 Document 对象）转换为字符串
        # TODO: 优化上下文压缩
        context_str = "\n---\n".join([doc.content for doc in context if hasattr(doc, 'content')])
        
        if not context_str:
            logger.warning(f"[{self.agent_id}] No string content found in context for task {task.task_id}. Skipping.")
            return

        context_map = {
            "symbol": symbol,
            "context": context_str
        }
        
        agent_prompt_name = "l1_fundamental_analyst" # 对应于 l1_fundamental_analyst.json

        try:
            # 2. 异步调用 LLM
            # 我们假设 llm_client 有一个 'run_llm_task' 方法
            logger.debug(f"[{self.agent_id}] Calling LLM with prompt: {agent_prompt_name} for symbol: {symbol}")
            
            response_str = await self.llm_client.run_llm_task(
                agent_prompt_name=agent_prompt_name,
                context_map=context_map
            )
            
            if not response_str:
                logger.warning(f"[{self.agent_id}] LLM returned no response for task {task.task_id}.")
                return

            # 3. 解析和验证 JSON 输出
            logger.debug(f"[{self.agent_id}] Received LLM response (raw): {response_str[:200]}...")
            response_data = json.loads(response_str)
            
            # 4. 验证并生成 EvidenceItem
            # Pydantic 模型的 model_validate 会自动验证 schema
            evidence = EvidenceItem.model_validate(response_data)
            
            # 确保 agent_id 被正确设置
            evidence.agent_id = self.agent_id
            
            logger.info(f"[{self.agent_id}] Successfully generated evidence for {symbol} with confidence {evidence.confidence}")
            yield evidence

        except json.JSONDecodeError as e:
            logger.error(f"[{self.agent_id}] Failed to decode LLM JSON response for task {task.task_id}. Error: {e}")
            logger.debug(f"[{self.agent_id}] Faulty JSON string: {response_str}")
            return
        except Exception as e:
            # 捕获其他潜在错误（例如 Pydantic 验证错误, API 调用失败）
            logger.error(f"[{self.agent_id}] An error occurred during agent run for task {task.task_id}. Error: {e}")
            return
