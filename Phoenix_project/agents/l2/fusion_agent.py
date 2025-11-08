import json
import logging
from typing import List, Any, AsyncGenerator, Optional

from Phoenix_project.agents.l2.base import L2Agent
from Phoenix_project.core.schemas.task_schema import Task
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem
from Phoenix_project.core.schemas.fusion_result import FusionResult
from pydantic import ValidationError

# 获取日志记录器
logger = logging.getLogger(__name__)

class FusionAgent(L2Agent):
    """
    L2 智能体：融合 (Fusion)
    审查所有 L1 和 L2 的证据，融合冲突信息，并得出最终的 L2 决策。
    """

    async def run(self, task: Task, dependencies: List[Any]) -> AsyncGenerator[FusionResult, None]:
        """
        异步运行智能体，以融合所有 L1/L2 证据。
        此智能体运行一次（不使用 gather），产出一个 FusionResult。
        
        参数:
            task (Task): 当前任务。
            dependencies (List[Any]): 来自 L1 和 L2 (Adversary, Critic) 的 EvidenceItem 列表。

        收益:
            AsyncGenerator[FusionResult, None]: 异步生成 *单一* 的 FusionResult 对象。
        """
        logger.info(f"[{self.agent_id}] Running FusionAgent for task: {task.task_id}")

        # 1. 过滤出所有有效的证据
        evidence_items = [
            item for item in dependencies 
            if isinstance(item, EvidenceItem)
        ]

        if not evidence_items:
            logger.warning(f"[{self.agent_id}] No EvidenceItems found in dependencies for task {task.task_id}. Skipping fusion.")
            return

        logger.info(f"[{self.agent_id}] Found {len(evidence_items)} total evidence items (L1+L2) to fuse.")

        agent_prompt_name = "l2_fusion"

        try:
            # 2. 准备 Prompt 上下文
            # [关键] 序列化整个证据列表。
            # 我们使用 .model_dump() 并设置 default=str 来处理 UUID 和 datetime
            evidence_json_list = json.dumps(
                [item.model_dump() for item in evidence_items], 
                default=str
            )
            
            context_map = {
                "symbols_list_str": json.dumps(task.symbols),
                "evidence_json_list": evidence_json_list
            }

            # 3. 异步调用 LLM (只调用一次)
            logger.debug(f"[{self.agent_id}] Calling LLM with prompt: {agent_prompt_name} for fusion task.")
            
            response_str = await self.llm_client.run_llm_task(
                agent_prompt_name=agent_prompt_name,
                context_map=context_map
            )

            if not response_str:
                logger.warning(f"[{self.agent_id}] LLM returned no response for fusion task {task.task_id}.")
                return

            # 4. 解析和验证 FusionResult
            logger.debug(f"[{self.agent_id}] Received LLM fusion response (raw): {response_str[:200]}...")
            response_data = json.loads(response_str)
            fusion_result = FusionResult.model_validate(response_data)
            
            logger.info(f"[{self.agent_id}] Successfully generated FusionResult for {task.symbols} with sentiment: {fusion_result.overall_sentiment}")
            
            # 5. [关键] Yield 单一的 FusionResult
            # 我们不转换它，L3 (AlphaAgent) 会直接消费这个对象
            yield fusion_result

        except json.JSONDecodeError as e:
            logger.error(f"[{self.agent_id}] Failed to decode LLM JSON response for fusion task. Error: {e}")
            logger.debug(f"[{self.agent_id}] Faulty JSON string for fusion: {response_str}")
            return
        except ValidationError as e:
            logger.error(f"[{self.agent_id}] Failed to validate FusionResult schema. Error: {e}")
            logger.debug(f"[{self.agent_id}] Faulty JSON data for fusion: {response_data}")
            return
        except Exception as e:
            logger.error(f"[{self.agent_id}] An unexpected error occurred during fusion task. Error: {e}")
            return
