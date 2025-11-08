import json
import logging
import asyncio
from typing import List, Any, AsyncGenerator, Optional

from Phoenix_project.agents.l2.base import L2Agent
from Phoenix_project.core.schemas.task_schema import Task
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem
from Phoenix_project.core.schemas.critic_result import CriticResult
from pydantic import ValidationError

# 获取日志记录器
logger = logging.getLogger(__name__)

class CriticAgent(L2Agent):
    """
    L2 智能体：批评家 (Critic)
    评估 L1 证据的质量、清晰度、偏见和可信度。
    """

    async def _critique_evidence(self, evidence: EvidenceItem) -> Optional[EvidenceItem]:
        """
        (内部) 异步评估单个 EvidenceItem 的质量。
        
        参数:
            evidence (EvidenceItem): L1 智能体生成的证据。

        返回:
            Optional[EvidenceItem]: 一个新的、类型为 "Critic" 的 EvidenceItem，
                                    如果评估失败则返回 None。
        """
        logger.debug(f"[{self.agent_id}] Critiquing evidence: {evidence.evidence_id} from agent {evidence.agent_id}")
        
        agent_prompt_name = "l2_critic"
        
        try:
            # 1. 准备 Prompt 上下文
            context_map = {
                "agent_id": evidence.agent_id,
                "original_evidence_json": evidence.model_dump_json(),
                "original_evidence_id": str(evidence.evidence_id),
                "symbols_list_str": json.dumps(evidence.symbols)
            }

            # 2. 异步调用 LLM
            response_str = await self.llm_client.run_llm_task(
                agent_prompt_name=agent_prompt_name,
                context_map=context_map
            )

            if not response_str:
                logger.warning(f"[{self.agent_id}] LLM returned no response for critiquing evidence {evidence.evidence_id}")
                return None

            # 3. 解析和验证 CriticResult
            response_data = json.loads(response_str)
            critic_result = CriticResult.model_validate(response_data)

            # 4. [关键] 将 CriticResult 转换为 EvidenceItem 以便 Executor 处理
            # 我们使用 'quality_score' 作为新证据的 'confidence'
            critique_evidence = EvidenceItem(
                symbols=critic_result.symbols,
                evidence_type="Critic",
                content=critic_result.critique,
                confidence=critic_result.quality_score, # 映射质量分
                data_horizon=evidence.data_horizon, # 继承原始数据范围
                parent_evidence_id=evidence.evidence_id, # 追踪溯源
                metadata={
                    "original_evidence_id": str(critic_result.original_evidence_id),
                    "quality_score": critic_result.quality_score,
                    "clarity_score": critic_result.clarity_score,
                    "bias_score": critic_result.bias_score,
                    "relevance_score": critic_result.relevance_score,
                    "suggestions": critic_result.suggestions_for_improvement or "None"
                }
                # evidence_id 和 created_at 将由 Pydantic 自动生成
            )
            
            logger.info(f"[{self.agent_id}] Successfully generated critique for {evidence.evidence_id} (Quality: {critic_result.quality_score})")
            return critique_evidence

        except json.JSONDecodeError as e:
            logger.error(f"[{self.agent_id}] Failed to decode LLM JSON response for critique task. Error: {e}")
            logger.debug(f"[{self.agent_id}] Faulty JSON string for critique: {response_str}")
            return None
        except ValidationError as e:
            logger.error(f"[{self.agent_id}] Failed to validate CriticResult schema. Error: {e}")
            logger.debug(f"[{self.agent_id}] Faulty JSON data for critique: {response_data}")
            return None
        except Exception as e:
            logger.error(f"[{self.agent_id}] An unexpected error occurred during critique task. Error: {e}")
            return None

    async def run(self, task: Task, dependencies: List[Any]) -> AsyncGenerator[EvidenceItem, None]:
        """
        异步运行智能体，并行评估所有 L1 证据的质量。
        
        参数:
            task (Task): 当前任务。
            dependencies (List[Any]): 来自 L1 智能体的 EvidenceItem 列表。

        收益:
            AsyncGenerator[EvidenceItem, None]: 异步生成类型为 "Critic" 的新 EvidenceItem。
        """
        logger.info(f"[{self.agent_id}] Running CriticAgent for task: {task.task_id}")

        # 1. 过滤出有效的 L1 证据 (和 Adversary 证据)
        evidence_items = [
            item for item in dependencies 
            if isinstance(item, EvidenceItem)
        ]

        if not evidence_items:
            logger.warning(f"[{self.agent_id}] No EvidenceItems found in dependencies for task {task.task_id}. Skipping.")
            return

        logger.info(f"[{self.agent_id}] Found {len(evidence_items)} evidence items to critique.")

        # 2. 为每个证据创建并行的评估任务
        critique_tasks = [
            self._critique_evidence(evidence) for evidence in evidence_items
        ]

        # 3. 异步并行执行所有评估
        results = await asyncio.gather(*critique_tasks, return_exceptions=True)

        # 4. 处理结果并 Yield
        for result in results:
            if isinstance(result, Exception):
                # 记录在 gather 中发生的异常
                logger.error(f"[{self.agent_id}] An exception occurred in a critique task: {result}")
            elif isinstance(result, EvidenceItem):
                # 这是一个成功的评估，yield 新的 "Critic" 证据
                result.agent_id = self.agent_id
                yield result
            elif result is None:
                # 内部评估失败（已在 _critique_evidence 中记录）
                continue
