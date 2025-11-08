import json
import logging
import asyncio
from typing import List, Any, AsyncGenerator, Optional

from Phoenix_project.agents.l2.base import L2Agent
from Phoenix_project.core.schemas.task_schema import Task
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem
from Phoenix_project.core.schemas.adversary_result import AdversaryResult
from pydantic import ValidationError

# 获取日志记录器
logger = logging.getLogger(__name__)

class AdversaryAgent(L2Agent):
    """
    L2 智能体：诘难者 (Adversary)
    审查 L1 证据并生成反驳论点。
    """

    async def _challenge_evidence(self, evidence: EvidenceItem) -> Optional[EvidenceItem]:
        """
        (内部) 异步挑战单个 EvidenceItem。
        
        参数:
            evidence (EvidenceItem): L1 智能体生成的证据。

        返回:
            Optional[EvidenceItem]: 一个新的、类型为 "Adversary" 的 EvidenceItem，
                                    如果挑战失败则返回 None。
        """
        logger.debug(f"[{self.agent_id}] Challenging evidence: {evidence.evidence_id} from agent {evidence.agent_id}")
        
        agent_prompt_name = "l2_adversary"
        
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
                logger.warning(f"[{self.agent_id}] LLM returned no response for challenging evidence {evidence.evidence_id}")
                return None

            # 3. 解析和验证 AdversaryResult
            response_data = json.loads(response_str)
            adversary_result = AdversaryResult.model_validate(response_data)

            # 4. [关键] 将 AdversaryResult 转换为 EvidenceItem 以便 Executor 处理
            counter_evidence = EvidenceItem(
                symbols=adversary_result.symbols,
                evidence_type="Adversary",
                content=adversary_result.counter_argument,
                confidence=adversary_result.confidence,
                data_horizon=evidence.data_horizon, # 继承原始数据范围
                parent_evidence_id=evidence.evidence_id, # 追踪溯源
                metadata={
                    **adversary_result.metadata,
                    "original_evidence_id": str(adversary_result.original_evidence_id),
                    "risk_level": adversary_result.risk_level
                }
                # evidence_id 和 created_at 将由 Pydantic 自动生成
            )
            
            logger.info(f"[{self.agent_id}] Successfully generated counter-evidence for {evidence.evidence_id}")
            return counter_evidence

        except json.JSONDecodeError as e:
            logger.error(f"[{self.agent_id}] Failed to decode LLM JSON response for challenge task. Error: {e}")
            logger.debug(f"[{self.agent_id}] Faulty JSON string for challenge: {response_str}")
            return None
        except ValidationError as e:
            logger.error(f"[{self.agent_id}] Failed to validate AdversaryResult schema. Error: {e}")
            logger.debug(f"[{self.agent_id}] Faulty JSON data for challenge: {response_data}")
            return None
        except Exception as e:
            logger.error(f"[{self.agent_id}] An unexpected error occurred during challenge task. Error: {e}")
            return None

    async def run(self, task: Task, dependencies: List[Any]) -> AsyncGenerator[EvidenceItem, None]:
        """
        异步运行智能体，并行挑战所有 L1 证据。
        
        参数:
            task (Task): 当前任务。
            dependencies (List[Any]): 来自 L1 智能体的 EvidenceItem 列表。

        收益:
            AsyncGenerator[EvidenceItem, None]: 异步生成类型为 "Adversary" 的新 EvidenceItem。
        """
        logger.info(f"[{self.agent_id}] Running AdversaryAgent for task: {task.task_id}")

        # 1. 过滤出有效的 L1 证据
        l1_evidence_items = [item for item in dependencies if isinstance(item, EvidenceItem)]

        if not l1_evidence_items:
            logger.warning(f"[{self.agent_id}] No L1 EvidenceItems found in dependencies for task {task.task_id}. Skipping.")
            return

        logger.info(f"[{self.agent_id}] Found {len(l1_evidence_items)} L1 evidence items to challenge.")

        # 2. 为每个 L1 证据创建并行的挑战任务
        challenge_tasks = [
            self._challenge_evidence(evidence) for evidence in l1_evidence_items
        ]

        # 3. 异步并行执行所有挑战
        results = await asyncio.gather(*challenge_tasks, return_exceptions=True)

        # 4. 处理结果并 Yield
        for result in results:
            if isinstance(result, Exception):
                # 记录在 gather 中发生的异常
                logger.error(f"[{self.agent_id}] An exception occurred in a challenge task: {result}")
            elif isinstance(result, EvidenceItem):
                # 这是一个成功的挑战，yield 新的 "Adversary" 证据
                result.agent_id = self.agent_id
                yield result
            elif result is None:
                # 内部挑战失败（已在 _challenge_evidence 中记录）
                continue
