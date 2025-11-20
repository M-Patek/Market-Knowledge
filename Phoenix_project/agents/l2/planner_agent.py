import json
import logging
from typing import List, Any, AsyncGenerator

from Phoenix_project.agents.l2.base import L2Agent
from Phoenix_project.core.schemas.task_schema import Task
from Phoenix_project.core.pipeline_state import PipelineState

logger = logging.getLogger(__name__)

class PlannerAgent(L2Agent):
    """
    L2 智能体：规划者 (Planner)
    负责将高层目标分解为 L1 智能体的具体子任务。
    (通常在循环开始时运行)
    """

    async def run(self, state: PipelineState, dependencies: List[Any]) -> AsyncGenerator[Task, None]:
        """
        [Refactored Phase 3.2] 适配 PipelineState，生成 Task 列表。
        """
        # Planner 的输入通常是 state 中的 main_task_query
        main_query = state.main_task_query
        target_symbol = main_query.get("symbol", "UNKNOWN")
        description = main_query.get("description", "Analyze market conditions")
        
        logger.info(f"[{self.agent_id}] Planning tasks for: {target_symbol}")

        agent_prompt_name = "l2_planner" # 假设存在此 prompt

        try:
            context_map = {
                "target_symbol": target_symbol,
                "task_description": description,
                "current_date": state.current_time.isoformat()
            }

            # 3. 调用 LLM
            # 注意：Planner 可能会生成多个 Task
            response_str = await self.llm_client.run_llm_task(
                agent_prompt_name=agent_prompt_name,
                context_map=context_map
            )

            if not response_str:
                logger.warning(f"[{self.agent_id}] Empty plan. Using default.")
                yield self._create_default_task(state, target_symbol)
                return

            response_data = json.loads(response_str)
            tasks_data = response_data.get("tasks", [])

            if not tasks_data:
                 yield self._create_default_task(state, target_symbol)
                 return

            for task_dict in tasks_data:
                # 确保 Task ID 唯一性
                task_dict["task_id"] = f"{state.run_id}_{state.step_index}_{task_dict.get('task_id', 'subtask')}"
                task = Task(**task_dict)
                yield task

        except Exception as e:
            logger.error(f"[{self.agent_id}] Planning failed: {e}", exc_info=True)
            yield self._create_default_task(state, target_symbol)

    def _create_default_task(self, state: PipelineState, symbol: str) -> Task:
        """创建默认的基础分析任务。"""
        return Task(
            task_id=f"{state.run_id}_{state.step_index}_default",
            description=f"Perform basic technical analysis for {symbol}",
            task_type="analysis",
            agent_id="technical_analyst", # Default L1
            symbols=[symbol],
            status="pending",
            dependencies={}
        )
