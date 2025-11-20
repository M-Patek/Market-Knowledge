import json
import logging
from typing import Any, Dict, AsyncGenerator

from Phoenix_project.agents.l1.base import L1Agent, logger
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem, EvidenceSource
from Phoenix_project.core.pipeline_state import PipelineState

class InnovationTrackerAgent(L1Agent):
    """
    L1 智能体，专注于创新追踪。
    监控研发动态、专利申请和新产品发布。
    """
    def __init__(
        self,
        agent_id: str,
        llm_client: Any,
        data_manager: Any,
        config: Dict[str, Any] = None
    ):
        super().__init__(
            agent_id=agent_id,
            role="Innovation Tracker",
            prompt_template_name="l1_innovation_tracker",
            llm_client=llm_client,
            data_manager=data_manager
        )
        self.config = config or {}
        logger.info(f"[{self.agent_id}] Initialized.")

    async def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> AsyncGenerator[EvidenceItem, None]:
        """
        [Refactored Phase 3.1] 适配 PipelineState 和 DataManager。
        """
        symbol = dependencies.get("symbol")
        task_id = dependencies.get("task_id", "unknown_task")
        event_id = dependencies.get("event_id", "unknown_event")
        
        if not symbol:
            return

        logger.info(f"[{self.agent_id}] Tracking innovation for: {symbol}")

        # 1. 获取创新相关新闻
        try:
            query = f"{symbol} new product patent R&D technology launch"
            news_items = await self.data_manager.fetch_news_data(query=query)
            
            if not news_items:
                logger.debug(f"[{self.agent_id}] No innovation news for {symbol}.")
                return

            news_summary = "\n".join([
                f"- {item.headline} ({item.timestamp})" for item in news_items[:5]
            ])

        except Exception as e:
            logger.error(f"[{self.agent_id}] Error fetching news: {e}", exc_info=True)
            return

        # 2. 渲染 Prompt
        prompt_data = {
            "symbol": symbol,
            "innovation_news": news_summary,
            "current_date": state.current_time.isoformat()
        }

        try:
            prompt = self.render_prompt(prompt_data)
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error rendering prompt: {e}", exc_info=True)
            return

        # 3. 调用 LLM
        try:
            llm_response = await self.llm_client.generate(prompt)
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error calling LLM: {e}", exc_info=True)
            return

        # 4. 解析结果
        try:
            response_json = json.loads(llm_response)
            
            analysis = response_json.get("analysis", "N/A")
            innovation_score = response_json.get("innovation_score", 5) # 1-10

            content = (
                f"**Innovation Report:** {analysis}\n"
                f"**Innovation Score:** {innovation_score}/10"
            )

            evidence = EvidenceItem(
                agent_id=self.agent_id,
                task_id=task_id,
                event_id=event_id,
                symbol=symbol,
                headline="Innovation & Tech Update",
                content=content,
                data_source=EvidenceSource.AGENT_ANALYSIS,
                timestamp=state.current_time.timestamp(),
                tags=["innovation", "technology", f"score_{innovation_score}"],
                raw_data=response_json
            )

            yield evidence

        except json.JSONDecodeError:
             yield EvidenceItem(
                agent_id=self.agent_id,
                task_id=task_id,
                event_id=event_id,
                symbol=symbol,
                headline="Innovation Analysis (Raw)",
                content=f"Raw Output: {llm_response}",
                data_source=EvidenceSource.AGENT_ANALYSIS,
                timestamp=state.current_time.timestamp(),
                tags=["innovation", "raw"],
                raw_data={"raw_text": llm_response}
            )
