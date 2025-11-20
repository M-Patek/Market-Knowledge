import json
import logging
from typing import Any, Dict, AsyncGenerator

from Phoenix_project.agents.l1.base import L1Agent, logger
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem, EvidenceSource
from Phoenix_project.core.pipeline_state import PipelineState

class CatalystMonitorAgent(L1Agent):
    """
    L1 智能体，专注于催化剂监控。
    识别可能引发价格剧烈波动的事件（如财报、FDA批准、并购）。
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
            role="Catalyst Monitor",
            prompt_template_name="l1_catalyst_monitor",
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

        logger.info(f"[{self.agent_id}] Monitoring catalysts for: {symbol}")

        # 1. 获取事件驱动的新闻
        try:
            query = f"{symbol} earnings merger acquisition FDA approval lawsuit guidance"
            news_items = await self.data_manager.fetch_news_data(query=query)
            
            if not news_items:
                logger.debug(f"[{self.agent_id}] No catalyst news found for {symbol}.")
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
            "catalyst_news": news_summary,
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
            
            catalyst_type = response_json.get("catalyst_type", "None")
            probability = response_json.get("probability", "Low")
            impact = response_json.get("expected_impact", "Unknown")

            content = (
                f"**Catalyst Identified:** {catalyst_type}\n"
                f"**Probability:** {probability} | **Impact:** {impact}"
            )

            evidence = EvidenceItem(
                agent_id=self.agent_id,
                task_id=task_id,
                event_id=event_id,
                symbol=symbol,
                headline=f"Catalyst Alert: {catalyst_type}",
                content=content,
                data_source=EvidenceSource.AGENT_ANALYSIS,
                timestamp=state.current_time.timestamp(),
                tags=["catalyst", catalyst_type.lower(), probability.lower()],
                raw_data=response_json
            )

            yield evidence

        except json.JSONDecodeError:
             yield EvidenceItem(
                agent_id=self.agent_id,
                task_id=task_id,
                event_id=event_id,
                symbol=symbol,
                headline="Catalyst Monitor (Raw)",
                content=f"Raw Output: {llm_response}",
                data_source=EvidenceSource.AGENT_ANALYSIS,
                timestamp=state.current_time.timestamp(),
                tags=["catalyst", "raw"],
                raw_data={"raw_text": llm_response}
            )
