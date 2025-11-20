import json
import logging
from typing import Any, Dict, AsyncGenerator

from Phoenix_project.agents.l1.base import L1Agent, logger
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem, EvidenceSource
from Phoenix_project.core.pipeline_state import PipelineState

class GeopoliticalAnalystAgent(L1Agent):
    """
    L1 智能体，专注于地缘政治分析。
    评估政治事件、制裁和国际关系对市场的风险。
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
            role="Geopolitical Analyst",
            prompt_template_name="l1_geopolitical_analyst",
            llm_client=llm_client,
            data_manager=data_manager
        )
        self.config = config or {}
        logger.info(f"[{self.agent_id}] Initialized.")

    async def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> AsyncGenerator[EvidenceItem, None]:
        """
        [Refactored Phase 3.1] 适配 PipelineState 和 DataManager。
        """
        symbol = dependencies.get("symbol", "GLOBAL")
        task_id = dependencies.get("task_id", "unknown_task")
        event_id = dependencies.get("event_id", "unknown_event")
        
        logger.info(f"[{self.agent_id}] Running geopolitical analysis. Context: {symbol}")

        # 1. 获取地缘政治新闻
        try:
            # 构造查询
            query = "geopolitics sanctions trade war election conflict"
            if symbol != "GLOBAL":
                query = f"{symbol} {query}"
            
            news_items = await self.data_manager.fetch_news_data(query=query)
            
            if not news_items:
                logger.debug(f"[{self.agent_id}] No geopolitical news found.")
                return

            news_summary = "\n".join([
                f"- {item.headline}" for item in news_items[:5]
            ])

        except Exception as e:
            logger.error(f"[{self.agent_id}] Error fetching news: {e}", exc_info=True)
            return

        # 2. 渲染 Prompt
        prompt_data = {
            "symbol": symbol,
            "geopolitical_news": news_summary,
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
            impact_severity = response_json.get("impact_severity", "Low")

            content = (
                f"**Geopolitical Analysis:** {analysis}\n"
                f"**Impact Severity:** {impact_severity}"
            )

            evidence = EvidenceItem(
                agent_id=self.agent_id,
                task_id=task_id,
                event_id=event_id,
                symbol=symbol,
                headline="Geopolitical Risk Assessment",
                content=content,
                data_source=EvidenceSource.AGENT_ANALYSIS,
                timestamp=state.current_time.timestamp(),
                tags=["geopolitics", "risk", impact_severity.lower()],
                raw_data=response_json
            )

            yield evidence

        except json.JSONDecodeError:
             yield EvidenceItem(
                agent_id=self.agent_id,
                task_id=task_id,
                event_id=event_id,
                symbol=symbol,
                headline="Geopolitical Analysis (Raw)",
                content=f"Raw Output: {llm_response}",
                data_source=EvidenceSource.AGENT_ANALYSIS,
                timestamp=state.current_time.timestamp(),
                tags=["geopolitics", "raw"],
                raw_data={"raw_text": llm_response}
            )
