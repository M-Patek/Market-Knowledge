import json
import logging
from typing import Any, Dict, AsyncGenerator

from Phoenix_project.agents.l1.base import L1Agent, logger
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem, EvidenceSource
from Phoenix_project.core.pipeline_state import PipelineState

class MacroStrategistAgent(L1Agent):
    """
    L1 智能体，专注于宏观经济策略。
    分析利率、通胀、GDP 和央行政策对市场的影响。
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
            role="Macro Strategist",
            prompt_template_name="l1_macro_strategist",
            llm_client=llm_client,
            data_manager=data_manager
        )
        self.config = config or {}
        logger.info(f"[{self.agent_id}] Initialized.")

    async def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> AsyncGenerator[EvidenceItem, None]:
        """
        [Refactored Phase 3.1] 适配 PipelineState 和 DataManager。
        """
        symbol = dependencies.get("symbol", "GLOBAL") # Macro might not be symbol-specific
        task_id = dependencies.get("task_id", "unknown_task")
        event_id = dependencies.get("event_id", "unknown_event")
        
        logger.info(f"[{self.agent_id}] Running macro analysis. Context Symbol: {symbol}")

        # 1. 获取宏观新闻/数据
        try:
            # 定义宏观关键词
            macro_keywords = "interest rates, inflation, central bank, GDP, unemployment, market sentiment"
            query = f"{macro_keywords} {symbol if symbol != 'GLOBAL' else ''}"
            
            # 使用 DataManager 获取新闻
            news_items = await self.data_manager.fetch_news_data(query=query)
            
            if not news_items:
                logger.warning(f"[{self.agent_id}] No macro news found.")
                return # Macro agent requires news context

            # 摘要前 5 条新闻
            news_summary = "\n".join([
                f"- [{item.timestamp}] {item.headline}: {item.summary[:200]}..." 
                for item in news_items[:5]
            ])

        except Exception as e:
            logger.error(f"[{self.agent_id}] Error fetching macro data: {e}", exc_info=True)
            return

        # 2. 渲染 Prompt
        prompt_data = {
            "symbol": symbol,
            "macro_news_summary": news_summary,
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
            
            analysis_text = response_json.get("analysis", "No analysis provided.")
            risk_level = response_json.get("risk_level", "Medium")
            
            content = (
                f"**Macro Strategy:** {analysis_text}\n"
                f"**Risk Level:** {risk_level}"
            )

            evidence = EvidenceItem(
                agent_id=self.agent_id,
                task_id=task_id,
                event_id=event_id,
                symbol=symbol,
                headline="Macro Strategy Update",
                content=content,
                data_source=EvidenceSource.AGENT_ANALYSIS,
                timestamp=state.current_time.timestamp(),
                tags=["macro", "strategy", risk_level.lower()],
                raw_data=response_json
            )

            logger.info(f"[{self.agent_id}] Generated macro evidence.")
            yield evidence

        except json.JSONDecodeError:
             # Fallback
            yield EvidenceItem(
                agent_id=self.agent_id,
                task_id=task_id,
                event_id=event_id,
                symbol=symbol,
                headline="Macro Analysis (Raw)",
                content=f"Raw Output: {llm_response}",
                data_source=EvidenceSource.AGENT_ANALYSIS,
                timestamp=state.current_time.timestamp(),
                tags=["macro", "raw", "error"],
                raw_data={"raw_text": llm_response}
            )
