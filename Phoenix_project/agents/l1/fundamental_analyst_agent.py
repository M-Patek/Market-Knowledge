import json
import logging
from typing import Any, Dict, List, Optional

from Phoenix_project.agents.l1.base import L1Agent, logger
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem, EvidenceSource
from Phoenix_project.core.pipeline_state import PipelineState

class FundamentalAnalystAgent(L1Agent):
    """
    L1 智能体，专注于基本面分析。
    分析财务报表、估值指标 (P/E, P/B) 和盈利能力。
    """
    def __init__(
        self,
        agent_id: str,
        llm_client: Any,
        data_manager: Any,
        prompt_manager: Any = None,
        audit_manager: Any = None,
        retriever: Any = None, # [Registry Fix] Must accept retriever
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            llm_client=llm_client,
            data_manager=data_manager,
            role="Fundamental Analyst",
            prompt_template_name="l1_fundamental_analyst",
            prompt_manager=prompt_manager,
            audit_manager=audit_manager,
            retriever=retriever,
            **kwargs
        )
        logger.info(f"[{self.agent_id}] Initialized.")

    async def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> List[EvidenceItem]:
        """
        [Refactored Phase 3.1] 适配 PipelineState 和 DataManager。
        [Fix] Changed return type to List[EvidenceItem] and enforce Time Machine.
        """
        symbol = dependencies.get("symbol")
        task_id = dependencies.get("task_id", "unknown_task")
        event_id = dependencies.get("event_id", "unknown_event")
        
        logger.info(f"[{self.agent_id}] Running task: {task_id} for symbol: {symbol}")

        if not symbol:
            logger.warning(f"[{self.agent_id}] Task {task_id} missing 'symbol'.")
            return []

        # 1. 获取基本面数据 (通过 DataManager)
        data_summary = "{}"
        try:
            # [Time Machine] Pass current_time to prevent future data leakage
            fund_data = await self.data_manager.get_fundamental_data(symbol, as_of_date=state.current_time)
            
            if fund_data:
                # 序列化数据以供 Prompt 使用
                if hasattr(fund_data, 'model_dump_json'):
                    data_summary = fund_data.model_dump_json()
                else:
                    data_summary = json.dumps(fund_data, default=str)
            else:
                 logger.warning(f"[{self.agent_id}] No fundamental data found for {symbol} as of {state.current_time}.")
                 # Continue execution to let LLM say "No Data" or use latent knowledge if allowed,
                 # but usually better to provide explicit "Missing Data" context.
                 data_summary = json.dumps({"status": "missing_data", "note": "No fundamental records found."})

        except Exception as e:
            logger.error(f"[{self.agent_id}] Error fetching fundamental data: {e}", exc_info=True)
            data_summary = json.dumps({"error": str(e)})

        # 2. 渲染 Prompt
        prompt_data = {
            "symbol": symbol,
            "fundamental_data": data_summary,
            "current_date": state.current_time.isoformat()
        }

        try:
            prompt = await self.render_prompt(prompt_data)
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error rendering prompt: {e}", exc_info=True)
            return []

        # 3. 调用 LLM
        try:
            llm_response = await self.llm_client.generate(prompt)
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error calling LLM: {e}", exc_info=True)
            return []

        # 4. 解析结果并生成 Evidence
        try:
            response_json = json.loads(llm_response)
            
            analysis_text = response_json.get("analysis", "No analysis provided.")
            sentiment = response_json.get("sentiment", "Neutral").capitalize()
            metrics = response_json.get("key_metrics", {})

            content = (
                f"**Fundamental Analysis ({sentiment}):** {analysis_text}\n"
                f"**Key Metrics:** {json.dumps(metrics)}"
            )

            evidence = EvidenceItem(
                agent_id=self.agent_id,
                task_id=task_id,
                event_id=event_id,
                symbol=symbol,
                headline="Fundamental Analysis Report",
                content=content,
                data_source=EvidenceSource.AGENT_ANALYSIS,
                timestamp=state.current_time.timestamp(), # [Time Machine]
                tags=["fundamental", "financials", sentiment.lower()],
                raw_data=response_json
            )

            logger.info(f"[{self.agent_id}] Generated evidence for {symbol}")
            return [evidence]

        except json.JSONDecodeError:
            logger.warning(f"[{self.agent_id}] JSON Decode failed. Raw: {llm_response}")
            evidence = EvidenceItem(
                agent_id=self.agent_id,
                task_id=task_id,
                event_id=event_id,
                symbol=symbol,
                headline="Fundamental Analysis (Raw)",
                content=f"Raw Output: {llm_response}",
                data_source=EvidenceSource.AGENT_ANALYSIS,
                timestamp=state.current_time.timestamp(),
                tags=["fundamental", "raw", "error"],
                raw_data={"raw_text": llm_response}
            )
            return [evidence]
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error processing response: {e}", exc_info=True)
            return []
