import json
import logging
from typing import Any, Dict, List, Optional
from datetime import timedelta

from Phoenix_project.agents.l1.base import L1Agent, logger
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem, EvidenceSource
from Phoenix_project.core.pipeline_state import PipelineState

class MacroStrategistAgent(L1Agent):
    """
    L1 智能体，专注于宏观经济策略。
    分析利率、通胀、GDP、央行政策等宏观指标。
    """
    def __init__(
        self,
        agent_id: str,
        llm_client: Any,
        data_manager: Any,
        prompt_manager: Any = None,
        audit_manager: Any = None,
        retriever: Any = None, # [Registry Fix]
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            llm_client=llm_client,
            data_manager=data_manager,
            role="Macro Strategist",
            prompt_template_name="l1_macro_strategist",
            prompt_manager=prompt_manager,
            audit_manager=audit_manager,
            retriever=retriever,
            **kwargs
        )
        logger.info(f"[{self.agent_id}] Initialized.")

    async def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> List[EvidenceItem]:
        """
        Fetch macro news/data, analyze global context, return evidence.
        """
        task_id = dependencies.get("task_id", "unknown_task")
        
        # 1. 收集宏观数据 (News & Retriever)
        macro_context = ""
        try:
            # A. Fetch recent macro news via DataManager
            end_time = state.current_time
            start_time = end_time - timedelta(days=7) # Look back 7 days for macro news
            
            # Use fetch_news_data from DataManager
            news_items = await self.data_manager.fetch_news_data(
                start_time=start_time,
                end_time=end_time,
                limit=10
            )
            
            # Filter for macro keywords if fetch_news_data is generic
            macro_keywords = ["inflation", "rate", "fed", "central bank", "gdp", "cpi", "unemployment", "treasury"]
            relevant_news = [
                n for n in news_items 
                if any(k in (n.get("headline", "") + n.get("content", "")).lower() for k in macro_keywords)
            ]
            
            # B. Use Retriever (RAG) if available for deeper context
            rag_docs = []
            if self.retriever:
                # Retrieve based on generic macro query or current market regime
                rag_docs = await self.retriever.retrieve(
                    query="current global macroeconomic conditions interest rates inflation",
                    top_k=3,
                    # [Time Machine] Ideally retriever should filter by date, assuming docs have timestamps
                    # For now, we assume VectorStore handles insertion order or metadata filtering if implemented.
                )
            
            # C. Combine Context
            macro_context = f"Recent Macro News (Last 7 Days):\n"
            for news in relevant_news[:5]:
                macro_context += f"- {news.get('date', 'N/A')}: {news.get('headline', 'N/A')}\n"
            
            if rag_docs:
                macro_context += "\nRetrieved Context (RAG):\n"
                for doc in rag_docs:
                    macro_context += f"- {doc.get('content', '')[:200]}...\n"

        except Exception as e:
            logger.error(f"[{self.agent_id}] Data fetch error: {e}", exc_info=True)
            macro_context = "Error fetching macro data."

        # 2. Render Prompt
        prompt_data = {
            "macro_data": macro_context,
            "current_date": state.current_time.isoformat()
        }
        
        try:
            prompt = await self.render_prompt(prompt_data)
        except Exception as e:
            logger.error(f"[{self.agent_id}] Prompt error: {e}")
            return []

        # 3. Call LLM
        try:
            llm_response = await self.llm_client.generate(prompt)
        except Exception as e:
            logger.error(f"[{self.agent_id}] LLM error: {e}")
            return []

        # 4. Parse Response
        try:
            response_json = json.loads(llm_response)
            
            analysis = response_json.get("analysis", "No analysis")
            regime = response_json.get("current_regime", "Unknown")
            risks = response_json.get("risks", [])

            content = (
                f"**Macro Regime:** {regime}\n"
                f"**Analysis:** {analysis}\n"
                f"**Key Risks:** {', '.join(risks)}"
            )

            evidence = EvidenceItem(
                agent_id=self.agent_id,
                task_id=task_id,
                headline=f"Macro Strategy: {regime}",
                content=content,
                data_source=EvidenceSource.AGENT_ANALYSIS,
                timestamp=state.current_time.timestamp(),
                tags=["macro", "economy", regime.lower()],
                raw_data=response_json
            )
            
            return [evidence]

        except json.JSONDecodeError:
            # Fallback
            return [EvidenceItem(
                agent_id=self.agent_id,
                task_id=task_id,
                headline="Macro Analysis (Raw)",
                content=f"Raw Output: {llm_response}",
                data_source=EvidenceSource.AGENT_ANALYSIS,
                timestamp=state.current_time.timestamp(),
                tags=["macro", "raw"],
                raw_data={"raw_text": llm_response}
            )]
        except Exception as e:
            logger.error(f"[{self.agent_id}] Processing error: {e}")
            return []
