import json
import logging
from typing import Any, Dict, List
from datetime import timedelta

from Phoenix_project.agents.l1.base import L1Agent, logger
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem, EvidenceSource
from Phoenix_project.core.pipeline_state import PipelineState

class GeopoliticalAnalystAgent(L1Agent):
    """
    L1 智能体，专注于地缘政治风险分析。
    分析战争、贸易冲突、选举、政策制裁等对市场的影响。
    """
    def __init__(
        self,
        agent_id: str,
        llm_client: Any,
        data_manager: Any,
        prompt_manager: Any = None,
        audit_manager: Any = None,
        retriever: Any = None,
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            llm_client=llm_client,
            data_manager=data_manager,
            role="Geopolitical Analyst",
            prompt_template_name="l1_geopolitical_analyst",
            prompt_manager=prompt_manager,
            audit_manager=audit_manager,
            retriever=retriever,
            **kwargs
        )
        logger.info(f"[{self.agent_id}] Initialized.")

    async def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> List[EvidenceItem]:
        task_id = dependencies.get("task_id", "unknown_task")
        
        # 1. 收集数据
        geo_context = ""
        try:
            # A. News Fetch
            end_time = state.current_time
            start_time = end_time - timedelta(days=5)
            
            news_items = await self.data_manager.fetch_news_data(
                start_time=start_time,
                end_time=end_time,
                limit=15
            )
            
            geo_keywords = ["war", "conflict", "election", "sanction", "trade deal", "protest", "policy", "government"]
            relevant_news = [
                n for n in news_items 
                if any(k in (n.get("headline", "") + n.get("content", "")).lower() for k in geo_keywords)
            ]
            
            # B. Retriever
            rag_docs = []
            if self.retriever:
                rag_docs = await self.retriever.retrieve(
                    query="current geopolitical conflicts sanctions trade wars",
                    top_k=3
                )

            # C. Combine
            geo_context = f"Recent Geopolitical Events (Last 5 Days):\n"
            for news in relevant_news[:5]:
                geo_context += f"- {news.get('date', 'N/A')}: {news.get('headline', 'N/A')}\n"
            
            if rag_docs:
                geo_context += "\nRelated Historical Context (RAG):\n"
                for doc in rag_docs:
                    geo_context += f"- {doc.get('content', '')[:200]}...\n"

        except Exception as e:
            logger.error(f"[{self.agent_id}] Data fetch error: {e}", exc_info=True)
            geo_context = "Error fetching geopolitical data."

        # 2. Render Prompt
        prompt_data = {
            "geopolitical_data": geo_context,
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
            
            summary = response_json.get("summary", "No summary")
            impact_level = response_json.get("impact_level", "Low") # High, Medium, Low
            affected_regions = response_json.get("affected_regions", [])

            content = (
                f"**Geopolitical Impact:** {impact_level}\n"
                f"**Summary:** {summary}\n"
                f"**Regions:** {', '.join(affected_regions)}"
            )

            evidence = EvidenceItem(
                agent_id=self.agent_id,
                task_id=task_id,
                headline=f"Geopolitical Report ({impact_level} Impact)",
                content=content,
                data_source=EvidenceSource.AGENT_ANALYSIS,
                timestamp=state.current_time.timestamp(),
                tags=["geopolitics", "risk", impact_level.lower()],
                raw_data=response_json
            )
            
            return [evidence]

        except json.JSONDecodeError:
            return [EvidenceItem(
                agent_id=self.agent_id,
                task_id=task_id,
                headline="Geopolitical Analysis (Raw)",
                content=f"Raw Output: {llm_response}",
                data_source=EvidenceSource.AGENT_ANALYSIS,
                timestamp=state.current_time.timestamp(),
                tags=["geopolitics", "raw"],
                raw_data={"raw_text": llm_response}
            )]
        except Exception as e:
            logger.error(f"[{self.agent_id}] Processing error: {e}")
            return []
