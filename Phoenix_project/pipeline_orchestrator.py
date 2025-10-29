"""
Phoenix Project - Agent Dispatch Center & Pipeline Orchestrator
This module is the "brain" of the system, responsible for coordinating
L1 Agents, L2 Fusion, and L3 Meta-Cognition as defined in the project tasks.
"""

import json
import asyncio
from typing import List, Dict, Any
from datetime import datetime

# Internal dependencies
from observability import get_logger
from ai.validation import EvidenceItem
from ai.retriever import HybridRetriever     # For Task 1.1
from ai.prompt_renderer import render_prompt # For Task 0.3
from api.gemini_pool_manager import query_model # For Task 0.3
from ai.bayesian_fusion_engine import BayesianFusionEngine, SourceCredibilityStore # For Task 2.1
from data_manager import BaseAltDataClient   # For Task 3.2
from audit_manager import AuditManager       # For Task 4.1

logger = get_logger(__name__)

class PipelineOrchestrator:
    """Coordinates the full analysis pipeline from L1 Agent dispatch to L3 learning."""

    def __init__(self):
        logger.info("Initializing PipelineOrchestrator...")
        # Initialize components
        # TODO: The DB clients (Vector, Temporal, Tabular) need to be properly
        # initialized and passed here. Using None as a placeholder.
        self.retriever = HybridRetriever(
            vector_db_client=None, temporal_db_client=None, tabular_db_client=None, rerank_config={}
        )
        # self.gemini_pool = gemini_pool_manager.GeminiPoolManager() # We will call the static query_model
        self.fusion_engine = BayesianFusionEngine() # Task 2.1
        # self.audit_manager = AuditManager() # Task 4.1
        pass

    # --- Phase 0: Foundation ---

    async def run_l1_agent(self, agent_name: str, context: Dict[str, Any]) -> List[EvidenceItem]:
        """
        Task 0.3: Loads config, renders prompt, calls API, and validates output.
        """
        logger.info(f"Running L1 Agent: {agent_name} with context keys: {context.keys()}...")
        evidence_list = []
        
        try:
            # 1. Load the corresponding JSON config
            config_path = f"Phoenix_project/prompts/{agent_name}.json"
            with open(config_path, 'r') as f:
                agent_config = json.load(f)

            # 2. Use ai/prompt_renderer.py to fill the Prompt template
            # We merge the config template context with the dynamic runtime context
            full_context = {**agent_config, **context}
            prompt = render_prompt(agent_config['prompt_template'], full_context)

            # 3. Call the API via api/gemini_pool_manager.py
            # We can use the agent_name to select the model tier (Task 5.2)
            raw_llm_output = await query_model(prompt, agent_name=agent_name)

            # 4. Parse and validate the LLM text output
            raw_json = json.loads(raw_llm_output)
            
            # The output must be a list of items, per our JSON template's output_instructions
            if isinstance(raw_json, list):
                for item_dict in raw_json:
                    evidence_list.append(EvidenceItem.model_validate(item_dict))
            else: # Handle single-item outputs just in case
                evidence_list.append(EvidenceItem.model_validate(raw_json))

        except FileNotFoundError:
            logger.error(f"Agent config file not found: {config_path}. Master must create this file.")
        except Exception as e:
            logger.error(f"Error in L1 agent {agent_name} execution: {e}")

        logger.info(f"L1 Agent {agent_name} finished, produced {len(evidence_list)} EvidenceItem(s).")
        return evidence_list

    # --- Phase 1: L1 Core Alpha Cluster ---

    async def initiate_analysis(self, ticker: str) -> List[EvidenceItem]:
        """
        Task 1.1 & 1.3: Run Fundamental, Technical, and Catalyst agents.
        """
        logger.info(f"Initiating L1 Core Alpha analysis for {ticker}...")
        all_evidence = []

        # 1. Call ai/retriever.py for foundational data
        try:
            # Formulate a generic query for foundational data
            retrieval_query = f"Retrieve foundational data, recent news, and technical data for {ticker}."
            retrieved_data = await self.retriever.retrieve(query=retrieval_query, ticker=ticker)
            # We'd format this data to be passed as context
            document_snippets = "\n".join([doc.get('content', '') for doc in retrieved_data.get('evidence_documents', [])])
        except Exception as e:
            logger.error(f"Error during data retrieval for {ticker}: {e}")
            retrieved_data = {}
            document_snippets = "Data retrieval failed."

        # 2. Concurrently call L1 agents (Task 1.1 & 1.3)
        context = {
            "ticker": ticker,
            "document_snippets": document_snippets,
            "retrieved_data": retrieved_data # Pass full object for agents that need KG, etc.
        }
        
        agent_tasks = [
            self.run_l1_agent("fundamental_analyst", context), # Task 1.1
            self.run_l1_agent("technical_analyst", context),   # Task 1.3
            self.run_l1_agent("catalyst_monitor", context)     # Task 1.3
        ]

        results = await asyncio.gather(*agent_tasks, return_exceptions=True)

        # 3. Collect and return all EvidenceItems
        for res in results:
            if isinstance(res, list):
                all_evidence.extend(res)
            elif isinstance(res, Exception):
                logger.error(f"Error in L1 agent task: {res}")

        logger.info(f"L1 Core Alpha analysis for {ticker} complete. {len(all_evidence)} items generated.")
        return all_evidence

    async def run_adversarial_check(self, evidence_items: List[EvidenceItem]) -> List[EvidenceItem]:
        """
        Task 1.2: Run Fact-Checker Adversary on a list of EvidenceItems.
        """
        logger.info(f"Running L1 Adversarial Check on {len(evidence_items)} items...")
        adversarial_tasks = []

        # 1. For each EvidenceItem:
        for item in evidence_items:
            # 2. Execute the fact_checker_adversary Agent
            # We pass the original item's reasoning as context
            # This matches the template we defined in `fact_checker_adversary.json`
            context = {
                "input_hypothesis": item.hypothesis,
                "input_reasoning_chain": item.reasoning_chain,
                "input_score": item.score,
                # Pass relevant snippets if they were stored in metadata
                "document_snippets": item.metadata.get("document_snippets", "") 
            }
            adversarial_tasks.append(self.run_l1_agent("fact_checker_adversary", context))

        results = await asyncio.gather(*adversarial_tasks, return_exceptions=True)

        # 3. Return a list of all original + adversarial evidence
        all_evidence = list(evidence_items) # Start with the original list
        for res in results:
            if isinstance(res, list):
                all_evidence.extend(res) # Add the new adversarial items
            elif isinstance(res, Exception):
                logger.error(f"Error in L1 adversarial task: {res}")

        logger.info(f"L1 Adversarial Check complete. Total items (original + adversarial): {len(all_evidence)}.")
        return all_evidence

    # --- Phase 3: L1 Extended Cluster ---

    async def run_contextual_analysis(self) -> List[EvidenceItem]:
        """
        Task 3.1: Run Macro and Geopolitical agents.
        """
        logger.info("Running L1 Contextual Analysis (Macro, Geopolitical)...")
        all_evidence = []

        # These agents may not need ticker-specific context, but a global one.
        context = {
            "current_date": datetime.utcnow().isoformat(),
            "global_news_snippets": "..." # In a real implementation, we'd retrieve global news
        }

        # 1. Concurrently execute macro_strategist and geopolitical_analyst
        agent_tasks = [
            self.run_l1_agent("macro_strategist", context),
            self.run_l1_agent("geopolitical_analyst", context)
        ]

        results = await asyncio.gather(*agent_tasks, return_exceptions=True)

        # 2. Return their EvidenceItems
        for res in results:
            if isinstance(res, list):
                all_evidence.extend(res)
            elif isinstance(res, Exception):
                logger.error(f"Error in L1 contextual task: {res}")

        logger.info(f"L1 Contextual Analysis complete. {len(all_evidence)} items generated.")
        return all_evidence

    # --- Phase 5: Overall Integration ---

    async def run_full_analysis_pipeline(self, ticker: str):
        """
        Task 5.1: Main function connecting all phases.
        """
        logger.info(f"--- Starting Full Analysis Pipeline for {ticker} ---")

        # Define a core hypothesis for the L2 Engine
        core_hypothesis = f"Assessment of {ticker}'s short-term price movement."

        # Phase 1: Core Cluster + Adversary
        core_evidence = await self.initiate_analysis(ticker)
        evidence_pool = await self.run_adversarial_check(core_evidence)

        # Phase 3: Contextual Cluster
        contextual_evidence = await self.run_contextual_analysis()
        evidence_pool.extend(contextual_evidence)

        # --- Task 3.2: Gating Condition Logic ---
        # Run a preliminary fusion to check uncertainty
        logger.info(f"Running preliminary L2 fusion (Evidence count: {len(evidence_pool)})...")
        preliminary_fusion = self.fusion_engine.fuse(core_hypothesis, evidence_pool)
        uncertainty_score = preliminary_fusion.get("cognitive_uncertainty_score", 0.0)
        logger.info(f"Preliminary Cognitive Uncertainty: {uncertainty_score}")

        # Gating Condition: Only run high-cost agents if uncertainty is high
        if uncertainty_score > 0.6: # Threshold from Task 3.2
            logger.warning(f"High uncertainty detected ({uncertainty_score} > 0.6). Triggering high-cost Alternative Data Cluster.")
            
            # We'd fetch real data here, but for now we just call the agents
            # In a real system: self.data_manager.load_and_process_alternative_data(...)
            
            alt_data_context = {"ticker": ticker} # Context for alt data agents
            alt_data_tasks = [
                self.run_l1_agent("supply_chain_intel", alt_data_context),
                self.run_l1_agent("innovation_tracker", alt_data_context)
            ]
            results = await asyncio.gather(*alt_data_tasks, return_exceptions=True)
            
            for res in results:
                if isinstance(res, list):
                    evidence_pool.extend(res)
                elif isinstance(res, Exception):
                    logger.error(f"Error in L1 Alt Data task: {res}")
        else:
            logger.info(f"Uncertainty {uncertainty_score} is below threshold. Skipping high-cost agents.")

        # Phase 2: L2 Fusion
        logger.info(f"Running final L2 fusion (Evidence count: {len(evidence_pool)})...")
        final_fusion_result = self.fusion_engine.fuse(core_hypothesis, evidence_pool)

        # Phase 4: L3 Meta-Cognition (Logging)
        # self.audit_manager.log_decision(decision_id=..., evidence_items=evidence_pool, fusion_result=final_fusion_result, ...)

        logger.info(f"--- Full Analysis Pipeline for {ticker} Complete ---")
        logger.info(f"Final Fusion Result: {final_fusion_result}")
        
        return final_fusion_result

if __name__ == '__main__':
    # Example of how to run the orchestrator
    async def main():
        orchestrator = PipelineOrchestrator()
        await orchestrator.run_full_analysis_pipeline("NVDA")

    asyncio.run(main())
