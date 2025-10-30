# pipeline_orchestrator.py
# Main file for coordinating the L1 -> L2 -> L3 pipeline.

import os
import json
import asyncio
import yaml # Task 4.2
from typing import List, Dict, Any
from datetime import datetime
import logging

# Internal dependencies
from ai.validation import EvidenceItem            # Base data model
from ai.retriever import HybridRetriever     # For Task 1.1
from ai.prompt_renderer import render_prompt # For Task 0.3
from api.gemini_pool_manager import query_model # For Task 0.3
from ai.prompt_manager import PromptManager
from ai.tabular_db_client import TabularDBClient # Placeholder
from ai.bayesian_fusion_engine import BayesianFusionEngine, SourceCredibilityStore # For Task 2.1
from data_manager import BaseAltDataClient   # For Task 3.2
from audit_manager import AuditManager       # For Task 4.1

# Configure logging
logger = logging.getLogger("PhoenixProject.Orchestrator")


class PipelineOrchestrator:
    """
    Coordinates the entire analysis pipeline from L1 agent execution
    to L2 fusion and L3 meta-cognition triggering.
    """
    
    def __init__(self):
        """
        Initialize all necessary components.
        """
        logger.info("Initializing PipelineOrchestrator...")
        
        # --- Component Initialization ---
        # Note: In a real app, these would be injected dependencies.
        
        # Task 1.1: Hybrid Retriever
        self.retriever = HybridRetriever(
            # We'd pass real clients here
        )
        # self.gemini_pool = gemini_pool_manager.GeminiPoolManager() # We will call the static query_model
        self.fusion_engine = BayesianFusionEngine() # Task 2.1
        
        # --- Task 3.3: Initialize PromptManager ---
        db_client_placeholder = TabularDBClient() # In a real scenario, this would be properly configured and injected.
        self.prompt_manager = PromptManager(db_client=db_client_placeholder)
        # self.audit_manager = AuditManager() # Task 4.1
        pass

    # --- Phase 1: L1 Agent Execution ---
    
    async def run_l1_agent(self, agent_name: str, context: Dict[str, Any]) -> List[EvidenceItem]:
        """
        Task 0.3: Dynamically executes a single L1 agent.
        
        1. Loads the agent's JSON config.
        2. Renders the prompt with the provided context.
        3. Calls the LLM (Gemini).
        4. Validates the output against the EvidenceItem model.
        
        Returns:
            A list of EvidenceItem objects, or an empty list if errors occur.
        """
        logger.info(f"L1 Agent '{agent_name}' starting...")
        evidence_list = []
        
        try:
            # --- Task 3.3: Refactor to use PromptManager ---
            # 1. Load the corresponding JSON config
            agent_config = self.prompt_manager.get_prompt(agent_name)

            # 2. Use ai/prompt_renderer.py to fill the Prompt template
            # We merge the config template context with the dynamic runtime context
            final_context = {
                **(agent_config.get("template_context", {})),
                **context
            }
            system_prompt, user_prompt = render_prompt(
                agent_config["system_prompt"],
                agent_config["user_prompt_template"],
                final_context
            )
            
            # 3. Call the LLM (e.g., Gemini)
            # We assume query_model can handle the specific model, temp, etc.
            model_config = agent_config.get("model_config", {})
            
            raw_json_output = await query_model(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=model_config.get("model", "gemini-1.5-pro-latest"),
                client_name="gemini", # Or "anthropic", "openai"
                temperature=model_config.get("temperature", 0.5),
                json_output=True # Request structured JSON output
            )
            
            # 4. Validate the output(s)
            if isinstance(raw_json_output, list): # Handle multiple items
                for item in raw_json_output:
                    evidence_list.append(EvidenceItem.model_validate(item))
            else: # Handle single-item outputs just in case
                evidence_list.append(EvidenceItem.model_validate(raw_json_output))
        
        except Exception as e:
            logger.error(f"Error in L1 agent {agent_name} execution (fetch/render/validate): {e}")

        logger.info(f"L1 Agent {agent_name} finished, produced {len(evidence_list)} EvidenceItem(s).")
        return evidence_list

    # --- Phase 1.B: Specialized L1 Task Groups ---
    
    async def initiate_analysis(self, ticker: str) -> List[EvidenceItem]:
        """
        Task 1.2: Runs the core L1 agents (Fundamental, Technical, Catalyst).
        """
        logger.info(f"Initiating core L1 analysis for {ticker}...")
        
        # 1. Task 1.1: Run Hybrid Retriever
        # We fetch data *once* and pass it to all agents.
        retrieved_data = self.retriever.retrieve(
            query=f"Analyze financial outlook for {ticker}",
            ticker=ticker
        )
        
        base_context = {
            "ticker": ticker,
            "current_date": datetime.utcnow().isoformat(),
            "document_snippets": retrieved_data.get("documents", []),
            "retrieved_data": retrieved_data.get("structured_data", {})
        }
        
        # 2. Define and run tasks in parallel
        tasks = [
            self.run_l1_agent("fundamental_analyst", base_context),
            self.run_l1_agent("technical_analyst", base_context),
            self.run_l1_agent("catalyst_monitor", base_context)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 3. Collect results
        evidence_pool = []
        for res in results:
            if isinstance(res, list):
                evidence_pool.extend(res)
            elif isinstance(res, Exception):
                logger.error(f"Error in L1 'initiate_analysis' task: {res}")
                
        logger.info(f"Core L1 analysis complete. {len(evidence_pool)} items generated.")
        return evidence_pool

    async def run_adversarial_check(self, core_evidence: List[EvidenceItem]) -> List[EvidenceItem]:
        """
        Task 1.3: Runs the 'fact_checker_adversary' on all core evidence.
        """
        logger.info(f"Running adversarial check on {len(core_evidence)} items...")
        tasks = []
        
        # Create one adversary task for each piece of evidence
        for item in core_evidence:
            adversary_context = {
                "input_hypothesis": item.hypothesis,
                "input_reasoning_chain": item.reasoning_chain,
                "input_score": item.score,
                "input_source": item.source
            }
            tasks.append(self.run_l1_agent("fact_checker_adversary", adversary_context))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results - we add the original evidence AND the new checks
        evidence_pool = list(core_evidence) 
        for res in results:
            if isinstance(res, list):
                evidence_pool.extend(res) # Add the adversary's output
            elif isinstance(res, Exception):
                logger.error(f"Error in L1 'adversarial_check' task: {res}")
        
        logger.info(f"Adversarial check complete. Total items: {len(evidence_pool)}.")
        return evidence_pool

    async def run_contextual_analysis(self) -> List[EvidenceItem]:
        """
        Task 3.1: Runs the contextual L1 agents (Macro, Geopolitical).
        """
        logger.info("Running L1 contextual analysis cluster...")
        
        # These agents don't need ticker-specific context, just global data.
        context = {
            "current_date": datetime.utcnow().isoformat(),
            "global_news_summary": "..." # This would be fetched
        }
        
        tasks = [
            self.run_l1_agent("macro_strategist", context),
            self.run_l1_agent("geopolitical_analyst", context)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        evidence_pool = []
        for res in results:
            if isinstance(res, list):
                evidence_pool.extend(res)
            elif isinstance(res, Exception):
                logger.error(f"Error in L1 'contextual_analysis' task: {res}")
                
        logger.info(f"L1 Contextual Analysis complete. {len(evidence_pool)} items generated.")
        return evidence_pool

    async def _handle_escalations(self, evidence_pool: List[EvidenceItem]) -> List[EvidenceItem]:
        """
        Task 2.2: Detects and handles high-confidence contradictions by escalating to the Arbitrator.
        """
        # Thresholds defined as per task (example values)
        CONFIDENCE_THRESHOLD = 0.8
        CREDIBILITY_THRESHOLD = 0.8 # Mean credibility (alpha / (alpha + beta))
        
        # We need to detect contradictions *before* fusion
        contradiction_pairs = self.fusion_engine.contradiction_detector.detect(evidence_pool)
        if not contradiction_pairs:
            return evidence_pool # No escalations needed
        
        logger.info(f"Found {len(contradiction_pairs)} contradiction pairs. Checking for escalation...")
        
        # We need a way to track items to remove/add without modifying the list while iterating.
        items_to_remove = set()
        items_to_add = []
        
        for item_a, item_b in contradiction_pairs:
            # Avoid re-processing items that are already part of an escalated pair
            if item_a in items_to_remove or item_b in items_to_remove:
                continue
            
            # 1. Check Provenance Confidence
            if item_a.provenance_confidence > CONFIDENCE_THRESHOLD and \
               item_b.provenance_confidence > CONFIDENCE_THRESHOLD:
                
                # 2. Check Source Credibility
                cred_a_alpha, cred_a_beta = self.fusion_engine.credibility_store.get_credibility_params(item_a.source)
                cred_b_alpha, cred_b_beta = self.fusion_engine.credibility_store.get_credibility_params(item_b.source)
                
                credibility_a = cred_a_alpha / (cred_a_alpha + cred_b_beta)
                credibility_b = cred_b_alpha / (cred_b_alpha + cred_b_beta)
                
                if credibility_a > CREDIBILITY_THRESHOLD and credibility_b > CREDIBILITY_THRESHOLD:
                    # --- TRIGGER ESCALATION ---
                    logger.warning(f"Escalating high-confidence contradiction between {item_a.source} and {item_b.source}.")
                    
                    # Prepare context for arbitrator
                    arbitrator_context = {
                        "evidence_item_A": item_a.model_dump_json(), # Use model_dump_json for serialization
                        "evidence_item_B": item_b.model_dump_json()
                    }
                    
                    # Call arbitrator
                    try:
                        ruling_items = await self.run_l1_agent("arbitrator", arbitrator_context)
                        if ruling_items:
                            ruling = ruling_items[0] # Arbitrator returns one item
                            logger.info(f"Arbitrator ruled. Replacing pair with new EvidenceItem (Score: {ruling.score}).")
                            items_to_add.append(ruling)
                            items_to_remove.add(item_a)
                            items_to_remove.add(item_b)
                        else:
                            logger.error("Arbitrator failed to return a ruling. Keeping original items.")
                    except Exception as e:
                        logger.error(f"Error during arbitrator execution: {e}. Keeping original items.")
        
        # Apply changes to the evidence pool
        if items_to_remove:
            evidence_pool = [item for item in evidence_pool if item not in items_to_remove]
            evidence_pool.extend(items_to_add)
            logger.info(f"Escalation complete. Removed {len(items_to_remove)} items, added {len(items_to_add)} rulings.")
        
        return evidence_pool

    # --- Phase 5: Overall Integration ---
    
    async def run_full_analysis_pipeline(self, ticker: str, workflow_file_path: str = "workflows/full_analysis.yaml"):
        """
        Task 4.2: Main function, refactored to be a declarative workflow interpreter.
        It loads and executes the stages defined in the provided YAML file.
        """
        logger.info(f"--- Starting Declarative Analysis Pipeline for {ticker} using {workflow_file_path} ---")

        # Load and parse the workflow YAML
        try:
            with open(workflow_file_path, 'r') as f:
                workflow = yaml.safe_load(f)
            logger.info(f"Successfully loaded workflow: {workflow.get('workflow_name')}")
        except FileNotFoundError:
            logger.error(f"Workflow file not found: {workflow_file_path}")
            return
        except yaml.YAMLError as e:
            logger.error(f"Error parsing workflow YAML: {e}")
            return

        evidence_pool: List[EvidenceItem] = []
        stages_to_skip = set()
        # Define a core hypothesis for the L2 Engine
        # TODO: This should probably also be in the YAML
        core_hypothesis = f"Assessment of {ticker}'s short-term price movement."
        
        # Base context for all agents
        # In a real system, we'd fetch this once.
        base_context = {
            "ticker": ticker,
            "current_date": datetime.utcnow().isoformat(),
            # We can still run the retriever to provide a base context
            "document_snippets": "...", 
            "retrieved_data": {}
        }

        for stage in workflow.get('stages', []):
            stage_name = stage.get('name')
            if stage_name in stages_to_skip:
                logger.info(f"Skipping conditional stage: {stage_name}")
                continue

            logger.info(f"--- Executing Stage: {stage_name} ---")
            stage_type = stage.get('type', 'agent_execution')

            if stage_type == 'agent_execution':
                tasks = []
                if stage.get('target_agent'): # Handle special case like 'L1_Adversarial_Check'
                    agent_name = stage['target_agent']
                    # We'd need a more robust way to get inputs, but for this task:
                    if stage.get('input_from_stage'): 
                        logger.info(f"Running {agent_name} on {len(evidence_pool)} items from previous stages...")
                        for item in evidence_pool:
                            context = {**base_context, "input_hypothesis": item.hypothesis, "input_reasoning_chain": item.reasoning_chain, "input_score": item.score}
                            tasks.append(self.run_l1_agent(agent_name, context))
                else: # Handle standard agent lists
                    for agent_name in stage.get('agents', []):
                        tasks.append(self.run_l1_agent(agent_name, base_context))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in results:
                    if isinstance(res, list):
                        evidence_pool.extend(res)
                    elif isinstance(res, Exception):
                        logger.error(f"Error in L1 agent task during stage {stage_name}: {res}")

            elif stage_type == 'fusion':
                logger.info(f"Running L2 fusion (Evidence count: {len(evidence_pool)})...")
                fusion_result = self.fusion_engine.fuse(core_hypothesis, evidence_pool)

                # Handle Gating Logic (Task 3.2 logic, now driven by YAML)
                if 'gate_on_uncertainty' in stage:
                    gate_config = stage['gate_on_uncertainty']
                    uncertainty_score = fusion_result.get("cognitive_uncertainty_score", 0.0)
                    threshold = gate_config.get('threshold', 0.6)
                    
                    if uncertainty_score <= threshold:
                        stage_to_skip = gate_config.get('trigger_stage')
                        if stage_to_skip:
                            logger.warning(f"Uncertainty {uncertainty_score} <= {threshold}. Skipping conditional stage: {stage_to_skip}.")
                            stages_to_skip.add(stage_to_skip)
                    else:
                         logger.info(f"Uncertainty {uncertainty_score} > {threshold}. Proceeding with conditional stage.")
                
                # If this is the final fusion, return the result
                if stage_name == 'L2_Final_Fusion': # This is a bit brittle, but matches the YAML
                    logger.info(f"--- Full Analysis Pipeline for {ticker} Complete ---")
                    logger.info(f"Final Fusion Result: {fusion_result}")
                    # Phase 4: L3 Meta-Cognition (Logging)
                    # self.audit_manager.log_decision(decision_id=..., evidence_items=evidence_pool, fusion_result=final_fusion_result, ...)
                    return fusion_result

            elif stage_type == 'arbitration':
                # Handle Arbitration Logic (Task 2.2 logic, now driven by YAML)
                logger.info(f"Checking for high-confidence contradictions before final fusion...")
                evidence_pool = await self._handle_escalations(evidence_pool)
            
            else:
                logger.warning(f"Unknown stage type '{stage_type}' in stage {stage_name}. Skipping.")

        logger.warning(f"Workflow completed without a final fusion stage. Returning raw evidence pool.")
        return evidence_pool
