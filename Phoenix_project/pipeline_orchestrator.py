"""
Pipeline Orchestrator (Layer 9)

Reads the defined workflow from workflow_config.yaml and executes each stage,
passing the PipelineState context.
"""

import yaml
from registry import registry
from observability import get_logger
from pipeline_state import PipelineState
from knowledge_graph_service import KnowledgeGraphService

# Configure logger for this module (Layer 12)
logger = get_logger(__name__)

class PipelineOrchestrator:
    
    def __init__(self, config_path="workflow_config.yaml"):
        logger.info(f"Loading pipeline workflow from: {config_path}")
        try:
            with open(config_path, 'r') as f:
                workflow = yaml.safe_load(f)
                self.stages = workflow.get('stages', [])
            logger.info(f"Pipeline stages loaded: {self.stages}")
        except FileNotFoundError:
            logger.error(f"Workflow config file not found at: {config_path}")
            self.stages = []
        
        # Resolve the KG service once (Layer 11)
        self.kg_service: KnowledgeGraphService = registry.resolve("knowledge_graph_service")

    def run_pipeline(self, initial_data: dict) -> PipelineState:
        """
        Runs the full cognitive pipeline for a given ticker or event.
        """
        logger.info(f"Starting pipeline run for: {initial_data.get('ticker')}")
        context = PipelineState(ticker=initial_data.get('ticker', 'UNKNOWN'))

        for stage_name in self.stages:
            
            # --- Layer 10: KG Enhancement ---
            if stage_name == 'L2_fusion':
                # Before L2, retrieve context based on L1 entities (Task 2)
                # We assume a previous stage (L1) has populated 'L1_agents_entities'
                l1_entities = context.get_data("L1_agents_entities")
                if l1_entities:
                    logger.info("Retrieving KG context for L2_fusion...")
                    kg_context = self.kg_service.retrieve_context(l1_entities)
                    context.add_data("kg_context_for_l2", kg_context)
            # --- End Layer 10 ---

            context = self._execute_stage(stage_name, context)

            # --- Layer 10: KG Extraction ---
            if stage_name == 'L1_agents':
                # After L1, extract and store entities (Task 1)
                l1_outputs = context.get_data(stage_name) # Assuming L1 populates this
                if l1_outputs:
                    logger.info("Extracting and storing L1 entities to KG...")
                    self.kg_service.extract_and_store(l1_outputs)
                    # We'll mock that the L1 stage also produced entities for L2
                    context.add_data("L1_agents_entities", ["NVDA", "AI"]) 
            # --- End Layer 10 ---

        logger.info(f"Pipeline run finished for: {context.ticker}")
        return context

    def _execute_stage(self, stage_name: str, context: PipelineState) -> PipelineState:
        """
        Executes a single pipeline stage.
        Each stage reads and returns the updated PipelineState.
        """
        # Placeholder for stage execution logic (L1, L2, L3...)
        
        # In a real implementation:
        # stage_service = registry.resolve(f"{stage_name}_service")
        # context = stage_service.process(context)
        
        # Mock: L1_agents stage adds its output to the state
        if stage_name == 'L1_agents':
            context.add_data(stage_name, [{"source": "mock", "insight": "NVDA is good"}])

        logger.info(f"Executing stage: {stage_name}")
        logger.debug(f"Stage {stage_name} complete.")
        return context
