"""
Phoenix_project/agents/l1/base.py
[Refactor] Unified L1 Agent Interface Contract.
[Fix] Renamed BaseL1Agent to L1Agent to match subclass usage.
[Fix] safe_run now normalizes List, Generator, and Single Item into List[EvidenceItem].
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import logging
import traceback
import inspect
from pydantic import ValidationError

from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem, EvidenceType

logger = logging.getLogger(__name__)

class L1Agent(ABC):
    """
    Abstract Base Class for all L1 agents.
    L1 agents are specialized LLM-driven experts that generate EvidenceItems.
    """
    
    def __init__(self, agent_id: str, llm_client: Any, data_manager: Any = None, **kwargs):
        """
        Initializes the L1 agent.
        
        Args:
            agent_id (str): The unique identifier for the agent.
            llm_client (Any): An instance of an LLM client.
            data_manager (Any): Data Manager instance.
            **kwargs: Additional args like 'role', 'prompt_template_name', 'prompt_manager', etc.
        """
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.data_manager = data_manager
        self.role = kwargs.get('role', 'Agent')
        self.prompt_template_name = kwargs.get('prompt_template_name')
        
        # [Optional] Inject prompt handling dependencies if provided
        self.prompt_manager = kwargs.get('prompt_manager')
        self.prompt_renderer = kwargs.get('prompt_renderer')

    @abstractmethod
    async def run(
        self, state: PipelineState, dependencies: Dict[str, Any]
    ) -> Union[EvidenceItem, List[EvidenceItem], AsyncGenerator[EvidenceItem, None]]:
        """
        The main execution method for the agent.
        Can return a single Item, a List of Items, or an AsyncGenerator.
        """
        pass

    async def render_prompt(self, variables: Dict[str, Any]) -> str:
        """
        Helper to render prompt using injected prompt_renderer or prompt_manager.
        """
        if self.prompt_renderer and self.prompt_template_name:
            return await self.prompt_renderer.render(self.prompt_template_name, variables)
        elif self.prompt_manager and self.prompt_template_name:
            # Fallback if only manager is present
            template = self.prompt_manager.get_prompt(self.prompt_template_name)
            return template.format(**variables) # Simple format
        else:
            # Fallback for when no prompt system is injected (testing)
            logger.warning(f"[{self.agent_id}] No PromptRenderer found. Using raw variable dump.")
            return str(variables)

    async def safe_run(self, state: PipelineState, dependencies: Dict[str, Any]) -> List[EvidenceItem]:
        """
        [Unified Interface] A defensive wrapper that executes the agent and 
        normalizes the output into a List[EvidenceItem].
        """
        results: List[EvidenceItem] = []
        try:
            logger.info(f"Agent {self.agent_id} ({self.role}) starting execution.")
            
            # Execute run
            raw_result = await self.run(state, dependencies)
            
            # [Normalization Strategy]
            if inspect.isasyncgen(raw_result):
                # Case 1: Async Generator (Streaming)
                async for item in raw_result:
                    if self._validate_item(item):
                        results.append(item)
                        
            elif isinstance(raw_result, list):
                # Case 2: Batch List
                for item in raw_result:
                    if self._validate_item(item):
                        results.append(item)
                        
            elif isinstance(raw_result, EvidenceItem):
                # Case 3: Single Item
                results.append(raw_result)
                
            elif raw_result is None:
                logger.info(f"Agent {self.agent_id} returned None.")
                
            else:
                logger.warning(f"Agent {self.agent_id} returned unknown type: {type(raw_result)}")

            return results

        except (ValueError, KeyError, TypeError, AttributeError, ValidationError) as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Agent {self.agent_id} CRASHED: {error_msg}\n{stack_trace}")
            
            # Return valid ERROR EvidenceItem in a list
            error_item = EvidenceItem(
                agent_id=self.agent_id,
                headline=f"Agent Crash: {self.role}",
                content=f"CRITICAL FAILURE: Agent crashed. Error: {error_msg}",
                evidence_type=EvidenceType.GENERIC, 
                confidence=0.0,
                data_horizon="Immediate",
                symbols=[],
                metadata={
                    "status": "CRASHED",
                    "error_type": type(e).__name__,
                    "stack_trace": stack_trace
                }
            )
            return [error_item]
        
        except Exception as e:
            # Re-raise unexpected system errors (let Retry mechanism handle)
            logger.error(f"Agent {self.agent_id} System Error: {e}", exc_info=True)
            raise

    def _validate_item(self, item: Any) -> bool:
        """Helper to validate if an item is a valid EvidenceItem."""
        if isinstance(item, EvidenceItem):
            return True
        logger.warning(f"[{self.agent_id}] Ignored invalid output item type: {type(item)}")
        return False

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id='{self.agent_id}')>"

# Alias for backward compatibility if needed, though most imports updated
BaseL1Agent = L1Agent
