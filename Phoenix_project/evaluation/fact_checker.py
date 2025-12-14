import asyncio
import copy 
from typing import List, Dict, Any, Optional

from ..ai.ensemble_client import EnsembleClient
from ..ai.prompt_manager import PromptManager
from ..ai.prompt_renderer import PromptRenderer
from ..core.schemas.evidence_schema import Evidence, FactCheckResult
from ..monitor.logging import get_logger

logger = get_logger(__name__)

class FactChecker:
    """
    事实核查器 (FactChecker) 负责验证由 L1 智能体
    生成的分析中的具体、可核查的声明。
    
    它现在使用 L2 Critic 提示来执行此操作，以确保与核心智能体一致。
    """

    def __init__(self, 
                 llm_client: EnsembleClient, 
                 prompt_manager: PromptManager, 
                 prompt_renderer: PromptRenderer,
                 config: Optional[Dict[str, Any]] = None):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.prompt_renderer = prompt_renderer
        self.config = config or {}
        
        # [Task 4.1] Configuration Decoupling
        self.model_name = self.config.get("evaluation", {}).get("fact_checker", {}).get("model_name", "gemini-1.5-flash-latest")
        
        self.search_tool = {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Searches for relevant documents, news, and reports based on a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        
        self.prompt = copy.deepcopy(self.prompt_manager.get_prompt("l2_critic"))
        
        if not self.prompt:
             logger.error("Failed to load 'l2_critic' prompt for FactChecker. Check prompts directory.", exc_info=True)
             raise FileNotFoundError("FactChecker prompt 'l2_critic' not found.")
        
        # [Task 6.1] Anti-Hallucination
        search_instruction = (
            "\n\n---\n"
            "MANDATORY INSTRUCTION: You MUST use the 'search_documents' tool to verify the claims. "
            "If no evidence is found after searching, you MUST report the claim as 'Unverified' or 'No Evidence Found'. "
            "Do NOT fabricate sources or evidence to satisfy the request. "
            "Do not rely on internal knowledge."
            " You are allowed to report that no evidence was found. Do not feel pressured to find supporting facts if they do not exist."
            "\n---\n"
        )
        
        # [Task 3.3] Time-Travel Constraint Injection
        time_instruction = (
            "\n\n---\n"
            "TEMPORAL CONSTRAINT: Current Simulation Time is {current_time}. "
            "You are strictly PROHIBITED from using or searching for any information, news, or data dated AFTER {current_time}. "
            "Treat the current time as the absolute present. Future events are unknown."
            "\n---\n"
        )
        
        full_instruction = search_instruction + time_instruction

        if "system" in self.prompt:
            if isinstance(self.prompt["system"], list):
                self.prompt["system"].append(full_instruction)
            else:
                self.prompt["system"] = self.prompt["system"] + full_instruction
        else:
            self.prompt["system"] = full_instruction.strip()
            
        logger.info(f"FactChecker initialized with 'l2_critic' prompt and model '{self.model_name}'.")

    async def check_facts(self, claims: List[str], symbol: str = "Unknown") -> List[FactCheckResult]:
        """
        核查一系列声明。
        [Task 3.3] Injects time context.
        """
        if not claims:
            return []
            
        logger.info(f"Fact-checking {len(claims)} claims using 'l2_critic' prompt...")
        
        claims_str = "\n".join([f"- {claim}" for claim in claims])
        
        try:
            # [Task 3.3] Let PromptRenderer handle time injection (via auto-injection or explicit pass if needed)
            # However, prompt template uses {current_time} inside the string we just appended.
            # We rely on PromptRenderer's auto-injection or we must ensure 'current_time' is in prompt_context.
            # PromptRenderer auto-injects if missing, assuming it has TimeProvider.
            
            prompt_context = {
                "evidence_items": claims_str, 
                "symbol": symbol
            }
            
            # Note: PromptRenderer.render will inject 'current_time' if initialized with TimeProvider.
            # If not, it defaults to system time. 
            prompt = self.prompt_renderer.render(
                self.prompt, context=prompt_context # Keyword arg for clarity
            )
            
            # [Task 3.3] Backtest Logic: Disable/Limit Search Tool
            # Check if we are in simulation mode (heuristic: look for time provider on renderer)
            tools_to_use = [self.search_tool]
            if self.prompt_renderer.time_provider and self.prompt_renderer.time_provider.is_simulation_mode():
                # In strict backtest, we might want to disable live search entirely OR ensure the search tool 
                # implementation (in EnsembleClient/GeminiPool) respects the time filter.
                # For now, we rely on the Prompt Constraint added above.
                pass

            response_json = await self.llm_client.run_chain_structured(
                prompt,
                tools=tools_to_use,
                model_name=self.model_name
            )
            
            if isinstance(response_json, dict):
                response_json = [response_json]

            if not isinstance(response_json, list):
                logger.error(f"Fact-checker LLM returned non-list response: {response_json}")
                raise ValueError("Fact-checker response is not a list of results.")

            results = []
            for item in response_json:
                if isinstance(item, dict):
                    evidence = Evidence(
                        source=item.get('source_url', 'Unknown'),
                        snippet=item.get('evidence_snippet', 'N/A')
                    )
                    results.append(FactCheckResult(
                        claim=item.get('claim', 'N/A'),
                        verified=bool(item.get('verified', False)),
                        evidence=evidence,
                        confidence=float(item.get('confidence', 0.0))
                    ))
                else:
                    logger.warning(f"Skipping invalid item in fact-checker response: {item}")
                    
            logger.info(f"Fact-checking complete. Found {len(results)} results.")
            return results

        except Exception as e:
            logger.error(f"Error during fact-checking: {e}", exc_info=True)
            return [
                FactCheckResult(
                    claim=claim,
                    verified=False,
                    evidence=Evidence(source="Fact-Check Error", snippet=str(e)),
                    confidence=0.0
                ) for claim in claims
            ]
