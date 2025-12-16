import json
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional, Union

# [Task P2-BASE-01] Optional tiktoken import for precise counting
try:
    import tiktoken
except ImportError:
    tiktoken = None

logger = logging.getLogger(__name__)

class L2Agent(ABC):
    """
    Abstract Base Class for L2 agents.
    Provides common context management and token counting utilities.
    """
    def __init__(self, agent_id: str, llm_client: Any, data_manager: Any):
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.data_manager = data_manager
        # Attempt to get model name from client config, default to gpt-4 for encoding
        self.model_name = getattr(llm_client, "model_name", "gpt-4")

    @abstractmethod
    async def run(self, state: Any, dependencies: List[Any]) -> Any:
        """
        Core logic to be implemented by concrete agents.
        """
        pass

    # [Task P2-BASE-01] Token Counting Logic
    def _count_tokens(self, text: str) -> int:
        """
        Counts tokens using tiktoken if available, otherwise approximates.
        """
        if tiktoken:
            try:
                try:
                    encoding = tiktoken.encoding_for_model(self.model_name)
                except KeyError:
                    encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception as e:
                logger.debug(f"Tiktoken error: {e}. Falling back to approximation.")
        
        # Fallback: Approx 4 characters per token for English text
        return len(text) // 4

    # [Task P2-BASE-01] Structure-Safe Context Preparation
    def _safe_prepare_context(self, data: Any, max_tokens: int = 6000) -> str:
        """
        Serializes data to JSON, ensuring the result stays within token limits.
        For lists, it drops items from the tail to maintain valid JSON structure.
        """
        # 1. Handle Lists Structurally
        if isinstance(data, list):
            if not data:
                return "[]"
            
            # Fast path: Check full JSON
            full_json = json.dumps(data)
            if self._count_tokens(full_json) <= max_tokens:
                return full_json

            # Iterative Construction
            logger.info(f"[{self.agent_id}] Context list exceeds {max_tokens} tokens. Truncating items safely...")
            
            included_items = []
            current_tokens = 2 # Overhead for "[]"
            
            for i, item in enumerate(data):
                item_str = json.dumps(item)
                # Add 1 token roughly for comma/spacing overhead
                item_cost = self._count_tokens(item_str) + 1
                
                if current_tokens + item_cost > max_tokens:
                    logger.warning(
                        f"[{self.agent_id}] Truncated context at item {i}/{len(data)} "
                        f"to maintain limit ({current_tokens + item_cost} > {max_tokens})."
                    )
                    break
                
                included_items.append(item)
                current_tokens += item_cost
            
            return json.dumps(included_items)

        # 2. Handle Non-List types (Dict, etc.) - Fallback to simple check
        else:
            try:
                full_json = json.dumps(data)
                if self._count_tokens(full_json) <= max_tokens:
                    return full_json
                else:
                    logger.warning(f"[{self.agent_id}] Non-list context exceeds limit. Returning empty object/string.")
                    return "{}" if isinstance(data, dict) else ""
            except Exception as e:
                logger.error(f"[{self.agent_id}] Serialization failed: {e}")
                return ""

    def _sanitize_symbol(self, symbol: Any) -> str:
        if not symbol:
            return "UNKNOWN"
        return str(symbol).strip().upper()

    def _sanitize_general_input(self, text: Any) -> str:
        if not text:
            return ""
        return str(text).replace("\n", " ").strip()
