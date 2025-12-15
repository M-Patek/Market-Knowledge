import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class L1BaseAgent:
    """
    Base class for all L1 Agents.
    Provides common functionality for prompt rendering, tool execution, and memory interaction.
    """
    
    # [Task P1-004] Prompt Safety Constants
    MAX_PROMPT_TOKENS = 4000
    
    def __init__(self, name: str, config: Dict[str, Any], prompt_renderer: Any = None):
        self.name = name
        self.config = config
        self.prompt_renderer = prompt_renderer
        self.logger = logger
        
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimates token count using tiktoken if available, otherwise char approximation.
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: Approx 4 chars per token
            return len(text) // 4
        except Exception:
            return 0

    def _truncate_content(self, content: Any, max_tokens: int) -> Any:
        """
        [Task P1-004] Hard truncation for prompt injection defense.
        """
        if not isinstance(content, str):
            return content
            
        # Quick check using char length (optimization)
        # 1 token >= 1 char, so if chars < max_tokens, definitely safe (conservative)
        if len(content) < max_tokens:
            return content
            
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(content)
            
            if len(tokens) > max_tokens:
                self.logger.warning(f"Content truncated: {len(tokens)} tokens > {max_tokens} limit.")
                truncated_text = encoding.decode(tokens[:max_tokens])
                return truncated_text + " ... [TRUNCATED]"
            return content
            
        except ImportError:
            # Fallback char limit (avg 4 chars/token)
            char_limit = max_tokens * 4
            if len(content) > char_limit:
                self.logger.warning(f"Content truncated (char limit): {len(content)} > {char_limit}.")
                return content[:char_limit] + " ... [TRUNCATED]"
            return content
        except Exception as e:
            self.logger.error(f"Truncation failed: {e}")
            return content

    def render_prompt(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Renders the prompt using the PromptRenderer.
        [Task P1-004] Applied Input Sanitization & Length Truncation.
        """
        if not self.prompt_renderer:
            self.logger.error("PromptRenderer not initialized.")
            return ""

        # [Task P1-004] Sanitize Context
        # Allow variables to take up to 75% of MAX_PROMPT_TOKENS to leave room for instructions
        var_limit = int(self.MAX_PROMPT_TOKENS * 0.75)
        
        safe_context = {}
        for key, value in context.items():
            safe_context[key] = self._truncate_content(value, var_limit)
        
        try:
            rendered = self.prompt_renderer.render(template_name, safe_context)
            
            # Handle return format (Dict or Str)
            prompt_str = ""
            if isinstance(rendered, dict):
                prompt_str = rendered.get("full_prompt_template", "")
            else:
                prompt_str = str(rendered)
                
            return prompt_str
            
        except Exception as e:
            self.logger.error(f"Failed to render prompt '{template_name}': {e}")
            return ""
