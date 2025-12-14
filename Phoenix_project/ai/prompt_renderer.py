"""
Phoenix_project/ai/prompt_renderer.py
[Task B.3] Removed string.Template
[Task 1.5] Security: Prompt Injection Defense (Escaping & XML Tagging).
[Task 3.3] Time Context Injection.
"""
import json
import os
import html
from typing import Dict, Any, Union, List, Optional
from datetime import datetime, timezone

from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.monitor.logging import get_logger
# [Task 3.3] Optional TimeProvider import (to avoid circular deps, use type checking only or lazy)
# from Phoenix_project.core.time_provider import TimeProvider

logger = get_logger(__name__)

class PromptRenderer:
    """
    Dynamically renders prompt templates using provided context.
    Includes security mechanisms to prevent Prompt Injection.
    """
    
    def __init__(self, prompt_manager: PromptManager, time_provider: Any = None):
        """
        Initializes the PromptRenderer.
        
        Args:
            prompt_manager: An instance of PromptManager to load templates.
            time_provider: Optional TimeProvider to inject current_time automatically.
        """
        self.prompt_manager = prompt_manager
        self.time_provider = time_provider # [Task 3.3]
        logger.info("PromptRenderer initialized.")

    def render(
        self, 
        prompt_name: str, 
        context: Dict[str, Any]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        [任务 B.3] Implemented.
        Renders a prompt template by name, filling placeholders with context.
        [Task 1.5] Applies security sanitization to context data.
        [Task 3.3] Injects current_time if available.
        """
        logger.debug(f"Rendering prompt '{prompt_name}'...")
        
        try:
            # 1. 获取原始模板
            template_data = self.prompt_manager.get_prompt(prompt_name)
            
            if not template_data:
                raise KeyError(f"Prompt template '{prompt_name}' returned empty data.")
            
            # [Task 3.3] Auto-inject time if missing
            if 'current_time' not in context:
                if self.time_provider:
                    current_time = self.time_provider.get_current_time()
                    context['current_time'] = current_time.isoformat()
                else:
                    # Fallback to system time (UTC) if no provider
                    context['current_time'] = datetime.now(timezone.utc).isoformat()

            # 2. [Security] Sanitize Context (Defense against Injection)
            safe_context = self._sanitize_context(context)

            # 3. 递归地渲染模板
            rendered_data = self._render_recursive(template_data, safe_context)
            
            logger.debug(f"Successfully rendered prompt '{prompt_name}'.")
            return rendered_data

        except KeyError:
            logger.error(f"Prompt template '{prompt_name}' not found.")
            raise
        except Exception as e:
            logger.error(f"Failed to render prompt '{prompt_name}': {e}", exc_info=True)
            raise

    def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        [Task 1.5] Deeply sanitizes context dictionary.
        1. Escapes special characters (HTML/XML style).
        2. Wraps content in <user_content> tags to demarcate data from instructions.
        """
        sanitized = {}
        for k, v in context.items():
            # [Task 3.3] Skip sanitization for trusted system variables like 'current_time'
            if k in ['current_time']:
                sanitized[k] = v
                continue

            if isinstance(v, str):
                escaped_val = html.escape(v)
                sanitized[k] = f"<user_content>{escaped_val}</user_content>"
            elif isinstance(v, dict):
                sanitized[k] = self._sanitize_context(v)
            elif isinstance(v, list):
                sanitized[k] = [
                    self._sanitize_context(i) if isinstance(i, dict) else 
                    (f"<user_content>{html.escape(i)}</user_content>" if isinstance(i, str) else i)
                    for i in v
                ]
            else:
                sanitized[k] = v
        return sanitized

    def _render_recursive(
        self, 
        template_part: Any, 
        context: Dict[str, Any]
    ) -> Any:
        """
        Helper function to recursively render parts of a prompt structure.
        """
        if isinstance(template_part, str):
            try:
                # [Fix B.4] Switch to .format() 
                return template_part.format(**context)
            except KeyError as e:
                # Log but re-raise to fail fast
                logger.error(f"Missing context variable during rendering: {e}")
                raise
            except Exception as e:
                logger.warning(f"Error substituting template string: {e}")
                raise

        elif isinstance(template_part, dict):
            rendered_dict = {}
            for key, value in template_part.items():
                rendered_dict[key] = self._render_recursive(value, context)
            return rendered_dict

        elif isinstance(template_part, list):
            rendered_list = []
            for item in template_part:
                rendered_list.append(self._render_recursive(item, context))
            return rendered_list

        else:
            return template_part
