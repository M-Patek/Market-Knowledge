"""
Phoenix_project/ai/prompt_renderer.py
[Task B.3] Removed string.Template
[Task 1.5] Security: Prompt Injection Defense (Escaping & XML Tagging).
"""
import json
import os
import html
from typing import Dict, Any, Union, List

from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class PromptRenderer:
    """
    Dynamically renders prompt templates using provided context.
    Includes security mechanisms to prevent Prompt Injection.
    """
    
    def __init__(self, prompt_manager: PromptManager):
        """
        Initializes the PromptRenderer.
        
        Args:
            prompt_manager: An instance of PromptManager to load templates.
        """
        self.prompt_manager = prompt_manager
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
        """
        logger.debug(f"Rendering prompt '{prompt_name}'...")
        
        try:
            # 1. 获取原始模板
            template_data = self.prompt_manager.get_prompt(prompt_name)
            
            # [Task 1.5] PromptManager now raises exception, but double check doesn't hurt
            if not template_data:
                raise KeyError(f"Prompt template '{prompt_name}' returned empty data.")
            
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
            if isinstance(v, str):
                # Escape chars like <, >, &, " to prevent breaking XML/JSON structure
                escaped_val = html.escape(v)
                # Encapsulate to prevent instruction override
                # Note: We apply this to ALL string inputs from context as they are considered untrusted
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
        # Case 1: 模板部分是字符串 -> 渲染它
        if isinstance(template_part, str):
            try:
                # [Fix B.4] Switch to .format() 
                return template_part.format(**context)
            except KeyError as e:
                # [Robustness] Log missing key but keep running if possible, or re-raise?
                # Task says "Fail-Fast" for loading, but for rendering partial might be dangerous.
                # Let's fail fast to warn developer about mismatch.
                logger.error(f"Missing context variable during rendering: {e}")
                raise
            except Exception as e:
                logger.warning(f"Error substituting template string: {e}")
                raise # Re-raise to prevent silent failure

        # Case 2: 模板部分是字典 -> 递归渲染它的值
        elif isinstance(template_part, dict):
            rendered_dict = {}
            for key, value in template_part.items():
                rendered_dict[key] = self._render_recursive(value, context)
            return rendered_dict

        # Case 3: 模板部分是列表 -> 递归渲染它的项
        elif isinstance(template_part, list):
            rendered_list = []
            for item in template_part:
                rendered_list.append(self._render_recursive(item, context))
            return rendered_list

        # Case 4: 其他类型 (int, bool, etc.) -> 按原样返回
        else:
            return template_part
