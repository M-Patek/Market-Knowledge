import json
import os
from typing import Dict, Any, Union, List
# [Task B.3] Removed string.Template

from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class PromptRenderer:
    """
    Dynamically renders prompt templates using provided context.
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
        
        It handles both dictionary (e.g., {"system": "...", "user": "..."})
        and list (e.g., [{"role": "system", "content": "..."}, ...])
        JSON template structures.

        Args:
            prompt_name: The name of the prompt (e.g., 'l1_technical_analyst').
            context: A dictionary of key-value pairs to fill in.
            
        Returns:
            The rendered prompt structure (dict or list).
            
        Raises:
            KeyError: If the prompt_name is not found.
            TypeError: If the loaded prompt is not a dict or list.
        """
        logger.debug(f"Rendering prompt '{prompt_name}'...")
        
        try:
            # 1. 获取原始模板 (可能是 dict 或 list)
            template_data = self.prompt_manager.get_prompt(prompt_name)
            
            if not template_data:
                logger.error(f"Prompt template '{prompt_name}' is empty or None.")
                raise KeyError(f"Prompt template '{prompt_name}' is empty or None.")
            
            # 2. 递归地渲染模板
            rendered_data = self._render_recursive(template_data, context)
            
            logger.debug(f"Successfully rendered prompt '{prompt_name}'.")
            return rendered_data

        except KeyError:
            logger.error(f"Prompt template '{prompt_name}' not found.")
            raise
        except Exception as e:
            logger.error(f"Failed to render prompt '{prompt_name}': {e}", exc_info=True)
            # 重新抛出异常，以便调用者可以处理
            raise

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
                # [Fix B.4] Switch to .format() to match {variable} syntax in JSON templates
                return template_part.format(**context)
            except Exception as e:
                logger.warning(f"Error substituting template string: {e}")
                return template_part # 返回原始字符串

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
