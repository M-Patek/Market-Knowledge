"""
Phoenix_project/ai/prompt_manager.py
[Task 1.5] Security: Fail-Fast on Prompt Loading.
Prevent "Lobotomy" (Running without instructions) by raising exceptions instead of returning empty dicts.
[Phase 5 Task 6] Remove Legacy Interface.
Removed get_legacy_rendered_template to close security loophole.
"""
import json
import os
from typing import Dict, Any, Optional

class PromptManager:
    """
    管理从 JSON 文件加载和渲染提示模板。
    """
    
    def __init__(self, prompts_dir: str = "prompts/"):
        """
        初始化 PromptManager。
        """
        self.prompts_dir = prompts_dir
        if not os.path.isdir(prompts_dir):
            # [Task 1.5] Warn loudly
            print(f"CRITICAL WARNING: Prompt directory not found: {prompts_dir}")

    def _load_prompt_json(self, prompt_name: str) -> Dict:
        """
        私有辅助方法：加载并解析指定名称的提示 JSON 文件。
        [Task 1.5] Modified to Raise Exceptions (Fail-Fast).
        """
        # [Task 5.2] Path Traversal Defense
        safe_prompt_name = os.path.basename(prompt_name)
        if safe_prompt_name != prompt_name:
            raise ValueError(f"Security Alert: Path traversal detected in prompt name '{prompt_name}'")
            
        file_path = os.path.join(self.prompts_dir, f"{safe_prompt_name}.json")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Critical: Prompt file missing: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Critical: Corrupted JSON in prompt file {file_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Critical: Unexpected error loading prompt {file_path}: {e}")

    def get_system_prompt(self, prompt_name: str) -> str:
        """
        从 JSON 文件中加载静态系统提示。
        """
        prompt_data = self._load_prompt_json(prompt_name)
        prompt_str = prompt_data.get("system_prompt", "")
        
        if not prompt_str:
            # [Task 1.5] Fail fast if critical system prompt is missing
            raise ValueError(f"Missing 'system_prompt' field in {prompt_name}.json")
            
        return prompt_str

    def get_prompt(self, prompt_name: str) -> Dict[str, Any]: 
        """
        [Task 4] 已修改
        加载原始提示 JSON 文件。
        
        [Task 1.5] Updated signature: No longer Optional, must return Dict or raise.
        """
        prompt_data = self._load_prompt_json(prompt_name)
        
        # Double check validity
        if not prompt_data:
            raise ValueError(f"Prompt {prompt_name}.json is empty.")
            
        return prompt_data 

    # [Task 6] Removed get_legacy_rendered_template to enforce secure rendering via PromptRenderer.
