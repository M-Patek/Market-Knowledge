import json
import os
from typing import Dict, Any, Optional

# 假设日志记录器已经设置好
# from monitor.logging import log

class PromptManager:
    """
    管理从 JSON 文件加载和渲染提示模板。
    """
    
    def __init__(self, prompts_dir: str = "prompts/"):
        """
        初始化 PromptManager。

        Args:
            prompts_dir (str): 存放提示 JSON 文件的目录路径。
                               默认为 "prompts/"。
        """
        self.prompts_dir = prompts_dir
        if not os.path.isdir(prompts_dir):
            print(f"Warning: 提示目录不存在: {prompts_dir}。将使用相对路径。")
            pass
        # log.info(f"PromptManager initialized with directory: {self.prompts_dir}")

    def _load_prompt_json(self, prompt_name: str) -> Dict:
        """
        私有辅助方法：加载并解析指定名称的提示 JSON 文件。
        """
        # [Task 5.2] Path Traversal Defense
        # Strictly enforce that the file comes from the prompts directory
        safe_prompt_name = os.path.basename(prompt_name)
        if safe_prompt_name != prompt_name:
            print(f"Warning: Path traversal attempt detected. Sanitized '{prompt_name}' to '{safe_prompt_name}'")
            
        file_path = os.path.join(self.prompts_dir, f"{safe_prompt_name}.json")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Prompt file not found at {file_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {file_path}")
            return {}
        except Exception as e:
            print(f"Error: Unknown error loading prompt {file_path}: {e}")
            return {}

    def get_system_prompt(self, prompt_name: str) -> str:
        """
        从 JSON 文件中加载静态系统提示。
        """
        prompt_data = self._load_prompt_json(prompt_name)
        prompt_str = prompt_data.get("system_prompt", "")
        
        if not prompt_str:
            print(f"Warning: 'system_prompt' key not found in {prompt_name}.json or file failed to load.")
            
        return prompt_str

    def get_prompt(self, prompt_name: str) -> Optional[Dict[str, Any]]: # [Task 4] 修改
        """
        [Task 4] 已修改
        加载原始提示 JSON 文件。
        渲染由此模块的 'PromptRenderer' 处理。

        Args:
            prompt_name (str): 提示的名称。

        Returns:
            Optional[Dict[str, Any]]: 原始 JSON 内容。
        """
        prompt_data = self._load_prompt_json(prompt_name)
        
        if not prompt_data:
            print(f"Warning: 'prompt_name' {prompt_name}.json not found or empty.")
            return None
            
        return prompt_data # [Task 4] 返回整个字典

    # [Task 4] 保留旧的 .format() 行为作为遗留方法
    def get_legacy_rendered_template(self, prompt_name: str, context_data: Optional[Dict[str, Any]] = None) -> str:
        """
        [Task 4] 遗留方法
        加载提示并仅渲染 "prompt_template" 字段。
        (旧的 get_prompt 方法)
        """
        if context_data is None:
            context_data = {}
            
        prompt_data = self._load_prompt_json(prompt_name)
        template_str = prompt_data.get("prompt_template", "")
        
        if not template_str:
            print(f"Warning: 'prompt_template' key not found in {prompt_name}.json")
            return ""
            
        try:
            # 使用 .format() 进行键值替换
            return template_str.format(**context_data)
        except KeyError as e:
            print(f"Error rendering prompt '{prompt_name}': Missing key {e} in context")
            return template_str
        except Exception as e:
            print(f"Error: Unknown error rendering prompt '{prompt_name}': {e}")
            return template_str

# 示例用法 (用于测试)
if __name__ == "__main__":
    # ... (测试代码保持不变) ...
    pass
