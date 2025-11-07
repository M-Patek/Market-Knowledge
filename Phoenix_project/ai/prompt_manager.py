import json
import os
from typing import Dict, Any, Optional

# 假设日志记录器已经设置好
# from monitor.logging import log

class PromptManager:
    """
    管理从 JSON 文件加载和渲染提示模板。

    该类提供方法来加载静态系统提示和动态渲染
    包含上下文数据的提示模板。
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
            # log.warning(f"提示目录不存在: {prompts_dir}。将使用相对路径。")
            # 允许在测试或不同工作目录中运行
            pass
        # log.info(f"PromptManager initialized with directory: {self.prompts_dir}")

    def _load_prompt_json(self, prompt_name: str) -> Dict:
        """
        私有辅助方法：加载并解析指定名称的提示 JSON 文件。

        Args:
            prompt_name (str): 提示的名称（不含 .json 扩展名）。

        Returns:
            Dict: 解析后的 JSON 内容。如果文件未找到或解码失败，
                  则返回空字典并记录错误。
        """
        file_path = os.path.join(self.prompts_dir, f"{prompt_name}.json")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # log.error(f"提示文件未找到: {file_path}")
            print(f"Error: Prompt file not found at {file_path}")
            return {}
        except json.JSONDecodeError:
            # log.error(f"无法解码 {file_path} 中的 JSON")
            print(f"Error: Failed to decode JSON from {file_path}")
            return {}
        except Exception as e:
            # log.error(f"加载提示 {file_path} 时发生未知错误: {e}")
            print(f"Error: Unknown error loading prompt {file_path}: {e}")
            return {}

    def get_system_prompt(self, prompt_name: str) -> str:
        """
        从 JSON 文件中加载静态系统提示。

        假定 JSON 文件包含一个名为 "system_prompt" 的键。

        Args:
            prompt_name (str): 提示的名称。

        Returns:
            str: 系统提示文本。如果键未找到或文件加载失败，
                 则返回空字符串。
        """
        prompt_data = self._load_prompt_json(prompt_name)
        prompt_str = prompt_data.get("system_prompt", "")
        
        if not prompt_str:
            # log.warning(f"在 {prompt_name}.json 中未找到 'system_prompt' 键或文件加载失败。")
            print(f"Warning: 'system_prompt' key not found in {prompt_name}.json or file failed to load.")
            
        return prompt_str

    def get_prompt(self, prompt_name: str, context_data: Optional[Dict[str, Any]] = None) -> str:
        """
        加载提示模板并使用上下文数据进行动态渲染。

        假定 JSON 文件包含一个名为 "prompt_template" 的键，该键
        包含一个使用 Python .format() 语法的模板字符串
        （例如 "Hello, {name}"）。

        Args:
            prompt_name (str): 提示的名称。
            context_data (Optional[Dict[str, Any]]): 一个字典，包含
                用于渲染模板的占位符和值。如果为 None，
                将使用空字典。

        Returns:
            str: 渲染后的提示文本。如果模板未找到，
                 将返回空字符串。如果渲染失败（例如缺少键），
                 将返回未渲染的模板字符串并记录错误。
        """
        if context_data is None:
            context_data = {}
            
        prompt_data = self._load_prompt_json(prompt_name)
        template_str = prompt_data.get("prompt_template", "")
        
        if not template_str:
            # log.warning(f"在 {prompt_name}.json 中未找到 'prompt_template' 键。")
            print(f"Warning: 'prompt_template' key not found in {prompt_name}.json")
            return ""
            
        try:
            # 使用 .format() 进行键值替换
            return template_str.format(**context_data)
        except KeyError as e:
            # log.error(f"渲染提示 '{prompt_name}' 时出错: 模板中缺少键 {e}")
            print(f"Error rendering prompt '{prompt_name}': Missing key {e} in context")
            # 返回未渲染的模板以便调试
            return template_str
        except Exception as e:
            # log.error(f"渲染提示 '{prompt_name}' 时发生未知错误: {e}")
            print(f"Error: Unknown error rendering prompt '{prompt_name}': {e}")
            return template_str

# 示例用法 (用于测试)
if __name__ == "__main__":
    # 假设在 Phoenix_project 根目录运行
    # 并且 prompts/ 目录存在
    # 我们需要创建临时的 JSON 文件用于测试
    
    temp_dir = "temp_prompts_test"
    os.makedirs(temp_dir, exist_ok=True)
    
    system_prompt_file = os.path.join(temp_dir, "test_system.json")
    template_prompt_file = os.path.join(temp_dir, "test_template.json")
    
    try:
        # 1. 创建测试文件
        with open(system_prompt_file, 'w') as f:
            json.dump({"system_prompt": "You are a helpful assistant."}, f)
            
        with open(template_prompt_file, 'w') as f:
            json.dump({"prompt_template": "Analyze the following data: {data_chunk} for user {user_id}"}, f)

        # 2. 初始化 Manager
        pm = PromptManager(prompts_dir=temp_dir)
        
        # 3. 测试 get_system_prompt
        system_msg = pm.get_system_prompt("test_system")
        print(f"System Prompt: {system_msg}")
        assert system_msg == "You are a helpful assistant."
        
        # 4. 测试 get_prompt (成功)
        context = {"data_chunk": "[...data...]", "user_id": "123"}
        template_msg = pm.get_prompt("test_template", context)
        print(f"Rendered Prompt: {template_msg}")
        assert template_msg == "Analyze the following data: [...data...] for user 123"
        
        # 5. 测试 get_prompt (缺少键)
        print("\nTesting missing key:")
        context_fail = {"user_id": "456"}
        failed_msg = pm.get_prompt("test_template", context_fail)
        print(f"Failed Render (expected fallback): {failed_msg}")
        assert failed_msg == "Analyze the following data: {data_chunk} for user {user_id}"

        # 6. 测试文件未找到
        print("\nTesting file not found:")
        not_found_msg = pm.get_system_prompt("non_existent")
        print(f"Not Found Msg: '{not_found_msg}'")
        assert not_found_msg == ""

        print("\nAll tests passed!")

    finally:
        # 清理
        if os.path.exists(system_prompt_file):
            os.remove(system_prompt_file)
        if os.path.exists(template_prompt_file):
            os.remove(template_prompt_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
