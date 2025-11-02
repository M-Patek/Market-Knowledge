import json
import os
from typing import Dict, Any
# 修复：将相对导入 'from ..monitor.logging...' 更改为绝对导入
from monitor.logging import get_logger

logger = get_logger(__name__)

class PromptManager:
    """
    Manages loading and retrieving prompt templates from the '/prompts' directory.
    
    Assumes prompts are stored as JSON files (e.g., 'analyst.json', 'fact_checker.json')
    where each file contains at least a "system_prompt" key.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the PromptManager and loads all prompts from the specified directory.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary. Expects 'prompt_directory'.
        """
        self.prompt_directory = config.get('prompt_directory', 'prompts')
        # Ensure the path is relative to this file's location or an absolute path
        if not os.path.isabs(self.prompt_directory):
            self.prompt_directory = os.path.join(
                os.path.dirname(__file__), 
                '..', 
                self.prompt_directory
            )
            
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, Dict[str, Any]]:
        """
        Scans the prompt directory and loads all .json files.
        The filename (without extension) is used as the prompt 'role' (e.g., 'analyst').
        """
        loaded_prompts = {}
        if not os.path.exists(self.prompt_directory):
            logger.error(f"Prompt directory not found: {self.prompt_directory}")
            return {}

        logger.info(f"Loading prompts from: {self.prompt_directory}")
        for filename in os.listdir(self.prompt_directory):
            if filename.endswith(".json"):
                role_name = filename.split('.')[0]
                filepath = os.path.join(self.prompt_directory, filename)
                try:
                    with open(filepath, 'r') as f:
                        prompt_data = json.load(f)
                        if "system_prompt" not in prompt_data:
                            logger.warning(f"Prompt file {filename} is missing 'system_prompt' key.")
                            continue
                        loaded_prompts[role_name] = prompt_data
                        logger.debug(f"Loaded prompt for role: {role_name}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON from prompt file: {filename}")
                except Exception as e:
                    logger.error(f"Failed to load prompt file {filename}: {e}")
                    
        logger.info(f"Successfully loaded {len(loaded_prompts)} prompts.")
        return loaded_prompts

    def get_system_prompt(self, role: str) -> str:
        """
        Retrieves the system prompt string for a given agent role.
        
        Args:
            role (str): The name of the agent role (e.g., 'analyst').
            
        Returns:
            str: The system prompt. Returns a default error message if not found.
        """
        prompt_data = self.prompts.get(role)
        if prompt_data:
            return prompt_data["system_prompt"]
        else:
            logger.warning(f"No system prompt found for role: {role}")
            return f"Error: System prompt for '{role}' not found."

    def get_all_system_prompts(self) -> Dict[str, str]:
        """
        Retrieves a dictionary of all loaded system prompts, keyed by role.
        
        Returns:
            Dict[str, str]: A map of {role_name: system_prompt_string}.
        """
        return {role: data["system_prompt"] for role, data in self.prompts.items()}

    def get_prompt_metadata(self, role: str) -> Dict[str, Any]:
        """
        Retrieves any additional metadata stored in the prompt JSON (e.g., version, model_prefs).
        
        Args:
            role (str): The name of the agent role.
            
        Returns:
            Dict[str, Any]: The metadata, excluding the 'system_prompt' itself.
        """
        prompt_data = self.prompts.get(role, {}).copy()
        prompt_data.pop("system_prompt", None) # Remove the main prompt string
        return prompt_data
