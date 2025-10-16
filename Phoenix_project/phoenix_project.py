import asyncio
import logging
import os
import yaml
from typing import Dict, Any, MutableMapping

from .cognitive.engine import CognitiveEngine
from .events.risk_filter import RiskFilter
from .events.event_distributor import EventDistributor
# from .data_manager import DataManager (被流处理器取代)

def _load_and_override_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML config and overrides values from environment variables."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    def _recursive_override(sub_config: MutableMapping[str, Any]):
        """Recursively traverses the config dict to find and replace env var placeholders."""
        for key, value in list(sub_config.items()):
            if isinstance(value, dict):
                _recursive_override(value)
            elif isinstance(key, str) and key.endswith('_env_var'):
                env_var_name = value
                env_var_value = os.getenv(env_var_name)
                if env_var_value:
                    # Replace the placeholder key with the actual config key
                    # e.g., 'api_key_env_var' becomes 'api_key'
                    new_key = key.replace('_env_var', '')
                    sub_config[new_key] = env_var_value
                    del sub_config[key] # Remove the original placeholder
                    logging.info(f"Configuration override: Found and set '{new_key}' from environment variable '{env_var_name}'.")

    _recursive_override(config)
    return config

class PhoenixProject:
    """
    项目的主应用类，初始化并连接所有核心组件。
    现在以事件驱动模式运行。
    """
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes the PhoenixProject application.

        Args:
            config_path (str): Path to the base configuration YAML file.
        """
        # Setup logging first
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("PhoenixProject")

        # [V2.0] Load base config and override with environment variables
        self.config = _load_and_override_config(config_path)

        # --- 主应用逻辑 ---
        self.cognitive_engine = CognitiveEngine(self.config)
        self.risk_filter = RiskFilter(self.config)
        self.event_distributor = EventDistributor(self.cognitive_engine, self.risk_filter)
        self.logger.info("PhoenixProject组件已初始化。")

    async def run(self):
        """
        主运行循环，现在是事件驱动的。
        """
        self.logger.info("--- PhoenixProject切换到事件驱动模式 ---")
        try:
            # EventDistributor的循环现在无限期运行，消费实时流
            await self.event_distributor.run_event_loop()
        except KeyboardInterrupt:
            self.logger.info("收到关闭信号。正在退出。")
        except Exception as e:
            self.logger.critical(f"事件驱动运行循环出现严重错误: {e}", exc_info=True)

if __name__ == "__main__":
    # 这是一个简化的启动器
    project = PhoenixProject(config_path="config.yaml")
    asyncio.run(project.run())

