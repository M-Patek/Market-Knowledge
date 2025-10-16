import asyncio
import logging
from typing import Dict, Any

from .cognitive.engine import CognitiveEngine
from .events.risk_filter import RiskFilter
from .events.event_distributor import EventDistributor
# from .data_manager import DataManager (被流处理器取代)

class PhoenixProject:
    """
    项目的主应用类，初始化并连接所有核心组件。
    现在以事件驱动模式运行。
    """
    def __init__(self, config: Dict[str, Any] = None):
        # ... (日志和配置加载保持不变) ...
        self.config = config or {} 
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("PhoenixProject")

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
    project = PhoenixProject()
    asyncio.run(project.run())
