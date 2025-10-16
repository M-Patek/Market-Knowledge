import logging
from typing import Dict, Any
from .stream_processor import StreamProcessor

class EventDistributor:
    """
    接收事件，通过风险过滤器，然后分发到认知引擎。
    现在作为一个实时流消费者运行。
    """
    def __init__(self, cognitive_engine, risk_filter):
        self.logger = logging.getLogger("PhoenixProject.EventDistributor")
        self.cognitive_engine = cognitive_engine
        self.risk_filter = risk_filter
        # 分发器现在拥有一个流处理器
        self.stream_processor = StreamProcessor()
        self.logger.info("EventDistributor已初始化。")

    async def run_event_loop(self):
        """
        运行主事件循环，从流处理器消费事件。
        """
        async for event in self.stream_processor.event_stream():
            self.logger.debug(f"收到事件: {event['event_type']} ({event['event_id']})")
            await self._process_single_event(event)

    async def _process_single_event(self, event: Dict[str, Any]):
        """
        处理单个事件，将其通过风险过滤器，
        然后传递给认知引擎。现在是一个异步方法。
        """
        self.logger.debug(f"处理事件: {event}")
        if not self.risk_filter.is_event_safe(event):
            self.logger.warning(f"事件未通过风险过滤器: {event}")
            return

        self.logger.info(f"事件已传递给认知引擎: {event['event_type']}")
        # 假设认知引擎现在是异步的
        await self.cognitive_engine.handle_event(event)
