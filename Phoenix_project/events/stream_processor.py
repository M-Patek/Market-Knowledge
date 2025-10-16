# Phoenix_project/events/stream_processor.py
import asyncio
import random
from datetime import datetime, timezone
import logging

class StreamProcessor:
    """
    模拟与实时事件流（如Kafka、Pulsar）的连接。
    在真实系统中，这将是一个消息总线的消费者客户端。
    """
    def __init__(self):
        self.logger = logging.getLogger("PhoenixProject.StreamProcessor")
        self.logger.info("StreamProcessor已初始化 (模拟模式)。")

    async def event_stream(self):
        """
        一个异步生成器，产生连续的模拟市场事件流。
        """
        self.logger.info("开始监听实时事件...")
        event_id = 0
        while True:
            # 模拟每0.5到2秒随机接收一个新事件
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            event_type = random.choice(["MARKET_DATA", "NEWS", "MACRO_SIGNAL"])
            event = {
                "event_id": event_id,
                "event_type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": self._generate_payload(event_type)
            }
            yield event
            event_id += 1

    def _generate_payload(self, event_type: str) -> dict:
        """为给定的事件类型生成一个合理的载荷。"""
        if event_type == "MARKET_DATA":
            return {"symbol": "SPY", "price": round(random.uniform(500, 550), 2), "volume": random.randint(1000, 10000)}
        elif event_type == "NEWS":
            return {"headline": "美联储暗示可能调整利率", "source": "主流新闻媒体"}
        elif event_type == "MACRO_SIGNAL":
            return {"indicator": "CPI", "value": round(random.uniform(0.1, 0.3), 2), "status": "初步数据"}
        return {}
