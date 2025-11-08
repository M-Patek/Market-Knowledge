import os
import asyncio
import json
import logging
from kafka import KafkaProducer
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.common import URL
from datetime import datetime

# 假设的配置加载器和数据模式
# 在实际应用中，您会从您的项目中导入这些
try:
    from Phoenix_project.config.loader import ConfigLoader
    from Phoenix_project.core.schemas.data_schema import MarketData, NewsData
except ImportError:
    logging.warning("无法导入 Phoenix_project 组件。将使用基本的 dict。")
    # --- Fallback Schemas (用于独立运行) ---
    from pydantic import BaseModel # 假设 pydantic 可用
    class ConfigLoader:
        def __init__(self, path):
            self.config_path = path
        def load_config(self, name):
            # 模拟加载，在实际部署中会从 system.yaml 读取
            logging.info(f"模拟加载配置: {name}")
            return {
                "system": {"environment": "production"},
                "broker": {"base_url": "https://paper-api.alpaca.markets"}
            }
    class MarketData(BaseModel):
        symbol: str
        timestamp: datetime
        open: float
        high: float
        low: float
        close: float
        volume: float
    class NewsData(BaseModel):
        id: str
        source: str
        timestamp: datetime
        symbols: list
        content: str
        headline: str
        metadata: dict = {}

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [DataProducer] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Kafka 主题 ---
TOPIC_MARKET_DATA = "phoenix_market_data"
TOPIC_NEWS_EVENTS = "phoenix_news_events"

# --- Kafka 生产者 ---
def get_kafka_producer():
    kafka_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
    logger.info(f"正在连接到 Kafka: {kafka_servers}")
    try:
        producer = KafkaProducer(
            bootstrap_servers=kafka_servers.split(','),
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
        )
        logger.info("Kafka 生产者连接成功。")
        return producer
    except Exception as e:
        logger.critical(f"无法连接到 Kafka: {e}", exc_info=True)
        return None

# --- Alpaca 回调 ---

async def on_trade(trade):
    """
    处理实时行情数据 (trade)。
    按照计划，我们将一个 trade 转换为 MarketData 模式。
    """
    global producer # 确保回调可以访问全局生产者
    logger.debug(f"收到 Trade: {trade}")
    try:
        # 关键逻辑：将 Trade 映射到 MarketData (OHLCV)
        # 这是一个模拟，因为 trade 只有一个价格点。
        market_data_obj = MarketData(
            symbol=trade.symbol,
            timestamp=trade.timestamp.to_pydatetime(), # 转换为 Pydantic 兼容的 datetime
            open=trade.price,
            high=trade.price,
            low=trade.price,
            close=trade.price,
            volume=trade.size
        )
        # .model_dump() 是 Pydantic v2+ 的方法
        # .dict() 是 v1 的方法
        market_data_dict = market_data_obj.dict() if hasattr(market_data_obj, 'dict') else market_data_obj.model_dump()

        producer.send(TOPIC_MARKET_DATA, market_data_dict)
        logger.info(f"已发送 MarketData (来自 Trade) 到 Kafka: {trade.symbol} @ {trade.price}")
    except Exception as e:
        logger.error(f"处理 Trade 失败: {e}", exc_info=True)

async def on_news(news):
    """
    处理实时新闻数据。
    """
    global producer # 确保回调可以访问全局生产者
    logger.debug(f"收到 News: {news.headline}")
    try:
        # NewsData 模式与 Alpaca news 对象的字段匹配
        news_data_obj = NewsData(
            id=str(news.id), # 确保 ID 是字符串
            source=news.source,
            timestamp=news.created_at.to_pydatetime(), # 转换
            symbols=news.symbols,
            content=news.content,
            headline=news.headline
        )
        # .model_dump() 是 Pydantic v2+ 的方法
        # .dict() 是 v1 的方法
        news_data_dict = news_data_obj.dict() if hasattr(news_data_obj, 'dict') else news_data_obj.model_dump()
        
        producer.send(TOPIC_NEWS_EVENTS, news_data_dict)
        logger.info(f"已发送 News 到 Kafka: {news.headline[:50]}...")
    except Exception as e:
        logger.error(f"处理 News 失败: {e}", exc_info=True)


async def main():
    logger.info("启动数据生产者服务...")
    
    # 1. 加载配置 (用于 Alpaca base_url)
    config_loader = ConfigLoader(os.environ.get('PHOENIX_CONFIG_PATH', 'config'))
    system_config = config_loader.load_config('system.yaml')
    
    # 2. 获取 Alpaca 凭证
    api_key = os.environ.get('ALPACA_API_KEY')
    api_secret = os.environ.get('ALPACA_API_SECRET')
    
    if not api_key or not api_secret:
        logger.critical("未找到 ALPACA_API_KEY 或 ALPACA_API_SECRET 环境变量。正在退出。")
        return

    # 3. 确定 base_url (纸上交易或真实交易)
    base_url = system_config.get("broker", {}).get("base_url", "https://paper-api.alpaca.markets")
    logger.info(f"使用 Alpaca base_url: {base_url}")
    
    # 4. 初始化 Alpaca Stream
    stream = Stream(
        api_key,
        api_secret,
        base_url=URL(base_url),
        data_feed='iex' # 'iex' 或 'sip' 取决于订阅
    )

    # 5. 订阅流
    # (按照计划订阅 AAPL 和 MSFT 的行情，以及所有新闻)
    logger.info("正在订阅 Alpaca topics...")
    stream.subscribe_trades(on_trade, "AAPL", "MSFT")
    stream.subscribe_news(on_news, "*")

    logger.info("Alpaca Stream 订阅完成。正在运行...")
    await stream.run()

if __name__ == "__main__":
    producer = get_kafka_producer() # 将生产者设为全局变量
    if producer:
        try:
            asyncio.run(main())
        except (KeyboardInterrupt, SystemExit):
            logger.info("数据生产者正在关闭...")
        finally:
            producer.flush()
            producer.close()
            logger.info("Kafka 生产者已关闭。")
    else:
        logger.critical("无法初始化 Kafka 生产者。服务未启动。")
