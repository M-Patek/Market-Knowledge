import asyncio
import random
from typing import Callable, Coroutine, Any, Dict

from .core.schemas.data_schema import TradeData, QuoteData, MarketData, MarketDataSource
from .monitor.logging import get_logger

logger = get_logger(__name__)

class DataProducer:
    """
    模拟来自数据源（如交易所）的实时市场数据流。
    这个类用于测试和演示，模拟生成数据并将其推送到事件流。
    """
    
    def __init__(self, 
                 event_callback: Callable[[MarketData], Coroutine[Any, Any, None]],
                 data_source: MarketDataSource = MarketDataSource.SIMULATED):
        """
        初始化 DataProducer。

        Args:
            event_callback: 一个异步回调函数，用于处理生成的 MarketData 事件。
            data_source: 数据源标识。
        """
        self.event_callback = event_callback
        self.data_source = data_source
        self._running = False
        self._demo_task = None
        self.last_prices: Dict[str, float] = {}
        logger.info(f"DataProducer initialized with data source: {self.data_source}")

    async def on_trade(self, trade: TradeData):
        """
        处理传入的 TradeData 并将其转换为 MarketData (OHLCV) 事件。

        备注：这是一个更真实的模拟。在真实的系统中，
        这个生产者可能会聚合一分钟内的多次交易来构建一个K线 (bar)，
        或者直接从数据源接收K线。
        为了进行合理的模拟，我们基于当前交易价格和最后价格来估算一个OHLCV K线。
        """
        try:
            symbol = trade.symbol
            price = trade.price
            
            # 从缓存中获取最后的价格，如果不存在则使用当前价格
            open_price = self.last_prices.get(symbol, price)
            
            # 模拟高点和低点
            # 假设高点是开盘价和收盘价中的较高者，再加一点随机波动
            # 假设低点是开盘价和收盘价中的较低者，再减一点随机波动
            high_price = max(open_price, price) + (price * random.uniform(0.0001, 0.0005))
            low_price = min(open_price, price) - (price * random.uniform(0.0001, 0.0005))
            
            # 确保 H >= L
            if low_price > high_price:
                low_price, high_price = high_price, low_price
            
            # 确保价格在 [L, H] 范围内
            open_price = max(low_price, min(high_price, open_price))
            close_price = max(low_price, min(high_price, price)) # close price is the trade price
            
            # 模拟交易量
            simulated_volume = (trade.volume or 1.0) * random.uniform(1.0, 5.0)
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=trade.timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=simulated_volume,
                source=self.data_source,
                data_type='trade_derived_bar'
            )
            
            # 更新最后的价格
            self.last_prices[symbol] = price
            
            await self.event_callback(market_data)
            logger.debug(f"Processed simulated bar for trade: {market_data}")
            
        except Exception as e:
            logger.error(f"Error in on_trade: {e}", exc_info=True)

    async def on_quote(self, quote: QuoteData):
        """
        处理传入的 QuoteData 并将其转换为 MarketData (OHLCV) 事件。

        备注：报价数据 (bid/ask) 通常不直接生成K线。
        这个模拟使用中间价 (mid-price) 来估算一个OHLCV K线。
        """
        try:
            symbol = quote.symbol
            if quote.bid_price <= 0 or quote.ask_price <= 0:
                logger.warning(f"Skipping quote with invalid prices: {quote}")
                return
                
            mid_price = (quote.bid_price + quote.ask_price) / 2.0
            
            # 从缓存中获取最后的价格，如果不存在则使用中间价
            open_price = self.last_prices.get(symbol, mid_price)
            
            # 模拟高点和低点
            spread = (quote.ask_price - quote.bid_price)
            high_price = max(open_price, mid_price) + (spread * random.uniform(0.1, 0.5))
            low_price = min(open_price, mid_price) - (spread * random.uniform(0.1, 0.5))
            
            if low_price > high_price:
                low_price, high_price = high_price, low_price
            
            open_price = max(low_price, min(high_price, open_price))
            close_price = max(low_price, min(high_price, mid_price))
            
            # 模拟交易量 (基于报价大小)
            volume = (quote.bid_size + quote.ask_size) * random.uniform(0.05, 0.1)

            market_data = MarketData(
                symbol=symbol,
                timestamp=quote.timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                source=self.data_source,
                data_type='quote_derived_bar'
            )
            
            # 更新最后的价格
            self.last_prices[symbol] = mid_price
            
            await self.event_callback(market_data)
            logger.debug(f"Processed simulated bar for quote: {market_data}")

        except Exception as e:
            logger.error(f"Error in on_quote: {e}", exc_info=True)

    async def _run_demo_feed(self):
        """
        一个内部循环，用于在演示模式下生成模拟数据。
        """
        logger.info("Starting demo data feed...")
        self._running = True
        
        symbols = ["BTC/USD", "ETH/USD"]
        base_prices = {"BTC/USD": 60000.0, "ETH/USD": 3000.0}
        
        while self._running:
            try:
                await asyncio.sleep(random.uniform(0.5, 2.0))
                
                symbol = random.choice(symbols)
                base_price = base_prices[symbol]
                
                # 生成模拟交易
                trade_price = base_price + random.uniform(-100, 100)
                trade_volume = random.uniform(0.1, 5.0)
                trade = TradeData(
                    symbol=symbol,
                    price=trade_price,
                    volume=trade_volume,
                    timestamp=asyncio.get_event_loop().time()
                )
                await self.on_trade(trade)
                
                # 更新基准价格以模拟市场波动
                base_prices[symbol] = trade_price
                
                # 生成模拟报价 (偶尔)
                if random.random() < 0.3:
                    bid_price = trade_price - random.uniform(1, 5)
                    ask_price = trade_price + random.uniform(1, 5)
                    quote = QuoteData(
                        symbol=symbol,
                        bid_price=bid_price,
                        ask_price=ask_price,
                        bid_size=random.uniform(1, 10),
                        ask_size=random.uniform(1, 10),
                        timestamp=asyncio.get_event_loop().time()
                    )
                    await self.on_quote(quote)

            except asyncio.CancelledError:
                logger.info("Demo data feed cancelled.")
                self._running = False
            except Exception as e:
                logger.error(f"Error in demo feed loop: {e}", exc_info=True)
                await asyncio.sleep(5) # 发生错误时稍作等待

    def start_demo(self):
        """启动模拟数据流。"""
        if not self._running:
            self._demo_task = asyncio.create_task(self._run_demo_feed())
            logger.info("Demo data producer started.")
        else:
            logger.warning("Demo data producer is already running.")

    def stop_demo(self):
        """停止模拟数据流。"""
        if self._running and self._demo_task:
            self._demo_task.cancel()
            self._running = False
            logger.info("Demo data producer stopped.")
        else:
            logger.warning("Demo data producer is not running.")
