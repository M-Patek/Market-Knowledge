import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod
import os

from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.execution.interfaces import IBrokerAdapter
from Phoenix_project.core.schemas.data_schema import Order, OrderStatus

# [任务 B.2] 导入 Alpaca 客户端
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logger = get_logger(__name__)

# --- 市场数据接口 ---

class AlpacaAdapter(IBrokerAdapter):
    """
    Adapter for Alpaca serving as both a market data provider
    and an execution broker.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Alpaca adapter.
        
        Args:
            config: Configuration dictionary, expects 'alpaca_api_key'
                    and 'alpaca_api_secret'.
        """
        self.api_key = config.get('alpaca_api_key') or os.environ.get('ALPACA_API_KEY')
        self.api_secret = config.get('alpaca_api_secret') or os.environ.get('ALPACA_API_SECRET')
        self.paper = config.get('paper_trading', True)
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API key/secret not provided or found in environment.")
            
        try:
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.paper
            )
            
            # [任务 B.2] 初始化历史数据客户端
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                raw_data=False # False = 自动转换为 Pydantic/Pandas
            )
            
            account = self.trading_client.get_account()
            logger.info(f"Alpaca connection successful. Account: {account.account_number}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}", exc_info=True)
            raise

    # --- IMarketData 实现 ---

    def get_market_data(
        self, 
        symbols_list: List[str], 
        start_date: datetime, 
        end_date: datetime,
        timeframe_str: str = "1D"
    ) -> Dict[str, pd.DataFrame]:
        """
        [任务 B.2] Implemented.
        Fetches historical market data (bars) from Alpaca.
        
        Args:
            symbols_list: List of stock tickers.
            start_date: Start of the historical period.
            end_date: End of the historical period.
            timeframe_str: "1D", "1H", "1Min", etc.
            
        Returns:
            A dictionary mapping symbol -> DataFrame of market data.
        """
        logger.info(f"Fetching Alpaca market data for {symbols_list} from {start_date} to {end_date}...")
        
        # 1. 转换时间框架字符串
        tf_map = {
            "1Min": TimeFrame.Minute,
            "1H": TimeFrame.Hour,
            "1D": TimeFrame.Day
        }
        alpaca_tf = tf_map.get(timeframe_str)
        if not alpaca_tf:
            logger.warning(f"Unsupported timeframe '{timeframe_str}'. Defaulting to '1D'.")
            alpaca_tf = TimeFrame.Day

        try:
            # 2. 构建请求
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols_list,
                timeframe=alpaca_tf,
                start=start_date,
                end=end_date
            )
            
            # 3. 调用 API
            bars_response = self.data_client.get_stock_bars(request_params)
            
            # 4. 转换为所需的 Dict[str, pd.DataFrame] 格式
            bars_df = bars_response.df
            
            if bars_df.empty:
                logger.warning(f"Alpaca returned no data for symbols {symbols_list}.")
                return {}

            data_dict = {}
            if isinstance(bars_df.index, pd.MultiIndex):
                # 多股票：按 'symbol' 级别拆分
                for symbol in bars_df.index.get_level_values('symbol').unique():
                    data_dict[symbol] = bars_df.loc[symbol]
            else:
                # 单股票：直接分配
                if symbols_list:
                    data_dict[symbols_list[0]] = bars_df
                else:
                    logger.warning("Data fetched but original symbols list was empty.")

            logger.info(f"Successfully fetched {len(data_dict)} symbols from Alpaca.")
            return data_dict

        except Exception as e:
            logger.error(f"Failed to get market data from Alpaca: {e}", exc_info=True)
            return {}

    # --- IExecutionBroker 实现 ---

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def subscribe_fills(self, callback) -> None:
        pass

    def subscribe_order_status(self, callback) -> None:
        pass

    def get_all_open_orders(self) -> List[Order]:
        return []

    def get_portfolio_value(self) -> float:
        return 0.0

    def get_cash_balance(self) -> float:
        return 0.0

    def get_position(self, symbol: str) -> float:
        return 0.0

    def place_order(self, order: Order, price: Optional[float] = None) -> str:
        """
        [Task 1.2] Adapter method to conform to IBrokerAdapter interface.
        Delegates to existing submit_order.
        """
        order_data = {
            "symbol": order.symbol,
            "quantity": order.quantity,
            "order_type": order.order_type.lower(),
            "limit_price": order.limit_price
        }
        result = self.submit_order(order_data)
        if result.get("status") == "success":
            return str(result.get("order_id"))
        raise RuntimeError(f"Alpaca placement failed: {result.get('message')}")

    def submit_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submits an order to Alpaca.
        """
        symbol = order_data.get('symbol')
        qty = order_data.get('quantity')
        order_type = order_data.get('order_type', 'market')
        limit_price = order_data.get('limit_price')
        
        if not symbol or not qty:
            msg = "Order submission failed: Symbol and Quantity are required."
            logger.error(msg)
            return {"status": "error", "message": msg}

        side = OrderSide.BUY if qty > 0 else OrderSide.SELL
        abs_qty = abs(qty)

        try:
            if order_type == 'market':
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=abs_qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
            elif order_type == 'limit':
                if not limit_price:
                    msg = "Limit order requires 'limit_price'."
                    logger.error(msg)
                    return {"status": "error", "message": msg}
                request = LimitOrderRequest(
                    symbol=symbol,
                    qty=abs_qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )
            else:
                msg = f"Unsupported order type: {order_type}"
                logger.error(msg)
                return {"status": "error", "message": msg}

            logger.info(f"Submitting {order_type} order to Alpaca: {side} {abs_qty} {symbol}")
            order = self.trading_client.submit_order(order_data=request)
            
            logger.info(f"Order submitted successfully. Order ID: {order.id}")
            return {"status": "success", "order_id": order.id, "data": order.dict()}

        except Exception as e:
            logger.error(f"Failed to submit order to Alpaca: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Gets the status of a specific order from Alpaca.
        """
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return {"status": "success", "data": order.dict()}
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancels an open order.
        """
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Cancel request sent for order {order_id}.")
            return {"status": "success", "order_id": order_id}
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def get_account_info(self) -> Dict[str, Any]:
        """
        Retrieves account information (balance, positions, etc.).
        """
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            
            return {
                "status": "success",
                "account": account.dict(),
                "positions": [p.dict() for p in positions]
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
