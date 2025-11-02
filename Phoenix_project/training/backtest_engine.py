# (原: backtesting/engine.py)
import backtrader as bt
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd

# --- [修复] ---
# 导入路径 '..' 依然正确 (training/ -> Phoenix_project/ -> core/)
# --- [修复结束] ---
from ..core.pipeline_state import PipelineState
from ..monitor.logging import get_logger

logger = get_logger(__name__)

class BacktestingEngine:
    """
    使用 backtrader 的核心回测引擎的封装。
    它可以被 WalkForwardTrainer 用来评估模型。
    """
    def __init__(self, config: Dict[str, Any]):
        self.cerebro = bt.Cerebro()
        self.config = config.get('backtesting', {})
        self.start_cash = self.config.get('start_cash', 100000.0)
        self.cerebro.broker.setcash(self.start_cash)
        
        # 添加分析器
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        logger.info(f"BacktestingEngine (backtrader) 已初始化，初始资金: {self.start_cash}")

    def add_strategy(self, strategy_class: bt.Strategy, **params):
        """
        向引擎添加一个 backtrader 策略。
        """
        self.cerebro.addstrategy(strategy_class, **params)
        logger.info(f"策略 {strategy_class.__name__} 已添加。")

    def add_data(self, data_feed: pd.DataFrame, name: str, start_date: datetime, end_date: datetime):
        """
        向引擎添加 Pandas DataFrame 格式的数据。
        """
        data = bt.feeds.PandasData(
            dataname=data_feed,
            fromdate=start_date,
            todate=end_date
        )
        self.cerebro.adddata(data, name=name)
        logger.debug(f"数据 {name} 已添加。")

    def run_backtest(self) -> Dict[str, Any]:
        """
        运行回测并返回结果。
        """
        logger.info("--- Backtrader 回测开始 ---")
        try:
            results = self.cerebro.run()
            strategy_result = results[0] # 获取第一个策略的实例
            
            # 提取分析结果
            sharpe = strategy_result.analyzers.sharpe.getanalysis().get('sharperatio', 0.0)
            drawdown = strategy_result.analyzers.drawdown.getanalysis().max.drawdown
            trades = strategy_result.analyzers.trades.getanalysis()

            final_value = self.cerebro.broker.getvalue()
            
            output = {
                "start_value": self.start_cash,
                "final_value": final_value,
                "pnl_pct": (final_value - self.start_cash) / self.start_cash * 100,
                "sharpe_ratio": sharpe,
                "max_drawdown_pct": drawdown,
                "total_trades": trades.get('total', {}).get('total', 0),
                "win_rate_pct": (trades.get('won', {}).get('total', 0) / (trades.get('total', {}).get('total', 1) or 1)) * 100
            }
            
            logger.info("--- Backtrader 回测完成 ---")
            logger.info(f"最终价值: {output['final_value']:.2f}, 夏普: {output['sharpe_ratio']:.2f}")
            
            return output
        
        except Exception as e:
            logger.error(f"Backtrader 运行失败: {e}", exc_info=True)
            return {"error": str(e)}

    def plot(self, filename: str = "backtest_plot.png"):
        """
        绘制回测结果。
        """
        try:
            self.cerebro.plot(style='candlestick', iplot=False, savefig=True, figfilename=filename)
            logger.info(f"回测图表已保存到: {filename}")
        except Exception as e:
            logger.warning(f"无法绘制图表: {e} (可能在无头服务器上运行)")

# 示例：一个简单的策略 (用于测试)
class SimpleMovingAverageStrategy(bt.Strategy):
    params = (
        ('maperiod', 15),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod
        )

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.dataclose[0] > self.sma[0]:
                self.order = self.buy()
        else:
            if self.dataclose[0] < self.sma[0]:
                self.order = self.sell()
