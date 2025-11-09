import pandas as pd
# 修复：[FIX-15] 添加 'Callable' 用于类型提示
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime # [任务 A.2] 导入
import numpy as np # [任务 A.2] 导入

# 修正：将 'core.pipeline_state' 转换为 'Phoenix_project.core.pipeline_state'
from Phoenix_project.core.pipeline_state import PipelineState
# 修正：将 'data.data_iterator' 转换为 'Phoenix_project.data.data_iterator'
from Phoenix_project.data.data_iterator import DataIterator
# 修正：将 'controller.orchestrator' 转换为 'Phoenix_project.controller.orchestrator'
from Phoenix_project.controller.orchestrator import Orchestrator
# 修正：将 'execution.trade_lifecycle_manager' 转换为 'Phoenix_project.execution.trade_lifecycle_manager'
from Phoenix_project.execution.trade_lifecycle_manager import TradeLifecycleManager
# 修正：将 'monitor.logging' 转换为 'Phoenix_project.monitor.logging'
from Phoenix_project.monitor.logging import get_logger
# 修复：[FIX-15] 导入 'render_report' (函数)，而不是 'ReportRenderer' (类)
# 修正：将 'output.renderer' 转换为 'Phoenix_project.output.renderer'
from Phoenix_project.output.renderer import render_report

# [任务 A.2] 导入 DRL 智能体和数据加载器
from Phoenix_project.data_manager import DataManager
from Phoenix_project.features.store import FeatureStore
from Phoenix_project.agents.l3.alpha_agent import AlphaAgent
from Phoenix_project.agents.l3.risk_agent import RiskAgent
from Phoenix_project.agents.l3.execution_agent import ExecutionAgent
from Phoenix_project.training.drl.multi_agent_trainer import load_nutrition_data


logger = get_logger(__name__)

class BacktestingEngine:
    """
    Drives the simulation by iterating through historical data and feeding
    it to the Orchestrator, simulating the passage of time.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        data_iterator: DataIterator,
        orchestrator: Orchestrator,
        trade_lifecycle_manager: TradeLifecycleManager,
        # 修复：[FIX-15] 更新类型提示和参数名称
        report_renderer_func: Callable
    ):
        """
        Initializes the BacktestingEngine.
        
        Args:
            config: The main strategy configuration.
            data_iterator: Provides the historical event stream (data).
            orchestrator: The "brain" that processes events and generates signals.
            trade_lifecycle_manager: Manages the simulated portfolio state.
            report_renderer_func: 生成报告的函数。
        """
        self.config = config
        self.backtest_config = config.get('backtesting', {})
        self.data_iterator = data_iterator
        self.orchestrator = orchestrator
        self.trade_lifecycle_manager = trade_lifecycle_manager
        # 修复：[FIX-15]
        self.report_renderer = report_renderer_func
        
        self.start_date = pd.to_datetime(self.backtest_config.get('start_date'))
        self.end_date = pd.to_datetime(self.backtest_config.get('end_date'))
        
        logger.info(f"BacktestingEngine initialized for period: {self.start_date} to {self.end_date}")

    async def run(self):
        """
        Executes the (event-driven) backtest.
        """
        logger.info("--- Starting (Event-Driven) Backtest Run ---")
        
        # 1. Load initial historical data for all components
        logger.info(f"Preloading historical data up to {self.start_date}...")
        await self.orchestrator.preload_historical_data(self.start_date)
        
        # 2. Initialize the data iterator for the backtest period
        self.data_iterator.setup(self.start_date, self.end_date)
        
        # 3. Main Event Loop
        logger.info("Starting event loop...")
        event_count = 0
        timestamp = self.start_date # 修复：[FIX-10] 初始化 timestamp
        try:
            # The DataIterator yields (timestamp, [events_at_this_timestamp])
            async for timestamp, events_batch in self.data_iterator:
                
                # A. Update the Orchestrator's market state with the latest data
                # This includes both market data (prices) and news events
                await self.orchestrator.update_state_from_batch(timestamp, events_batch)
                
                # B. Trigger the cognitive pipeline for each *new* market event
                # (Price updates usually don't trigger the full AI pipeline,
                # but MarketEvents (news) do.)
                news_events = [e for e in events_batch if e.get('type') == 'MarketEvent']
                
                for event_data in news_events:
                    # The orchestrator processes the event, runs the AI,
                    # generates a signal, and applies trades.
                    # This is the core "tick" of the strategy.
                    await self.orchestrator.process_event_and_execute(event_data)
                    event_count += 1

                # C. Mark-to-market at the end of the time step
                # (e.g., end of day)
                # We need to get the latest prices from the orchestrator's state
                latest_prices = self.orchestrator.get_latest_prices()
                self.trade_lifecycle_manager.mark_to_market(timestamp, latest_prices)

                if event_count % 100 == 0 and event_count > 0:
                    logger.info(f"Processed {event_count} events. Current sim time: {timestamp}")

        except Exception as e:
            logger.error(f"Backtest loop failed at {timestamp}: {e}", exc_info=True)
            if hasattr(self.orchestrator, 'audit_manager'):
                self.orchestrator.audit_manager.log_system_error(e, "backtest_loop")
            else:
                logger.error("Orchestrator has no audit_manager. Cannot log system error.")

        finally:
            logger.info(f"--- Backtest Run Finished ---")
            logger.info(f"Total events processed: {event_count}")
            
            # 4. Generate Performance Report
            self.generate_report()
            
    # [任务 A.2] 实现新的 DRL 回测方法
    def run_backtest(
        self,
        data_manager: DataManager,
        feature_store: FeatureStore,
        alpha_agent: AlphaAgent,
        risk_agent: RiskAgent,
        execution_agent: ExecutionAgent,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 100000.0
    ) -> Dict[str, Any]:
        """
        [任务 A.2]
        运行一个用于评估 L3 DRL 智能体的向量化回测循环。
        
        注意：这与上面的 'async def run' 不同，后者是用于事件驱动的
        L1/L2 认知回测。这个方法是为 L3 DRL 评估而设计的。
        """
        logger.info(f"--- Starting DRL Agent Backtest for {symbol} ---")
        
        try:
            # 1. & 2. 加载 L1 (价格) 和 L2 (营养) 数据
            logger.info("Loading L1/L2 data...")
            merged_df = load_nutrition_data(
                data_manager, feature_store, symbol, start_date, end_date
            )
            
            if merged_df.empty:
                logger.error("No data found for the specified period. Aborting backtest.")
                return {"error": "No data."}

            # 3. 初始化状态
            balance = initial_balance
            holdings = 0.0
            portfolio_values = []
            trade_cost_pct = 0.001 # 模拟交易成本 (滑点+佣金)

            logger.info(f"Data loaded. Starting time loop ({len(merged_df)} steps)...")
            
            # 4. 实现时间循环
            for timestamp, row in merged_df.iterrows():
                
                price = row['price']
                sentiment = row['l2_sentiment']
                confidence = row['l2_confidence']
                
                # a. 构建 5-dim 状态 (Obs)
                # [balance, shares_held, price, l2_sentiment, l2_confidence]
                # 标准化状态以获得更好的智能体性能
                norm_balance = (balance / initial_balance) - 1.0
                norm_holdings = holdings * price / initial_balance # 持仓占总初始资本的百分比
                norm_price = (price / merged_df['price'].iloc[0]) - 1.0 # 相对于初始价格
                
                obs = np.array([norm_balance, norm_holdings, norm_price, sentiment, confidence], dtype=np.float32)

                # b. 智能体决策
                alpha_action = alpha_agent.compute_action(obs)
                risk_action = risk_agent.compute_action(obs)
                exec_action = execution_agent.compute_action(obs) # (暂时未使用)
                
                # c. 模拟执行
                
                # 假设 alpha_action[0] 是目标 *权重* (-1.0 到 1.0)
                target_weight = np.clip(alpha_action[0], -1.0, 1.0)
                # 假设 risk_action[0] 是风险 *标量* (0.0 到 1.0)
                risk_scalar = np.clip(risk_action[0], 0.0, 1.0)
                
                final_target_weight = target_weight * risk_scalar
                
                # 计算当前投资组合价值
                current_value = balance + (holdings * price)
                
                # 计算目标持仓
                target_dollar_value = current_value * final_target_weight
                target_holdings = target_dollar_value / price
                
                # 计算交易量
                trade_qty = target_holdings - holdings
                
                # 模拟交易成本和执行
                if abs(trade_qty) > 1e-6: # 如果需要交易
                    trade_cost = abs(trade_qty) * price * trade_cost_pct
                    balance -= trade_cost # 扣除成本
                    balance -= trade_qty * price # 现金变动
                    holdings += trade_qty # 持仓变动
                
                # d. 记录指标 (记录交易后的当前价值)
                portfolio_values.append(balance + (holdings * price))

            logger.info("Time loop complete. Calculating metrics...")
            
            # 5. 计算结果
            if not portfolio_values:
                logger.warning("Backtest completed but no portfolio values were recorded.")
                return {"error": "No portfolio values."}

            results_df = pd.Series(portfolio_values, index=merged_df.index)
            returns = results_df.pct_change().dropna()
            
            total_return = (results_df.iloc[-1] / results_df.iloc[0]) - 1
            
            if returns.std() == 0 or np.isnan(returns.std()):
                sharpe_ratio = 0.0
            else:
                # 假设日度数据，年化
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
            
            # 计算最大回撤
            cumulative_max = results_df.cummax()
            drawdown = (results_df - cumulative_max) / cumulative_max
            max_drawdown = drawdown.min()

            results = {
                "total_return_pct": total_return * 100,
                "annualized_sharpe_ratio": sharpe_ratio,
                "max_drawdown_pct": max_drawdown * 100,
                "final_portfolio_value": results_df.iloc[-1]
            }
            
            logger.info(f"DRL Backtest Complete. Results: {results}")
            return results

        except Exception as e:
            logger.error(f"DRL Backtest failed: {e}", exc_info=True)
            return {"error": str(e)}


    def generate_report(self):
        """
        Generates and saves the final performance report.
        """
        logger.info("Generating performance report...")
        try:
            pnl_history = self.trade_lifecycle_manager.get_pnl_history()
            trade_log = self.trade_lifecycle_manager.get_trade_log()
            
            if pnl_history.empty:
                logger.warning("No PnL history found. Cannot generate report.")
                return

            # 修复：[FIX-15] 直接调用函数，
            # 而不是 'generate_html_report' 方法
            report_html = self.report_renderer(
                pnl_history=pnl_history,
                trade_log=trade_log,
                config=self.config
            )
            
            report_path = self.backtest_config.get('report_output_path', 'logs/backtest_report.html')
            # 确保目录存在
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_html)
                
            logger.info(f"Performance report saved to: {report_path}")

        except Exception as e:
            logger.error(f"Failed to generate report: {e}", exc_info=True)
