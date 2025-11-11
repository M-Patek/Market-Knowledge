# Phoenix_project/cognitive/portfolio_constructor.py
# [主人喵的修复 11.11] 实现了 FIXME (TBD 风险管理逻辑)。
# [主人喵的修复 11.12] 实现了 TBD (从 DataManager 加载初始投资组合)。

import logging
from omegaconf import DictConfig
from cognitive.risk_manager import RiskManager
from sizing.base import BaseSizingStrategy
from core.pipeline_state import PipelineState
from core.schemas.risk_schema import RiskReport
from context_bus import ContextBus

logger = logging.getLogger(__name__)

class PortfolioConstructor:
    """
    [已实现]
    负责根据 L3 智能体的 Alpha 信号，
    结合 RiskManager 的约束和 SizingStrategy，
    计算出目标投资组合。
    """

    def __init__(self, config: DictConfig, context_bus: ContextBus, risk_manager: RiskManager, sizing_strategy: BaseSizingStrategy, data_manager):
        self.config = config.get("portfolio_constructor", {})
        self.context_bus = context_bus
        self.risk_manager = risk_manager
        self.sizing_strategy = sizing_strategy
        self.data_manager = data_manager # [新] 需要 DataManager 来获取当前状态
        
        self.current_portfolio = self._load_initial_portfolio()
        
        # [新] 订阅投资组合更新 (来自 OrderManager) 以保持状态同步
        self.context_bus.subscribe("PORTFOLIO_UPDATE", self._handle_portfolio_update)
        logger.info("PortfolioConstructor initialized and subscribed to PORTFOLIO_UPDATE.")


    def _load_initial_portfolio(self):
        """
        [主人喵的修复 11.12] 实现了 TBD，从 DataManager 加载初始投资组合。
        """
        logger.info("Loading initial portfolio from DataManager...")
        try:
            # [TBD 已修复]
            # 假设 DataManager 有一个方法可以返回当前持仓
            # (注意：这需要 DataManager 能够访问持久化状态，例如通过 OrderManager 或快照)
            current_portfolio = self.data_manager.get_current_portfolio() 
            
            if current_portfolio and "positions" in current_portfolio and "cash" in current_portfolio:
                logger.info(f"Loaded initial portfolio from DataManager: {len(current_portfolio['positions'])} positions.")
                return current_portfolio
            else:
                logger.warning("DataManager returned no initial portfolio or data was malformed. Starting with empty.")
                return {"cash": self.config.get("initial_cash", 1_000_000), "positions": {}}
        except Exception as e:
            logger.error(f"Failed to load initial portfolio from DataManager: {e}. Starting with empty.", exc_info=True)
            return {"cash": self.config.get("initial_cash", 1_000_000), "positions": {}}

    def _handle_portfolio_update(self, portfolio_data: dict):
        """
        [新] 回调函数，用于在订单执行后更新内部的 'current_portfolio' 状态。
        """
        logger.debug(f"Received PORTFOLIO_UPDATE. Updating internal state.")
        self.current_portfolio = portfolio_data

    def construct_portfolio(self, pipeline_state: PipelineState):
        """
        [已实现] 
        根据 L3 Alpha 信号和风险约束生成目标投资组合。
        """
        logger.info("Starting portfolio construction...")
        
        # 1. 获取 L3 Alpha 信号 (目标权重)
        alpha_signals = pipeline_state.l3_alpha_signal
        if not alpha_signals:
            logger.warning("No L3 alpha signals (target weights) found in pipeline_state. No construction possible.")
            # (我们仍然发布一个空的目标组合吗？还是什么都不做？)
            # (最好是什么都不做)
            return None

        # 2. 获取当前市场数据 (用于风险评估和规模计算)
        market_data = pipeline_state.market_data_batch 
        if not market_data:
            logger.warning("No market data batch found in pipeline_state. Cannot construct portfolio.")
            return None
        
        # 3. [FIXME 已修复]
        # (将提议的权重传递给 RiskManager 并获取调整)
        
        target_weights = alpha_signals # 初始提议
        adjusted_weights = target_weights
        risk_report = None
        
        try:
            logger.debug(f"Evaluating target weights with RiskManager: {target_weights}")
            # [实现] risk_manager.evaluate 接受提议的权重、当前组合和市场数据
            risk_report = self.risk_manager.evaluate(
                target_weights, 
                self.current_portfolio, 
                market_data
            )
            
            if risk_report.adjustments_made:
                logger.info(f"RiskManager evaluation triggered adjustments. Reason: {risk_report.reason}")
                adjusted_weights = risk_report.adjusted_weights
            else:
                logger.info("RiskManager evaluation passed. Using original target weights.")
                adjusted_weights = risk_report.adjusted_weights # (即使没有调整，也使用报告中的版本)
            
        except Exception as e:
            logger.error(f"RiskManager evaluation failed: {e}. Falling back to zero weights.", exc_info=True)
            # [安全模式] 风险管理失败，清空所有目标头寸
            adjusted_weights = {asset: 0.0 for asset in target_weights}
            risk_report = RiskReport(
                passed=False, 
                adjustments_made=True, 
                adjusted_weights=adjusted_weights, 
                reason=f"RiskManager FAILED: {str(e)}"
            )

        # 4. [已实现] 计算目标投资组合 (从权重到股数/规模)
        
        try:
            logger.debug(f"Calculating target portfolio shares with SizingStrategy for weights: {adjusted_weights}")
            target_portfolio = self.sizing_strategy.calculate_target_portfolio(
                adjusted_weights, 
                self.current_portfolio,
                market_data,
                self.data_manager # (Sizing 可能需要访问最新价格)
            )
            
            logger.info(f"Target portfolio constructed: {target_portfolio.get('positions')}")
            
        except Exception as e:
            logger.error(f"SizingStrategy failed: {e}. Aborting portfolio construction.", exc_info=True)
            return None

        
        # 5. [已实现] 将最终的目标投资组合发布到 ContextBus
        # OrderManager 应该订阅这个
        self.context_bus.publish("TARGET_PORTFOLIO", {
            "target": target_portfolio,
            "risk_report": risk_report.to_dict() if risk_report else None,
            "alpha_signal": alpha_signals
        })
        
        return target_portfolio
