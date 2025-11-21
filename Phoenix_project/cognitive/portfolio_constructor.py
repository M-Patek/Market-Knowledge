# Phoenix_project/cognitive/portfolio_constructor.py
# [主人喵的修复 11.11] 实现了 FIXME (TBD 风险管理逻辑)。
# [主人喵的修复 11.12] 实现了 TBD (从 DataManager 加载初始投资组合)。
# [Code Opt Expert Fix] Task 3: Clean Architecture / Remove ContextBus Communication

import logging
from omegaconf import DictConfig
from cognitive.risk_manager import RiskManager
from Phoenix_project.sizing.base import IPositionSizer
from core.pipeline_state import PipelineState
from core.schemas.risk_schema import RiskReport
from Phoenix_project.core.schemas.data_schema import TargetPortfolio, TargetPosition
from context_bus import ContextBus

logger = logging.getLogger(__name__)

class PortfolioConstructor:
    """
    [已实现]
    负责根据 L3 智能体的 Alpha 信号，
    结合 RiskManager 的约束和 SizingStrategy，
    计算出目标投资组合。
    """

    def __init__(self, config: DictConfig, context_bus: ContextBus, risk_manager: RiskManager, sizing_strategy: IPositionSizer, data_manager):
        self.config = config.get("portfolio_constructor", {})
        self.context_bus = context_bus
        self.risk_manager = risk_manager
        self.sizing_strategy = sizing_strategy
        self.data_manager = data_manager # [新] 需要 DataManager 来获取当前状态
        
        self.current_portfolio = self._load_initial_portfolio()
        logger.info("PortfolioConstructor initialized.")


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
            logger.debug(f"Sizing positions for adjusted weights: {adjusted_weights}")
            
            # [Task 3] Adapter: Convert weights dict to Sizing candidates list
            # Note: Sizing strategies typically expect a list of dicts
            candidates = [{"ticker": symbol, "weight": weight} for symbol, weight in adjusted_weights.items()]
            
            # [Task 3] Call correct interface method: size_positions
            # Assuming max_total_allocation is 1.0 (100%) unless configured otherwise
            allocation_results = self.sizing_strategy.size_positions(
                candidates=candidates, 
                max_total_allocation=1.0
            )
            
            # [Task 3] Adapter: Convert Sizing results back to TargetPortfolio object
            # FIX: Use TargetPosition, not Position (which is for current holdings)
            target_positions_list = []
            for res in allocation_results:
                target_positions_list.append(TargetPosition(
                    symbol=res.get("ticker"),
                    target_weight=res.get("capital_allocation_pct", 0.0),
                    reasoning="Sizing Strategy Output"
                ))
            
            target_portfolio = TargetPortfolio(positions=target_positions_list, metadata={"source": "PortfolioConstructor"})
            logger.info(f"Target portfolio constructed with {len(target_positions_list)} positions.")
            
        except Exception as e:
            logger.error(f"SizingStrategy/Adapter failed: {e}. Aborting.", exc_info=True)
            return None

        return target_portfolio
