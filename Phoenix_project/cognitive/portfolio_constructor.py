# Phoenix_project/cognitive/portfolio_constructor.py
# [主人喵的修复 11.11] 实现了 FIXME (TBD 风险管理逻辑)。
# [主人喵的修复 11.12] 实现了 TBD (从 DataManager 加载初始投资组合)。
# [Code Opt Expert Fix] Task 3: Clean Architecture / Remove ContextBus Communication
# [Phase III Fix] Risk Gating & Signal Mapping

import logging
from omegaconf import DictConfig
from Phoenix_project.cognitive.risk_manager import RiskManager
from Phoenix_project.sizing.base import IPositionSizer
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.risk_schema import RiskReport
from Phoenix_project.core.schemas.data_schema import TargetPortfolio, TargetPosition
from Phoenix_project.context_bus import ContextBus

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
                # [Safety Fix] Phantom Initialization Prevention
                err_msg = "DataManager returned no initial portfolio or data was malformed. Cannot safely start."
                logger.critical(err_msg)
                raise RuntimeError(err_msg)
        except Exception as e:
            # [Safety Fix] Prevent starting with phantom zero positions if data service is down
            logger.critical(f"Failed to load initial portfolio from DataManager: {e}. System HALT.", exc_info=True)
            raise RuntimeError(f"Critical Data Failure: {e}")

    async def construct_portfolio(self, pipeline_state: PipelineState):
        """
        [已实现] 
        根据 L3 Alpha 信号和风险约束生成目标投资组合。
        """
        logger.info("Starting portfolio construction...")
        
        # [Phase 3 Fix] Risk Gate: Check L3 Risk Decision
        # 如果 L3 Risk Agent 发出熔断指令，立即中止
        if pipeline_state.l3_decision:
            risk_action = pipeline_state.l3_decision.get("risk_action", "CONTINUE")
            if risk_action == "HALT_TRADING":
                logger.warning("Portfolio Construction ABORTED by L3 Risk Agent (HALT_TRADING).")
                return None
        
        # 1. 获取 L3 Alpha 信号 (目标权重)
        alpha_signals = pipeline_state.l3_alpha_signal
        
        if not alpha_signals:
            # [Phase 3 Fix] Signal Mapping: Try to extract from l3_decision
            # 如果没有明确的 alpha_signal 字典，尝试从 decision 中提取原始动作
            if pipeline_state.l3_decision and pipeline_state.l3_decision.get("alpha_action") is not None:
                l3_d = pipeline_state.l3_decision
                sym = l3_d.get("symbol")
                raw_act = l3_d.get("alpha_action")
                
                # Handle list or scalar
                # 通常 Alpha Agent 输出一个连续值 (权重)
                weight = raw_act[0] if isinstance(raw_act, list) and len(raw_act) > 0 else raw_act
                
                try:
                    # 将单一信号映射为 {Symbol: Weight}
                    alpha_signals = {sym: float(weight)}
                    logger.info(f"Mapped L3 Alpha Signal: {sym} -> {weight}")
                except (ValueError, TypeError):
                    logger.warning(f"Failed to map L3 alpha action: {raw_act}")

        if not alpha_signals:
            logger.warning("No L3 alpha signals (target weights) found. No construction possible.")
            return None

        # 2. 获取当前市场数据 (用于风险评估和规模计算)
        market_data = pipeline_state.market_data_batch 
        if not market_data:
            logger.warning("No market data batch found in pipeline_state. Cannot construct portfolio.")
            return None
        
        # 3. [Safety Fix] Validate Allocations (Async & Type Safe)
        target_weights = alpha_signals
        
        try:
            logger.debug(f"Validating target weights with RiskManager: {target_weights}")
            risk_report = await self.risk_manager.validate_allocations(
                target_weights,
                self.current_portfolio,
                market_data
            )
            
            if not risk_report.passed:
                logger.warning(f"Risk validation failed: {risk_report.adjustments_made}. HALTING construction.")
                return None

            # Pass-through if validated
            adjusted_weights = target_weights
            
        except Exception as e:
            logger.error(f"RiskManager validation error: {e}. Aborting construction.", exc_info=True)
            # [Safety Fix] Do NOT return zero weights (Liquidation Risk). Return None (Halt).
            return None

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
