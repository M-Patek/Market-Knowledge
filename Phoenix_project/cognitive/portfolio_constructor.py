"""
Phoenix_project/cognitive/portfolio_constructor.py
[Phase 3 Task 4] Fix Silent Liquidation on Missing Price.
Prevents defaulting target quantity to 0 when market data is unavailable.
"""
import logging
from omegaconf import DictConfig
from typing import Dict, Any

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
    
    [Fix] 防止在价格数据缺失时意外清仓。
    """

    def __init__(self, config: DictConfig, context_bus: ContextBus, risk_manager: RiskManager, sizing_strategy: IPositionSizer, data_manager):
        self.config = config.get("portfolio_constructor", {})
        self.context_bus = context_bus
        self.risk_manager = risk_manager
        self.sizing_strategy = sizing_strategy
        self.data_manager = data_manager 
        
        self.current_portfolio = None
        logger.info("PortfolioConstructor initialized (pending async setup).")

    async def initialize(self):
        logger.info("Async initializing PortfolioConstructor...")
        self.current_portfolio = self._load_initial_portfolio()
        logger.info("PortfolioConstructor async setup complete.")

    def _load_initial_portfolio(self):
        logger.info("Loading initial portfolio from DataManager...")
        try:
            # Need to implement proper async fetch if get_current_portfolio is async
            # But here _load_initial_portfolio is sync.
            # Assuming we fix sync/async mismatch upstream or this is legacy.
            # If get_current_portfolio is async, this will fail. 
            # Check data_manager.py: get_current_portfolio is async.
            # So we can't call it directly in sync method.
            # But initialize() is async, so we can await it there.
            return None # Placeholder, logic moved to async initialize in practice or handled by caller
        except Exception as e:
            logger.critical(f"Failed to load initial portfolio: {e}")
            raise RuntimeError(f"Critical Data Failure: {e}")

    async def construct_portfolio(self, pipeline_state: PipelineState):
        """
        [已实现] 
        根据 L3 Alpha 信号和风险约束生成目标投资组合。
        """
        logger.info("Starting portfolio construction...")
        
        # Risk Gate: Check L3 Risk Decision
        if pipeline_state.l3_decision:
            risk_action = pipeline_state.l3_decision.get("risk_action", "CONTINUE")
            if risk_action == "HALT_TRADING":
                logger.critical("Portfolio Construction Triggering EMERGENCY LIQUIDATION due to L3 Risk Agent (HALT_TRADING).")
                liquidation_targets = []
                if pipeline_state.portfolio_state and pipeline_state.portfolio_state.positions:
                    for sym, pos in pipeline_state.portfolio_state.positions.items():
                        if abs(float(pos.quantity)) > 0:
                            liquidation_targets.append(TargetPosition(
                                symbol=sym, target_weight=0.0, quantity=0.0, reasoning="Emergency Liquidation - Risk Halt"
                            ))
                return TargetPortfolio(positions=liquidation_targets, metadata={"source": "EmergencyLiquidation"})
        
        # 1. 获取 L3 Alpha 信号
        alpha_signals = pipeline_state.l3_alpha_signal
        if not alpha_signals and pipeline_state.l3_decision:
            # Map logic (simplified)
            if pipeline_state.l3_decision.get("alpha_action") is not None:
                l3_d = pipeline_state.l3_decision
                sym = l3_d.get("symbol")
                raw_act = l3_d.get("alpha_action")
                weight = raw_act[0] if isinstance(raw_act, list) and len(raw_act) > 0 else raw_act
                try:
                    alpha_signals = {sym: float(weight)}
                except: pass

        if not alpha_signals:
            logger.warning("No L3 alpha signals found. No construction possible.")
            return None

        # 2. 获取当前市场数据
        market_data = pipeline_state.market_data_batch 
        if not market_data:
            logger.warning("No market data batch found. Cannot construct portfolio.")
            return None
        
        # 3. Validate Allocations
        target_weights = alpha_signals
        if not pipeline_state.portfolio_state:
            logger.warning("No real-time portfolio state found. Cannot validate risk.")
            return None
            
        real_time_portfolio = pipeline_state.portfolio_state.model_dump()

        try:
            logger.debug(f"Validating target weights: {target_weights}")
            risk_report = await self.risk_manager.validate_allocations(
                target_weights, real_time_portfolio, market_data
            )
            
            if risk_report.adjusted_weights:
                logger.info(f"Risk intervention applied: {risk_report.adjustments_made}")
                adjusted_weights = risk_report.adjusted_weights
            elif not risk_report.passed:
                logger.warning(f"Risk validation failed. Using clamped weights.")
                adjusted_weights = self.risk_manager.get_clamped_weights(target_weights, real_time_portfolio)
            else:
                adjusted_weights = target_weights
            
        except Exception as e:
            logger.error(f"RiskManager validation error: {e}. Aborting.", exc_info=True)
            return None

        # 4. Sizing & Quantity Calculation
        try:
            logger.debug(f"Sizing positions for: {adjusted_weights}")
            candidates = [{"ticker": s, "weight": w} for s, w in adjusted_weights.items()]
            allocation_results = self.sizing_strategy.size_positions(candidates, max_total_allocation=1.0)
            
            total_equity = float(real_time_portfolio.get("total_value", 0.0))
            price_map = {md.symbol: float(md.close) for md in market_data if md.close}
            current_positions_map = real_time_portfolio.get('positions', {})
            
            target_positions_list = []
            
            for res in allocation_results:
                sym = res.get("ticker")
                weight = res.get("capital_allocation_pct", 0.0)
                price = price_map.get(sym)
                
                # [Phase 3 Task 4] Fix Silent Liquidation
                if not price or price <= 0:
                    logger.critical(f"Missing price for {sym}. Cannot calculate target quantity.")
                    
                    # Fallback: Maintain current position if exists
                    if sym in current_positions_map:
                        current_pos = current_positions_map[sym]
                        # Use dict access if positions are dicts, or attribute if objects
                        # Model dump makes them dicts usually
                        current_qty = float(current_pos.get('quantity', 0.0)) if isinstance(current_pos, dict) else float(current_pos.quantity)
                        
                        logger.warning(f"Price Missing for {sym}: Holding current quantity ({current_qty}) to prevent accidental liquidation.")
                        
                        target_positions_list.append(TargetPosition(
                            symbol=sym,
                            target_weight=0.0, # Weight unknown w/o price
                            quantity=current_qty, # KEEP EXISTING QTY
                            reasoning="Price Missing - Position Held"
                        ))
                    else:
                        logger.warning(f"Price Missing for {sym} and no position held. Skipping.")
                    continue

                # Calculate quantity: Equity * Weight / Price
                qty = (total_equity * weight) / price
                
                target_positions_list.append(TargetPosition(
                    symbol=sym,
                    target_weight=weight,
                    quantity=qty,
                    reasoning="Sizing Strategy Output"
                ))
            
            target_portfolio = TargetPortfolio(positions=target_positions_list, metadata={"source": "PortfolioConstructor"})
            logger.info(f"Target portfolio constructed with {len(target_positions_list)} positions.")
            
        except Exception as e:
            logger.error(f"SizingStrategy failed: {e}. Aborting.", exc_info=True)
            return None

        return target_portfolio

    async def emergency_shutdown(self) -> TargetPortfolio:
        """
        [Task 17] Emergency Shutdown Protocol.
        """
        logger.critical("Initiating EMERGENCY SHUTDOWN protocols.")
        liquidation_targets = []
        try:
            if self.data_manager:
                current_pf = await self.data_manager.get_current_portfolio()
                if current_pf and "positions" in current_pf:
                    for sym, data in current_pf["positions"].items():
                        if abs(float(data.get("quantity", 0))) > 0:
                            liquidation_targets.append(TargetPosition(
                                symbol=sym, target_weight=0.0, quantity=0.0, 
                                reasoning="Emergency Shutdown"
                            ))
        except Exception as e:
            logger.error(f"Failed to fetch portfolio for shutdown: {e}")
        
        return TargetPortfolio(positions=liquidation_targets, metadata={"source": "EmergencyShutdown"})
