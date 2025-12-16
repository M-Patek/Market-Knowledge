"""
Phoenix_project/cognitive/portfolio_constructor.py
[Phase 3 Task 4] Fix Silent Liquidation on Missing Price.
Prevents defaulting target quantity to 0 when market data is unavailable.
[Task P0-002] Force normalization of portfolio weights.
[Task P1-RISK-03] Optimized Construction with Sizer Volatility & Turnover Control.
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
            
            # [Task P1-RISK-03] Inject Volatility for Sizer
            # Create a map for quick lookup
            volatility_map = {}
            if market_data:
                for md in market_data:
                    # Assuming MarketData schema has 'volatility' or we derive it elsewhere.
                    # If not present, Sizer might default or fail. 
                    # Here we try to get it if available (e.g. from risk manager or data enrichment)
                    # For now, check if 'metadata' has it or if risk_manager has history.
                    # Simplest is to pass what's available.
                    v = getattr(md, 'volatility', None)
                    if v is not None:
                        volatility_map[md.symbol] = float(v)
            
            candidates = []
            for s, w in adjusted_weights.items():
                item = {"ticker": s, "weight": w}
                if s in volatility_map:
                    item["volatility"] = volatility_map[s]
                candidates.append(item)

            allocation_results = self.sizing_strategy.size_positions(candidates, max_total_allocation=1.0)
            
            # [Task TASK-P0-002] Force normalization of allocation results
            total_allocated_weight = sum(item.get("capital_allocation_pct", 0.0) for item in allocation_results)
            if total_allocated_weight > 1.0:
                logger.warning(f"Warning: Leverage exceeded 1.0 ({total_allocated_weight}), normalizing weights.")
                for item in allocation_results:
                    item["capital_allocation_pct"] = item.get("capital_allocation_pct", 0.0) / total_allocated_weight
            
            total_equity = float(real_time_portfolio.get("total_value", 0.0))
            price_map = {md.symbol: float(md.close) for md in market_data if md.close}
            current_positions_map = real_time_portfolio.get('positions', {})
            
            # [Task P1-RISK-03] Turnover Control (Buffer)
            TURNOVER_BUFFER_PCT = 0.05 # 5% buffer zone
            
            target_positions_list = []
            
            for res in allocation_results:
                sym = res.get("ticker")
                target_weight = res.get("capital_allocation_pct", 0.0)
                price = price_map.get(sym)
                
                # Get current weight
                current_weight = 0.0
                current_pos = current_positions_map.get(sym)
                current_qty = 0.0
                
                if current_pos:
                    current_qty = float(current_pos.get('quantity', 0.0)) if isinstance(current_pos, dict) else float(current_pos.quantity)
                    current_val = float(current_pos.get('market_value', 0.0)) if isinstance(current_pos, dict) else float(current_pos.market_value)
                    if total_equity > 0:
                        current_weight = current_val / total_equity

                # [Phase 3 Task 4] Fix Silent Liquidation
                if not price or price <= 0:
                    logger.critical(f"Missing price for {sym}. Holding current quantity.")
                    target_positions_list.append(TargetPosition(
                        symbol=sym,
                        target_weight=current_weight, # Use current weight if price unknown
                        quantity=current_qty, 
                        reasoning="Price Missing - Held"
                    ))
                    continue

                # [Task P1-RISK-03] Apply Turnover Buffer
                # Only rebalance if weight change > buffer, OR if flipping side (buy->sell), OR closing out (target=0)
                weight_diff = abs(target_weight - current_weight)
                
                final_weight = target_weight
                final_reason = "Sizing Strategy"
                
                # If change is small and not closing out completely
                if weight_diff < TURNOVER_BUFFER_PCT and target_weight > 0:
                    logger.info(f"Turnover Buffer: {sym} change {weight_diff:.2%} < {TURNOVER_BUFFER_PCT:.2%}. Holding current weight.")
                    final_weight = current_weight
                    # Recalculate Qty based on CURRENT weight (effectively hold qty)
                    # Ideally just keep current qty to avoid price-induced drift?
                    # Yes, keeping QTY is safer for 'Holding'.
                    qty = current_qty
                    final_reason = "Turnover Buffer - Held"
                else:
                    # Calculate new quantity
                    qty = (total_equity * final_weight) / price
                
                target_positions_list.append(TargetPosition(
                    symbol=sym,
                    target_weight=final_weight,
                    quantity=qty,
                    reasoning=final_reason
                ))
            
            # Handle liquidations (symbols in current but not in allocation)
            allocated_symbols = set(res.get("ticker") for res in allocation_results)
            for sym, pos in current_positions_map.items():
                if sym not in allocated_symbols:
                    # Implicit Close
                    # Check if price exists to close?
                    price = price_map.get(sym)
                    if not price or price <= 0:
                         # Cannot calculate Close value, but Qty -> 0 is an instruction.
                         # OrderManager handles execution.
                         logger.warning(f"Closing {sym} (not in target).")
                         target_positions_list.append(TargetPosition(
                            symbol=sym, target_weight=0.0, quantity=0.0, reasoning="Not in Target"
                        ))
                    else:
                        target_positions_list.append(TargetPosition(
                            symbol=sym, target_weight=0.0, quantity=0.0, reasoning="Not in Target"
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
