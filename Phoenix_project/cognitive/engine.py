# cognitive/engine.py
import logging
import pandas as pd
from datetime import date
from typing import List, Dict, Optional, Any

from phoenix_project import StrategyConfig, PositionSizerConfig
from sizing.base import IPositionSizer
from sizing.fixed_fraction import FixedFractionSizer
from sizing.volatility_parity import VolatilityParitySizer
from .risk_manager import RiskManager
from .portfolio_constructor import PortfolioConstructor
# L5: Import schemas needed for signal integration
from schemas.fusion_result import FusionResult

class CognitiveEngine:
    def __init__(self,
                 config: StrategyConfig,
                 l3_rules: List[Dict[str, Any]],
                 asset_analysis_data: Optional[Dict[date, Dict]] = None,
                 sentiment_data: Optional[Dict[date, float]] = None,
                 ai_mode: str = "processed"):
        self.config = config
        self.l3_rules = l3_rules or []
        self.logger = logging.getLogger("PhoenixProject.CognitiveEngine")
        self.risk_manager = RiskManager(config)
        self.portfolio_constructor = PortfolioConstructor(config, asset_analysis_data, mode=config.ai_mode)
        self.position_sizer = self._create_sizer(config.position_sizer)
        if self.l3_rules:
            self.logger.info(f"Loaded {len(self.l3_rules)} L3 heuristic rules into the CognitiveEngine.")

    def _apply_l3_rules(self,
                          worthy_targets: List[Dict],
                          effective_max_allocation: float) -> (List[Dict], float):
        """
        应用已加载的 L3 启发式规则来修改目标、分数或策略边界。
        (Task 2.3 - Rule Application)
        """
        if not self.l3_rules:
            return worthy_targets, effective_max_allocation
        
        self.logger.debug(f"Applying {len(self.l3_rules)} L3 rules...")
        
        # 这是一个复杂规则引擎的占位符。
        # 一个真实的实现会解析规则条件 (例如, "IF market_is_volatile")
        # 并应用行动 (例如, "THEN reduce max_allocation by 20%").
        # 目前, 我们只记录并返回未修改的内容。
        
        self.logger.info("L3 rules application placeholder: No modifications made.")
        return worthy_targets, effective_max_allocation

    def _create_sizer(self, sizer_config: PositionSizerConfig) -> IPositionSizer:
        method = sizer_config.method
        params = sizer_config.parameters
        self.logger.info(f"Initializing position sizer: '{method}' with params: {params}")
        if method == "fixed_fraction":
            return FixedFractionSizer(**params)
        elif method == "volatility_parity":
            return VolatilityParitySizer(**params)
        else:
            raise ValueError(f"Unknown position sizer method: {method}")

    def determine_allocations(self,
                              candidate_analysis: List[Dict],
                              current_date: date,
                              total_portfolio_value: float,
                              historical_returns: Optional[pd.DataFrame] = None,
                              adv_data: Optional[Dict[str, float]] = None,
                              emergency_factor: Optional[float] = None,
                              rl_output: Optional[Any] = None) -> List[Dict[str, Any]]: # (L5) Added rl_output
        self.logger.info("--- [Cognitive Engine Call: Marshal Coordination] ---")
        
        # --- 紧急覆盖 ---
        if emergency_factor is not None:
            self.logger.warning(f"EMERGENCY FACTOR '{emergency_factor}' ACTIVATED. Overriding standard logic.")
            battle_plan = self.position_sizer.emergency_resize(emergency_factor)
            return battle_plan

        # [NEW] 1. 从有价值的目标中计算当日的平均认知不确定性
        worthy_targets = self.portfolio_constructor.identify_opportunities(candidate_analysis, current_date)
        
        daily_uncertainty = 0.0
        if worthy_targets and self.config.ai_mode != 'off':
            daily_asset_analysis = self.portfolio_constructor.asset_analysis_data.get(current_date, {})
            uncertainties = [daily_asset_analysis.get(t['ticker'], {}).get('final_conclusion', {}).get('posterior_variance', 0.0) for t in worthy_targets]
            valid_uncertainties = [u for u in uncertainties if u is not None]
            if valid_uncertainties:
                daily_uncertainty = sum(valid_uncertainties) / len(valid_uncertainties)

        # 2. 根据这种不确定性获取资本修正因子
        capital_modifier = self.risk_manager.get_capital_modifier(daily_uncertainty)
        effective_max_allocation = self.config.max_total_allocation * capital_modifier
        
        # 3. [NEW] 应用 L3 启发式规则 (Task 2.3)
        # 规则可以修改目标或资本策略边界
        worthy_targets, effective_max_allocation = self._apply_l3_rules(worthy_targets, effective_max_allocation)
        
        # 3. [V2.0+] 使用专门的仓位大小器确定资本分配
        # 这正确地将 "什么" (worthy_targets) 与 "多少" (sizer) 分开。
        initial_battle_plan = self.position_sizer.size_positions(worthy_targets, effective_max_allocation)

        # 4. [V2.0+] 首先应用流动性约束
        liquidity_constrained_plan = self.risk_manager.apply_liquidity_constraints(
            initial_battle_plan, adv_data or {}, total_portfolio_value
        )

        # 5. (L5 Task 3) Integrate DRL signal if provided
        if rl_output is not None:
            battle_plan = self.integrate_rl_signal(liquidity_constrained_plan, rl_output)
        else:
            battle_plan = liquidity_constrained_plan

        # 5. [V2.0+] 在流动性调整后的计划上强制执行 CVaR 约束
        if self.risk_manager.risk_config.cvar_enabled and historical_returns is not None and battle_plan:
            portfolio_weights = {p['ticker']: p['capital_allocation_pct'] for p in battle_plan}
            portfolio_cvar = self.risk_manager.calculate_portfolio_cvar(portfolio_weights, historical_returns)

            if portfolio_cvar is not None and portfolio_cvar > self.risk_manager.risk_config.cvar_max_threshold:
                self.logger.warning(
                    f"Portfolio CVaR ({portfolio_cvar:.2%}) exceeds threshold "
                    f"({self.risk_manager.risk_config.cvar_max_threshold:.2%}). Scaling down."
                )
                # 简单缩放：按 CVaR 超出部分的比例减少分配
                scale_factor = self.risk_manager.risk_config.cvar_max_threshold / portfolio_cvar
                for position in battle_plan:
                    position['capital_allocation_pct'] *= scale_factor
                
                self.logger.info(f"Scaled down allocations by a factor of {scale_factor:.2f}.")

        self.logger.info("--- [Cognitive Engine Call: Concluded] ---")
        return battle_plan

    def calculate_opportunity_score(self, current_price: float, sma: float, rsi: float) -> float:
        """
        根据技术指标计算专有的 '机会分数'。
        分数被归一化到 0 到 100 之间。
        """
        # 成分 1: 趋势 (价格 vs. SMA)
        # 我们希望价格在 SMA 之上，但又不要太远。
        price_vs_sma = (current_price - sma) / sma
        # 使用类高斯函数来奖励略高于 SMA
        trend_score = 100 * (1 - abs(price_vs_sma - 0.05)) # 在高于 SMA 5% 处达到峰值
        trend_score = max(0, trend_score) # 确保非负

        # 成分 2: 动量/均值回归 (RSI)
        # 我们希望在 RSI 未超买时买入。
        rsi_score = 100 - rsi if rsi > self.config.rsi_overbought_threshold else 100
        
        # 组合分数。给趋势更多权重。
        final_score = (0.6 * trend_score) + (0.4 * rsi_score)
        return max(0, min(100, final_score))

    def integrate_rl_signal(self,
                              bayesian_target_weights: List[Dict],
                              rl_output: Any) -> List[Dict[str, Any]]:
        """
        (L5 Task 3) Combines the Bayesian-driven target weights from the position sizer
        with the policy output from the DRL agent.

        Args:
            bayesian_target_weights: The list of allocations from determine_allocations.
            rl_output: The raw output from the DRL agent (e.g., a single float or dict).

        Returns:
            A final, mixed list of allocation decisions.
        """
        self.logger.info(f"Integrating DRL signal ({rl_output}) with Bayesian plan...")

        # Placeholder Logic:
        # A simple merge could be to use the RL output as a "scaling factor"
        # on the Bayesian-derived weights.
        # e.g., if rl_output is a float from [0, 1] representing "risk-on".
        try:
            rl_scaling_factor = float(rl_output) # Assuming rl_output is a simple scalar
            mixed_plan = bayesian_target_weights.copy()
            for position in mixed_plan:
                position['capital_allocation_pct'] *= rl_scaling_factor
            return mixed_plan
        except Exception as e:
            self.logger.error(f"Failed to integrate DRL signal: {e}. Returning original Bayesian plan.")
            return bayesian_target_weights
