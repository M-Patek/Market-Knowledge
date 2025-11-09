# controller/orchestrator.py
import asyncio
from typing import Dict, Any

from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.exceptions import CognitiveError, DataError
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.data_manager import DataManager
from Phoenix_project.cognitive.engine import CognitiveEngine
from Phoenix_project.cognitive.portfolio_constructor import PortfolioConstructor
from Phoenix_project.execution.order_manager import OrderManager

# [任务 2.2] 导入 L3 DRL 智能体
from Phoenix_project.agents.l3.alpha_agent import AlphaAgent
from Phoenix_project.agents.l3.risk_agent import RiskAgent
from Phoenix_project.agents.l3.execution_agent import ExecutionAgent

# [任务 2.3] 导入 L3 决策所需的数据 Schema
import numpy as np
from Phoenix_project.core.schemas.data_schema import TargetPortfolio, TargetPosition, PortfolioState
from Phoenix_project.core.schemas.fusion_result import FusionResult


class Orchestrator:
    """
    (Orchestrator 文档字符串...)
    
    [任务 2.2] 更新:
    现在注入了 L3 DRL 智能体 (Alpha, Risk, Exec) 并使用它们
    在 run_main_cycle 中替换旧的 PortfolioConstructor 决策逻辑。
    """
    def __init__(
        self,
        config: Dict[str, Any],
        data_manager: DataManager,
        cognitive_engine: CognitiveEngine,
        portfolio_constructor: PortfolioConstructor,
        order_manager: OrderManager,
        # [任务 2.2] 注入 L3 DRL 智能体
        alpha_agent: AlphaAgent,
        risk_agent: RiskAgent,
        execution_agent: ExecutionAgent,
        loop_manager: Any = None # (假设 LoopManager 存在)
    ):
        self.config = config
        self.data_manager = data_manager
        self.cognitive_engine = cognitive_engine
        self.order_manager = order_manager
        
        # [任务 2.3] 我们仍然保留 PC，以防它有其他辅助功能 (例如计算)，
        # 但它的 .construct() 决策逻辑将被 L3 替换。
        self.portfolio_constructor = portfolio_constructor 
        
        # [任务 2.2] 保存 L3 智能体
        self.alpha_agent = alpha_agent
        self.risk_agent = risk_agent
        self.execution_agent = execution_agent
        
        self.loop_manager = loop_manager
        self.logger = get_logger(self.__class__.__name__)

    async def run_main_cycle(self, state: PipelineState):
        """
        执行一个完整的“感知-认知-行动”周期。
        """
        self.logger.info(f"--- [Cycle {state.cycle_id}] Orchestrator Main Cycle START ---")
        
        try:
            # --- 1. 感知 (Perception) ---
            # (获取 L0/L1 数据, 例如市场数据, 新闻等)
            # (假设 data_manager.process_perception_cycle 更新了 state)
            
            # --- 2. L2 认知 (Cognition) ---
            # (运行 L2 智能体 (Fusion, Critic 等) 来生成 L2 分析)
            # (假设 cognitive_engine.process_cognitive_cycle 返回 L2 结果)
            
            # [任务 2.3] 确保 L2 结果 (fusion_results) 被保存到 state
            # (我们假设 L2 结果是一个字典: Dict[str, FusionResult])
            # self.cognitive_engine.process...
            # state.set_value("l2_fusion_results", l2_results_dict)
            
            # (模拟 L2 步骤 - 在实际代码中，这将由 cognitive_engine 完成)
            if not state.get_value("l2_fusion_results"):
                 self.logger.warning("L2 认知未运行，模拟 L2 结果...")
                 state.set_value("l2_fusion_results", {
                     "AAPL": FusionResult(symbol="AAPL", final_decision="BUY", confidence=0.8, sentiment_score=0.8, reasoning="Simulated L2 Buy"),
                     "GOOG": FusionResult(symbol="GOOG", final_decision="SELL", confidence=0.7, sentiment_score=-0.7, reasoning="Simulated L2 Sell")
                 })
            
            # --- 3. 获取当前投资组合状态 ---
            # (这对于 L3 DRL 智能体至关重要)
            self.logger.info("Step 3: Fetching current portfolio and market state...")
            current_portfolio_state = await self.order_manager.get_current_portfolio_state()
            state.set_value("current_portfolio_state", current_portfolio_state)
            
            # (模拟 L0 市场数据 - 在实际代码中，这将由 data_manager 完成)
            market_data = {"AAPL": 150.0, "GOOG": 175.0}
            state.set_value("current_market_data", market_data)

            # --- 4. L3 认知 (DRL 决策) [任务 2.3 核心替换] ---
            self.logger.info("Step 4: Engaging L3 DRL Agents for portfolio decision...")
            
            l2_fusion_results: Dict[str, FusionResult] = state.get_value("l2_fusion_results")

            target_positions = []
            
            if not all([current_portfolio_state, market_data, l2_fusion_results]):
                self.logger.error("L3 DRL: 缺少 投资组合/市场/L2 数据！跳过 L3 决策。")
                target_portfolio = TargetPortfolio(positions=[])
            
            else:
                # 遍历 L2 评估过的每个资产
                for symbol, fusion_result in l2_fusion_results.items():
                    if symbol not in market_data:
                        self.logger.warning(f"L3 DRL: 缺少 {symbol} 的市场数据，跳过。")
                        continue

                    # (准备特定于资产的 5-d 状态)
                    current_holding = current_portfolio_state.positions.get(symbol)
                    symbol_state_data = {
                        "balance": current_portfolio_state.cash,
                        "holdings": current_holding.quantity if current_holding else 0.0,
                        "price": market_data[symbol]
                    }

                    # 1. Alpha Agent: 决定目标 *权重* (例如 0.0 到 1.0)
                    # (我们假设 Alpha Agent 学会了 L2 'BUY' -> >0, L2 'SELL' -> 0.0)
                    alpha_obs = self.alpha_agent.format_observation(symbol_state_data, fusion_result)
                    alpha_action = self.alpha_agent.compute_action(alpha_obs)
                    target_weight = np.clip(alpha_action[0], 0.0, 1.0) # (假设 [0] 是权重)

                    # 2. Risk Agent: 决定风险调整 *标量* (例如 0.0 到 1.0)
                    risk_obs = self.risk_agent.format_observation(symbol_state_data, fusion_result)
                    risk_action = self.risk_agent.compute_action(risk_obs)
                    risk_scalar = np.clip(risk_action[0], 0.0, 1.0) # (假设 [0] 是缩放因子)

                    # 3. (Execution Agent: 暂时忽略, 假设 OM 处理执行)
                    
                    final_weight = target_weight * risk_scalar

                    if final_weight > 0.01: # (最小权重阈值)
                        target_positions.append(
                            TargetPosition(
                                symbol=symbol,
                                target_weight=final_weight,
                                reasoning=f"L2({fusion_result.final_decision}) | L3_Alpha({target_weight:.2f}) | L3_Risk({risk_scalar:.2f}) -> {final_weight:.2f}"
                            )
                        )
            
            # (创建最终的 TargetPortfolio 对象)
            target_portfolio = TargetPortfolio(positions=target_positions)
            
            self.logger.info(f"L3 DRL Agents constructed target portfolio.")
            state.set_value("target_portfolio", target_portfolio)

            # --- 5. L3 行动 (Execution) ---
            # (旧的 Step 5 (PC.construct) 已被上面的 Step 4 替换)
            # (我们现在进入执行阶段)
            self.logger.info("Step 5: Processing portfolio targets into orders...")
            
            # (OrderManager 现在使用 L3 DRL 智能体 (如果需要) 
            #  来执行目标投资组合)
            # (我们假设 ExecutionAgent 被注入到 OrderManager 中，
            #  或者 OrderManager 在这里调用它)
            
            # (暂时假设 OM 只需要 TargetPortfolio)
            await self.order_manager.process_target_portfolio(
                current_portfolio_state,
                target_portfolio,
                market_data
            )
            
            self.logger.info(f"--- [Cycle {state.cycle_id}] Orchestrator Main Cycle END ---")

        except CognitiveError as e:
            self.logger.error(f"Cycle {state.cycle_id} failed during Cognitive step: {e}")
            # (错误处理...)
        except DataError as e:
            self.logger.error(f"Cycle {state.cycle_id} failed during Data step: {e}")
            # (错误处理...)
        except Exception as e:
            self.logger.critical(f"Cycle {state.cycle_id} CRITICAL FAILURE: {e}", exc_info=True)
            # (关键错误处理...)
