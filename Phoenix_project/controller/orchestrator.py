import logging
from typing import Dict, Any, Optional
import asyncio

from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.context_bus import ContextBus
from Phoenix_project.controller.loop_manager import LoopManager
from Phoenix_project.cognitive.engine import CognitiveEngine
from Phoenix_project.execution.order_manager import OrderManager
from Phoenix_project.cognitive.portfolio_constructor import PortfolioConstructor
from Phoenix_project.agents.l3.alpha_agent import AlphaAgent
from Phoenix_project.agents.l3.risk_agent import RiskAgent
from Phoenix_project.agents.l3.execution_agent import ExecutionAgent
from Phoenix_project.core.schemas.fusion_result import FusionResult

class Orchestrator:
    """
    (Controller) 系统的主要协调器。
    
    [RAG 架构]：
    Orchestrator 是 "Control & Orchestration Layer" (控制与编排层) 的核心。
    它不直接执行 RAG，但它协调依赖于 RAG (通过 L1/L2 智能体) 的智能体 (L3)。
    
    职责:
    1. 启动和停止所有主要组件 (LoopManager, CognitiveEngine, OrderManager)。
    2. 管理主要的认知-执行循环 (Cognitive-Execution Loop)。
    3. 充当 L3 智能体 (Alpha, Risk, Execution) 的 "宿主" (Host)。
    4. 协调 L3 智能体与 CognitiveEngine (L2) 和 OrderManager (执行) 之间的交互。
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        context_bus: ContextBus,
        loop_manager: LoopManager,
        cognitive_engine: CognitiveEngine,
        order_manager: OrderManager,
        # (任务 2.3) L3 智能体现在被注入，而不是由 Orchestrator 创建
        alpha_agent: AlphaAgent,
        risk_agent: RiskAgent,
        execution_agent: ExecutionAgent,
    ):
        self.config = config
        self.context_bus = context_bus
        
        # (任务 2.3) 我们保留了对 L2 引擎的引用，以获取 L3 的输入
        self.cognitive_engine = cognitive_engine
        self.order_manager = order_manager
        
        # (任务 2.3) 我们仍然保留 PC，以防它有其他辅助功能 (例如计算)，
        # 但它不再是 L3 循环的主要驱动力。
        # (我们可能需要重构 PortfolioConstructor 或将其功能移至 L3 RiskAgent)
        self.portfolio_constructor = PortfolioConstructor(config, order_manager)
        
        # (任务 2.3) 存储对 L3 智能体的引用
        self.alpha_agent = alpha_agent
        self.risk_agent = risk_agent
        self.execution_agent = execution_agent
        
        # [FIX] Load default symbols from config
        self.default_symbols = config.get("trading", {}).get("default_symbols", ["AAPL", "GOOG"])
        
        self.loop_manager = loop_manager
        self.logger = get_logger(self.__class__.__name__)
        self.is_running = False

    async def start(self):
        """
        启动 Orchestrator 和所有受管组件。
        """
        if self.is_running:
            self.logger.warning("Orchestrator is already running.")
            return
            
        self.logger.info("Starting Orchestrator...")
        
        # 1. 启动 L0 (数据) 和 L1/L2 (认知) 循环
        # (这是由 LoopManager 管理的)
        # (LoopManager 应该已经在 main.py 中启动了)
        
        # 2. 启动 L3 (决策) 循环
        # (这是由 Orchestrator 管理的)
        self.is_running = True
        asyncio.create_task(self.run_main_loop())
        
        self.logger.info("Orchestrator started.")

    async def stop(self):
        """
        停止 Orchestrator。
        (注意：LoopManager 和其他组件的停止是在 main.py 中处理的)
        """
        self.logger.info("Stopping Orchestrator...")
        self.is_running = False
        # (循环将在下一次迭代时停止)
        self.logger.info("Orchestrator stopped.")

    async def run_main_loop(self):
        """
        (L3) 主要的认知-执行循环 (Cognitive-Execution Loop)。
        
        [任务 2.3] 此循环现在由 L3 智能体 (Alpha, Risk, Execution) 驱动。
        """
        self.logger.info("L3 Main Loop started.")
        
        # (定义 L3 循环的频率，例如每 5 分钟)
        # (这应该来自配置)
        l3_loop_frequency_seconds = self.config.get("controller", {}).get("l3_loop_frequency_sec", 300)

        while self.is_running:
            try:
                self.logger.info("[L3 Loop] Starting new L3 cycle...")
                
                # 1. (L3 Alpha) - 生成交易信号
                
                # (L3 Alpha 需要 L2 的输出)
                # (我们从 L2 Cognitive Engine 获取最新的融合决策)
                final_decision_obj: Optional[FusionResult] = self.cognitive_engine.get_latest_fusion_result()
            
                if final_decision_obj and isinstance(final_decision_obj, FusionResult):
                    # 这是一个临时修复，以解决设计不匹配的问题
                    # (L2 (FusionResult) 是全局的/跨资产的)
                    # (L3 (AlphaAgent) 可能是特定于资产的)
                    
                    # (我们需要知道要处理哪些符号，这应该来自 L0 或配置)
                    # (暂时依赖 L3 循环中的模拟数据)
                    # [FIX] Use default symbols from config instead of hardcoded list
                    symbols_to_process = self.default_symbols
            
                    for symbol in symbols_to_process:
                        symbol_fusion_result = final_decision_obj.model_copy(deep=True)
                        symbol_fusion_result.symbol = symbol
                        
                        # (在 L3 AlphaAgent 中运行 L2 结果)
                        alpha_signal = await self.alpha_agent.generate_signal(symbol_fusion_result)
                        
                        if alpha_signal:
                            self.logger.info(f"[L3 Alpha] Generated signal for {symbol}: {alpha_signal.signal_type} @ {alpha_signal.confidence}")
                            
                            # 2. (L3 Risk) - 调整信号 (例如调整大小)
                            # (RiskAgent 还需要 Portfolio 上下文)
                            portfolio_context = self.portfolio_constructor.get_context()
                            risk_adjusted_signal = await self.risk_agent.manage_risk(alpha_signal, portfolio_context)
                            
                            if risk_adjusted_signal:
                                # 3. (L3 Execution) - 执行信号
                                await self.execution_agent.execute_signal(risk_adjusted_signal)
                            else:
                                self.logger.info(f"[L3 Risk] Signal for {symbol} was vetoed by RiskAgent.")
                        else:
                            self.logger.info(f"[L3 Alpha] No signal generated for {symbol}.")
                
                else:
                    self.logger.warning("[L3 Loop] No L2 FusionResult available yet. Skipping L3 cycle.")

                # (等待下一个 L3 循环)
                self.logger.info(f"[L3 Loop] L3 cycle finished. Waiting {l3_loop_frequency_seconds}s...")
                await asyncio.sleep(l3_loop_frequency_seconds)

            except asyncio.CancelledError:
                self.logger.info("L3 Main Loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in L3 Main Loop: {e}", exc_info=True)
                # (在非致命错误时继续)
                await asyncio.sleep(l3_loop_frequency_seconds) # (发生错误时也等待)
                
        self.logger.info("L3 Main Loop stopped.")
