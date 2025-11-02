"""
策略处理器 (Strategy Handler)
(这是一个遗留或特定策略的封装器，似乎已在 phoenix_project.py 中被绕过)

它封装了 CognitiveEngine 和 PortfolioConstructor，
似乎是用于响应特定的外部信号（StrategySignal）？
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

# (核心组件)
from cognitive.engine import CognitiveEngine
from cognitive.portfolio_constructor import PortfolioConstructor
from core.pipeline_state import PipelineState
from core.schemas.data_schema import Order, PortfolioState

# FIX (E6): 从 signal_protocol 导入 (不存在的) StrategySignal
# 我们将在 signal_protocol.py 中添加一个占位符
from execution.signal_protocol import StrategySignal 

# (AI/RAG 组件 - 用于初始化)
from ai.retriever import Retriever
from ai.ensemble_client import EnsembleClient
from ai.metacognitive_agent import MetacognitiveAgent
from ai.reasoning_ensemble import ReasoningEnsemble
from evaluation.arbitrator import Arbitrator
from evaluation.fact_checker import FactChecker
from ai.prompt_manager import PromptManager
from api.gateway import IAPIGateway
from api.gemini_pool_manager import GeminiPoolManager
from memory.vector_store import VectorStore
from memory.cot_database import CoTDatabase
from config.loader import ConfigLoader
from sizing.base import IPositionSizer
from sizing.fixed_fraction import FixedFractionSizer # 示例

from monitor.logging import get_logger

logger = get_logger(__name__)

class StrategyHandler:
    """
    封装了从外部信号到订单的完整认知和构造流程。
    """
    def __init__(self, config_loader: ConfigLoader, position_sizer: Optional[IPositionSizer] = None):
        
        self.config_loader = config_loader
        self.log_prefix = "StrategyHandler:"
        
        # 1. 初始化 AI 栈 (与 phoenix_project.py 中的 _setup_cognitive_engine 类似)
        try:
            gemini_pool = GeminiPoolManager()
            api_gateway = APIGateway(gemini_pool)
            prompt_manager = PromptManager(config_loader.config_path)
            vector_store = VectorStore()
            cot_db = CoTDatabase()
            
            self.retriever = Retriever(vector_store, cot_db)
            
            agent_registry = config_loader.get_agent_registry()
            self.ensemble_client = EnsembleClient(api_gateway, prompt_manager, agent_registry)
            self.metacognitive_agent = MetacognitiveAgent(api_gateway, prompt_manager)
            self.arbitrator = Arbitrator(api_gateway, prompt_manager)
            self.fact_checker = FactChecker(api_gateway, prompt_manager)
            
            self.reasoning_ensemble = ReasoningEnsemble(
                retriever=self.retriever,
                ensemble_client=self.ensemble_client,
                metacognitive_agent=self.metacognitive_agent,
                arbitrator=self.arbitrator,
                fact_checker=self.fact_checker
            )
            
            # FIX (E5): CognitiveEngine 构造函数参数错误
            # 原本传入了不正确的依赖项
            self.cognitive_engine = CognitiveEngine(
                reasoning_ensemble=self.reasoning_ensemble,
                fact_checker=self.fact_checker
            )
            
            # 2. 初始化投资组合构造器
            self.position_sizer = position_sizer or FixedFractionSizer() # 使用默认
            
            self.portfolio_constructor = PortfolioConstructor(
                position_sizer=self.position_sizer
            )
            logger.info(f"{self.log_prefix} Initialized.")
            
        except Exception as e:
            logger.error(f"{self.log_prefix} Initialization failed: {e}", exc_info=True)
            raise

    def process_signals(self, signal: StrategySignal, state: PipelineState) -> List[Order]:
        """
        处理传入的 (外部) 策略信号。
        """
        logger.info(f"{self.log_prefix} Processing external signal for {signal.symbol} at {signal.timestamp}")
        
        # 1. 运行认知推理
        # (注意：这里的 target_symbols 和 timestamp 来自外部信号)
        fusion_result = self.cognitive_engine.run_inference(
            target_symbols=[signal.symbol],
            timestamp=signal.timestamp
        )
        
        if not fusion_result:
            logger.warning(f"{self.log_prefix} Cognitive engine produced no result for {signal.symbol}")
            return []

        # 2. 转换为内部信号
        signal_obj = self.portfolio_constructor.translate_decision_to_signal(fusion_result)
        
        if not signal_obj:
            logger.warning(f"{self.log_prefix} Portfolio constructor failed to translate decision {fusion_result.id}")
            return []
            
        # 3. 获取当前投资组合状态
        portfolio_state = state.get_latest_portfolio_state()
        if not portfolio_state:
            logger.error(f"{self.log_prefix} Cannot generate orders, PipelineState has no PortfolioState.")
            # (在真实系统中，我们可能需要从 TradeLifecycleManager 初始化一个)
            return []

        # 4. 生成订单
        orders = self.portfolio_constructor.generate_orders(
            signals=[signal_obj],
            portfolio_state=portfolio_state
        )
        
        logger.info(f"{self.log_prefix} Generated {len(orders)} orders for {signal.symbol}")
        
        return orders
