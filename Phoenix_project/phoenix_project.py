"""
Phoenix Project (Market Knowledge) - 主入口点
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional
import hydra
import redis
from omegaconf import DictConfig, OmegaConf

from Phoenix_project.monitor.logging import setup_logging
from Phoenix_project.api.gemini_pool_manager import GeminiPoolManager 
from Phoenix_project.ai.gemini_search_adapter import GeminiSearchAdapter
from Phoenix_project.memory.vector_store import VectorStore
from Phoenix_project.memory.cot_database import CoTDatabase
from Phoenix_project.context_bus import ContextBus
from Phoenix_project.ai.graph_db_client import GraphDBClient
from Phoenix_project.ai.tabular_db_client import TabularDBClient
from Phoenix_project.ai.temporal_db_client import TemporalDBClient
from Phoenix_project.ai.gnn_inferencer import GNNInferencer
from Phoenix_project.ai.retriever import Retriever
from Phoenix_project.ai.data_adapter import DataAdapter
from Phoenix_project.ai.relation_extractor import RelationExtractor
from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.ai.prompt_renderer import PromptRenderer
from Phoenix_project.ai.ensemble_client import EnsembleClient
from Phoenix_project.ai.embedding_client import EmbeddingClient
from Phoenix_project.agents.executor import AgentExecutor
from Phoenix_project.agents.l3.alpha_agent import AlphaAgent
from Phoenix_project.agents.l3.risk_agent import RiskAgent
from Phoenix_project.agents.l3.execution_agent import ExecutionAgent
from Phoenix_project.registry import Registry 
from Phoenix_project.agents.l2.fusion_agent import FusionAgent
from Phoenix_project.evaluation.fact_checker import FactChecker
from Phoenix_project.evaluation.arbitrator import Arbitrator
from Phoenix_project.evaluation.voter import Voter
from Phoenix_project.cognitive.engine import CognitiveEngine
from Phoenix_project.cognitive.risk_manager import RiskManager
from Phoenix_project.cognitive.portfolio_constructor import PortfolioConstructor
from Phoenix_project.execution.order_manager import OrderManager
from Phoenix_project.execution.trade_lifecycle_manager import TradeLifecycleManager
from Phoenix_project.execution.adapters import AlpacaAdapter
from Phoenix_project.execution.interfaces import IBrokerAdapter
from Phoenix_project.events.event_distributor import EventDistributor
from Phoenix_project.events.risk_filter import EventRiskFilter
from Phoenix_project.ai.reasoning_ensemble import ReasoningEnsemble
from Phoenix_project.ai.market_state_predictor import MarketStatePredictor
from Phoenix_project.controller.orchestrator import Orchestrator
from Phoenix_project.controller.loop_manager import LoopManager
from Phoenix_project.data_manager import DataManager
from Phoenix_project.knowledge_injector import KnowledgeInjector
from Phoenix_project.interfaces.api_server import APIServer
from Phoenix_project.audit_manager import AuditManager

# 设置日志 (在最顶部)
setup_logging()
logger = logging.getLogger(__name__)

class PhoenixProject:
    """
    Phoenix Project 主应用类，负责服务初始化和生命周期管理。
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        logger.info(f"Initializing PhoenixProject with run_mode: {cfg.get('run_mode', 'default')}")
        self.services = self._create_services(cfg)
        self.systems = self._create_main_systems(self.services)

    def _create_services(self, config: DictConfig) -> Dict[str, Any]:
        logger.info("Initializing core services...")
        # --- 1. 基础管理器与工具 (Managers & Utils) ---

        # [Infra] Redis Client (用于 ContextBus, DataManager, EventDistributor)
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        redis_port = int(os.environ.get("REDIS_PORT", 6379))
        
        # [Beta FIX] Re-enabled decode_responses=False for binary support (Pickle)
        # Note: Binary data must be handled explicitly if introduced later
        redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=False)
        logger.info(f"Redis client initialized at {redis_host}:{redis_port} (Binary Mode)")
    
        # [Infra] ContextBus (状态持久化)
        context_bus = ContextBus(redis_client=redis_client, config=config.get('context_bus'))
    
        # [Fix IV.1] 将 PromptManager 移至顶部，作为 EnsembleClient 的依赖项
        prompt_manager = PromptManager(prompt_directory="prompts")
        prompt_renderer = PromptRenderer(prompt_manager=prompt_manager)

        # [Fix IV.1] 全局 Registry (用于 V2 智能体)
        global_registry = Registry()

        # --- 2. API 客户端与 LLM (Clients & LLM) ---
    
        # [Fix IV.1] 替换 APIGateway (Server) 为 GeminiPoolManager (Client)
        gemini_manager = GeminiPoolManager(config=config.api_gateway)

        # [Fix IV.1] 直接从 env 获取 API 密钥
        gemini_api_key_list = os.environ.get("GEMINI_API_KEYS", "").split(',')
        gemini_api_key = gemini_api_key_list[0].strip() if gemini_api_key_list else None
        if not gemini_api_key:
            logger.warning("GEMINI_API_KEYS 未在 .env 中设置！EmbeddingClient 可能失败。")

        # [RAG] (RAG) 嵌入客户端 (Embedding Client)
        embedding_client = EmbeddingClient(
            provider=config.api_gateway.embedding_model.provider,
            model_name=config.api_gateway.embedding_model.model_name,
            api_key=gemini_api_key # [Fix IV.1]
        )
    
        # (RAG) 搜索工具 (Search Tool)
        search_client = GeminiSearchAdapter(gemini_manager=gemini_manager)
        logger.info("Search Client: Switched to Gemini Grounding Adapter (Google Search).")
    
        # (RAG) LLM 客户端 (Ensemble Client)
        ensemble_client = EnsembleClient(
           gemini_manager=gemini_manager,
           prompt_manager=prompt_manager,
           agent_registry=config.l1_agents, 
           global_registry=global_registry
        )

        # --- 3. 数据库和内存 (Databases & Memory) ---
    
        # (RAG) 向量存储 (Vector Store)
        vector_store_client = VectorStore(
            config=config.memory.vector_store,
            embedding_client=embedding_client
        )
    
        # (RAG) 知识图谱 (Knowledge Graph)
        graph_db_client = GraphDBClient()
    
        # (RAG) 表格数据库 (Tabular DB)
        tabular_db_client = TabularDBClient(config=config.data_manager.tabular_db, llm_client=gemini_manager, prompt_manager=prompt_manager, prompt_renderer=prompt_renderer)
    
        # (RAG) 时序数据库 (Temporal DB)
        temporal_db_client = TemporalDBClient(config=config.data_manager.temporal_db)
    
        # (CoT) 审计/思维链数据库 (Audit/CoT Database)
        cot_db_client = CoTDatabase(config=config.memory.cot_database)

        # --- 4. 核心 AI 组件 (Core AI Components) ---
        
        # (RAG) 数据管理器 (Data Manager) - [Task 4.2] Moved to services to inject into OrderManager
        data_manager = DataManager(
            config=config.data_manager,
            redis_client=redis_client,
            tabular_db=tabular_db_client,
            temporal_db=temporal_db_client,
        )
    
        # (RAG) 关系提取器 (Relation Extractor)
        relation_extractor = RelationExtractor(
            llm_client=ensemble_client, 
            config=config.ai_components.relation_extractor
        )

        # (RAG) 数据适配器 (Data Adapter)
        data_adapter = DataAdapter()

        # [GNN Plan Task 1.3] Instantiate GNNInferencer Service
        gnn_model_path = config.ai_components.gnn.model_path if 'gnn' in config.ai_components else '/app/models/default_gnn'
        gnn_inferencer_service = GNNInferencer(model_path=gnn_model_path)

        # (RAG) 检索器 (Retriever)
        retriever = Retriever(
            vector_store=vector_store_client,
            graph_db=graph_db_client,
            temporal_db=temporal_db_client,
            tabular_db=tabular_db_client,
            search_client=search_client, 
            ensemble_client=ensemble_client,
            config=config.ai_components.retriever, 
            gnn_inferencer=gnn_inferencer_service,
            prompt_manager=prompt_manager,
            prompt_renderer=prompt_renderer
        )

        # L1 智能体执行器 (L1 Agent Executor)
        agent_executor = AgentExecutor(
            ensemble_client=ensemble_client,
            retriever=retriever, 
            prompt_manager=prompt_manager,
            prompt_renderer=prompt_renderer,
            cot_db=cot_db_client,
            config_path="agents/registry.yaml"
        )

        # --- 5. 核心逻辑引擎 (Core Logic Engines) ---
    
        # (L2) 认知引擎 (Cognitive Engine)
        cognitive_engine = CognitiveEngine(
            agent_executor=agent_executor,
            ensemble_client=ensemble_client,
            prompt_manager=prompt_manager,
            prompt_renderer=prompt_renderer,
            cot_db=cot_db_client,
            retriever=retriever 
        )
    
        # (L3) 风险管理器 (Risk Manager)
        risk_manager = RiskManager(config=config.trading) 
    
        # (L3) 投资组合构建器 (Portfolio Constructor)
        portfolio_constructor = PortfolioConstructor(
            sizer_config=config.portfolio.sizer
        )

        # (L3) 交易生命周期管理器 (Trade Lifecycle Manager)
        initial_capital = config.trading.get('initial_capital', 100000.0)
        trade_lifecycle_manager = TradeLifecycleManager(
            initial_cash=initial_capital,
            tabular_db=tabular_db_client # [Task 4.1] Inject DB for persistence
        )

        # (L3) 券商适配器 (Broker Adapter)
        broker_adapter: IBrokerAdapter = AlpacaAdapter(
            config=config.broker
        )

        # (L3) 订单管理器 (Order Manager)
        order_manager = OrderManager(
            broker=broker_adapter,
            trade_lifecycle_manager=trade_lifecycle_manager,
            data_manager=data_manager # [Task 4.2] Inject DataManager
        )
    
        # (L3) 核心 DRL 智能体 (Core DRL Agents)
        # 注意：这里假设配置中包含 agent 的配置路径
        alpha_agent = AlphaAgent(config=config.agents.l3.alpha) if 'agents' in config and 'l3' in config.agents else None
        risk_agent = RiskAgent(config=config.agents.l3.risk) if 'agents' in config and 'l3' in config.agents else None
        execution_agent = ExecutionAgent(config=config.agents.l3.execution) if 'agents' in config and 'l3' in config.agents else None

        # --- (L3) 创建 Orchestrator 依赖 (Task 6) ---
        # [主人喵 Phase 0 修复] 移除错误的 config 参数
        event_distributor = EventDistributor()
        
        event_filter = EventRiskFilter(
            config=config.events.risk_filter 
        )
        
        # [Task 3.3] Instantiate L2/Eval components properly
        fusion_agent = FusionAgent(agent_id="l2_fusion", llm_client=ensemble_client)
        fact_checker = FactChecker(llm_client=ensemble_client)
        arbitrator = Arbitrator(llm_client=ensemble_client)
        voter = Voter(llm_client=ensemble_client)

        reasoning_ensemble = ReasoningEnsemble(
            fusion_agent=fusion_agent,
            alpha_agent=alpha_agent,
            voter=voter,
            arbitrator=arbitrator,
            fact_checker=fact_checker,
            data_manager=data_manager # [Task 3.3] Inject DataManager
        )
        
        market_state_predictor = MarketStatePredictor(
            llm_client=ensemble_client
        )

        logger.info("Core services initialized.")
        return {
            "config": config,
            "redis_client": redis_client,
            "context_bus": context_bus,
            "data_manager": data_manager, # [Task 4.2] Return data_manager
            "gemini_manager": gemini_manager,
            "embedding_client": embedding_client,
            "ensemble_client": ensemble_client,
            "vector_store": vector_store_client,
            "graph_db": graph_db_client,
            "tabular_db": tabular_db_client,
            "temporal_db": temporal_db_client,
            "cot_db": cot_db_client,
            "prompt_manager": prompt_manager,
            "prompt_renderer": prompt_renderer,
            "data_adapter": data_adapter,
            "relation_extractor": relation_extractor,
            "retriever": retriever,
            "gnn_inferencer": gnn_inferencer_service,
            "agent_executor": agent_executor,
            "cognitive_engine": cognitive_engine,
            "risk_manager": risk_manager,
            "portfolio_constructor": portfolio_constructor,
            "order_manager": order_manager,
            "trade_lifecycle_manager": trade_lifecycle_manager,
            "broker_adapter": broker_adapter,
            "event_distributor": event_distributor,
            "event_filter": event_filter,
            "reasoning_ensemble": reasoning_ensemble,
            "market_state_predictor": market_state_predictor,
            "alpha_agent": alpha_agent,
            "risk_agent": risk_agent,
            "execution_agent": execution_agent
        }

    def _create_main_systems(self, services: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Initializing main systems...")
    
        # [修复 喵!] 审计管理器必须在 Orchestrator 之前创建
        audit_manager = AuditManager(cot_db=services["cot_db"])

        # (L3) 编排器 (Orchestrator)
        orchestrator = Orchestrator(
            config=services["config"],
            context_bus=services["context_bus"], # [Task 2.3] Inject ContextBus
            data_manager=services["data_manager"], # [Task 4.2] Injected from services
            cognitive_engine=services["cognitive_engine"],
            event_distributor=services["event_distributor"],
            event_filter=services["event_filter"],
            market_state_predictor=services["market_state_predictor"],
            portfolio_constructor=services["portfolio_constructor"],
            order_manager=services.get("order_manager"), 
            audit_manager=audit_manager,
            alpha_agent=services["alpha_agent"],
            risk_agent=services["risk_agent"],
            execution_agent=services["execution_agent"]
        )

        # (L0/L1/L2) 循环管理器 (Loop Manager)
        loop_manager = LoopManager(
            agent_executor=services["agent_executor"],
            cognitive_engine=services["cognitive_engine"],
            config=services["config"].controller
        )
    
        # (RAG) 知识注入器 (Knowledge Injector)
        knowledge_injector = KnowledgeInjector(
            data_manager=services["data_manager"], # [Task 4.2] Use from services
            vector_store=services["vector_store"],
            graph_db=services["graph_db"],
            relation_extractor=services["relation_extractor"],
            data_adapter=services["data_adapter"]
        )

        logger.info("Main systems initialized.")
        return {
            "orchestrator": orchestrator,
            "loop_manager": loop_manager,
            "data_manager": services["data_manager"],
            "knowledge_injector": knowledge_injector,
            "audit_manager": audit_manager,
            "trade_lifecycle_manager": services["trade_lifecycle_manager"]
        }

    def run_backtest(self, cli_args):
        """
        [CLI] 运行回测模式
        """
        logger.info(f"Starting Backtest with args: {cli_args}")
        # TODO: Implement generic backtest execution logic here or in a separate engine
        pass

    def run_training(self, cli_args):
        """
        [CLI] 运行训练模式
        """
        logger.info(f"Starting Training with args: {cli_args}")
        pass

    def run_live(self):
        """
        [CLI/Main] 启动实时交易循环 (同步入口，内部运行异步循环)
        """
        asyncio.run(self._async_run_live())

    async def _async_run_live(self):
        """
        系统主异步循环逻辑。
        """
        logger.info("--- Phoenix Project Starting (Live Mode) ---")
        
        try:
            # --- 3. 获取核心系统 ---
            loop_manager = self.systems["loop_manager"]
            orchestrator = self.systems["orchestrator"]
            trade_lifecycle_manager = self.systems["trade_lifecycle_manager"]
            
            # 暂时使用 APIServer (Flask) 作为主 API
            self.api_server = APIServer(
                audit_manager=self.systems["audit_manager"],
                orchestrator=orchestrator 
            )

            # Start API Server explicitly [Fixed Dormant API]
            api_task = asyncio.create_task(self.api_server.start())

            # --- 4. 启动后台任务 (Loops) ---
            logger.info("Starting background processing loops...")
            
            # 启动 L0/L1 循环 (数据摄入/L1 智能体)
            l0_l1_task = asyncio.create_task(loop_manager.start_l0_l1_loop())
            
            # [Task 4.1] 初始化持久化账本 (Async Init)
            await trade_lifecycle_manager.initialize()

            # 启动 L2 循环 (L2 智能体 - 融合/批评)
            l2_task = asyncio.create_task(loop_manager.start_l2_loop())
            
            # 启动 L3 循环 (L3 智能体 - Alpha/Risk)
            l3_task = asyncio.create_task(orchestrator.start_l3_loop(
                frequency_sec=self.cfg.controller.get('l3_loop_frequency_sec', 300)
            ))

            tasks = [l0_l1_task, l2_task, l3_task, api_task]
            
            # --- 5. (优雅关闭) ---
            await asyncio.gather(*tasks) # 等待所有循环完成
            
            logger.info("--- Phoenix Project Shutting Down ---")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.critical(f"Fatal error in main: {e}", exc_info=True)

        finally:
            # [Code Opt Expert Fix] Task 6: Ensure cleanup uses self.services and runs in finally block
            logger.info("Cleaning up resources...")
            if hasattr(self, 'api_server'):
                await self.api_server.stop()

            if hasattr(self, 'services'):
                if "graph_db" in self.services:
                    await self.services["graph_db"].close()
                if "vector_store" in self.services:
                    await self.services["vector_store"].close()


@hydra.main(version_base=None, config_path="config", config_name="system")
def main(cfg: DictConfig):
    """
    主入口点：由 Hydra 处理配置加载。
    """
    # 1. 初始化应用
    app = PhoenixProject(cfg)
    # 2. 运行 (默认 Live 模式)
    app.run_live()

if __name__ == "__main__":
    main()
