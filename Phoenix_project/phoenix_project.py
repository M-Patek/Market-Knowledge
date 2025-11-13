"""
Phoenix Project (Market Knowledge) - 主入口点
"""

import asyncio
import logging
from typing import Dict, Any

from config.loader import ConfigLoader
from monitor.logging import setup_logging
from api.gateway import APIGateway
from memory.vector_store import VectorStore
from memory.cot_database import CoTDatabase
from ai.graph_db_client import GraphDBClient
from ai.tabular_db_client import TabularDBClient
from ai.temporal_db_client import TemporalDBClient
from ai.gnn_inferencer import GNNInferencer
from ai.retriever import Retriever
from ai.data_adapter import DataAdapter
from ai.relation_extractor import RelationExtractor
from ai.prompt_manager import PromptManager
from ai.prompt_renderer import PromptRenderer
from ai.ensemble_client import EnsembleClient
from ai.embedding_client import EmbeddingClient
from agents.executor import AgentExecutor
from cognitive.engine import CognitiveEngine
from cognitive.risk_manager import RiskManager
from cognitive.portfolio_constructor import PortfolioConstructor
from execution.order_manager import OrderManager
# --- Task 5 Imports 喵! ---
from execution.trade_lifecycle_manager import TradeLifecycleManager
from execution.adapters import AlpacaAdapter
from execution.interfaces import IBrokerAdapter
# --- Task 6 Imports 喵! ---
from events.event_distributor import EventDistributor
from events.risk_filter import EventRiskFilter
from ai.reasoning_ensemble import ReasoningEnsemble
from ai.market_state_predictor import MarketStatePredictor
from controller.orchestrator import Orchestrator
from controller.loop_manager import LoopManager
from data_manager import DataManager
from knowledge_injector import KnowledgeInjector
from interfaces.api_server import APIServer
from audit_manager import AuditManager

# 设置日志 (在最顶部)
setup_logging()
logger = logging.getLogger(__name__)


def create_services(config: ConfigLoader) -> Dict[str, Any]:
    """
    创建并初始化所有核心服务。
    """
    logger.info("Initializing core services...")

    # --- 1. 基础 API 和客户端 ---
    api_gateway = APIGateway(config=config.get_config('api_gateway'))
    
    # [RAG] (RAG) 嵌入客户端 (Embedding Client)
    embedding_client = EmbeddingClient(
        provider=config.get('api_gateway.embedding_model.provider'),
        model_name=config.get('api_gateway.embedding_model.model_name'),
        # dimensions=config.get('api_gateway.embedding_model.dimensions'), # <--- 移除
        api_key=api_gateway.get_api_key('gemini') # <--- 修改为 'gemini' 密钥
    )
    
    # (RAG) 搜索工具 (Search Tool) - 示例
    search_client = api_gateway.get_search_client() # 假设 APIGateway 抽象了
    
    # (RAG) LLM 客户端 (Ensemble Client)
    ensemble_client = EnsembleClient(
        llm_provider=api_gateway.get_llm_client('gemini'), # 假设
        embedding_client=embedding_client
    )

    # --- 2. 数据库和内存 (Databases & Memory) ---
    
    # (RAG) 向量存储 (Vector Store)
    vector_store_client = VectorStore(
        config=config.get_config('memory.vector_store'),
        embedding_client=embedding_client
    )
    
    # (RAG) 知识图谱 (Knowledge Graph)
    graph_db_client = GraphDBClient()
    
    # (RAG) 表格数据库 (Tabular DB)
    tabular_db_client = TabularDBClient(config=config.get_config('data_manager.tabular_db'))
    
    # (RAG) 时序数据库 (Temporal DB)
    temporal_db_client = TemporalDBClient(config=config.get_config('data_manager.temporal_db'))
    
    # (CoT) 审计/思维链数据库 (Audit/CoT Database)
    cot_db_client = CoTDatabase(config=config.get_config('memory.cot_database'))

    # --- 3. 核心 AI 组件 (Core AI Components) ---
    
    # (RAG) Prompt 管理器 (Prompt Manager)
    prompt_manager = PromptManager(prompt_directory="prompts")
    prompt_renderer = PromptRenderer(prompt_manager=prompt_manager)
    
    # (RAG) 关系提取器 (Relation Extractor)
    relation_extractor = RelationExtractor(
        llm_client=ensemble_client, # 使用 Flash 模型
        config=config.get('ai_components.relation_extractor')
    )

    # (RAG) 数据适配器 (Data Adapter)
    data_adapter = DataAdapter()

    # [GNN Plan Task 1.3] Instantiate GNNInferencer Service
    gnn_model_path = config.get('ai_components.gnn.model_path', '/app/models/default_gnn')
    gnn_inferencer_service = GNNInferencer(model_path=gnn_model_path)

    # (RAG) 检索器 (Retriever)
    # [GNN Plan Task 2.1] Pass GNN service to Retriever
    retriever = Retriever(
        vector_store=vector_store_client,
        graph_db=graph_db_client,
        llm_client=ensemble_client.get_client("gemini"), # 假设
        # TODO: 修复 RAG 依赖 (Temporal, Tabular, Search)
        temporal_db=temporal_db_client,
        tabular_db=tabular_db_client,
        search_client=search_client,
        ensemble_client=ensemble_client,
        config=config.get_config('ai_components.retriever'), # [修复] 移除硬编码
        gnn_inferencer=gnn_inferencer_service,
        prompt_manager=prompt_manager,
        prompt_renderer=prompt_renderer
    )

    # L1 智能体执行器 (L1 Agent Executor)
    agent_executor = AgentExecutor(
        ensemble_client=ensemble_client,
        retriever=retriever, # L1 智能体使用 RAG
        prompt_manager=prompt_manager,
        prompt_renderer=prompt_renderer,
        cot_db=cot_db_client,
        config_path="agents/registry.yaml"
    )

    # --- 4. 核心逻辑引擎 (Core Logic Engines) ---
    
    # (L2) 认知引擎 (Cognitive Engine)
    cognitive_engine = CognitiveEngine(
        agent_executor=agent_executor,
        ensemble_client=ensemble_client,
        prompt_manager=prompt_manager,
        prompt_renderer=prompt_renderer,
        cot_db=cot_db_client,
        retriever=retriever # L2 智能体也使用 RAG
    )
    
    # (L3) 风险管理器 (Risk Manager)
    risk_manager = RiskManager(config=config.get_config('trading')) # 示例
    
    # (L3) 投资组合构建器 (Portfolio Constructor)
    portfolio_constructor = PortfolioConstructor(
        sizer_config=config.get_config('portfolio.sizer')
    )

    # (L3) 交易生命周期管理器 (Trade Lifecycle Manager)
    # (喵~ TLM 可能需要 portfolio_constructor 或其他状态管理器)
    initial_capital = config.get('trading.initial_capital', 100000.0)
    trade_lifecycle_manager = TradeLifecycleManager(
        initial_cash=initial_capital
    )

    # (L3) 券商适配器 (Broker Adapter)
    # (喵~ 适配器可能需要 config 和 api_gateway)
    broker_adapter: IBrokerAdapter = AlpacaAdapter(
        config=config.get_config('broker')
    )

    # (L3) 订单管理器 (Order Manager)
    order_manager = OrderManager(
        broker=broker_adapter,
        trade_lifecycle_manager=trade_lifecycle_manager
    )
    
    # --- (L3) 创建 Orchestrator 依赖 (Task 6) ---
    event_distributor = EventDistributor(
        config=config.get_config('events.distributor') # 假设
    )
    event_filter = EventRiskFilter(
        config=config.get_config('events.risk_filter') # 假设
    )
    reasoning_ensemble = ReasoningEnsemble(
        llm_client=ensemble_client,
        prompt_manager=prompt_manager,
        prompt_renderer=prompt_renderer
    )
    market_state_predictor = MarketStatePredictor(
        llm_client=ensemble_client
    )
    
    # (AuditManager 在 create_main_systems 中创建，因为它依赖 cot_db)
    # ...



    logger.info("Core services initialized.")
    return {
        "config": config,
        "api_gateway": api_gateway,
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
        # --- Task 5 Additions 喵! ---
        "trade_lifecycle_manager": trade_lifecycle_manager,
        "broker_adapter": broker_adapter,
        # --- Task 6 Additions 喵! ---
        "event_distributor": event_distributor,
        "event_filter": event_filter,
        "reasoning_ensemble": reasoning_ensemble,
        "market_state_predictor": market_state_predictor
    }

def create_main_systems(services: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建并初始化主要系统 (Orchestrator, DataManager, etc.)。
    """
    logger.info("Initializing main systems...")
    
    # [修复 喵!] 审计管理器必须在 Orchestrator 之前创建
    audit_manager = AuditManager(cot_db=services["cot_db"])

    # (L3) 编排器 (Orchestrator)
    # [修复 喵!] 使用 orchestrator.py 中定义的 9 个参数
    orchestrator = Orchestrator(
        config=services["config"],
        cognitive_engine=services["cognitive_engine"],
        event_distributor=services["event_distributor"],
        event_filter=services["event_filter"],
        reasoning_ensemble=services["reasoning_ensemble"],
        market_state_predictor=services["market_state_predictor"],
        portfolio_constructor=services["portfolio_constructor"],
        order_manager=services.get("order_manager"), # 使用 .get()
        audit_manager=audit_manager
    )

    # (L0/L1/L2) 循环管理器 (Loop Manager)
    loop_manager = LoopManager(
        agent_executor=services["agent_executor"],
        cognitive_engine=services["cognitive_engine"],
        config=services["config"].get_config('controller')
    )

    # (RAG) 数据管理器 (Data Manager)
    data_manager = DataManager(
        config=services["config"].get_config('data_manager'),
        tabular_db=services["tabular_db"],
        temporal_db=services["temporal_db"],
        api_gateway=services["api_gateway"] # 用于获取外部数据
    )
    
    # (RAG) 知识注入器 (Knowledge Injector)
    knowledge_injector = KnowledgeInjector(
        data_manager=data_manager,
        vector_store=services["vector_store"],
        graph_db=services["graph_db"],
        relation_extractor=services["relation_extractor"],
        data_adapter=services["data_adapter"]
    )

    # 审计管理器 (Audit Manager)
    # audit_manager = AuditManager(cot_db=services["cot_db"]) # 移动到 Orchestrator 之前

    logger.info("Main systems initialized.")
    return {
        "orchestrator": orchestrator,
        "loop_manager": loop_manager,
        "data_manager": data_manager,
        "knowledge_injector": knowledge_injector,
        "audit_manager": audit_manager,
    }


async def main():
    """
    系统主异步入口。
    """
    logger.info("--- Phoenix Project Starting ---")
    
    try:
        # --- 1. 加载配置 ---
        config = ConfigLoader(
            system_config_path="config/system.yaml",
            rules_config_path="config/symbolic_rules.yaml"
        )
        logger.info(f"Configuration loaded. System mode: {config.get_system_mode()}")

        # --- 2. 创建服务和系统 ---
        services = create_services(config)
        systems = create_main_systems(services)

        # --- 3. 获取核心系统 ---
        loop_manager = systems["loop_manager"]
        orchestrator = systems["orchestrator"]
        api_server = APIServer(
            audit_manager=systems["audit_manager"],
            orchestrator=orchestrator
        )

        # --- 4. 启动后台任务 (Loops) ---
        logger.info("Starting background processing loops...")
        
        # 启动 L0/L1 循环 (数据摄入/L1 智能体)
        l0_l1_task = asyncio.create_task(loop_manager.start_l0_l1_loop())
        
        # 启动 L2 循环 (L2 智能体 - 融合/批评)
        l2_task = asyncio.create_task(loop_manager.start_l2_loop())
        
        # 启动 L3 循环 (L3 智能体 - Alpha/Risk)
        l3_task = asyncio.create_task(orchestrator.start_l3_loop(
            frequency_sec=config.get('controller.l3_loop_frequency_sec', 300)
        ))

        tasks = [l0_l1_task, l2_task, l3_task]

        # --- 5. 启动 API 服务器 (前台) ---
        logger.info("Starting FastAPI server...")
        api_server.run() # Call directly; it starts in a background thread

        # --- 6. (优雅关闭) ---
        logger.info("--- Phoenix Project Shutting Down ---")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # 清理数据库连接
        await services["graph_db"].close()
        await services["vector_store"].close()
        # ... 其他清理

    except Exception as e:
        logger.critical(f"Fatal error in main: {e}", exc_info=True)

if __name__ == "__main__":
    # (如果直接运行，例如在 Docker 中)
    asyncio.run(main())
