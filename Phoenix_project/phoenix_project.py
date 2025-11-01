import yaml
from pathlib import Path  # 修正：导入 Path
from cognitive.engine import CognitiveEngine
from data_manager import DataManager
from audit_manager import AuditManager
from ai.prompt_manager import PromptManager
from ai.source_credibility import SourceCredibilityStore
from api.gemini_pool_manager import GeminiPoolManager
from registry import registry
from monitor.logging import get_logger
from knowledge_graph_service import KnowledgeGraphService
from ai.embedding_client import EmbeddingClient
from backtesting.engine import BacktestingEngine
from ai.reasoning_ensemble import ReasoningEnsemble # Was BayesianFusionEngine
from ai.retriever import HybridRetriever, VectorDBClient # Imports dummy VectorDBClient
from ai.temporal_db_client import TemporalDBClient
from ai.tabular_db_client import TabularDBClient

# Configure logger for this module (Layer 12)
logger = get_logger(__name__)

# 修正：定义项目的绝对根路径，确保路径始终正确
PROJECT_ROOT = Path(__file__).parent.resolve()


def setup_dependencies():
    """
    Centralized service instantiation and registration (Layer 11).
    Loads configurations and registers all core services with the singleton registry.
    """
    # Load main config
    # 修正：使用基于 PROJECT_ROOT 的绝对路径加载配置
    config_path = PROJECT_ROOT / "config" / "system.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
    except FileNotFoundError:
        logger.error(f"CRITICAL: Configuration file not found at {config_path}")
        raise
    
    # Instantiate and register services
    # 修正：确保所有从配置中读取的路径都相对于 PROJECT_ROOT 解析
    audit_log_path_str = config.get('audit_log_path', 'logs/audit_log.jsonl')
    audit_log_path = PROJECT_ROOT / audit_log_path_str
    audit_manager = AuditManager(audit_log_path)
    registry.register("audit_manager", audit_manager)

    data_catalog_path_str = config.get('data_catalog_path', 'data_catalog.json')
    data_catalog_path = PROJECT_ROOT / data_catalog_path_str
    data_manager = DataManager(data_catalog_path)
    registry.register("data_manager", data_manager)

    prompt_manager = PromptManager()
    registry.register("prompt_manager", prompt_manager)

    credibility_store = SourceCredibilityStore()
    registry.register("credibility_store", credibility_store)

    cognitive_engine = CognitiveEngine(data_manager)
    registry.register("cognitive_engine", cognitive_engine)

    gemini_pool = GeminiPoolManager()
    registry.register("gemini_pool", gemini_pool)

    # Register Layer 10 service
    knowledge_graph_service = KnowledgeGraphService()
    registry.register("knowledge_graph_service", knowledge_graph_service)

    # Register Layer 13 service
    embedding_client = EmbeddingClient()
    registry.register("embedding_client", embedding_client)

    # Register Layer 14 service
    backtesting_engine = BacktestingEngine()
    registry.register("backtesting_engine", backtesting_engine)

    # Register Layer 13 (L2) service
    reasoning_ensemble_service = ReasoningEnsemble(config) # Instantiated with config
    registry.register("reasoning_ensemble", reasoning_ensemble_service)
    logger.info("ReasoningEnsemble registered as 'reasoning_ensemble'.")

    # --- RAG SERVICE REGISTRATION ---
    logger.info("Registering RAG services...")
    vector_db_client = VectorDBClient() # Uses dummy client from retriever.py
    
    temporal_db_client = TemporalDBClient(config.get('temporal_db', {}))
    tabular_db_client = TabularDBClient(config.get('tabular_db', {}))
    
    rerank_config = config.get('ai', {}).get('retriever', {}).get('rerank', {})
    
    hybrid_retriever = HybridRetriever(
        vector_db_client=vector_db_client,
        temporal_db_client=temporal_db_client,
        tabular_db_client=tabular_db_client,
        rerank_config=rerank_config
    )
    registry.register("hybrid_retriever", hybrid_retriever)
    logger.info("HybridRetriever (RAG) service registered.")


    logger.info("All core services instantiated and registered.")


if __name__ == "__main__":
    # Set up all application dependencies first
    setup_dependencies()

    # Resolve the main engine from the registry
    cognitive_engine: CognitiveEngine = registry.resolve("cognitive_engine")

    logger.info("Starting Phoenix Project simulation...")
    cognitive_engine.run_simulation()
    logger.info("Phoenix Project simulation finished.")
