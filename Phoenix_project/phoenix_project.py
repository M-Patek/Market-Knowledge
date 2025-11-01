import yaml
from cognitive.engine import CognitiveEngine
from data_manager import DataManager
from audit_manager import AuditManager
from ai.prompt_manager import PromptManager
from ai.source_credibility import SourceCredibilityStore
from api.gemini_pool_manager import GeminiPoolManager
# NOTE: 'l3_rules_engine.py' 在项目文件列表中不存在。
# 必须保持注释，否则会导致 ModuleNotFoundError。
# from l3_rules_engine import L3RulesEngine
from registry import registry
# FIXED: 'observability.py' 不存在。'get_logger' 位于 'monitor/logging.py'。
from monitor.logging import get_logger
from knowledge_graph_service import KnowledgeGraphService
from ai.embedding_client import EmbeddingClient
from backtesting.engine import BacktestingEngine
# NOTE: 'ai/bayesian_fusion_engine.py' 在项目文件列表中不存在。
# FIXED: Refactored to ai.reasoning_ensemble.py per analysis
from ai.reasoning_ensemble import ReasoningEnsemble # Was BayesianFusionEngine

# NEW IMPORTS FOR RAG (Task 7)
from ai.retriever import HybridRetriever, VectorDBClient # Imports dummy VectorDBClient
from ai.temporal_db_client import TemporalDBClient
from ai.tabular_db_client import TabularDBClient

# Configure logger for this module (Layer 12)
logger = get_logger(__name__)


def setup_dependencies():
    """
    Centralized service instantiation and registration (Layer 11).
    Loads configurations and registers all core services with the singleton registry.
    """
    # Load main config
    # FIXED: 修正了配置文件的路径
    with open("config/system.yaml", 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded.")
    
    # Instantiate and register services
    audit_manager = AuditManager(config.get('audit_log_path', 'logs/audit_log.jsonl'))
    registry.register("audit_manager", audit_manager)

    data_manager = DataManager(config.get('data_catalog_path', 'data_catalog.json'))
    registry.register("data_manager", data_manager)

    prompt_manager = PromptManager()
    registry.register("prompt_manager", prompt_manager)

    credibility_store = SourceCredibilityStore()
    registry.register("credibility_store", credibility_store)

    cognitive_engine = CognitiveEngine(data_manager)
    registry.register("cognitive_engine", cognitive_engine)

    gemini_pool = GeminiPoolManager()
    registry.register("gemini_pool", gemini_pool)

    # NOTE: 已注释掉，因为 'l3_rules_engine.py' 文件缺失
    # l3_rules_engine = L3RulesEngine()
    # registry.register("l3_rules_engine", l3_rules_engine)
    # FIXED: Registering the ReasoningEnsemble (aliased as bayesian_fusion_engine)
    # under the old "l3_rules_engine" key as a compatibility shim.
    # This relies on bayesian_fusion_engine being registered first (below).

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
    # NOTE: 已注释掉，因为 'ai/bayesian_fusion_engine.py' 文件缺失
    bayesian_fusion_engine = ReasoningEnsemble(config) # Instantiated with config
    registry.register("bayesian_fusion_engine", bayesian_fusion_engine)
    
    # Register L3RulesEngine compatibility shim *after* bayesian_fusion_engine
    registry.register("l3_rules_engine", bayesian_fusion_engine)
    logger.info("ReasoningEnsemble registered under compatibility keys 'bayesian_fusion_engine' and 'l3_rules_engine'.")

    # --- NEW RAG SERVICE REGISTRATION (Task 7) ---
    logger.info("Registering RAG services...")
    # Instantiate clients for the retriever
    vector_db_client = VectorDBClient() # Uses dummy client from retriever.py
    
    # Pass in config subsections safely
    temporal_db_client = TemporalDBClient(config.get('temporal_db', {}))
    tabular_db_client = TabularDBClient(config.get('tabular_db', {}))
    
    # Get rerank config from the main system config
    rerank_config = config.get('ai', {}).get('retriever', {}).get('rerank', {})
    
    # Instantiate and register the HybridRetriever
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
