import logging
import importlib
import os
import redis
from omegaconf import DictConfig
from types import SimpleNamespace
from typing import Dict, Any, Optional, Type

# 核心基础设施 (无循环依赖风险)
from Phoenix_project.context_bus import ContextBus
from Phoenix_project.data_manager import DataManager
from Phoenix_project.data.data_iterator import DataIterator
from Phoenix_project.controller.orchestrator import Orchestrator
from Phoenix_project.memory.vector_store import VectorStore
from Phoenix_project.memory.cot_database import CoTDatabase
from Phoenix_project.ai.embedding_client import EmbeddingClient
from Phoenix_project.ai.ensemble_client import EnsembleClient
from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.ai.retriever import Retriever
from Phoenix_project.ai.data_adapter import DataAdapter
from Phoenix_project.ai.relation_extractor import RelationExtractor
from Phoenix_project.ai.graph_db_client import GraphDBClient
from Phoenix_project.ai.source_credibility import SourceCredibility
from Phoenix_project.api.gemini_pool_manager import GeminiPoolManager
from Phoenix_project.knowledge_graph_service import KnowledgeGraphService
from Phoenix_project.audit_manager import AuditManager

# 核心认知/执行组件
from Phoenix_project.cognitive.engine import CognitiveEngine
from Phoenix_project.cognitive.portfolio_constructor import PortfolioConstructor
from Phoenix_project.cognitive.risk_manager import RiskManager
from Phoenix_project.execution.order_manager import OrderManager
# [Task FIX-MED-001] Import missing execution components
from Phoenix_project.execution.trade_lifecycle_manager import TradeLifecycleManager
from Phoenix_project.execution.adapters import AlpacaAdapter

from Phoenix_project.events.stream_processor import StreamProcessor
from Phoenix_project.events.risk_filter import EventRiskFilter
from Phoenix_project.events.event_distributor import EventDistributor

# [Task 2.2] Import missing components for CognitiveEngine construction
from Phoenix_project.agents.executor import AgentExecutor
from Phoenix_project.ai.reasoning_ensemble import ReasoningEnsemble
from Phoenix_project.evaluation.voter import Voter
from Phoenix_project.evaluation.fact_checker import FactChecker
from Phoenix_project.evaluation.arbitrator import Arbitrator
from Phoenix_project.fusion.uncertainty_guard import UncertaintyGuard
from Phoenix_project.ai.prompt_renderer import PromptRenderer

# [Task FIX-HIGH-003] Time Machine Abstraction
from Phoenix_project.core.time_provider import SystemTimeProvider

logger = logging.getLogger(__name__)

class DependencyContainer:
    """
    [Task 1.1] 集中式依赖容器。
    持有核心服务的单例，并传递给工厂以解析依赖。
    """
    def __init__(self):
        # [Task FIX-HIGH-003] Time Provider
        self.time_provider: Optional[SystemTimeProvider] = None
        
        self.context_bus: Optional[ContextBus] = None
        self.data_manager: Optional[DataManager] = None
        self.gemini_manager: Optional[GeminiPoolManager] = None
        self.embedding_client: Optional[EmbeddingClient] = None
        self.ensemble_client: Optional[EnsembleClient] = None
        self.prompt_manager: Optional[PromptManager] = None
        self.audit_manager: Optional[AuditManager] = None
        self.vector_store: Optional[VectorStore] = None
        self.cot_database: Optional[CoTDatabase] = None
        self.graph_db_client: Optional[GraphDBClient] = None
        
        # DB Clients (Hosted in Container to share pools)
        self.tabular_db: Any = None
        self.temporal_db: Any = None
        
        # 子系统 (在 build_system 中初始化)
        self.retriever: Optional[Retriever] = None
        self.data_adapter: Optional[DataAdapter] = None
        self.knowledge_graph_service: Optional[KnowledgeGraphService] = None
        self.source_credibility: Optional[SourceCredibility] = None

class AgentFactory:
    """
    [Task 1.1] 用于延迟加载智能体的动态工厂。
    充当适配器 (Adapter)，将依赖注入到现有的构造函数签名中。
    """
    
    # 映射: Config Key -> (Module Path, Class Name)
    L1_MAP = {
        "catalyst_monitor": ("Phoenix_project.agents.l1.catalyst_monitor_agent", "CatalystMonitorAgent"),
        "fundamental_analyst": ("Phoenix_project.agents.l1.fundamental_analyst_agent", "FundamentalAnalystAgent"),
        "geopolitical_analyst": ("Phoenix_project.agents.l1.geopolitical_analyst_agent", "GeopoliticalAnalystAgent"),
        "innovation_tracker": ("Phoenix_project.agents.l1.innovation_tracker_agent", "InnovationTrackerAgent"),
        "macro_strategist": ("Phoenix_project.agents.l1.macro_strategist_agent", "MacroStrategistAgent"),
        "supply_chain_intelligence": ("Phoenix_project.agents.l1.supply_chain_intelligence_agent", "SupplyChainIntelligenceAgent"),
        "technical_analyst": ("Phoenix_project.agents.l1.technical_analyst_agent", "TechnicalAnalystAgent"),
    }
    
    L2_MAP = {
        "planner": ("Phoenix_project.agents.l2.planner_agent", "PlannerAgent"),
        "metacognitive": ("Phoenix_project.agents.l2.metacognitive_agent", "MetacognitiveAgent"),
        "fusion": ("Phoenix_project.agents.l2.fusion_agent", "FusionAgent"),
        "critic": ("Phoenix_project.agents.l2.critic_agent", "CriticAgent"),
        "adversary": ("Phoenix_project.agents.l2.adversary_agent", "AdversaryAgent"),
    }

    L3_MAP = {
        "alpha": ("Phoenix_project.agents.l3.alpha_agent", "AlphaAgent"),
        "risk": ("Phoenix_project.agents.l3.risk_agent", "RiskAgent"),
        "execution": ("Phoenix_project.agents.l3.execution_agent", "ExecutionAgent"),
    }

    @staticmethod
    def _import_agent_class(module_path: str, class_name: str) -> Type:
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import {class_name} from {module_path}: {e}")
            raise

    @staticmethod
    def create_agent(category_map: Dict, key: str, container: DependencyContainer) -> Any:
        if key not in category_map:
            return None
        
        module_path, class_name = category_map[key]
        agent_cls = AgentFactory._import_agent_class(module_path, class_name)
        
        # [Fix 1.1] 注入 agent_id 和核心依赖
        base_args = [key, container.ensemble_client, container.data_manager]
        
        # [Task 2.1] Use kwargs for optional dependencies to match BaseAgent signature
        common_kwargs = {
            "prompt_manager": container.prompt_manager,
            "audit_manager": container.audit_manager
        }
        
        if key == "technical_analyst":
            return agent_cls(*base_args, container.data_adapter, container.embedding_client, **common_kwargs)
        elif key in ["fundamental_analyst", "geopolitical_analyst", "innovation_tracker", "macro_strategist", "supply_chain_intelligence", "critic"]:
            return agent_cls(*base_args, container.retriever, **common_kwargs)
        elif key == "fusion":
            return agent_cls(*base_args, container.knowledge_graph_service, container.source_credibility, **common_kwargs)
        else:
            return agent_cls(*base_args, **common_kwargs)

    @staticmethod
    def create_l3_agent(key: str, config: DictConfig, container: DependencyContainer) -> Any:
        """使用 DRLAgentLoader 加载 L3 智能体的特殊加载器。"""
        from Phoenix_project.agents.l3.base import DRLAgentLoader
        
        if key not in AgentFactory.L3_MAP:
            return None

        module_path, class_name = AgentFactory.L3_MAP[key]
        agent_cls = AgentFactory._import_agent_class(module_path, class_name)
        
        agent_config = config.agents.l3.get(key)
        if not agent_config:
             logger.warning(f"Configuration for L3 agent '{key}' not found.")
        
        ckpt_path = getattr(agent_config, "checkpoint_path", None)
        policy_id = getattr(agent_config, "policy_id", "default_policy")

        if ckpt_path:
            logger.info(f"Loading L3 Agent {key} from {ckpt_path}")
            return DRLAgentLoader.load_agent(agent_class=agent_cls, checkpoint_path=ckpt_path, policy_id=policy_id)
        else:
            logger.warning(f"No checkpoint path for L3 Agent {key}. Initializing dummy/untrained agent.")
            return None

class Registry:
    """
    [Refactored Task 1.1]
    The Registry is responsible for building and providing access to all
    core components of the system, managing dependencies using a Container pattern.
    """
    def __init__(self, config: DictConfig):
        self.config = config
        self.container = DependencyContainer()
        self._build_core()

    def _build_core(self):
        """Builds components that are shared across the system."""
        logger.info("Building core components...")
        
        # [Task FIX-HIGH-003] Initialize TimeProvider first
        self.container.time_provider = SystemTimeProvider()
        
        # [Task 1.2] Fail Fast Initialization
        if not os.getenv("GEMINI_API_KEYS") and not os.getenv("SKIP_API_CHECK"):
             logger.warning("GEMINI_API_KEYS not found. System running in limited mode or check .env")
        
        self.container.gemini_manager = GeminiPoolManager(self.config.get("api_gateway", {}))
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            redis_client = redis.from_url(redis_url)
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Redis connection failed at {redis_url}. Cause: {str(e)}")

        # 1. 核心通信
        self.container.context_bus = ContextBus(redis_client=redis_client)
        
        # 2. 核心 AI / 数据服务
        # [Task FIX-CRIT-004] Fix instantiation and detect dimension
        api_key = os.getenv("GEMINI_API_KEYS")
        model_name = self.config.api_gateway.get("embedding_model", "models/text-embedding-004")
        
        self.container.embedding_client = EmbeddingClient(
            api_key=api_key,
            model_name=model_name,
            provider="google",
            config=self.config.api_gateway
        )
        
        # Dynamic Dimension Probe
        try:
            real_dim = self.container.embedding_client.get_output_dimension()
            logger.info(f"Registry: Detected embedding dimension {real_dim} for model {model_name}")
        except Exception as e:
            logger.error(f"Registry: Failed to detect dimension: {e}. Defaulting to 1536.")
            real_dim = 1536
        
        self.container.ensemble_client = EnsembleClient(
            gemini_manager=self.container.gemini_manager,
            prompt_manager=None, # Will be set next
            agent_registry=self.config.agents,
            global_registry=self # 注入 self
        )
        self.container.prompt_manager = PromptManager(self.config.paths.prompts)
        
        # Wiring up EnsembleClient deps that were just created
        self.container.ensemble_client.prompt_manager = self.container.prompt_manager

        # 3. 核心内存
        # [Task FIX-CRIT-004] Inject real dimension into VectorStore
        self.container.vector_store = VectorStore(
            config=self.config.memory.vector_store, 
            embedding_client=self.container.embedding_client,
            vector_size=real_dim # Explicit override
        )
        self.container.cot_database = CoTDatabase(self.config.memory.cot_database)
        self.container.graph_db_client = GraphDBClient(self.config.memory.graph_database)
        
        # 3b. 初始化 Shared DB Clients
        from Phoenix_project.ai.tabular_db_client import TabularDBClient
        from Phoenix_project.ai.temporal_db_client import TemporalDBClient
        
        self.container.tabular_db = TabularDBClient(self.config.data_manager.tabular_db)
        self.container.temporal_db = TemporalDBClient(self.config.data_manager.temporal_db)

        # 4. 核心数据管理
        # Inject the pre-initialized DB clients and TimeProvider
        self.container.data_manager = DataManager(
            config=self.config,
            redis_client=redis_client,
            tabular_db=self.container.tabular_db,
            temporal_db=self.container.temporal_db,
            time_provider=self.container.time_provider # [Task FIX-HIGH-003] Injected
        )

        # 5. 核心审计
        self.container.audit_manager = AuditManager(self.container.cot_database)

        logger.info("Core components built.")

    def build_system(self, config: DictConfig) -> SimpleNamespace:
        """
        Builds and wires together the full agentic pipeline.
        """
        logger.info("Building full Phoenix system pipeline...")
        
        c = self.container

        # --- AI/RAG 子系统 ---
        c.retriever = Retriever(
            vector_store=c.vector_store,
            graph_db=c.graph_db_client,
            temporal_db=c.temporal_db,
            tabular_db=c.tabular_db,
            search_client=None,
            ensemble_client=c.ensemble_client,
            config=config.ai_components.retriever,
            prompt_manager=c.prompt_manager,
            prompt_renderer=PromptRenderer()
        )
        c.data_adapter = DataAdapter()
        relation_extractor = RelationExtractor(c.ensemble_client, c.prompt_manager)
        c.source_credibility = SourceCredibility(config.ai_components.get("source_credibility"))
        
        c.knowledge_graph_service = KnowledgeGraphService(
            graph_db_client=c.graph_db_client,
            relation_extractor=relation_extractor,
            retriever=c.retriever,
            audit_manager=c.audit_manager
        )

        # --- L1 智能体 ---
        l1_agents = {}
        for key in AgentFactory.L1_MAP.keys():
            try:
                l1_agents[key] = AgentFactory.create_agent(AgentFactory.L1_MAP, key, c)
            except Exception as e:
                logger.error(f"Failed to initialize L1 Agent '{key}': {e}", exc_info=True)
        
        # --- L2 智能体 ---
        l2_agents = {}
        for key in AgentFactory.L2_MAP.keys():
            try:
                l2_agents[key] = AgentFactory.create_agent(AgentFactory.L2_MAP, key, c)
            except Exception as e:
                logger.error(f"Failed to initialize L2 Agent '{key}': {e}", exc_info=True)

        # --- L3 智能体 ---
        l3_agents = {}
        for key in ["alpha", "risk", "execution"]:
            try:
                l3_agents[key] = AgentFactory.create_l3_agent(key, config, c)
            except Exception as e:
                 logger.error(f"Failed to initialize L3 Agent '{key}': {e}", exc_info=True)
                 l3_agents[key] = None

        # [Task 2.2] Build Cognitive Engine Dependencies
        prompt_renderer = PromptRenderer()
        voter = Voter()
        
        fact_checker = FactChecker(
            llm_client=c.ensemble_client,
            prompt_manager=c.prompt_manager,
            prompt_renderer=prompt_renderer,
            config=config
        )
        
        arbitrator = Arbitrator(
            llm_client=c.ensemble_client,
            prompt_manager=c.prompt_manager,
            prompt_renderer=prompt_renderer
        )
        
        uncertainty_guard = UncertaintyGuard(config.get("cognitive_engine", {}).get("uncertainty_guard"))

        # [Task 2.2] Build Reasoning Ensemble
        reasoning_ensemble = ReasoningEnsemble(
            fusion_agent=l2_agents.get("fusion"),
            alpha_agent=l3_agents.get("alpha"),
            voter=voter,
            arbitrator=arbitrator,
            fact_checker=fact_checker,
            data_manager=c.data_manager
        )

        # [Task 2.2] Build Agent Executor
        all_agents = list(l1_agents.values()) + list(l2_agents.values()) + [a for a in l3_agents.values() if a]
        agent_executor = AgentExecutor(
            agent_list=all_agents,
            context_bus=c.context_bus,
            config=config
        )

        # --- 执行层 ---
        # [Task FIX-MED-001] Fully Initialize Execution Layer in Registry
        
        # 1. Trade Lifecycle Manager
        initial_capital = config.trading.get('initial_capital', 100000.0)
        trade_lifecycle_manager = TradeLifecycleManager(
            initial_cash=initial_capital,
            data_manager=c.data_manager,
            tabular_db=c.tabular_db,
            bus=c.context_bus
        )

        # 2. Broker Adapter
        broker_adapter = AlpacaAdapter(config=config.broker)

        # 3. Order Manager (Fully Wired)
        order_manager = OrderManager(
            broker=broker_adapter,
            trade_lifecycle_manager=trade_lifecycle_manager,
            data_manager=c.data_manager,
            bus=c.context_bus
        )

        # --- 认知层 (投资组合构建) ---
        portfolio_constructor = PortfolioConstructor(
            config=config.cognitive.portfolio, 
            context_bus=c.context_bus,
            risk_manager=None, # Set below
            sizing_strategy=None,
            data_manager=c.data_manager
        )
        
        risk_manager = RiskManager(
            config=config.trading, 
            redis_client=c.data_manager.redis_client, 
            data_manager=c.data_manager,
            initial_capital=initial_capital
        )
        
        # Wiring Portfolio Constructor deps
        portfolio_constructor.risk_manager = risk_manager

        # [Task 2.2] Cognitive Engine with correct dependencies
        cognitive_engine = CognitiveEngine(
            agent_executor=agent_executor,
            reasoning_ensemble=reasoning_ensemble,
            fact_checker=fact_checker,
            uncertainty_guard=uncertainty_guard,
            voter=voter,
            config=config.get("cognitive_engine", {})
        )

        # --- 事件流处理 ---
        stream_processor = StreamProcessor(
            config=config.events.stream_processor,
            l1_agents=l1_agents,
            data_adapter=c.data_adapter,
            audit_manager=c.audit_manager,
            context_bus=c.context_bus
        )
        
        risk_filter = EventRiskFilter(config.events.risk_filter)
        
        event_distributor = EventDistributor(
            stream_processor=stream_processor,
            risk_filter=risk_filter,
            context_bus=c.context_bus,
            redis_client=c.data_manager.redis_client
        )

        # --- 主协调器 ---
        orchestrator = Orchestrator(
            config=config.orchestrator,
            context_bus=c.context_bus,
            data_manager=c.data_manager,
            cognitive_engine=cognitive_engine,
            event_distributor=event_distributor,
            event_filter=risk_filter,
            market_state_predictor=None,
            portfolio_constructor=portfolio_constructor,
            order_manager=order_manager,
            audit_manager=c.audit_manager,
            trade_lifecycle_manager=trade_lifecycle_manager, # [Task FIX-MED-001] Injected directly
            # Inject Agents
            alpha_agent=l3_agents.get("alpha"),
            risk_agent=l3_agents.get("risk"),
            execution_agent=l3_agents.get("execution"),
            risk_manager=risk_manager
        )

        # --- 数据迭代器 ---
        data_iterator = DataIterator(config.data_manager.get("iterator", {}), c.data_manager)

        logger.info("Full Phoenix system built.")

        # 返回所有构建的组件
        return SimpleNamespace(
            config=config,
            context_bus=c.context_bus,
            data_manager=c.data_manager,
            data_iterator=data_iterator,
            audit_manager=c.audit_manager,
            orchestrator=orchestrator,
            l1_agents=l1_agents,
            l2_agents=l2_agents,
            l3_agents=l3_agents,
            agent_executor=agent_executor, 
            cognitive_engine=cognitive_engine,
            order_manager=order_manager,
            event_distributor=event_distributor,
            knowledge_graph_service=c.knowledge_graph_service,
            ensemble_client=c.ensemble_client,
            risk_manager=risk_manager,
            # [Task FIX-MED-001] Export new components
            trade_lifecycle_manager=trade_lifecycle_manager,
            broker_adapter=broker_adapter
        )

    def get_component(self, name: str):
        """Gets a component by name."""
        component = getattr(self.container, name, None)
        if not component:
            logger.error(f"Component '{name}' not found in container.")
        return component
