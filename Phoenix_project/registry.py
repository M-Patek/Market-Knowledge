# Phoenix_project/registry.py
import logging
import importlib
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
from Phoenix_project.knowledge_graph_service import KnowledgeGraphService
from Phoenix_project.audit_manager import AuditManager

# 核心认知/执行组件
from Phoenix_project.cognitive.engine import CognitiveEngine
from Phoenix_project.cognitive.portfolio_constructor import PortfolioConstructor
from Phoenix_project.cognitive.risk_manager import RiskManager
from Phoenix_project.execution.order_manager import OrderManager
from Phoenix_project.events.stream_processor import StreamProcessor
from Phoenix_project.events.risk_filter import RiskFilter
from Phoenix_project.events.event_distributor import EventDistributor


logger = logging.getLogger(__name__)

class DependencyContainer:
    """
    [Task 1.1] 集中式依赖容器。
    持有核心服务的单例，并传递给工厂以解析依赖。
    """
    def __init__(self):
        self.context_bus: Optional[ContextBus] = None
        self.data_manager: Optional[DataManager] = None
        self.embedding_client: Optional[EmbeddingClient] = None
        self.ensemble_client: Optional[EnsembleClient] = None
        self.prompt_manager: Optional[PromptManager] = None
        self.audit_manager: Optional[AuditManager] = None
        self.vector_store: Optional[VectorStore] = None
        self.cot_database: Optional[CoTDatabase] = None
        self.graph_db_client: Optional[GraphDBClient] = None
        
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
        
        # [Adapter Pattern] 根据类名注入特定的依赖
        # 标准基础依赖: (ensemble, prompt, audit)
        args = [container.ensemble_client, container.prompt_manager, container.audit_manager]
        
        if key == "technical_analyst":
            return agent_cls(*args, container.data_adapter, container.embedding_client)
        elif key in ["fundamental_analyst", "geopolitical_analyst", "innovation_tracker", "macro_strategist", "supply_chain_intelligence", "critic"]:
             # 分析师和批评家需要 Retriever
            return agent_cls(*args, container.retriever)
        elif key == "fusion":
            return agent_cls(*args, container.knowledge_graph_service, container.source_credibility)
        else:
            # 默认 (Catalyst, Planner, Metacognitive, Adversary)
            return agent_cls(*args)

    @staticmethod
    def create_l3_agent(key: str, config: DictConfig, container: DependencyContainer) -> Any:
        """使用 DRLAgentLoader 加载 L3 智能体的特殊加载器。"""
        # 延迟导入 DRL Loader 以避免循环依赖
        from Phoenix_project.agents.l3.base import DRLAgentLoader
        
        if key not in AgentFactory.L3_MAP:
            return None

        module_path, class_name = AgentFactory.L3_MAP[key]
        agent_cls = AgentFactory._import_agent_class(module_path, class_name)
        
        # 从配置中解析检查点路径
        # 假设配置结构: l3.<key>.checkpoint_path
        if not hasattr(config.l3, key):
             logger.warning(f"Configuration for L3 agent '{key}' not found.")
             return None

        ckpt_path = getattr(config.l3, key).checkpoint_path
        
        logger.info(f"Loading L3 Agent {key} from {ckpt_path}")
        return DRLAgentLoader.load_agent(agent_class=agent_cls, checkpoint_path=ckpt_path)

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
        
        # 1. 核心通信
        self.container.context_bus = ContextBus()
        
        # 2. 核心 AI / 数据服务
        self.container.embedding_client = EmbeddingClient(self.config.models.embedding)
        # 注意: EnsembleClient 现在需要注入 registry 本身 (Task 1.3), 但为了打破鸡生蛋问题，
        # 我们可能需要稍微调整 EnsembleClient 或在此处传递 self (如果 EnsembleClient 支持)
        # 假设 EnsembleClient 已经被 patched 以支持 DependencyContainer 或稍后设置
        # 为简单起见，这里按原样初始化，如果 EC 需要 registry，建议 setter 注入
        self.container.ensemble_client = EnsembleClient(
            gemini_manager=None, # TBD: Wired up in main
            prompt_manager=None, # Will be set next
            agent_registry=self.config.agents,
            global_registry=self # 注入 self
        )
        self.container.prompt_manager = PromptManager(self.config.paths.prompts)
        
        # Wiring up EnsembleClient deps that were just created
        self.container.ensemble_client.prompt_manager = self.container.prompt_manager

        # 3. 核心内存
        self.container.vector_store = VectorStore(self.config.memory.vector_store, self.container.embedding_client)
        self.container.cot_database = CoTDatabase(self.config.memory.cot_database)
        self.container.graph_db_client = GraphDBClient(self.config.memory.graph_database)

        # 4. 核心数据管理
        self.container.data_manager = DataManager(self.config.data, redis_client=None) # Redis TBD via wiring

        # 5. 核心审计
        self.container.audit_manager = AuditManager(self.config.audit, self.container.cot_database)

        logger.info("Core components built.")

    def build_system(self, config: DictConfig) -> SimpleNamespace:
        """
        Builds and wires together the full agentic pipeline.
        """
        logger.info("Building full Phoenix system pipeline...")
        
        c = self.container

        # --- AI/RAG 子系统 ---
        c.retriever = Retriever(c.vector_store, c.graph_db_client)
        c.data_adapter = DataAdapter()
        relation_extractor = RelationExtractor(c.ensemble_client, c.prompt_manager)
        c.source_credibility = SourceCredibility(config.ai.source_credibility)
        
        c.knowledge_graph_service = KnowledgeGraphService(
            graph_db_client=c.graph_db_client,
            relation_extractor=relation_extractor,
            retriever=c.retriever,
            audit_manager=c.audit_manager
        )

        # --- L1 智能体 (数据处理/分析) ---
        l1_agents = {}
        for key in AgentFactory.L1_MAP.keys():
             l1_agents[key] = AgentFactory.create_agent(AgentFactory.L1_MAP, key, c)
        
        # --- L2 智能体 (认知/融合) ---
        l2_agents = {}
        for key in AgentFactory.L2_MAP.keys():
            l2_agents[key] = AgentFactory.create_agent(AgentFactory.L2_MAP, key, c)

        # --- L3 智能体 (决策/DRL) ---
        l3_agents = {
            "alpha": AgentFactory.create_l3_agent("alpha", config, c),
            "risk": AgentFactory.create_l3_agent("risk", config, c),
            "execution": AgentFactory.create_l3_agent("execution", config, c),
        }

        # Check Load Status
        if not l3_agents["alpha"] or not l3_agents["risk"] or not l3_agents["execution"]:
            logger.error("One or more L3 DRL Agents failed to load from checkpoint.")

        # --- 执行层 ---
        order_manager = OrderManager(
            broker=None, # TBD: Injected from main
            trade_lifecycle_manager=None, # TBD
            data_manager=c.data_manager
        )

        # --- 认知层 (投资组合构建) ---
        portfolio_constructor = PortfolioConstructor(config.cognitive.portfolio, c.context_bus)
        
        risk_manager = RiskManager(
            config=config.cognitive.risk, 
            redis_client=None, # TBD
            data_manager=c.data_manager
        )

        cognitive_engine = CognitiveEngine(
            portfolio_constructor=portfolio_constructor,
            risk_manager=risk_manager,
            context_bus=c.context_bus
        )

        # --- 事件流处理 ---
        stream_processor = StreamProcessor(
            config=config.events.stream_processor,
            l1_agents=l1_agents,
            data_adapter=c.data_adapter,
            audit_manager=c.audit_manager,
            context_bus=c.context_bus
        )
        
        risk_filter = RiskFilter(config.events.risk_filter, c.context_bus)
        
        event_distributor = EventDistributor(
            config=config.events.distributor,
            stream_processor=stream_processor,
            risk_filter=risk_filter,
            context_bus=c.context_bus
        )

        # --- 主协调器 ---
        orchestrator = Orchestrator(
            config=config.orchestrator,
            context_bus=c.context_bus,
            data_manager=c.data_manager,
            cognitive_engine=cognitive_engine,
            event_distributor=event_distributor,
            event_filter=risk_filter,
            market_state_predictor=None, # TBD: Add if needed
            portfolio_constructor=portfolio_constructor,
            order_manager=order_manager,
            audit_manager=c.audit_manager,
            trade_lifecycle_manager=None, # TBD
            # Inject Agents
            alpha_agent=l3_agents["alpha"],
            risk_agent=l3_agents["risk"],
            execution_agent=l3_agents["execution"]
        )

        # --- 数据迭代器 (用于回测) ---
        data_iterator = DataIterator(config.data.iterator, c.data_manager)

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
            cognitive_engine=cognitive_engine,
            order_manager=order_manager,
            event_distributor=event_distributor,
            knowledge_graph_service=c.knowledge_graph_service,
            ensemble_client=c.ensemble_client,
            risk_manager=risk_manager
        )

    def get_component(self, name: str):
        """Gets a component by name."""
        component = getattr(self.container, name, None)
        if not component:
            logger.error(f"Component '{name}' not found in container.")
        return component
