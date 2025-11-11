# Phoenix_project/registry.py
import logging
from omegaconf import DictConfig
from types import SimpleNamespace # (主人喵的清洁计划 5.3) [新]

from context_bus import ContextBus
from data_manager import DataManager
from data.data_iterator import DataIterator # (主人喵的清洁计划 5.3) [新]
from controller.orchestrator import Orchestrator
from memory.vector_store import VectorStore
from memory.cot_database import CoTDatabase
from ai.embedding_client import EmbeddingClient
from ai.ensemble_client import EnsembleClient # (主人喵的清洁计划 5.3) [新]
from ai.prompt_manager import PromptManager
from ai.retriever import Retriever
from ai.data_adapter import DataAdapter # (主人喵的清洁计划 5.3) [新]
from ai.relation_extractor import RelationExtractor # (主人喵的清洁计划 5.3) [新]
from ai.graph_db_client import GraphDBClient # (主人喵的清洁计划 5.3) [新]
from ai.source_credibility import SourceCredibility # (主人喵的清洁计划 5.3) [新]
from knowledge_graph_service import KnowledgeGraphService # (主人喵的清洁计划 5.3) [新]
from audit_manager import AuditManager # (主人喵的清洁计划 5.3) [新]

# (主人喵的清洁计划 5.3) [新] L1 Agents
from agents.l1.catalyst_monitor_agent import CatalystMonitorAgent
from agents.l1.fundamental_analyst_agent import FundamentalAnalystAgent
from agents.l1.geopolitical_analyst_agent import GeopoliticalAnalystAgent
from agents.l1.innovation_tracker_agent import InnovationTrackerAgent
from agents.l1.macro_strategist_agent import MacroStrategistAgent
from agents.l1.supply_chain_intelligence_agent import SupplyChainIntelligenceAgent
from agents.l1.technical_analyst_agent import TechnicalAnalystAgent

# (主人喵的清洁计划 5.3) [新] L2 Agents
from agents.l2.planner_agent import PlannerAgent
from agents.l2.metacognitive_agent import MetacognitiveAgent
from agents.l2.fusion_agent import FusionAgent
from agents.l2.critic_agent import CriticAgent
from agents.l2.adversary_agent import AdversaryAgent

# (主人喵的清洁计划 5.3) [新] L3 Agents
from agents.l3.alpha_agent import AlphaAgent
from agents.l3.risk_agent import RiskAgent
from agents.l3.execution_agent import ExecutionAgent
from agents.l3.base import DRLAgentLoader # [主人喵的修复] 导入 DRL 加载器

# (主人喵的清洁计划 5.3) [新] Cognitive Layer
from cognitive.engine import CognitiveEngine
from cognitive.portfolio_constructor import PortfolioConstructor
from cognitive.risk_manager import RiskManager

# (主人喵的清洁计划 5.3) [新] Execution Layer
from execution.order_manager import OrderManager

# (主人喵的清洁计划 5.3) [新] Event Stream
from events.stream_processor import StreamProcessor
from events.risk_filter import RiskFilter
from events.event_distributor import EventDistributor


logger = logging.getLogger(__name__)

class Registry:
    """
    [主人喵的修复 11.10] 移除了 '(这是一个高级别的示例...)' 注释。
    
    The Registry is responsible for building and providing access to all
    core components of the system, managing dependencies.
    
    (这是对 5.3 计划的重大重构)
    """
    def __init__(self, config: DictConfig):
        self.config = config
        self.components = SimpleNamespace() # 使用 SimpleNamespace 来存放实例
        self._build_core()

    def _build_core(self):
        """Builds components that are shared across the system."""
        logger.info("Building core components...")
        
        # 1. 核心通信
        self.components.context_bus = ContextBus()
        
        # 2. 核心 AI / 数据服务
        self.components.embedding_client = EmbeddingClient(self.config.models.embedding)
        self.components.ensemble_client = EnsembleClient(self.config.models.llm_ensemble)
        self.components.prompt_manager = PromptManager(self.config.paths.prompts)

        # 3. 核心内存
        self.components.vector_store = VectorStore(self.config.memory.vector_store, self.components.embedding_client)
        self.components.cot_database = CoTDatabase(self.config.memory.cot_database)
        self.components.graph_db_client = GraphDBClient(self.config.memory.graph_database)

        # 4. 核心数据管理
        # (DataManager 现在负责加载所有数据，包括流数据和批数据)
        self.components.data_manager = DataManager(self.config.data, self.components.context_bus)

        # 5. 核心审计
        self.components.audit_manager = AuditManager(self.config.audit, self.components.cot_database)

        logger.info("Core components built.")

    def build_system(self, config: DictConfig) -> SimpleNamespace:
        """
        Builds and wires together the full agentic pipeline.
        This is separated to allow for different system configurations (e.g., test vs. prod).
        """
        logger.info("Building full Phoenix system pipeline...")
        
        # --- 获取核心组件 ---
        context_bus = self.components.context_bus
        ensemble_client = self.components.ensemble_client
        prompt_manager = self.components.prompt_manager
        vector_store = self.components.vector_store
        graph_db_client = self.components.graph_db_client
        data_manager = self.components.data_manager
        audit_manager = self.components.audit_manager
        embedding_client = self.components.embedding_client # (需要用于 L1)

        # --- AI/RAG 子系统 ---
        retriever = Retriever(vector_store, graph_db_client)
        data_adapter = DataAdapter()
        relation_extractor = RelationExtractor(ensemble_client, prompt_manager)
        source_credibility = SourceCredibility(config.ai.source_credibility)
        
        knowledge_graph_service = KnowledgeGraphService(
            graph_db_client=graph_db_client,
            relation_extractor=relation_extractor,
            retriever=retriever,
            audit_manager=audit_manager
        )

        # --- L1 智能体 (数据处理/分析) ---
        l1_agents = {
            "catalyst_monitor": CatalystMonitorAgent(ensemble_client, prompt_manager, audit_manager),
            "fundamental_analyst": FundamentalAnalystAgent(ensemble_client, prompt_manager, audit_manager, retriever),
            "geopolitical_analyst": GeopoliticalAnalystAgent(ensemble_client, prompt_manager, audit_manager, retriever),
            "innovation_tracker": InnovationTrackerAgent(ensemble_client, prompt_manager, audit_manager, retriever),
            "macro_strategist": MacroStrategistAgent(ensemble_client, prompt_manager, audit_manager, retriever),
            "supply_chain_intelligence": SupplyChainIntelligenceAgent(ensemble_client, prompt_manager, audit_manager, retriever),
            "technical_analyst": TechnicalAnalystAgent(ensemble_client, prompt_manager, audit_manager, data_adapter, embedding_client)
        }
        
        # --- L2 智能体 (认知/融合) ---
        l2_agents = {
            "planner": PlannerAgent(ensemble_client, prompt_manager, audit_manager),
            "metacognitive": MetacognitiveAgent(ensemble_client, prompt_manager, audit_manager),
            "fusion": FusionAgent(ensemble_client, prompt_manager, audit_manager, knowledge_graph_service, source_credibility),
            "critic": CriticAgent(ensemble_client, prompt_manager, audit_manager, retriever),
            "adversary": AdversaryAgent(ensemble_client, prompt_manager, audit_manager)
        }

        # --- L3 智能体 (决策/DRL) ---
        
        # --- 执行层 ---
        # (OrderManager 必须在 L3 ExecutionAgent 之前创建)
        order_manager = OrderManager(config.execution, context_bus)
        
        # [主人喵的修复 11.10] 移除了关于注入的 TODO 注释。
        # 当前架构 (OM 注入到 EA) 是一个有效的设计。

        # [主人喵的修复] (TBD): L3 Alpha 智能体的 DRL 特定配置。
        # 使用 DRLAgentLoader 从 system.yaml 中指定的检查点路径加载 L3 智能体。
        
        l3_agents = {
            "alpha": DRLAgentLoader.load_agent(
                agent_class=AlphaAgent,
                checkpoint_path=config.l3.alpha.checkpoint_path
            ),
            "risk": DRLAgentLoader.load_agent(
                agent_class=RiskAgent,
                checkpoint_path=config.l3.risk.checkpoint_path
            ),
            "execution": DRLAgentLoader.load_agent(
                agent_class=ExecutionAgent,
                checkpoint_path=config.l3.execution.checkpoint_path
            )
        }

        # [主人喵的修复] 检查 DRL 智能体是否加载成功
        if not l3_agents["alpha"] or not l3_agents["risk"] or not l3_agents["execution"]:
            logger.error("一个或多个 L3 DRL 智能体未能从检查点加载。请检查 config/system.yaml 中的 'checkpoint_path'。")
            # 我们可以根据配置决定是中止还是继续 (可能在没有 L3 的情况下运行)
            # raise RuntimeError("L3 DRL Agent loading failed.")

        # --- 认知层 (投资组合构建) ---
        portfolio_constructor = PortfolioConstructor(config.cognitive.portfolio, context_bus)
        
        # [主人喵的修复] (TBD): RiskAgent (决策) 和 RiskManager (认知) 之间的交互方式。
        # 解决方案: 将 L3 DRL RiskAgent 注入 RiskManager。
        # RiskManager (认知层) 将使用 L3 Agent (决策层) 的输出作为其规则的输入之一。
        risk_manager = RiskManager(
            config=config.cognitive.risk, 
            context_bus=context_bus,
            l3_drl_risk_agent=l3_agents["risk"] # <-- [修复] 在此注入
        )

        cognitive_engine = CognitiveEngine(
            portfolio_constructor=portfolio_constructor,
            risk_manager=risk_manager,
            context_bus=context_bus
            # [修复] L3 RiskAgent 已移至 RiskManager，此处不再需要
            # l3_risk_agent=l3_agents["risk"] 
        )

        # --- 事件流处理 ---
        # (这用于实时数据摄取)
        stream_processor = StreamProcessor(
            config=config.events.stream_processor,
            l1_agents=l1_agents,
            data_adapter=data_adapter,
            audit_manager=audit_manager,
            context_bus=context_bus
        )
        
        risk_filter = RiskFilter(config.events.risk_filter, context_bus)
        
        event_distributor = EventDistributor(
            config=config.events.distributor,
            stream_processor=stream_processor,
            risk_filter=risk_filter,
            context_bus=context_bus
        )

        # --- 主协调器 ---
        orchestrator = Orchestrator(
            l1_agents=l1_agents,
            l2_agents=l2_agents,
            l3_agents=l3_agents,
            cognitive_engine=cognitive_engine,
            context_bus=context_bus,
            audit_manager=audit_manager,
            data_adapter=data_adapter,
            knowledge_graph_service=knowledge_graph_service,
            config=config.orchestrator
        )

        # --- 数据迭代器 (用于回测) ---
        data_iterator = DataIterator(data_manager, config.data.iterator)

        logger.info("Full Phoenix system built.")

        # 返回所有构建的组件
        return SimpleNamespace(
            config=config,
            context_bus=context_bus,
            data_manager=data_manager,
            data_iterator=data_iterator,
            audit_manager=audit_manager,
            orchestrator=orchestrator,
            l1_agents=l1_agents,
            l2_agents=l2_agents,
            l3_agents=l3_agents,
            cognitive_engine=cognitive_engine,
            order_manager=order_manager,
            event_distributor=event_distributor,
            knowledge_graph_service=knowledge_graph_service,
            # (添加其他需要从外部访问的组件)
        )

    def get_component(self, name: str):
        """Gets a component by name."""
        component = getattr(self.components, name, None)
        if not component:
            logger.error(f"Component '{name}' not found in registry.")
            # (或者我们可以实现延迟加载)
        return component
