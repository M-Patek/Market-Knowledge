Phoenix (Market Knowledge) RAG 架构

本文档是 Phoenix_project (Market Knowledge) 中 RAG (Retrieval-Augmented Generation) 和 AI 系统的核心架构图。

1. RAG 核心理念

Phoenix 的 RAG 架构旨在通过从多种异构数据源 (Vector, Graph, Tabular, Temporal) 检索上下文，为 L1, L2, 和 L3 层的 AI 智能体 (Agents) 提供支持。

RAG 流程 (L1 智能体):

查询 (Query): L1 智能体 (例如 FundamentalAnalyst) 产生一个分析任务 (例如 "分析 AAPL 的供应链风险")。

检索 (Retrieve): Retriever 模块接收此任务，并将其分发到多个数据源：

VectorStore (Qdrant): 检索相关的新闻、报告 (非结构化数据)。

GraphDBClient (Neo4j): 检索相关的实体、关系 (例如 AAPL -> SUPPLIES -> Foxconn)。

TabularDBClient (Postgres): 检索相关的财务报表 (结构化数据)。

TemporalDBClient (Elasticsearch): 检索相关的时序事件 (例如 "Foxconn 工厂停工")。

SearchClient (Tavily / Gemini): 检索实时网络信息。

GNN Inferencer (via Celery): 检索基于图神经网络的深层隐性关系。

重排 (Rerank): Retriever 使用 CrossEncoder 模型对所有来源的证据 (Evidence) 进行融合和重排。

增强 (Augment): 将排名靠前的证据 (Top-K) 注入到 L1 智能体的提示 (Prompt) 中。

生成 (Generate): L1 智能体使用 EnsembleClient (Gemini) 基于增强的上下文生成其分析报告。

2. 核心组件 (Core Components)

1. 数据源 (Data Sources)

非结构化 (Unstructured): PDF, HTML, DOCX, TXT.

摄入 (Ingestion): KnowledgeInjector -> DataAdapter (Text-to-Chunks) -> EmbeddingClient -> VectorStore.

结构化 (Structured): SQL, CSV.

摄入: DataManager -> TabularDBClient (PostgreSQL).

RAG: Retriever -> TabularDBClient -> DataAdapter (Table-to-Text) -> Augment.

图 (Graph): 知识图谱 (Knowledge Graph).

摄入: KnowledgeInjector -> RelationExtractor (LLM) -> GraphDBClient (Neo4j).

RAG: Retriever -> GraphDBClient (Text-to-Cypher or Keyword) -> Augment.

时序 (Temporal): Events, Logs.

摄入: DataManager -> TemporalDBClient (Elasticsearch).

RAG: Retriever -> TemporalDBClient -> Augment.

外部 (External): Web Search.

RAG: Retriever -> SearchClient (Tavily/Gemini) -> Augment.

2. (RAG) 内存 / 数据库 (Memory / Databases)

memory/vector_store.py (VectorStore)

实现: Qdrant (通过 qdrant-client).

用途: 存储 L1 证据 (Evidence) 和 RAG 文档 (Documents) 的嵌入。

配置: config/system.yaml -> memory.vector_store (Host: phoenix_qdrant).

ai/graph_db_client.py (GraphDBClient)

实现: Neo4j (通过 neo4j-driver).

用途: 存储实体和关系的知识图谱 (Graph-RAG)。

配置: config/system.yaml -> knowledge_graph.

ai/tabular_db_client.py (TabularDBClient)

实现: PostgreSQL (通过 sqlalchemy / asyncpg).

用途: 存储结构化财务数据 (Tabular-RAG)。

配置: config/system.yaml -> data_manager.tabular_db.

ai/temporal_db_client.py (TemporalDBClient)

实现: Elasticsearch (通过 elasticsearch-async).

用途: 存储时序事件 (Temporal-RAG)。

配置: config/system.yaml -> data_manager.temporal_db.

memory/cot_database.py (CoTDatabase)

实现: MongoDB (通过 motor).

用途: 存储所有智能体 (L1/L2/L3) 的思维链 (Chain-of-Thought) 和决策轨迹，用于审计 (Audit) 和元认知 (Metacognition)。

配置: config/system.yaml -> memory.cot_database.

3. (RAG) 知识图谱 (Knowledge Graph)

知识图谱是 Phoenix RAG 战略的核心，用于连接非结构化数据和结构化数据。

ai/relation_extractor.py (RelationExtractor):

用途: 在 KnowledgeInjector 流程中，使用 LLM (例如 gemini-1.5-flash) 从文本块 (Chunks) 中提取 (Subject, Predicate, Object) 三元组。

配置: config/system.yaml -> ai_components.relation_extractor.

ai/graph_db_client.py (GraphDBClient):

用途:

存储 RelationExtractor 提取的三元组。

在 RAG Retriever 中执行 Cypher 查询。

ai/retriever.py (Retriever):

Text-to-Cypher: Retriever (L1/L2) 使用 EnsembleClient (LLM) 和 PromptRenderer (text_to_cypher.json) 动态生成 Cypher 查询 (Graph-RAG 的一种形式)。

GNN (Graph Neural Network):

Graph-RAG: GNN Inference is integrated into the retrieval process.

[GNN Offloading]: GNN inference is computationally intensive and is now offloaded to a dedicated Celery queue (`gnn_queue`) in `ai/retriever.py`. This ensures the main RAG pipeline is not blocked by heavy matrix operations.

4. (RAG) 检索器 (Retriever)

ai/retriever.py 是 RAG 流程的核心协调器。

__init__: 接收所有数据库客户端 (VectorStore, GraphDBClient 等) 和 GNNInferencer。

retrieve_and_rerank (已重构):

并行检索 (Parallel Retrieval): 使用 asyncio.gather 并行调用所有数据源：

search_vector_db() (Vector)

search_knowledge_graph() (Graph Text-to-Cypher)

_query_graph_with_gnn() (Graph GNN-enhanced via Celery)

search_temporal_db()

search_tabular_db()

search_web() (Web Search via Tavily or Gemini)

鲁棒性 (Robustness - Resilient Gathering): 实现了 "Resilient Gathering" 模式。即使某个数据源 (如 GNN 或 Web Search) 失败或超时，检索器也会捕获异常并继续处理其他来源的结果，防止整个 RAG 流程中断。

融合 (Fusion): 将所有来源的 Evidence 列表合并。

重排 (Reranking): 使用 CrossEncoder (_rerank_documents) 对合并的列表进行重排，以生成最终的 Top-K 上下文。

5. AI 模型和提示 (AI Models & Prompts)

api/gateway.py (APIGateway):

用途: 集中管理所有外部 API (LLM, Embedding, Tools) 的密钥、速率限制和池化。

配置: config/system.yaml -> api_gateway.

ai/embedding_client.py (EmbeddingClient):

用途: 专门处理文本嵌入。

模型: **models/text-embedding-004** (Google)。已修复硬编码维度，支持动态维度检测。

配置: config/system.yaml -> api_gateway.embedding_model.

ai/ensemble_client.py (EnsembleClient):

用途: (RAG 生成) 抽象 LLM (例如 Gemini) 的调用，是所有 L1/L2/L3 智能体用于 "思考" 和 "生成" 的主要接口。

ai/prompt_manager.py & ai/prompt_renderer.py:

用途: (RAG 提示) 从 prompts/ 目录 (JSON) 加载、管理和动态渲染提示模板。

示例: text_to_cypher.json 包含用于 Graph-RAG 的提示。

3. L1/L2/L3 智能体 RAG 流程 (Agent RAG Flow)

L1 智能体 (L1 Agents)

示例: FundamentalAnalystAgent, GeopoliticalAnalystAgent.

RAG 用途: 重度消费者 (Heavy Consumers)。L1 智能体将其任务 (Task) 作为查询 (Query) 发送给 Retriever，并使用检索到的上下文 (Context) 来生成其分析 (Evidence)。

流程: LoopManager (L1) -> AgentExecutor -> L1Agent.execute() -> self.retriever.retrieve_and_rerank() -> (Augment) -> self.llm.generate() -> Save to VectorStore & CoTDatabase.

L2 智能体 (L2 Agents)

示例: FusionAgent, CriticAgent.

RAG 用途: 元消费者 (Meta Consumers)。L2 智能体 (例如 FusionAgent) 从 VectorStore (L1 证据) 和 CoTDatabase (L1 思维链) 中检索上下文，以生成更高层次的综合洞察 (FusionResult)。

[新] 认知引擎 (Cognitive Engine): 引入了 `CognitiveEngine` (engine.py) 来协调 L2 流程，包括 L2 融合 (ReasoningEnsemble) 和不确定性防御 (UncertaintyGuard)。

流程: LoopManager (L2) -> CognitiveEngine.process_cognitive_cycle() -> ReasoningEnsemble -> UncertaintyGuard -> Save to CoTDatabase.

L3 智能体 (L3 Agents)

示例: AlphaAgent, RiskAgent.

RAG 用途: 轻度消费者 (Light Consumers)。L3 智能体主要消费 L2 的 FusionResult 和 CriticResult。它们也可以按需从 Retriever 检索特定的 L1 证据或 CoT 轨迹以进行决策。

流程: Orchestrator (L3) -> L3Agent.execute() -> (Consume L2 Results) -> Generate Decision.
