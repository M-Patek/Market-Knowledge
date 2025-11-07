Phoenix Project: RAG Infrastructure Architectural Design
Version: 2.1 (已根据代码库 v2.0 更新)
Status: Implemented (部分为 Mock)
Date: 2025-11-08

概述 & 设计哲学 (Overview & Design Philosophy)

本文档阐明了 Phoenix Project (V2.0) 的检索增强生成 (RAG) 基础设施架构。该系统旨在为 AI 认知层提供高度相关、及时且多模式的证据，以支持其分析任务。

核心哲学是一种混合、并行的检索架构。此设计承认金融数据是异构的，单一索引方法不足以应对。通过并发查询多个专业化索引并融合其结果，我们实现了比任何单一方法都更全面、上下文感知能力更强的检索。

该系统由三个专业检索索引和三个核心 AI 模块（Retriever, RelationExtractor, GraphEncoder）组成。

组件 1: 向量索引 (Vector Index - 语义搜索)

向量索引是理解非结构化文本语义和上下文的主要引擎。

技术 (Technology): pinecone-client (如 requirements.txt 所示) 或其他向量数据库 (如 memory/vector_store.py 中模拟的 Chroma/Mock)。env.example 中包含 PINECONE_API_KEY，表明 Pinecone 是预期的生产环境技术。

Embedding 模型 (Embedding Models):

非结构化文本 (News, Filings): text-embedding-004 (Google)。[根据 config/system.yaml 和 ai/embedding_client.py 确认]。

元数据存储 (Metadata Storage): 每个向量都存储有丰富的元数据负载（source_id, available_at, source_type 等），以便进行查询后过滤。

组件 2: 时序索引 (Temporal Index - 时间感知搜索)

时序索引是一种专门的倒排索引，设计用于对基于事件的数据进行快速、带时间窗口的查询。

技术 (Technology): elasticsearch>=8.13 (如 requirements.txt 和 config/system.yaml 所示)。

索引结构 (Index Structure): 存储包含 timestamp, entities (例如 'Federal Reserve', 'AAPL'), keywords 和 source_id 的文档。

查询能力 (Query Capability): 此索引支持 "查找在 2023 年 8 月至 9 月期间提及 'rate hike' 和 'Federal Reserve' 的所有文档" 之类的查询。这直接映射到 ai/temporal_db_client.py 的实现。

组件 3: 表格索引 (Tabular Index - 结构化数据搜索)

表格索引专为针对从表格（例如，财报、经济发布）中提取的结构化金融数据进行直接、精确的查询而设计。

技术 (Technology): PostgreSQL (如 requirements.txt 和 config/system.yaml 所示)。

数据模型 (Data Model): ai/tabular_db_client.py 中的查询逻辑表明，存在一个（如 financial_metrics）表，用于存储解析后的金融数据，包含 ticker, report_date, metric_name, metric_value 等字段。

查询能力 (Query Capability): 允许精确的 SQL 查询，例如：SELECT metric_value FROM financial_metrics WHERE ticker = 'AAPL' AND metric_name = 'Revenue' AND report_date > '2023-01-01'。

统一查询与 AI 层 (Unified Query & AI Layer)

该层充当整个检索和知识处理过程的协调器。

查询调度器 (Query Dispatcher - ai/retriever.py):

Retriever 类（在 ai/retriever.py 中实现）协调对各种数据存储的查询。

retrieve_relevant_context 方法调用 vector_store.search (用于向量) 和 cot_database.search_traces (用于历史推理)。

注意: RAG_ARCHITECTURE.md (V2.0) 中描述的 HyDE、RRF (Reciprocal Rank Fusion) 和 Cross-Encoder 深度重排 (Stage 1 & 2) 尚未在 ai/retriever.py 的当前代码中实现。config/system.yaml 中虽然定义了 rerank.model，但检索器代码目前仅执行简单的上下文组装。

知识图谱 (Knowledge Graph - ai/graph_encoder.py, ai/relation_extractor.py):

GraphEncoder 和 RelationExtractor 类使用 LLM 从非结构化文本中提取结构化关系（节点和边）。

KnowledgeGraphService (knowledge_graph_service.py) 协调这些模块，以更新图谱（在代码中目前为存根 graph_db_stub）。

requirements.txt 中包含 tensorflow-gnn，表明 GNN (图神经网络) 是此架构的预期部分，符合原始 V2.0 文档的 Stage 3 描述。

消费者 (Consumer - ai/reasoning_ensemble.py):

Retriever 的主要消费者是 ReasoningEnsemble (在 ai/reasoning_ensemble.py 中)。

ReasoningEnsemble 在其 reason 方法中调用 retriever.retrieve 和 retriever.format_context，然后将检索到的上下文和 L1/L2 智能体的决策传递给 MetacognitiveAgent (metacognitive_agent.supervise) 进行监督。
