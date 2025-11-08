Phoenix Project: RAG Infrastructure Architectural Design
Version: 2.2 (已根据代码库 v2.1 更新)
Status: Implemented
Date: 2025-11-09

概述 & 设计哲学 (Overview & Design Philosophy)

本文档阐明了 Phoenix Project (V2.1) 的检索增强生成 (RAG) 基础设施架构。该系统旨在为 AI 认知层提供高度相关、及时且多模式的证据，以支持其分析任务。

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

[已更新] 统一查询与 AI 层 (Unified Query & AI Layer)

该层充当整个检索和知识处理过程的协调器。

查询调度器 (Query Dispatcher - ai/retriever.py):

Retriever 类（在 ai/retriever.py 中实现）协调对所有数据存储（向量、时序、CoT 历史、图数据库）的并发异步查询。

[已确认] 高级检索已实现: 与旧版文档不同，高级检索功能 已在 ai/retriever.py 代码中完全实现。

1. RRF 融合: retrieve_relevant_context 方法调用 _apply_rrf，使用倒数排序融合 (Reciprocal Rank Fusion) 算法，将来自多个异构索引的搜索结果合并为一个统一的排序列表。

2. Cross-Encoder 重排: 随后，retrieve_relevant_context 方法调用 _apply_reranking，它使用 sentence-transformers 库中的 CrossEncoder 模型（在 __init__ 中加载，模型名称由 config/system.yaml 定义）对 RRF 融合后的结果进行深度的语义重排，以确保最高的上下文相关性。

知识图谱 (Knowledge Graph - ai/graph_encoder.py, ai/relation_extractor.py):

GraphEncoder 和 RelationExtractor 类使用 LLM 从非结构化文本中提取结构化关系（节点和边）。

KnowledgeGraphService (knowledge_graph_service.py) 协调这些模块，以更新图谱（在代码中目前为存根 graph_db_stub）。

requirements.txt 中包含 tensorflow-gnn，表明 GNN (图神经网络) 是此架构的预期部分。

消费者 (Consumer - ai/reasoning_ensemble.py):

Retriever 的主要消费者是 ReasoningEnsemble (在 ai/reasoning_ensemble.py 中)。

ReasoningEnsemble 在其 reason 方法中调用 retriever.retrieve_relevant_context 和 retriever.format_context_for_prompt，然后将检索到的上下文和 L1/L2 智能体的决策传递给 MetacognitiveAgent (agents/l2/metacognitive_agent.py) 进行监督。
