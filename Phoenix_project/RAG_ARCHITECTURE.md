Phoenix Project: RAG Infrastructure Architectural Design
Version: 2.0 (Final)
Status: Implemented
Date: 2025-10-17

Overview & Design Philosophy
This document specifies the architecture for the Retrieval-Augmented Generation (RAG) infrastructure of the Phoenix Project (V2.0). The system is designed to provide the AI cognitive layer with highly relevant, timely, and multi-modal evidence to support its analytical tasks.

The core philosophy is a hybrid, parallel retrieval architecture. This design acknowledges that financial data is heterogeneous and that no single indexing method is sufficient. By querying multiple, specialized indexes concurrently and fusing their results, we achieve a more comprehensive and contextually aware retrieval than any single method could provide.

The system is composed of three specialized retrieval indexes and a unified query, fusion, and re-ranking layer.

Component: Vector Index (Semantic Search)
The Vector Index is the primary engine for understanding the semantic meaning and context of unstructured text.

Technology: Pinecone Serverless. This provides a scalable, managed vector database with low-latency query performance and robust API support, minimizing operational overhead.

Embedding Models:

Unstructured Text (News, Filings): text-embedding-3-large (OpenAI), as defined in models/embedding_models/. Chosen for its state-of-the-art performance in capturing semantic meaning.

Query Enhancement (HyDE): To improve semantic retrieval, the system implements Hypothetical Document Embedding (HyDE). Before searching, a high-capability LLM (gemini-2.5-pro) generates a "hypothetical" document that perfectly answers the query. This document (not the query) is then embedded and used for the vector search, leading to more robust semantic matching.

Metadata Storage: Each vector is stored with a rich metadata payload (source_id, available_at, source_type, license) to facilitate post-query filtering.

Component: Temporal Index (Time-Aware Search)
The Temporal Index is a specialized inverted index designed for fast, time-windowed queries on event-based data.

Technology: Elasticsearch. Its powerful time-series capabilities, date math, and robust keyword/entity search make it the ideal choice.

Index Structure: Stores documents with timestamp, entities (e.g., 'Federal Reserve', 'AAPL'), keywords, and source_id.

Query Capability: This index supports queries like: "Find all documents mentioning 'rate hike' and 'Federal Reserve' between August and September 2023." This maps directly to the ai/temporal_db_client.py implementation.

Component: Tabular Index (Structured Data Search)
The Tabular Index is designed for direct, structured queries against financial data extracted from tables (e.g., earnings reports, economic releases).

Technology: PostgreSQL with NUMERIC and JSONB columns. This provides the power of precise SQL filtering with the flexibility of a document store.

Data Model: A dedicated table stores parsed financial data with ticker, report_date, metric_name, metric_value, and a metadata_jsonb field.

Query Capability: Allows for precise queries like: SELECT metric_value FROM financial_data WHERE ticker = 'AAPL' AND metric_name = 'Revenue' AND report_date > '2023-01-01'. This maps to the ai/tabular_db_client.py implementation.

Unified Query & Fusion Layer (V2.0)
This layer acts as the orchestrator for the entire retrieval process, as implemented in ai/retriever.py and consumed by the MetacognitiveAgent.

Query Dispatcher: A single query triggers parallel sub-queries to all three indexes (Vector, Temporal, Tabular) using asyncio.gather.

Stage 1: Multi-Modal Fusion (RRF): The results (lists of documents) from all three indexes are collected. They are then fused into a single, ranked candidate list using Reciprocal Rank Fusion (RRF). RRF effectively merges disparate ranking scores from different systems without needing score normalization.

Stage 2: Deep Re-ranking (Cross-Encoder): The fused candidate list from Stage 1 is passed to a Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2). The Cross-Encoder performs deep semantic re-ranking by comparing the original query directly against the content of each retrieved document, providing a highly accurate final relevance score.

Stage 3: Knowledge Graph Integration (GNN): Concurrently, entities from the query are used to query a Knowledge Graph. The resulting subgraph is passed to a GNNEncoder (using tensorflow-gnn's RGCNConv), which encodes the relational structure into a dense embedding vector. This vector is provided to the MetaLearner as a separate, fused context.
