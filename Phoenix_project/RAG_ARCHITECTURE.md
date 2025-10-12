Phoenix Project: RAG Infrastructure Architectural Design
Version: 1.0
Status: Proposed
Date: 2025-10-12

1. Overview & Design Philosophy
This document specifies the architecture for the Retrieval-Augmented Generation (RAG) infrastructure of the Phoenix Project. The system is designed to provide the AI cognitive layer with highly relevant, timely, and multi-modal evidence to support its analytical tasks.

The core philosophy is a hybrid, parallel retrieval architecture. This design acknowledges that financial data is heterogeneous and that no single indexing method is sufficient. By querying multiple, specialized indexes concurrently and fusing their results, we achieve a more comprehensive and contextually aware retrieval than any single method could provide.

The system is composed of three specialized retrieval indexes and a unified query and fusion layer.

2. Component: Vector Index (Semantic Search)
The Vector Index is the primary engine for understanding the semantic meaning and context of unstructured text.

Technology: We will deploy a managed Pinecone instance. This choice provides a scalable, serverless vector database with low-latency query performance and robust API support, minimizing our operational overhead.

Embedding Models: The system will support multiple, source-specific embedding models to capture the unique nuances of different data types.

Unstructured Text (News, Filings): text-embedding-3-large (OpenAI). Chosen for its state-of-the-art performance in capturing semantic meaning in long-form text.

Tabular Data: We will research and integrate a specialized model (e.g., TaPas or a custom-trained model) to generate embeddings from structured table rows.

Metadata Storage: Each vector will be stored with a rich metadata payload to facilitate post-query filtering and provide context to the fusion layer. The metadata will include:

source_id: The unique ID of the document.

available_at: The timestamp when the information became valid.

observed_at: The timestamp of ingestion.

source_type: (e.g., 'news', 'sec_filing').

license: Data usage license.

3. Component: Temporal Index (Time-Aware Search)
The Temporal Index is a specialized inverted index designed for fast, time-windowed queries. It answers the question of when events occurred.

Technology: Elasticsearch. Its powerful time-series capabilities, date math, and robust keyword search functionality make it the ideal choice for this component.

Index Structure: The index will store documents with the following key fields:

timestamp: The available_at timestamp of the event.

entities: A list of named entities (e.g., 'Federal Reserve', 'AAPL') extracted from the source document.

keywords: A list of key terms.

source_id: The unique ID of the document, to link back to the full content.

Query Capability: This index must support queries like: "Find all documents mentioning 'rate hike' and 'Federal Reserve' between August and September 2023."

4. Component: Tabular Index (Structured Data Search)
The Tabular Index is designed for direct, structured queries against financial data extracted from tables, such as earnings reports or economic releases.

Technology: PostgreSQL with a JSONB column. This provides the flexibility of a document store with the power of SQL for structured querying.

Data Model: A dedicated table will be created to store parsed financial data. Each row will represent a specific data point (e.g., one quarter's earnings for one company) and will contain:

source_id: The unique ID of the source document.

ticker: The relevant asset ticker.

report_date: The date of the report.

metric_name: (e.g., 'Revenue', 'EPS').

metric_value: The numerical value.

metadata_jsonb: A flexible JSONB field to store other relevant dimensions (e.g., 'currency', 'segment').

Query Capability: This index allows for precise queries like: "SELECT metric_value FROM financial_data WHERE ticker = 'AAPL' AND metric_name = 'Revenue' AND report_date > '2023-01-01'".

5. Unified Query & Fusion Layer
This layer acts as the orchestrator for the entire retrieval process.

Query Dispatcher: A single API endpoint will receive a user's query. The dispatcher will be responsible for translating this query into appropriate sub-queries for each of the three indexes (Vector, Temporal, Tabular) and executing them in parallel using asyncio.

Fusion Logic: The results from all three indexes will be collected and fused into a single candidate list. The initial fusion logic will be a simple union of all retrieved source_ids, with duplicates removed. This unified list will then be passed to the two-stage re-ranker.

API Contract: The layer will expose a clear, well-defined API for other services to use, abstracting away the complexity of the underlying hybrid index architecture.
