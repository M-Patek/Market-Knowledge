import os
import asyncio
from typing import List, Dict, Any, Optional

# FIX: Changed import from 'observability' to 'monitor.logging'
from monitor.logging import get_logger
from ai.embedding_client import EmbeddingClient
from ai.relation_extractor import RelationExtractor
from memory.vector_store import VectorStore
from knowledge_graph_service import KnowledgeGraphService
from data_manager import DataManager
from core.schemas.data_schema import NewsArticle, Document
from ai.prompt_manager import PromptManager

logger = get_logger('KnowledgeInjector')

class KnowledgeInjector:
    """
    Handles the ingestion pipeline:
    1. Receives raw data (e.g., text, documents).
    2. Processes and chunks the data.
    3. Generates embeddings.
    4. Extracts entities and relations for the Knowledge Graph.
    5. Stores data in the Vector Store and Knowledge Graph.
    """
    
    def __init__(self, 
                 config: Dict[str, Any], 
                 data_manager: DataManager,
                 vector_store: VectorStore,
                 kg_service: KnowledgeGraphService,
                 embedding_client: EmbeddingClient,
                 gemini_pool: Optional[Any] = None): # Optional GeminiPoolManager
        
        self.config = config.get('knowledge_injector', {})
        self.data_manager = data_manager
        self.vector_store = vector_store
        self.kg_service = kg_service
        self.embedding_client = embedding_client
        self.gemini_pool = gemini_pool # Use the shared pool

        # Initialize the Relation Extractor
        # It needs access to an LLM, so we pass the pool
        prompt_manager = PromptManager(config.get('prompts', {}))
        self.relation_extractor = RelationExtractor(
            gemini_pool=self.gemini_pool,
            prompt_manager=prompt_manager,
            config=config.get('relation_extractor', {})
        )
        
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
        
        logger.info("KnowledgeInjector initialized.")

    async def inject_document(self, filepath: str, source: str) -> str:
        """
        Main entry point for ingesting a file (PDF, TXT, MD, etc.).
        """
        logger.info(f"Starting ingestion for file: {filepath} from source: {source}")
        try:
            # 1. Load and parse data using DataManager
            documents: List[Document] = self.data_manager.load_document(filepath, source)
            
            if not documents:
                logger.warning(f"No documents extracted from file: {filepath}")
                return "Failed: No content extracted"

            # 2. Process documents in parallel
            tasks = [self._process_single_document(doc) for doc in documents]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = 0
            fail_count = 0
            for res in results:
                if isinstance(res, Exception):
                    logger.error(f"Failed to process a document chunk: {res}", exc_info=res)
                    fail_count += 1
                else:
                    success_count += 1

            logger.info(f"Ingestion complete for {filepath}. Success: {success_count}, Failed: {fail_count}")
            return f"Ingestion complete. Success: {success_count}, Failed: {fail_count}"

        except Exception as e:
            logger.error(f"High-level error during document injection for {filepath}: {e}", exc_info=True)
            return f"Failed: {str(e)}"

    async def inject_news_article(self, article: NewsArticle) -> bool:
        """
        Main entry point for ingesting a single NewsArticle object.
        """
        logger.info(f"Starting ingestion for NewsArticle: {article.article_id} ({article.headline})")
        try:
            # Convert NewsArticle to a standard Document for processing
            doc = Document(
                doc_id=article.article_id,
                content=f"{article.headline}\n\n{article.summary}\n\n{article.full_text}",
                metadata={
                    "source": article.source,
                    "published_at": article.published_at.isoformat(),
                    "url": article.url,
                    "type": "NewsArticle",
                    "symbols": ", ".join(article.symbols) if article.symbols else ""
                }
            )
            
            await self._process_single_document(doc)
            logger.info(f"Successfully ingested NewsArticle: {article.article_id}")
            return True

        except Exception as e:
            logger.error(f"Error during NewsArticle injection for {article.article_id}: {e}", exc_info=True)
            return False

    async def _process_single_document(self, doc: Document):
        """
        Processes a single Document object (which could be a chunk or a full doc).
        """
        try:
            # 1. Generate embedding for the document content
            # This embedding represents the entire chunk/document
            doc_embedding = await self.embedding_client.get_embedding(doc.content)
            
            if doc_embedding is None:
                logger.warning(f"Failed to generate embedding for doc: {doc.doc_id}")
                return

            # 2. Store in VectorStore
            await self.vector_store.upsert(
                doc_id=doc.doc_id,
                vector=doc_embedding,
                text=doc.content,
                metadata=doc.metadata
            )
            logger.debug(f"Upserted document chunk to VectorStore: {doc.doc_id}")

            # 3. Extract and Store Knowledge Graph triples
            # This is an async call to the RelationExtractor
            triples = await self.relation_extractor.extract(doc.content)
            
            if triples:
                await self.kg_service.add_triples(triples)
                logger.debug(f"Added {len(triples)} triples to KG from doc: {doc.doc_id}")
            else:
                logger.debug(f"No triples extracted from doc: {doc.doc_id}")

        except Exception as e:
            logger.error(f"Failed to process document {doc.doc_id}: {e}", exc_info=True)
            # Re-raise to be caught by asyncio.gather
            raise

# Example of how this might be run (e.g., in a worker or script)
async def main():
    from config.system import load_config
    from memory.vector_store import VectorStore
    from knowledge_graph_service import KnowledgeGraphService
    from ai.embedding_client import EmbeddingClient
    from api.gemini_pool_manager import GeminiPoolManager
    
    # --- Configuration ---
    config = load_config('config/system.yaml')
    
    # --- Dependencies ---
    data_manager = DataManager(config, None) # Pass None for PipelineState if not needed
    
    vector_store = VectorStore(config.get('vector_store', {}))
    await vector_store.initialize() # Initialize connection
    
    kg_service = KnowledgeGraphService(config.get('knowledge_graph', {}))
    await kg_service.initialize() # Initialize connection
    
    gemini_pool = GeminiPoolManager(
        api_key=os.environ.get("GEMINI_API_KEY"),
        pool_size=config.get('llm', {}).get('gemini_pool_size', 5)
    )
    
    embedding_client = EmbeddingClient(
        gemini_pool=gemini_pool,
        config=config.get('embedding_client', {})
    )

    # --- Injector ---
    injector = KnowledgeInjector(
        config=config,
        data_manager=data_manager,
        vector_store=vector_store,
        kg_service=kg_service,
        embedding_client=embedding_client,
        gemini_pool=gemini_pool
    )
    
    # --- Example Usage ---
    
    # 1. Inject a local file
    filepath = "path/to/your/document.pdf"
    if os.path.exists(filepath):
        result_file = await injector.inject_document(filepath, source="local_pdf_upload")
        print(result_file)
    else:
        print(f"File not found: {filepath}")

    # 2. Inject a NewsArticle object (e.g., from a stream)
    from datetime import datetime
    sample_article = NewsArticle(
        article_id="news_12345",
        source="Financial Times",
        headline="Major Tech Firm Announces Stock Split",
        summary="A major tech firm announced a 5-for-1 stock split...",
        full_text="...",
        published_at=datetime.utcnow(),
        url="https://example.com/news/12345",
        symbols=["TECHCO"]
    )
    result_news = await injector.inject_news_article(sample_article)
    print(f"News article injection success: {result_news}")

    # --- Shutdown ---
    await gemini_pool.close()
    await kg_service.close()
    logger.info("Shutdown complete.")

if __name__ == "__main__":
    # Note: Running this main requires an active event loop
    # In a real scenario, this would be part of a larger async application
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Failed to run KnowledgeInjector main: {e}")
