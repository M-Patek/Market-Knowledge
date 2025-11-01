import asyncio
from typing import List, Dict, Any

# 修复：
# 1. 移除 'from .'，因为此文件位于根目录
# 2. 'NewsArticle' 拼写错误改为 'NewsData'
from data_manager import DataManager
from memory.vector_store import VectorStore
from ai.embedding_client import EmbeddingClient
from core.schemas.data_schema import NewsData
from monitor.logging import get_logger

logger = get_logger(__name__)

class KnowledgeInjector:
    """
    A service that continuously streams data from the DataManager,
    processes it (e.g., generates embeddings), and injects it into
    a long-term memory store (like VectorStore).
    """

    def __init__(self,
                 data_manager: DataManager,
                 vector_store: VectorStore,
                 embedding_client: EmbeddingClient):
        """
        Initialize the injector with necessary clients.
        """
        self.data_manager = data_manager
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self._running = False
        logger.info("KnowledgeInjector initialized.")

    async def start(self):
        """
        Starts the continuous injection background tasks.
        """
        if self._running:
            logger.warning("KnowledgeInjector is already running.")
            return

        self._running = True
        logger.info("KnowledgeInjector started.")
        
        # Start background tasks for different data types
        # (Using asyncio.create_task to run them concurrently)
        self.news_task = asyncio.create_task(self.process_news_stream())
        # self.market_task = asyncio.create_task(self.process_market_data_stream())
        
        # Add more tasks for other data types (e.g., filings, social media)
        
        try:
            await asyncio.gather(self.news_task) # Add other tasks here
        except asyncio.CancelledError:
            logger.info("KnowledgeInjector tasks cancelled.")
        except Exception as e:
            logger.error(f"Error in KnowledgeInjector run loop: {e}", exc_info=True)
        finally:
            self._running = False
            logger.info("KnowledgeInjector stopped.")

    def stop(self):
        """
        Stops the running background tasks.
        """
        if not self._running:
            logger.warning("KnowledgeInjector is not running.")
            return
            
        logger.info("Stopping KnowledgeInjector...")
        if self.news_task:
            self.news_task.cancel()
        
        # Cancel other tasks...
        
        self._running = False

    async def process_news_stream(self):
        """
        Processes the news data stream.
        """
        logger.info("Starting news stream processing...")
        try:
            # data_manager.stream_data should be an async generator
            async for news_batch in self.data_manager.stream_data(data_type='news'):
                if not self._running:
                    break
                if not news_batch:
                    await asyncio.sleep(1) # Wait if batch is empty
                    continue
                
                # 修复：'NewsArticle' (拼写错误) 改为 'NewsData'
                articles: List[NewsData] = [NewsData(**item) for item in news_batch]
                
                texts_to_embed = [self._format_news_for_embedding(art) for art in articles]
                embeddings = await self.embedding_client.generate_embeddings(texts_to_embed)
                
                if not embeddings or len(embeddings) != len(articles):
                    logger.warning("Mismatch between articles and generated embeddings count.")
                    continue
                    
                # Prepare documents for vector store
                documents = []
                # 修正：使用 art_index 访问 texts_to_embed
                for art_index, (art, emb) in enumerate(zip(articles, embeddings)):
                    doc = {
                        "id": art.source_id,
                        "text": texts_to_embed[art_index], # 修正
                        "vector": emb,
                        "metadata": {
                            "symbol": art.symbols,
                            "timestamp": art.timestamp.isoformat(),
                            "source": art.source,
                            "headline": art.headline
                        }
                    }
                    documents.append(doc)

                # Upsert (update or insert) documents into the vector store
                await self.vector_store.upsert(documents)
                logger.debug(f"Injected {len(documents)} news articles into VectorStore.")
                
        except asyncio.CancelledError:
            logger.info("News stream processing cancelled.")
        except Exception as e:
            logger.error(f"Error processing news stream: {e}", exc_info=True)

    # 修复：'NewsArticle' (拼写错误) 改为 'NewsData'
    def _format_news_for_embedding(self, article: NewsData) -> str:
        """
        Formats a news article object into a single string for embedding.
        """
        # Combine key fields for better semantic meaning
        return f"Headline: {article.headline}\nSource: {article.source}\nSummary: {article.summary or 'N/A'}\nContent: {article.content[:500]}..." # Truncate content

    async def process_market_data_stream(self):
        """
        (Placeholder) Processes the market data stream.
        This might involve calculating real-time features or landmarks
        and storing them in the TemporalDB.
        """
        logger.info("Starting market data stream processing...")
        try:
            async for data_batch in self.data_manager.stream_data(data_type='market'):
                if not self._running:
                    break
                if not data_batch:
                    await asyncio.sleep(0.1)
                    continue
                
                # Example: Detect volatility spikes (a temporal event)
                for data in data_batch:
                    # (Logic to detect spike)
                    is_spike = False 
                    
                    if is_spike:
                        event = {
                            "timestamp": data['timestamp'],
                            "event_type": "volatility_spike",
                            "symbol": data['symbol'],
                            "metadata": {"volume": data['volume']}
                        }
                        # await self.temporal_db_client.insert_event(event) # Assumes we have this client

                pass # Placeholder
        
        except asyncio.CancelledError:
            logger.info("Market data stream processing cancelled.")
        except Exception as e:
            logger.error(f"Error processing market data stream: {e}", exc_info=True)

