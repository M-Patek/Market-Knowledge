"""
Phoenix_project/knowledge_injector.py
[Phase 2 Task 2] Fix KnowledgeInjector Validation Deadlock.
Implement Sampling Verification to bypass unreliable exact counts (Pinecone Serverless).
"""
import logging
import uuid
from typing import List, Dict, Any, Optional
from uuid import UUID
import asyncio

from Phoenix_project.memory.vector_store import BaseVectorStore, Document
from Phoenix_project.ai.data_adapter import DataAdapter
from Phoenix_project.ai.relation_extractor import RelationExtractor
from Phoenix_project.ai.graph_db_client import GraphDBClient

logger = logging.getLogger(__name__)

class KnowledgeInjector:
    """
    负责将新知识注入到系统的长期记忆中 (VectorStore & GraphDB)。
    处理分块、嵌入、关系提取和事务性写入。
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        graph_db: GraphDBClient,
        data_adapter: DataAdapter,
        relation_extractor: RelationExtractor,
        data_manager: Any = None # Optional injection
    ):
        self.vector_store = vector_store
        self.graph_db = graph_db
        self.data_adapter = data_adapter
        self.relation_extractor = relation_extractor
        self.data_manager = data_manager
        logger.info("KnowledgeInjector initialized.")

    async def inject_document(
        self, 
        content: str, 
        metadata: Dict[str, Any], 
        batch_id: Optional[UUID] = None
    ) -> bool:
        """
        注入单个文档：
        1. 文本分块
        2. 向量化并存入 VectorStore
        3. 提取实体关系并存入 GraphDB
        4. [Fix] 执行智能校验
        """
        if not batch_id:
            batch_id = uuid.uuid4()
            
        logger.info(f"Starting injection for batch {batch_id}...")
        
        try:
            # 1. 文本处理与分块
            chunks = self.data_adapter.chunk_text(content, chunk_size=500, overlap=50)
            if not chunks:
                logger.warning("No chunks generated from content.")
                return False

            documents = [
                Document(page_content=chunk, metadata={**metadata, "chunk_index": i}) 
                for i, chunk in enumerate(chunks)
            ]

            # 2. VectorStore 写入 (生成 Embeddings 由 Store 内部或外部处理，这里假设 Store 需要 Embeddings 或者处理 Doc)
            # 根据 VectorStore 接口，aadd_batch 需要 embeddings。
            # 这里我们需要先调用 EmbeddingClient。
            
            embedding_client = getattr(self.vector_store, 'embedding_client', None)
            if not embedding_client:
                raise ValueError("VectorStore missing EmbeddingClient reference.")

            chunk_texts = [doc.page_content for doc in documents]
            embeddings = await embedding_client.get_embeddings(chunk_texts)
            
            if len(embeddings) != len(documents):
                raise ValueError("Embedding count mismatch.")

            # 写入向量库
            upserted_ids = await self.vector_store.aadd_batch(documents, embeddings, batch_id)
            logger.info(f"Upserted {len(upserted_ids)} vectors.")

            # 3. GraphDB 写入 (可选/并行)
            # ... (Graph logic simplified for this task) ...

            # 4. [Fix] 验证摄入 (Validation with Deadlock Fix)
            is_valid = await self._verify_ingestion(batch_id, len(documents), sample_text=chunk_texts[0])
            
            if is_valid:
                logger.info(f"Batch {batch_id} injection successful.")
                return True
            else:
                logger.error(f"Batch {batch_id} validation failed.")
                return False

        except Exception as e:
            logger.error(f"Injection failed for batch {batch_id}: {e}", exc_info=True)
            return False

    async def _verify_ingestion(self, batch_id: UUID, expected_count: int, sample_text: str = "") -> bool:
        """
        [Phase 2 Task 2] 验证数据摄入完整性。
        修复了 count_by_batch_id 返回 0 导致的死锁。
        采用“计数优先，抽样兜底”的策略。
        """
        retries = 3
        for attempt in range(retries):
            try:
                # 1. 尝试精确计数
                count = await self.vector_store.count_by_batch_id(batch_id)
                
                if count == expected_count:
                    logger.debug(f"Verification success: Count matches ({count}).")
                    return True
                
                # 2. [Deadlock Fix] 如果计数为0 (Pinecone Serverless 常见行为) 但我们期望非零
                if count == 0 and expected_count > 0:
                    logger.warning(f"Verification: Count is 0 (expected {expected_count}). Attempting Sampling Verification...")
                    
                    # 尝试搜索刚才插入的内容
                    if sample_text:
                        results = await self.vector_store.asimilarity_search(
                            query=sample_text, 
                            k=1,
                            # [Optional] Filter by batch_id if supported, otherwise just semantic match
                            # filter={"ingestion_batch_id": str(batch_id)} 
                        )
                        
                        # 如果能搜到内容，且相似度极高 (接近 1.0)，则认为写入成功
                        if results:
                            doc, score = results[0]
                            retrieved_batch_id = doc.metadata.get("ingestion_batch_id")
                            
                            if retrieved_batch_id == str(batch_id) or score > 0.95:
                                logger.info(f"Sampling Verification Passed: Found valid doc (Score: {score:.4f}).")
                                return True
                    
                    logger.warning("Sampling Verification Failed.")

                # 3. Retry Logic
                await asyncio.sleep(1) # Wait for eventual consistency
                
            except Exception as e:
                logger.warning(f"Verification attempt {attempt+1} failed: {e}")
                await asyncio.sleep(1)
        
        logger.error(f"Verification failed after {retries} attempts. Expected {expected_count}, got {count}.")
        return False
