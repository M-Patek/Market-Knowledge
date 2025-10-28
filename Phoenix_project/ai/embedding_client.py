# ai/embedding_client.py
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from api.gemini_pool_manager import GeminiPoolManager # 导入我们的新池

class EmbeddingClient:
    """
    使用 GeminiPoolManager 处理文本和其他嵌入的创建。
    """
    def __init__(self, pool_manager: GeminiPoolManager, model_name: str = 'text-embedding-004', batch_size: int = 100):
        self.logger = logging.getLogger("PhoenixProject.EmbeddingClient")
        self.pool_manager = pool_manager
        self.model_name = model_name
        self.batch_size = batch_size
        self.logger.info(f"EmbeddingClient configured with model '{self.model_name}' and GeminiPoolManager.")


    async def create_text_embeddings(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为文档列表创建文本嵌入。

        Args:
            documents: 字典列表，每个字典代表一个文档
                       并且必须包含 'content' 键。

        Returns:
            相同的文档列表，每个字典添加了 'vector' 键，
            包含文档的嵌入。
        """
        try:
            texts_to_embed = [doc['content'] for doc in documents]
            
            # 异步调用池管理器以获取嵌入
            embedding_result = await self.pool_manager.embed_content(
                model_name=self.model_name,
                contents=texts_to_embed,
                task_type="RETRIEVAL_DOCUMENT" # 传递所需的参数
            )
            
            # 池管理器将返回与 genai 库相同的结构，
            # 这是一个包含 'embedding' 列表的字典。
            embeddings_list = embedding_result['embedding']
            for i, doc in enumerate(documents):
                doc['vector'] = embeddings_list[i]

            self.logger.info(f"Successfully created {len(documents)} text embeddings.")
            return documents
        except Exception as e:
            self.logger.error(f"Failed to create text embeddings: {e}")
            # 返回没有向量的文档，以免流水线完全停止。
            self.logger.warning("Returning documents without text vectors due to an error.")
            return documents

    async def create_query_embedding(self, query: str) -> Optional[List[float]]:
        """为单个查询字符串创建嵌入。"""
        try:
            # 注意：池管理器期望 contents 是一个 List[str]。
            result = await self.pool_manager.embed_content(
                model_name=self.model_name,
                contents=[query], # 将单个查询包装在列表中
                task_type="RETRIEVAL_QUERY"
            )
            return result['embedding'][0] # 返回单个嵌入
        except Exception as e:
            self.logger.error(f"Failed to create query embedding: {e}")
            return None

    def create_time_series_embeddings(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为时间序列数据创建嵌入。
        这是一个真实时间序列嵌入模型（如TS2Vec）的占位符。
        """
        try:
            # 在真实系统中，您会加载一个在金融数据上训练过的模型。
            # 对于此实现，我们使用一个预训练的通用模型。
            # 模型被加载一次并缓存在内存中以提高效率。
            if not hasattr(self, '_ts2vec_model'): 
                # 这个路径是一个占位符。真实的实现会下载或加载一个训练好的模型。
                self.logger.warning("Loading a placeholder TS2Vec model. For production, use a trained model.")
                # self._ts2vec_model = TS2Vec.load_from_checkpoint('path/to/pretrained/ts2vec/model.ckpt')
                # 目前，我们创建一个虚拟模型，输出正确维度的随机向量
                class DummyTS2Vec:
                    def encode(self, data, encoding_window):
                        return np.random.rand(data.shape[0], 320) # 一个常见的 TS2Vec 输出维度
                self._ts2vec_model = DummyTS2Vec()
 
            for doc in documents:
                ts_data = doc.get("time_series_data")
                if isinstance(ts_data, np.ndarray):
                    # TS2Vec 期望一个 3D 数组 (batch, timestamp, feature)
                    if ts_data.ndim == 2:
                        ts_data = np.expand_dims(ts_data, axis=0)
                    
                    # 为时间序列生成嵌入
                    embedding = self._ts2vec_model.encode(ts_data, encoding_window='full_series')
                    doc['vector'] = embedding.flatten().tolist()
                else:
                    self.logger.warning(f"Document '{doc.get('source_id')}' is missing valid 'time_series_data'.")

            return documents
        except Exception as e:
            self.logger.error(f"Failed to create time-series embeddings: {e}")
            self.logger.warning("Returning documents without time-series vectors due to an error.")
            return documents

    def _convert_tabular_to_text(self, tabular_row: Dict[str, Any]) -> str:
        """将表格数据的字典转换为单个描述性字符串。"""
        # 简单实现：连接键值对。
        # 更复杂的方法会使用模板或自然语言生成。
        return ". ".join([f"{key.replace('_', ' ')} is {value}" for key, value in tabular_row.items()])

    async def create_tabular_embeddings(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        通过首先将表格数据转换为文本表示来为其创建嵌入。
        """
        try:
            # 为每行表格数据创建一个文本表示。
            text_representations = [self._convert_tabular_to_text(doc.get("tabular_data", {})) for doc in documents]
            
            # 批量嵌入文本表示。
            result = await self.pool_manager.embed_content(
                model=self.model_name,
                contents=text_representations,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings = result['embedding']

            for doc, embedding in zip(documents, embeddings):
                doc['vector'] = embedding

            self.logger.info(f"Successfully created tabular embeddings for {len(documents)} documents.")
            return documents
        except Exception as e:
            self.logger.error(f"Failed to create tabular embeddings: {e}")
            self.logger.warning("Returning documents without tabular vectors due to an error.")
            return documents
