import os
import asyncio
import google.generativeai as genai
from typing import List, Union, Optional
from core.exceptions import EmbeddingError
# 假设日志记录器已经设置好
# from monitor.logging import log

class EmbeddingClient:
    """
    一个封装了 Google Generative AI Embedding 模型的客户端。
    (异步版本)

    该客户端处理 API 密钥配置，并提供一个统一的异步接口
    来为单个文本或批量文本生成 embedding。
    """
    
    def __init__(self, model_name: str = "text-embedding-004", api_key: Optional[str] = None):
        """
        初始化 EmbeddingClient。

        Args:
            model_name (str): 要使用的 embedding 模型的名称。
                              默认为 "text-embedding-004"，根据 RAG_ARCHITECTURE.md。
            api_key (Optional[str]): Google AI API 密钥。如果为 None，
                                     将尝试从 'GOOGLE_API_KEY' 环境变量中读取。
        
        Raises:
            ValueError: 如果 API 密钥既未提供也未在环境变量中设置。
        """
        self.model_name = model_name
        
        used_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not used_api_key:
            # log.error("GOOGLE_API_KEY 未设置。请设置环境变量或在初始化时提供 api_key。")
            raise ValueError("API key for Google AI not provided or set in environment variables.")
            
        try:
            genai.configure(api_key=used_api_key)
            # log.info(f"EmbeddingClient initialized with model: {self.model_name}")
        except Exception as e:
            # log.error(f"配置 Google AI API 密钥时出错: {e}")
            raise EmbeddingError(f"Failed to configure Google AI: {e}")

    async def get_embedding(self, text: str) -> List[float]:
        """
        (异步) 为单个文本字符串生成 embedding。

        Args:
            text (str): 需要生成 embedding 的输入文本。

        Returns:
            List[float]: 表示文本的 embedding 向量。

        Raises:
            EmbeddingError: 如果 API 调用失败。
        """
        if not text:
            # log.warning("get_embedding 接收到空文本字符串。")
            return []
            
        try:
            # 使用 asyncio.to_thread 运行同步的 SDK 调用
            result = await asyncio.to_thread(
                genai.embed_content,
                model=self.model_name,
                content=text,
                task_type="retrieval_document" # 假设用于检索任务
            )
            return result['embedding']
        except Exception as e:
            # log.error(f"为文本生成 embedding 时出错: {e}")
            raise EmbeddingError(f"Error generating embedding: {e}")

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        (异步) 为一批文本字符串生成 embedding。

        Args:
            texts (List[str]): 需要生成 embedding 的输入文本列表。

        Returns:
            List[List[float]]: 对应每个输入文本的 embedding 向量列表。

        Raises:
            EmbeddingError: 如果 API 调用失败或返回的 embedding 数量不匹配。
        """
        if not texts:
            # log.warning("get_embeddings 接收到空文本列表。")
            return []
            
        try:
            # 使用 asyncio.to_thread 运行同步的 SDK 调用
            result = await asyncio.to_thread(
                genai.embed_content,
                model=self.model_name,
                content=texts,
                task_type="retrieval_document"
            )
            
            embeddings = result['embedding']
            
            if len(embeddings) != len(texts):
                # log.error(f"Embedding 数量不匹配: 需要 {len(texts)}, 得到 {len(embeddings)}")
                raise EmbeddingError(f"Mismatch in embedding count: expected {len(texts)}, got {len(embeddings)}")
                
            return embeddings
        except Exception as e:
            # log.error(f"为批量文本生成 embedding 时出错: {e}")
            raise EmbeddingError(f"Error generating embeddings batch: {e}")

# 示例用法 (用于测试)
async def main_test():
    """异步测试函数"""
    try:
        # 确保设置了 GOOGLE_API_KEY 环境变量
        client = EmbeddingClient()
        
        # 测试单个 embedding
        text1 = "Hello, world!"
        embedding1 = await client.get_embedding(text1)
        print(f"Embedding for '{text1}': {embedding1[:5]}... (dim: {len(embedding1)})")
        
        # 测试批量 embedding
        texts = ["The quick brown fox jumps over the lazy dog.", "A journey of a thousand miles begins with a single step."]
        embeddings = await client.get_embeddings(texts)
        
        print(f"\nBatch embeddings generated: {len(embeddings)}")
        for i, emb in enumerate(embeddings):
            print(f"Embedding {i+1}: {emb[:5]}... (dim: {len(emb)})")
            
    except (ValueError, EmbeddingError) as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # 使用 asyncio.run 来执行异步测试函数
    asyncio.run(main_test())
