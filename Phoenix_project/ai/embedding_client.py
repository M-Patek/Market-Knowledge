import os
import asyncio
import google.generativeai as genai
from typing import List, Union, Optional
# 修复：将相对导入 'core.exceptions' 更改为绝对导入
from Phoenix_project.core.exceptions import EmbeddingError
# [主人喵的清洁计划 1.1] 导入 Phoenix logger 和类型
from Phoenix_project.monitor.logging import get_logger, ESLogger

class EmbeddingClient:
    """
    一个封装了 Google Generative AI Embedding 模型的客户端。
    (异步版本)

    该客户端处理 API 密钥配置，并提供一个统一的异步接口
    来为单个文本或批量文本生成 embedding。
    """
    
    # [主人喵的清洁计划 1.2 修复] __init__ 签名不变，但逻辑改变
    def __init__(
        self, 
        model_name: str = "text-embedding-004", 
        api_key: Optional[str] = None, # 接受注入的 API Key
        logger: Optional[ESLogger] = None
    ):
        """
        初始化 EmbeddingClient。

        Args:
            model_name (str): 要使用的 embedding 模型的名称。
                              默认为 "text-embedding-004"，根据 RAG_ARCHITECTURE.md。
            api_key (Optional[str]): Google AI API 密钥。
                                     [重构] 调用者(例如服务容器)负责从配置(system.yaml)
                                     和环境(os.getenv)中读取此密钥并将其注入。
            logger (Optional[ESLogger]): 外部日志记录器。
        
        Raises:
            ValueError: 如果 'api_key' 参数未提供 (为 None 或空字符串)。
            EmbeddingError: 如果 Google AI SDK 配置失败。
        """
        self.model_name = model_name
        # [主人喵的清洁计划 1.1 修复] 使用传入的 logger
        self.logger = logger or get_logger(__name__)
        
        # [主人喵的清洁计划 1.2 修复]
        # 移除: used_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        # 客户端不应了解环境变量。它只接受一个 api_key。
        
        if not api_key:
            # 更新了错误信息，使其不再引用特定的环境变量
            self.logger.log_error("EmbeddingClient 初始化失败：未提供 'api_key'。")
            raise ValueError("API key for Google AI was not provided to EmbeddingClient.")
            
        try:
            # 使用传入的 api_key
            genai.configure(api_key=api_key)
            self.logger.log_info(f"EmbeddingClient initialized with model: {self.model_name}")
        except Exception as e:
            self.logger.log_error(f"配置 Google AI API 密钥时出错: {e}")
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
            self.logger.log_warning("get_embedding 接收到空文本字符串。")
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
            self.logger.log_error(f"为文本生成 embedding 时出错: {e}")
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
            self.logger.log_warning("get_embeddings 接收到空文本列表。")
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
                self.logger.log_error(f"Embedding 数量不匹配: 需要 {len(texts)}, 得到 {len(embeddings)}")
                raise EmbeddingError(f"Mismatch in embedding count: expected {len(texts)}, got {len(embeddings)}")
                
            return embeddings
        except Exception as e:
            self.logger.log_error(f"为批量文本生成 embedding 时出错: {e}")
            raise EmbeddingError(f"Error generating embeddings batch: {e}")

# 示例用法 (用于测试)
async def main_test():
    """异步测试函数"""
    try:
        # [测试] 演示新的初始化流程
        # 调用者 (此处的 main_test) 负责从环境中读取密钥
        # 生产代码将从 config/system.yaml 中读取 'GEMINI_PRO_KEY'
        
        # 为了测试方便，我们检查两个可能的键
        api_key_for_test = os.getenv("GEMINI_PRO_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key_for_test:
            print("测试错误：请设置 GEMINI_PRO_KEY 或 GOOGLE_API_KEY 环境变量以运行测试。")
            return

        # 修复：为测试实例化一个 logger
        test_logger = get_logger("main_test")
        
        # 将密钥作为参数传递
        client = EmbeddingClient(api_key=api_key_for_test, logger=test_logger)
        
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
