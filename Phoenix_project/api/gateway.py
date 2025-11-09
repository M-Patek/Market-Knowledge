import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Union
from Phoenix_project.api.gemini_pool_manager import GeminiPoolManager
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.config.loader import ConfigLoader

logger = get_logger(__name__)

class APIGateway:
    """
    Main gateway for all external API calls, primarily to Google Gemini.
    Manages API key, model selection, request formatting, and error handling.
    Uses GeminiPoolManager for concurrent requests.
    """

    # [任务 5 已修改] __init__
    def __init__(self, config_loader: ConfigLoader, pool_size: int = 10):
        # api_key 是单数, 用于 genai.configure。
        # PoolManager 将从 os.environ 加载 API_KEYS (复数)。
        api_key = config_loader.get_secret("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment.")
            raise ValueError("GEMINI_API_KEY is not set.")
        
        genai.configure(api_key=api_key)
        
        # --- [任务 5 已实现] ---
        # 1. 从 config_loader 加载 system.yaml
        system_config = config_loader.load_config('system.yaml')
        if not system_config:
             logger.warning("api/gateway.py: 无法加载 system.yaml。将使用默认 QPM 限制。")
             system_config = {}

        # 2. 从 system.yaml (或回退) 中提取 QPM 限制
        # (注意: 'gemini_qpm_limits' 在 system.yaml 中不存在,
        # 所以 qpm_limits_from_config 将是一个空字典,
        # PoolManager 将使用其内部的 DEFAULT_QPM_LIMITS)
        qpm_limits_from_config = system_config.get("ai", {}).get("gemini_qpm_limits", {})
        if not qpm_limits_from_config:
             logger.warning("在 system.yaml 的 'ai.gemini_qpm_limits' 中未找到 QPM 限制。")

        gemini_config = {
            "gemini_qpm_limits": qpm_limits_from_config
            # (pool_size 似乎没有被 PoolManager 使用, 将其移除)
        }

        # 3. 修复 PoolManager 的实例化
        # (它只接受 'config', 不接受 'api_key' 或 'pool_size')
        self.pool_manager = GeminiPoolManager(
            config=gemini_config # 传递 config 字典
        )
        # --- [任务 5 结束] ---
        
        
        # [任务 5 修复] 从 system_config 加载, 而不是 config_loader
        self.default_safety_settings = system_config.get(
            "gemini_safety_settings",
            [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        )
        
        # [任务 5 修复] 从 system_config 加载
        self.default_generation_config = system_config.get(
            "gemini_generation_config",
            {
                "candidate_count": 1,
                "max_output_tokens": 4096,
                "temperature": 0.7,
                "top_p": 1.0,
                "top_k": 1,
            }
        )
        
        logger.info(f"APIGateway initialized.") # 移除了 pool_size

    async def send_request(
        self,
        model_name: str,
        prompt: Union[str, List[Union[str, Any]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        response_mime_type: Optional[str] = None # 允许指定 mime 类型
    ) -> str:
        """
        Sends a request to the Gemini API using the pool manager.
        
        Args:
            model_name (str): The name of the model (e.g., "gemini-pro").
            prompt (Union[str, List[Union[str, Any]]]): The prompt string or
                                                       a list of content parts.
            response_mime_type (Optional[str]): e.g., "application/json".
            
        Returns:
            str: The text response from the model.
        """
        
        # Override defaults if provided
        generation_config = self.default_generation_config.copy()
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        if stop_sequences is not None:
            generation_config["stop_sequences"] = stop_sequences
        if top_p is not None:
            generation_config["top_p"] = top_p
        if top_k is not None:
            generation_config["top_k"] = top_k
        if response_mime_type is not None:
            generation_config["response_mime_type"] = response_mime_type
            
        # Format the prompt content
        contents = [prompt] if isinstance(prompt, str) else prompt

        try:
            request_id = f"req_{asyncio.get_event_loop().time()}" # 简单的请求ID
            logger.debug(f"Sending request {request_id} to model '{model_name}' via pool...")

            # MODIFICATION: 修复方法调用和返回类型处理
            
            # 1. 使用上下文管理器
            async with self.pool_manager.get_client(model_name) as client:
                # 2. 调用更新后的 'generate_content_async'
                response_dict = await client.generate_content_async(
                    contents=contents,
                    generation_config=generation_config,
                    safety_settings=self.default_safety_settings,
                    request_id=request_id
                )

            logger.debug(f"Received response {request_id} from '{model_name}'.")

            # 3. 从字典中提取文本 (解决问题 3)
            # 注意：如果请求了 JSON，这里仍然返回 *序列化* 的 JSON 字符串。
            # 调用者 (例如 EnsembleClient) 负责解析它。
            # GeminiPoolManager 确保返回 {"text": "..."}
            if "text" not in response_dict:
                logger.error(f"Response from {model_name} missing 'text' key: {response_dict}")
                raise ValueError("Invalid response structure from model client.")
            
            return response_dict["text"] # 返回 str

        except Exception as e:
            logger.error(f"Error in send_request via pool for '{model_name}': {e}", exc_info=True)
            # Re-raise or return a specific error message
            raise

    async def send_embedding_request(
        self,
        model_name: str,
        texts: List[str],
        dimensions: Optional[int] = None
    ) -> List[List[float]]:
        """
        Sends a batch embedding request.
        
        Args:
            model_name (str): The embedding model name (e.g., "text-embedding-3-large").
            texts (List[str]): List of texts to embed.
            dimensions (Optional[int]): Requested embedding dimensions.
            
        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        
        # This uses the genai library directly, as the pool manager
        # is set up for `generate_content`. We could extend the pool manager
        # to also handle `embed_content`.
        
        # For now, use asyncio.to_thread for the blocking SDK call
        
        def blocking_embed_call():
            try:
                logger.debug(f"Embedding {len(texts)} texts with model '{model_name}'...")
                # Note: `output_dimensionality` is the arg name in genai
                result = genai.embed_content(
                    model=f"models/{model_name}",
                    content=texts,
                    task_type="retrieval_document", # Or "semantic_similarity" etc.
                    output_dimensionality=dimensions
                )
                logger.debug("Embedding call successful.")
                return result['embedding']
            except Exception as e:
                logger.error(f"Error during blocking embed call: {e}", exc_info=True)
                raise

        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, blocking_embed_call)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            return [[] for _ in texts]


# Example usage (if run directly, though it's not typical)
if __name__ == "__main__":
    # This requires a valid config setup
    logger.warning("APIGateway is not intended to be run directly.")
    
    # Example async test
    async def test_gateway():
        try:
            # Mock ConfigLoader for testing
            class MockConfigLoader:
                def get_secret(self, key):
                    import os
                    return os.environ.get(key) # Assumes GEMINI_API_KEY is set in env
                
                def get(self, key, default=None):
                    # Mock config loader 'get' method
                    if key == "gemini_qpm_limits":
                        return {} # Use defaults
                    return default
                
                # [任务 5 修复] 添加 load_config
                def load_config(self, filename):
                    if filename == 'system.yaml':
                        return {
                            "ai": {
                                # "gemini_qpm_limits": {"gemini-pro": 1} # (测试 QPM)
                            }
                        }
                    return {}

            
            config_loader = MockConfigLoader()
            if not config_loader.get_secret("GEMINI_API_KEY"):
                print("GEMINI_API_KEY not set in environment. Skipping test.")
                return

            gateway = APIGateway(config_loader, pool_size=2)
            
            prompt = "What is the capital of France?"
            response = await gateway.send_request("gemini-pro", prompt, temperature=0.0)
            print(f"Test Request 1:\nPrompt: {prompt}\nResponse: {response}\n")

            prompt_2 = "Explain the concept of asynchronous programming in Python."
            response_2 = await gateway.send_request("gemini-pro", prompt_2, max_tokens=150)
            print(f"Test Request 2:\nPrompt: {prompt_2}\nResponse: {response_2}\n")
            
            # Test embedding
            # texts = ["Hello world", "This is a test"]
            # embeddings = await gateway.send_embedding_request("text-embedding-3-large", texts, dimensions=256)
            # print(f"Test Embedding:\nTexts: {texts}\nEmbedding 1 dim: {len(embeddings[0])}\n")

        except Exception as e:
            print(f"Error during test: {e}")

    # import os
    # if "GEMINI_API_KEY" in os.environ:
    #     asyncio.run(test_gateway())
    # else:
    #     print("Set GEMINI_API_KEY environment variable to run the test.")
