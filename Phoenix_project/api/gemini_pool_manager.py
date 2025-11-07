import os
import asyncio
import json
from google.generativeai import GenerativeModel
import google.generativeai as genai
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager

# 修复：将相对导入 'from ..monitor.logging...' 更改为绝对导入
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

# --- Concurrency and Rate Limiting ---
# These limits are examples and should be tuned based on Google's quotas
# and the specific models being used.
# Gemini 1.5 Pro: 5 QPM (Queries Per Minute)
# Gemini 1.5 Flash: 15 QPM

DEFAULT_QPM_LIMITS = {
    "gemini-1.5-pro": 5,
    "gemini-1.5-flash": 15,
    "gemini-pro": 15, # 假设
    "default": 10
}
# We use a safety margin to avoid hitting the limit exactly
QPM_SAFETY_FACTOR = 0.8

class GeminiClient:
    """
    A wrapper around a single GenerativeModel instance that handles
    rate limiting using an asyncio.Semaphore.
    """
    def __init__(self, model_id: str, api_key: str, qpm_limit: int):
        self.model_id = model_id
        
        # MODIFICATION: 确保此客户端实例配置了密钥
        # 注意：genai.configure是全局的，但在APIGateway中已调用。
        # 在这里单独配置模型实例可能更安全，但genai库
        # 倾向于全局配置。我们依赖于 APIGateway 已经调用了 genai.configure。
        # self.model = GenerativeModel(model_id)
        
        # MODIFICATION: 不依赖全局 'genai' 状态，配置此实例
        # 尽管 genai.configure(api_key=api_key) 是全局的，
        # 再次调用它来设置正确的密钥是安全的。
        genai.configure(api_key=api_key)
        
        # 移除 system_instruction，因为它将在 generate_content_async 中传递
        self.model = GenerativeModel(model_id)
        
        # Calculate queries per second (QPS) for the semaphore
        qps_limit = (qpm_limit / 60.0) * QPM_SAFETY_FACTOR
        
        # The semaphore rate is the inverse of QPS
        self.semaphore_rate = 1.0 / qps_limit if qps_limit > 0 else 1.0
        
        self.semaphore = asyncio.Semaphore(1) # Controls access
        self.last_call_time = 0
        
        logger.info(f"GeminiClient for {model_id} initialized with QPM limit {qpm_limit} (Rate: {self.semaphore_rate:.2f}s/query)")

    # MODIFICATION: 更改 generate_content_async 签名
    async def generate_content_async(
        self,
        contents: List[Union[str, Any]], # 替换 system_prompt 和 user_prompt
        generation_config: Dict[str, Any] = None,
        safety_settings: List[Dict[str, Any]] = None, # 添加 safety_settings
        request_id: str = "N/A" # 保留 request_id
    ) -> Dict[str, Any]:
        """
        Generates content from the model, respecting rate limits.
        (Updated signature to match APIGateway)
        
        Returns:
            Dict[str, Any]: A dictionary containing the 'text' response.
        """
        async with self.semaphore:
            # Enforce time-based rate limit
            now = asyncio.get_event_loop().time()
            time_since_last_call = now - self.last_call_time
            
            if time_since_last_call < self.semaphore_rate:
                sleep_time = self.semaphore_rate - time_since_last_call
                logger.debug(f"Rate limiting {self.model_id} ({request_id}). Sleeping for {sleep_time:.2f}s.")
                await asyncio.sleep(sleep_time)
            
            self.last_call_time = asyncio.get_event_loop().time()
            
            try:
                logger.info(f"Sending request {request_id} to {self.model_id}")
                
                # MODIFICATION: 更新 API 调用
                # 使用 self.model (已在 __init__ 中初始化)
                response = await self.model.generate_content_async(
                    contents, # 传递 'contents' 列表
                    generation_config=generation_config,
                    safety_settings=safety_settings # 传递 'safety_settings'
                )
                
                # Extract and parse the text
                if not response.parts:
                    # 检查是否因为安全设置被阻止
                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                        logger.warning(f"Request {request_id} to {self.model_id} blocked. Reason: {response.prompt_feedback.block_reason}")
                        raise ValueError(f"Content generation blocked: {response.prompt_feedback.block_reason}")
                    raise ValueError(f"Model response ({request_id}) contained no parts.")
                    
                response_text = response.parts[0].text
                
                # 无论是否为JSON，我们都返回包含文本的字典
                # APIGateway 将处理这个字典。
                # 如果请求了JSON，response_text 将是JSON字符串。
                
                return {"text": response_text}

            except Exception as e:
                logger.error(f"Error calling Gemini API ({self.model_id}) for {request_id}: {e}", exc_info=True)
                # Re-raise to be caught by the ensemble client
                raise

class GeminiPoolManager:
    """
    Manages a pool of GeminiClient instances, one for each model ID required.
    Provides a context manager to acquire and release clients.
    """
    
    # MODIFICATION: 更改 __init__ 签名以匹配 APIGateway 的调用
    def __init__(self, api_key: str, config: Dict[str, Any] = None, pool_size: int = 10):
        if not api_key:
            logger.error("API key not provided to GeminiPoolManager.")
            raise ValueError("API_KEY not set")
        self.api_key = api_key # 使用传入的 api_key
            
        if config is None:
            config = {}

        self.pool: Dict[str, GeminiClient] = {}
        self.pool_lock = asyncio.Lock()
        
        # Load QPM limits from config, fallback to defaults
        qpm_limits_config = config.get('gemini_qpm_limits', {})
        self.qpm_limits = {**DEFAULT_QPM_LIMITS, **qpm_limits_config}
        
        # pool_size 在这个实现中没有真正使用，因为我们为每个
        # model_id 创建一个客户端，而客户端内部处理速率限制，
        # 而不是限制并发客户端的数量。
        logger.info(f"GeminiPoolManager initialized (pool_size arg ignored, manages clients by model_id).")


    async def _get_or_create_client(self, model_id: str) -> GeminiClient:
        """
        Initializes a client for a specific model_id if it doesn't exist.
        """
        # 优化：在不持有锁的情况下检查
        if model_id in self.pool:
            return self.pool[model_id]
            
        async with self.pool_lock:
            # 再次检查，以防在等待锁时被创建
            if model_id not in self.pool:
                logger.info(f"Creating new GeminiClient for model: {model_id}")
                qpm_limit = self.qpm_limits.get(model_id, self.qpm_limits['default'])
                
                self.pool[model_id] = GeminiClient(
                    model_id=model_id,
                    api_key=self.api_key,
                    qpm_limit=qpm_limit
                )
            return self.pool[model_id]

    @asynccontextmanager
    async def get_client(self, model_id: str) -> GeminiClient:
        """
        Asynchronous context manager to get a client from the pool.
        
        Usage:
            async with gemini_pool.get_client("gemini-1.5-pro") as client:
                await client.generate_content_async(...)
        """
        
        client = await self._get_or_create_client(model_id)
        
        try:
            # The client itself handles rate limiting internally,
            # so we just yield the client instance.
            yield client
        except Exception as e:
            logger.error(f"Exception during Gemini client usage for {model_id}: {e}", exc_info=True)
            raise
        finally:
            # No explicit release needed as the client is persistent
            # and manages its own semaphore.
            pass
