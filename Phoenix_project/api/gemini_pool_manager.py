import os
import asyncio
import json
import httpx
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager
from collections import defaultdict 

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
    "gemini-pro": 15, 
    "default": 10
}
# We use a safety margin to avoid hitting the limit exactly
QPM_SAFETY_FACTOR = 0.8

class GeminiClient:
    """
    A wrapper around Gemini REST API that handles rate limiting using an asyncio.Semaphore.
    Replaced Google GenAI SDK to eliminate global state race conditions.
    """
    def __init__(self, model_id: str, api_key: str, qpm_limit: int):
        self.model_id = model_id
        self.api_key = api_key 
        
        # Calculate queries per second (QPS) for the semaphore
        qps_limit = (qpm_limit / 60.0) * QPM_SAFETY_FACTOR
        
        # The semaphore rate is the inverse of QPS
        self.semaphore_rate = 1.0 / qps_limit if qps_limit > 0 else 1.0
        
        self.semaphore = asyncio.Semaphore(1) # Controls access
        self.last_call_time = 0
        
        logger.info(f"GeminiClient for {model_id} initialized with QPM limit {qpm_limit} (Rate: {self.semaphore_rate:.2f}s/query)")

    async def generate_content_async(
        self,
        contents: List[Union[str, Any]], 
        tools: List[Any] = None,
        generation_config: Dict[str, Any] = None,
        safety_settings: List[Dict[str, Any]] = None, 
        request_id: str = "N/A" 
    ) -> Dict[str, Any]:
        """
        Generates content from the model via REST API, respecting rate limits.
        
        Returns:
            Dict[str, Any]: A dictionary containing the 'text' response, 'function_call', and optional 'grounding_metadata'.
        """
        # [Optimized Concurrency] Calculate sleep time inside lock, sleep outside to prevent chained blocking
        async with self.semaphore:
            now = asyncio.get_event_loop().time()
            # Reserve slot: Current time or last slot + rate
            start_time = max(now, self.last_call_time + self.semaphore_rate)
            sleep_time = start_time - now
            self.last_call_time = start_time

        if sleep_time > 0:
            logger.debug(f"Rate limiting {self.model_id} ({request_id}). Sleeping for {sleep_time:.2f}s.")
            await asyncio.sleep(sleep_time)
            
        try:
            logger.info(f"Sending request {request_id} to {self.model_id}")
            
            # Switch to REST API via httpx to eliminate global state race conditions
            base_url = "https://generativelanguage.googleapis.com/v1beta/models"
            url = f"{base_url}/{self.model_id}:generateContent?key={self.api_key}"
            
            # Adapter for contents to ensure JSON compatibility
            formatted_contents = contents
            if isinstance(contents, str):
                formatted_contents = [{"role": "user", "parts": [{"text": contents}]}]
            elif isinstance(contents, list) and contents and isinstance(contents[0], str):
                formatted_contents = [{"role": "user", "parts": [{"text": c}]} for c in contents]

            payload = {
                "contents": formatted_contents,
                "generationConfig": generation_config or {},
                "safetySettings": safety_settings or []
            }
            if tools:
                payload["tools"] = tools

            async with httpx.AsyncClient() as client:
                # Using a new client per request ensures isolation
                response = await client.post(url, json=payload, timeout=60.0)

            if response.status_code != 200:
                logger.error(f"Gemini API Error ({response.status_code}): {response.text}")
                raise ValueError(f"Gemini API call failed with status {response.status_code}: {response.text}")

            response_data = response.json()
            
            # Basic error handling for prompt feedback (safety blocks)
            prompt_feedback = response_data.get("promptFeedback", {})
            if prompt_feedback.get("blockReason"):
                    block_reason = prompt_feedback.get("blockReason")
                    logger.warning(f"Request {request_id} blocked. Reason: {block_reason}")
                    raise ValueError(f"Content generation blocked: {block_reason}")

            candidates = response_data.get("candidates", [])
            if not candidates:
                    raise ValueError(f"Model response ({request_id}) contained no candidates.")

            first_candidate = candidates[0]
            parts = first_candidate.get("content", {}).get("parts", [])
            
            # [Fixed Swallow Bug] Iterate through parts to capture both text and function calls
            response_text = ""
            function_call = None
            for part in parts:
                if "text" in part:
                    response_text += part["text"]
                if "functionCall" in part:
                    function_call = part["functionCall"]
            
            # Extract grounding metadata if available
            grounding_metadata = first_candidate.get('groundingMetadata')

            # Return dictionary compatible with APIGateway, correctly exposing function calls
            return {
                "text": response_text,
                "function_call": function_call,
                "grounding_metadata": grounding_metadata
            }

        except Exception as e:
            logger.error(f"Error calling Gemini API ({self.model_id}) for {request_id}: {e}", exc_info=True)
            raise

class GeminiPoolManager:
    """
    Manages a pool of GeminiClient instances, one for each model ID required.
    Provides a context manager to acquire and release clients.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}

        # 1. 加载共享密钥池
        keys_str = os.environ.get("GEMINI_API_KEYS")
        if not keys_str:
            logger.error("环境变量 'GEMINI_API_KEYS' 未设置或为空。")
            raise ValueError("GEMINI_API_KEYS 环境变量未设置")
            
        self.shared_key_pool: List[str] = [key.strip() for key in keys_str.split(',')]
        if not self.shared_key_pool:
            logger.error("API 密钥池为空，请检查 'GEMINI_API_KEYS'。")
            raise ValueError("API 密钥池为空")

        logger.info(f"GeminiPoolManager 初始化，加载了 {len(self.shared_key_pool)} 个共享密钥。")

        # 2. 为每个模型ID设置独立的轮询计数器
        self.model_counters: Dict[str, int] = defaultdict(int)

        # 3. 为每个【密钥】缓存一个【GeminiClient】实例
        self.client_cache: Dict[str, GeminiClient] = {}
        
        # 4. 需要一个锁来保护 client_cache 的写入，防止竞态
        self.cache_lock = asyncio.Lock()

        # 5. 保留 QPM 限制的配置
        qpm_limits_config = config.get('gemini_qpm_limits', {})
        self.qpm_limits = {**DEFAULT_QPM_LIMITS, **qpm_limits_config}

    @asynccontextmanager
    async def get_client(self, model_id: str) -> GeminiClient:
        """
        根据 model_id 的独立循环，从共享池中获取一个密钥，
        并返回该密钥对应的、带速率限制的客户端。
        """
        
        # 步骤 1: 根据 model_id 独立循环，选择一个密钥
        current_index = self.model_counters[model_id]
        key_to_use = self.shared_key_pool[current_index]
        
        # 步骤 2: 更新该 model_id 的计数器（循环）
        self.model_counters[model_id] = (current_index + 1) % len(self.shared_key_pool)

        # 步骤 3: 检查此【密钥】是否已有缓存的客户端
        client = self.client_cache.get(key_to_use)

        if client is None:
            # 步骤 4: 如果没有，加锁并创建客户端
            async with self.cache_lock:
                # 再次检查，防止在等待锁时其他协程已创建
                client = self.client_cache.get(key_to_use)
                if client is None:
                    # 获取此模型（或默认）的 QPM 限制
                    qpm_limit = self.qpm_limits.get(model_id, self.qpm_limits['default'])
                    
                    logger.info(f"为 model '{model_id}' 创建新的 GeminiClient (使用密钥 ...{key_to_use[-4:]}, QPM: {qpm_limit})")
                    
                    # 创建客户端（它内部有自己的速率限制器）
                    client = GeminiClient(
                        model_id=model_id,
                        api_key=key_to_use,
                        qpm_limit=qpm_limit
                    )
                    
                    # 缓存这个【密钥】的客户端
                    self.client_cache[key_to_use] = client

        # 步骤 5: 产出客户端供 'async with' 使用
        try:
            yield client
        except Exception as e:
            logger.error(f"Gemini client (model: {model_id}, key: ...{key_to_use[-4:]}) 使用时出错: {e}", exc_info=True)
            raise
        finally:
            # (如原设计，无需释放)
            pass
