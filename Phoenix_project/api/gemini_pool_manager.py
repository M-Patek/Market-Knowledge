import os
import asyncio
from google.generativeai import GenerativeModel
import google.generativeai as genai
from typing import Dict, Any, List, Optional
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
        self.model = GenerativeModel(model_id)
        genai.configure(api_key=api_key)
        
        # Calculate queries per second (QPS) for the semaphore
        qps_limit = (qpm_limit / 60.0) * QPM_SAFETY_FACTOR
        
        # The semaphore rate is the inverse of QPS
        self.semaphore_rate = 1.0 / qps_limit if qps_limit > 0 else 1.0
        
        self.semaphore = asyncio.Semaphore(1) # Controls access
        self.last_call_time = 0
        
        logger.info(f"GeminiClient for {model_id} initialized with QPM limit {qpm_limit} (Rate: {self.semaphore_rate:.2f}s/query)")

    async def generate_content_async(
        self,
        system_prompt: str,
        user_prompt: str,
        request_id: str,
        generation_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generates content from the model, respecting rate limits.
        
        Args:
            system_prompt (str): The system-level instructions.
            user_prompt (str): The user's query.
            request_id (str): A unique ID for logging and tracing.
            generation_config (Dict, optional): e.g., {"response_mime_type": "application/json"}
            
        Returns:
            Dict[str, Any]: The parsed JSON response from the model.
        """
        async with self.semaphore:
            # Enforce time-based rate limit
            now = asyncio.get_event_loop().time()
            time_since_last_call = now - self.last_call_time
            
            if time_since_last_call < self.semaphore_rate:
                sleep_time = self.semaphore_rate - time_since_last_call
                logger.debug(f"Rate limiting {self.model_id}. Sleeping for {sleep_time:.2f}s.")
                await asyncio.sleep(sleep_time)
            
            self.last_call_time = asyncio.get_event_loop().time()
            
            try:
                logger.info(f"Sending request {request_id} to {self.model_id}")
                
                model_instance = GenerativeModel(
                    self.model_id,
                    system_instruction=system_prompt
                )
                
                # Generate content
                response = await model_instance.generate_content_async(
                    user_prompt,
                    generation_config=generation_config
                )
                
                # Extract and parse the JSON text
                if not response.parts:
                    raise ValueError("Model response contained no parts.")
                    
                response_text = response.parts[0].text
                
                # The Gemini API (with JSON mime type) should return a JSON *string*.
                # We need to parse it.
                if generation_config and generation_config.get("response_mime_type") == "application/json":
                    try:
                        import json
                        parsed_response = json.loads(response_text)
                        return parsed_response
                    except json.JSONDecodeError as json_err:
                        logger.error(f"Failed to decode JSON response from {self.model_id} for {request_id}. Error: {json_err}. Response text: {response_text[:500]}...")
                        raise ValueError(f"Invalid JSON response: {json_err}")
                
                # If not JSON, return as a simple dict
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
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.error("GEMINI_API_KEY environment variable not set.")
            raise ValueError("GEMINI_API_KEY not set")
            
        self.pool: Dict[str, GeminiClient] = {}
        self.pool_lock = asyncio.Lock()
        
        # Load QPM limits from config, fallback to defaults
        qpm_limits_config = config.get('gemini_qpm_limits', {})
        self.qpm_limits = {**DEFAULT_QPM_LIMITS, **qpm_limits_config}
        
        logger.info("GeminiPoolManager initialized.")

    async def _get_or_create_client(self, model_id: str) -> GeminiClient:
        """
        Initializes a client for a specific model_id if it doesn't exist.
        """
        async with self.pool_lock:
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
