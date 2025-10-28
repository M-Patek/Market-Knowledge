# api/gemini_pool_manager.py
import asyncio
import time
import google.generativeai as genai
from typing import List, Dict, Any, Optional

class GeminiPoolManager:
    """
    管理一个Gemini API密钥池，以提供高吞吐量和弹性。
    实现请求的轮询机制和智能断路器。 (Task 1.1 & 1.2)
    """
    def __init__(self, api_keys: List[str]):
        if not api_keys or len(api_keys) < 20:
            raise ValueError("至少需要20个API密钥。")
        
        self._keys = api_keys
        # 跟踪每个密钥的状态：
        # active: (bool) 用户是否启用了这个密钥
        # cooldown_until: (float) 临时速率限制的冷却时间戳
        # tripped: (bool) 是否因每日配额而跳闸
        self._key_states = {key: {"active": True, "cooldown_until": 0, "tripped": False} for key in api_keys}
        self._current_index = 0
        self._lock = asyncio.Lock()

    async def get_next_available_key(self) -> str:
        """
        使用轮询策略检索下一个可用的、活跃的API密钥。
        如果没有可用的密钥，将异步等待直到有密钥可用。
        """
        while True:
            async with self._lock:
                start_index = self._current_index
                
                for i in range(len(self._keys)):
                    idx = (start_index + i) % len(self._keys)
                    key = self._keys[idx]
                    state = self._key_states[key]
                    
                    current_time = time.time()
                    is_in_cooldown = state["cooldown_until"] > current_time
                    
                    if state["active"] and not state["tripped"] and not is_in_cooldown:
                        # 找到了一个有效的密钥。更新索引，以便下一次调用从 *下一个* 密钥开始。
                        self._current_index = (idx + 1) % len(self._keys)
                        return key
                
                # 如果完整遍历后没有找到密钥，则会跳出循环
            
            # 释放锁（通过退出 'with' 块隐式完成）并在重试前等待。
            # 这可以防止在所有密钥都不可用时出现忙循环。
            await asyncio.sleep(10) # 等待10秒后再次检查池

    async def generate_content(self, 
                               model_name: str, 
                               contents: List[Dict[str, Any]], 
                               generation_config: Optional[Dict[str, Any]] = None, 
                               max_retries: int = 3) -> Any:
        """
        使用池中的可用Gemini客户端异步生成内容。
        包含重试逻辑，并在失败时调用断路器。
        """
        request_options = None
        for attempt in range(max_retries):
            key = await self.get_next_available_key()
            # 使用 per-call 头部来传递API密钥，以避免全局 genai.configure 的线程安全问题
            request_options = {"headers": {"x-goog-api-key": key}}
            
            try:
                model = genai.GenerativeModel(model_name)
                response = await model.generate_content_async(
                   contents=contents, 
                   generation_config=generation_config, 
                   request_options=request_options
                )
                return response
            except Exception as e:
                print(f"Error with key {key[-4:]}: {e}")
                is_daily_quota_error = "daily quota" in str(e).lower()
                await self.trip_key(key, daily_quota=is_daily_quota_error)
        raise Exception("All retries failed. Could not generate content.")

    async def embed_content(self, model_name: str, contents: List[str], task_type: str, max_retries: int = 3) -> Any:
        """
        使用池中的可用Gemini客户端异步生成嵌入。
        包含重试逻辑，并在失败时调用断路器。
        """
        request_options = None
        for attempt in range(max_retries):
            key = await self.get_next_available_key()
            request_options = {"headers": {"x-goog-api-key": key}}

            try:
                # 注意: 嵌入也是通过 GenerativeModel 完成的
                model = genai.GenerativeModel(model_name)
                response = await model.embed_content_async(
                   content=contents, 
                   task_type=task_type, 
                   request_options=request_options
                )
                # genai 库会将其包装在一个字典中, e.g. {'embedding': [...]}
                return response
            except Exception as e:
                print(f"Error with key {key[-4:]}: {e}")
                is_daily_quota_error = "daily quota" in str(e).lower()
                await self.trip_key(key, daily_quota=is_daily_quota_error)
        raise Exception("All retries failed. Could not generate embeddings.")

    async def trip_key(self, key: str, daily_quota: bool = False):
        """
        因错误停用一个密钥，应用冷却或硬跳闸。
        此方法是异步的，以获取锁并确保原子状态更新。
        """
        async with self._lock:
            state = self._key_states.get(key)
            if state:
                if daily_quota:
                    print(f"Key {key[-4:]} has hit daily quota. Tripping until next reset.")
                    state["tripped"] = True
                else:
                    cooldown_seconds = 60
                    state["cooldown_until"] = time.time() + cooldown_seconds
                    print(f"Key {key[-4:]} hit a rate limit. Cooling down for {cooldown_seconds} seconds.")

    def reset_daily_quotas(self):
        """
        为因每日配额而被停用的密钥重置 'tripped' 状态。
        （应由外部调度程序在每天午夜调用）
        """
        # 这个方法不是异步的，因为它应该由一个单独的、同步的调度器调用
        # 但如果需要并发安全，它也可以是异步的
        print("Resetting all daily quota trips...")
        for key in self._keys:
            if self._key_states[key]["tripped"]:
                self._key_states[key]["tripped"] = False
                print(f"Key {key[-4:]} has been reset.")
