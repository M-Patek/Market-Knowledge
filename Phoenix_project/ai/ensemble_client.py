import asyncio
import logging
from typing import Dict, Any, List
from api.gemini_pool_manager import GeminiPoolManager


class _SingleAIClient:
    """
    单个 LLM 客户端实例的包装器，包含了可靠性特性
    如断路器和请求计时。
    """
    def __init__(self, client_name: str, config: Dict[str, Any], pool_manager: GeminiPoolManager):
        self.client_name = client_name
        self.pool_manager = pool_manager
        self.model_name = config.get('model_name', 'gemini-1.5-pro-latest') # 存储模型名称
        self.logger = logging.getLogger(f"PhoenixProject.AIClient.{self.client_name}")
        # llm_client 和 circuit_breaker 现在被 pool_manager 替换
        self.logger.info(f"Initialized AI client '{self.client_name}' with model '{self.model_name}' using GeminiPoolManager.")

    def update_client_config(self, new_config: Dict[str, Any]):
        """热重载底层 LLM 客户端的配置。"""
        self.model_name = new_config.get('model_name', self.model_name)
        self.logger.info(f"Updated config for AI client '{self.client_name}'. New model: {self.model_name}")

    async def execute_llm_call(self, prompt: str, temperature: float) -> Dict[str, Any]:
        """通过 GeminiPoolManager 执行异步 LLM 调用。"""
        try:
            generation_config = {"temperature": temperature}
            # Gemini API 期望 'contents' 具有特定的结构化格式。
            contents = [{"parts": [{"text": prompt}]}]
            
            response = await self.pool_manager.generate_content(
                model_name=self.model_name,
                contents=contents,
                generation_config=generation_config
            )
            # 假设池返回一个类字典对象，并添加 client_name 以便跟踪
            return {
                "response": response,
                "client_name": self.client_name,
                "status": "success"
            }
        except Exception as e:
            self.logger.error(f"LLM call failed for client '{self.client_name}': {e}", exc_info=True)
            return {
                "error": str(e),
                "client_name": self.client_name
            }


class AIEnsembleClient:
    """
    管理多个 AI 客户端的集成，优雅地分发请求和
    处理失败。
    """
    def __init__(self, ensemble_config: Dict[str, Any], pool_manager: GeminiPoolManager):
        self.logger = logging.getLogger("PhoenixProject.AIEnsembleClient")
        self.clients: Dict[str, _SingleAIClient] = {}
        self.pool_manager = pool_manager # 存储池管理器
        
        if 'clients' not in ensemble_config:
            self.logger.error("No 'clients' defined in ensemble configuration.")
            return

        for client_name, client_config in ensemble_config.get('clients', {}).items():
            self.clients[client_name] = _SingleAIClient(client_name, client_config, self.pool_manager)
            
        self.logger.info(f"AI Ensemble Client initialized with {len(self.clients)} clients: {list(self.clients.keys())}")

    def update_client_configs(self, new_ensemble_config: Dict[str, Any]):
        """
        更新所有受管客户端的配置。
        """
        self.logger.info("Starting hot-reload of AI client configurations...")
        for client_name, new_config in new_ensemble_config.get('clients', {}).items():
            if client_name in self.clients:
                self.clients[client_name].update_client_config(new_config)
            else:
                self.logger.warning(f"Config provided for unknown client '{client_name}'. Ignoring.")
        self.logger.info("AI client configuration reload complete.")

    async def execute_concurrent_calls(self, prompt: str, temperature: float = 0.5) -> List[Dict[str, Any]]:
        """
        在集成中的所有健康客户端上并发执行相同的提示。
        (Task 1.3 - Parallelism)
        """
        self.logger.info(f"Executing concurrent calls for prompt (length={len(prompt)}).")
        
        # 创建一个要并行运行的异步任务列表
        tasks = []
        for client_name, client in self.clients.items():
            tasks.append(client.execute_llm_call(prompt, temperature))

        # 并发运行所有任务
        results = await asyncio.gather(*tasks)

        successful_results = [r for r in results if 'error' not in r]
        failed_clients = [r['client_name'] for r in results if 'error' in r]
        
        self.logger.info(f"Concurrent calls complete. {len(successful_results)} successful, {len(failed_clients)} failed.")
        if failed_clients:
            self.logger.warning(f"Failed clients: {failed_clients}")
            
        return results
