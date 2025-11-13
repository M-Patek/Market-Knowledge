"""
AI 集成客户端
负责并行（或串行）调用多个 AI 智能体。
"""
import asyncio 
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from datetime import datetime
from pydantic import ValidationError

from Phoenix_project.core.schemas.fusion_result import AgentDecision
from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.api.gemini_pool_manager import GeminiPoolManager # [Fix IV.1]
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.core.pipeline_state import PipelineState
# (FIX 1.1) 导入 Registry 类型，但不是实例
from Phoenix_project.registry import Registry 

logger = get_logger(__name__)

class EnsembleClient:
    """
    管理 AI 智能体的池，并执行推理请求。
    """
    
    # (FIX 1.2) 更新 __init__ 以接收 global_registry
    def __init__(
        self, 
        gemini_manager: GeminiPoolManager, # [Fix IV.1]
        prompt_manager: PromptManager, 
        agent_registry: Dict[str, Any],
        global_registry: Registry  # <--- 注入的依赖
    ):
        """
        初始化 EnsembleClient。
        
        参数:
            gemini_manager (GeminiPoolManager): [Fix IV.1] 用于管理 Gemini 客户端和速率限制。
            prompt_manager (PromptManager): 用于管理和渲染提示。
            agent_registry (Dict[str, Any]): V1 智能体的配置 (来自 config)。
            global_registry (Registry): V2 智能体的依赖注入容器。
        """
        self.gemini_manager = gemini_manager # [Fix IV.1]
        self.prompt_manager = prompt_manager
        self.agent_registry = agent_registry # V1 (config) agents
        self.global_registry = global_registry # V2 (python) agents
        self.max_workers = agent_registry.get("config", {}).get("max_parallel_agents", 5)
        self.log_prefix = "EnsembleClient:"

    @property
    def api_gateway(self):
        """
        [Fix IV.1] [Compatibility] Expose self as the 'api_gateway' for Retriever.
        Retriever expects an object with 'generate_text', which EnsembleClient now implements.
        """
        return self

    async def generate_text(self, prompt: str, model_id: Optional[str] = None, use_json_mode: bool = False) -> Optional[str]:
        """
        [Fix IV.1] Generates text using the GeminiPoolManager.
        """
        target_model = model_id if model_id else "gemini-1.5-flash" # Default
        try:
            async with self.gemini_manager.get_client(target_model) as client:
                # Prepare contents
                contents = [prompt]
                generation_config = {"response_mime_type": "application/json"} if use_json_mode else None
                
                response = await client.generate_content_async(
                    contents=contents,
                    generation_config=generation_config
                )
                return response.get("text")
        except Exception as e:
            logger.error(f"{self.log_prefix} generate_text failed: {e}")
            return None

    # --- V2 逻辑 (同步执行, 内部使用线程池) ---
    
    def execute_ensemble_v2(self, state: PipelineState) -> List[AgentDecision]:
        """
        (V2) 执行基于 Python 类的智能体集成。
        这是一个同步方法，它在内部使用 ThreadPoolExecutor 来实现并行。
        """
        tasks = []
        results = []
        
        # 仅从配置中获取 V2 分析师智能体的 *名称*
        analyst_agents = [
            agent for agent in self.agent_registry.get("agents", [])
            if agent.get("role") == "analyst" and agent.get("version", "v1") == "v2"
        ]

        logger.info(f"{self.log_prefix} Executing ensemble V2 with {len(analyst_agents)} agents...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交 V2 任务
            for agent_config in analyst_agents:
                agent_name = agent_config["name"]
                
                tasks.append(
                    executor.submit(
                        self._run_agent_inference_v2,
                        state=state,
                        agent_name=agent_name
                    )
                )

            # 收集 V2 结果
            for future in as_completed(tasks):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"{self.log_prefix} Agent inference V2 failed: {e}", exc_info=True)
                    
        logger.info(f"{self.log_prefix} Ensemble execution V2 finished. Received {len(results)} decisions.")
        return results

    def _run_agent_inference_v2(self, state: PipelineState, agent_name: str) -> Optional[AgentDecision]:
        """ (V2) 运行单个基于 Python 的智能体 (同步) """
        try:
            start_time = time.time()
            
            # --- (FIX 1.3) 移除内部导入，使用注入的 registry ---
            # from Phoenix_project.registry import registry 
            # agent_instance = registry.resolve(agent_name)
            agent_instance = self.global_registry.resolve(agent_name)
            # --- (FIX 1.3 结束) ---
            
            if not agent_instance:
                 logger.error(f"{self.log_prefix} Agent '{agent_name}' not found in V2 registry.")
                 return None

            # V2 智能体应该有一个 'run' 方法
            if not hasattr(agent_instance, 'run'):
                logger.error(f"{self.log_prefix} Agent '{agent_name}' (V2) has no 'run' method.")
                return None

            decision: AgentDecision = agent_instance.run(state)
            
            duration = time.time() - start_time
            logger.info(f"{self.log_prefix} Agent '{agent_name}' (V2) completed in {duration:.2f}s")
            
            return decision

        except Exception as e:
            logger.error(f"{self.log_prefix} Agent '{agent_name}' (V2) inference failed: {e}", exc_info=True)
            return None

    # --- V1 逻辑 (异步) ---
    
    async def execute_ensemble(self, context: str, target_symbols: List[str]) -> List[AgentDecision]:
        """
        (V1) 异步执行基于配置/提示的智能体集成。
        """
        tasks = []
        results = []
        
        # 仅 V1 分析师智能体
        analyst_agents = [
            agent for agent in self.agent_registry.get("agents", [])
            if agent.get("role") == "analyst" and agent.get("version", "v1") == "v1"
        ]

        logger.info(f"{self.log_prefix} Executing ensemble V1 with {len(analyst_agents)} agents...")

        for agent_config in analyst_agents:
            agent_name = agent_config["name"]
            model_id = agent_config.get("model_id") # 允许覆盖模型
            
            # 渲染特定智能体的提示
            prompt_text = self.prompt_manager.render_prompt(
                template_name=agent_name,
                context=context,
                symbols=target_symbols
            )
            
            if not prompt_text:
                logger.warning(f"{self.log_prefix} Skipping agent {agent_name}: Could not render prompt.")
                continue

            tasks.append(
                self._run_agent_inference(
                    agent_name=agent_name,
                    prompt=prompt_text,
                    model_id=model_id
                )
            )
        
        # 并行执行所有 API 调用
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for result in completed_tasks:
            if isinstance(result, Exception):
                logger.error(f"{self.log_prefix} Agent inference V1 failed: {result}", exc_info=result)
            elif result:
                results.append(result)
                
        logger.info(f"{self.log_prefix} Ensemble execution V1 finished. Received {len(results)} decisions.")
        return results

    async def _run_agent_inference(self, agent_name: str, prompt: str, model_id: Optional[str] = None) -> Optional[AgentDecision]:
        """ (V1) 运行单个基于提示的智能体 (异步) """
        try:
            start_time = time.time()
            
            # [Fix IV.1] 使用 self.generate_text
            response_text = await self.generate_text(
                prompt=prompt,
                model_id=model_id,
                use_json_mode=True
            )
            
            if not response_text:
                logger.warning(f"{self.log_prefix} Agent {agent_name} (V1) returned no response.")
                return None
                
            response_data = json.loads(response_text)
            
            # 验证和解析
            try:
                decision = AgentDecision(
                    agent_name=agent_name,
                    timestamp=datetime.now(),
                    **response_data
                )
                duration = time.time() - start_time
                logger.info(f"{self.log_prefix} Agent '{agent_name}' (V1) completed in {duration:.2f}s")
                return decision
            except ValidationError as e:
                logger.error(f"{self.log_prefix} Agent {agent_name} (V1) output validation failed: {e}. Payload: {response_text[:200]}...")
                return None
                
        except Exception as e:
            logger.error(f"{self.log_prefix} Agent '{agent_name}' (V1) inference failed: {e}", exc_info=True)
            return None
