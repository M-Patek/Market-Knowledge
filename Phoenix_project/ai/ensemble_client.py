"""
AI 集成客户端
负责并行（或串行）调用多个 AI 智能体。
"""
import asyncio # <-- (FIX 2.2) 为 asyncio.gather 导入
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from datetime import datetime
from pydantic import ValidationError
from Phoenix_project.core.schemas.fusion_result import AgentDecision
from Phoenix_project.ai.prompt_manager import PromptManager
# --- (FIX 3) 修正导入：IAPIGateway -> APIGateway ---
from Phoenix_project.api.gateway import APIGateway
# --- (FIX 3 结束) ---
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.core.pipeline_state import PipelineState

logger = get_logger(__name__)
class EnsembleClient:
    """
    管理 AI 智能体的池，并执行推理请求。
    """
    
    # --- (FIX 3) 修正类型提示：IAPIGateway -> APIGateway ---
    def __init__(self, api_gateway: APIGateway, prompt_manager: PromptManager, agent_registry: Dict[str, Any]):
    # --- (FIX 3 结束) ---
        self.api_gateway = api_gateway
        self.prompt_manager = prompt_manager
        self.agent_registry = agent_registry 
        self.max_workers = agent_registry.get("config", {}).get("max_parallel_agents", 5)
        self.log_prefix = "EnsembleClient:"

    # --- V2 逻辑 (保持不变) ---
    def execute_ensemble_v2(self, state: PipelineState) -> List[AgentDecision]:
        # ... (此 V2 方法保持不变) ...
        tasks = []
        results = []
        
        analyst_agents = [
            agent for agent in self.agent_registry.get("agents", [])
            if agent.get("role") == "analyst"
        ]

        logger.info(f"{self.log_prefix} Executing ensemble V2 with {len(analyst_agents)} agents...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for agent_config in analyst_agents:
                agent_name = agent_config["name"]
                
                tasks.append(
                    executor.submit(
                        self._run_agent_inference_v2,
                        state=state,
                        agent_name=agent_name
                    )
                )

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
        # ... (此 V2 方法保持不变) ...
        try:
            start_time = time.time()
            
            from Phoenix_project.registry import registry 
            agent_instance = registry.resolve(agent_name)
            
            if not agent_instance:
                 logger.error(f"{self.log_prefix} Agent '{agent_name}' not found in registry.")
                 return None

            decision: AgentDecision = agent_instance.run(state)
            
            duration = time.time() - start_time
            logger.info(f"{self.log_prefix} Agent '{agent_name}' (V2) completed in {duration:.2f}s")
            
            return decision

        except Exception as e:
            logger.error(f"{self.log_prefix} Agent '{agent_name}' (V2) inference failed: {e}", exc_info=True)
            return None

    # --- (V1 - 旧版逻辑) ---

    # --- (FIX 2.2) 更改为 'async def' ---
    async def execute_ensemble(self, context: str, target_symbols: List[str]) -> List[AgentDecision]:
        """
        (V1) 并行执行在 agent_registry 中定义的所有 "analyst" 角色的智能体。
        (已重构为异步)
        """
        tasks = []
        
        analyst_agents = [
            agent for agent in self.agent_registry.get("agents", [])
            if agent.get("role") == "analyst"
        ]

        logger.info(f"{self.log_prefix} Executing ensemble (V1) for {target_symbols} with {len(analyst_agents)} agents...")

        # --- (FIX 2.2) 用 asyncio.gather 替换 ThreadPoolExecutor ---
        for agent_config in analyst_agents:
            agent_name = agent_config["name"]
            prompt_name = agent_config["prompt"]
            
            prompt_text = self.prompt_manager.render_prompt(
                prompt_name,
                context=context,
                target_symbols=target_symbols
            )
            
            if prompt_text:
                # 添加协程任务
                tasks.append(
                    self._run_agent_inference(
                        agent_name=agent_name,
                        prompt=prompt_text,
                        model_id=agent_config.get("model_id")
                    )
                )
            else:
                logger.warning(f"{self.log_prefix} Skipping agent {agent_name}: Failed to render prompt {prompt_name}")

        # 并发执行所有异步任务
        # return_exceptions=True 确保一个任务失败不会导致所有任务崩溃
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for result in all_results:
            if isinstance(result, AgentDecision):
                results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"{self.log_prefix} Agent inference (V1) failed in asyncio.gather: {result}", exc_info=result)
        # --- (FIX 2.2 结束) ---
                    
        logger.info(f"{self.log_prefix} Ensemble execution (V1) finished. Received {len(results)} decisions.")
        return results

    # --- (FIX 2.1) 更改为 'async def' ---
    async def _run_agent_inference(self, agent_name: str, prompt: str, model_id: Optional[str] = None) -> Optional[AgentDecision]:
        """
        (V1) 调用 LLM API 并将结果解析为 AgentDecision。
        (已重构为异步)
        """
        try:
            start_time = time.time()
            
            # --- (FIX 2.1) 适配异步 API (gateway.send_request) ---
            model_to_use = model_id if model_id else "gemini-pro"
            
            # 使用 'await' 而不是 'asyncio.run()'
            response_text = await self.api_gateway.send_request(
                model_name=model_to_use,
                prompt=prompt
            )
            # --- (FIX 2.1 结束) ---
            
            duration = time.time() - start_time
            logger.info(f"{self.log_prefix} Agent '{agent_name}' (V1) completed in {duration:.2f}s")
            
            if not response_text:
                logger.error(f"{self.log_prefix} Agent '{agent_name}' (V1) returned empty response")
                return None

            # 2. 将响应文本 (JSON) 解析为字典
            try:
                if "```json" in response_text:
                    json_text = response_text.split("```json")[1].split("```")[0].strip()
                else:
                    json_text = response_text
                
                response_data = json.loads(json_text)
            except json.JSONDecodeError:
                logger.error(f"{self.log_prefix} Agent '{agent_name}' (V1) returned non-JSON response: {response_text}")
                return None

            # 3. 验证字典并转换为 Pydantic 模型
            try:
                decision = AgentDecision(
                    agent_name=agent_name,
                    timestamp=datetime.utcnow(),
                    **response_data
                )
                return decision
            except ValidationError as e:
                logger.error(f"{self.log_prefix} Agent '{agent_name}' (V1) returned invalid JSON structure: {e}\nResponse: {response_data}")
                return None

        except Exception as e:
            logger.error(f"{self.log_prefix} Agent '{agent_name}' (V1) inference failed: {e}", exc_info=True)
            return None
