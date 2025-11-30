"""
AI 集成客户端
负责并行（或串行）调用多个 AI 智能体。
"""
import asyncio 
import re
from typing import List, Dict, Any, Optional, Union
import time
import json
from datetime import datetime
from pydantic import ValidationError

from Phoenix_project.core.schemas.fusion_result import AgentDecision
from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.api.gemini_pool_manager import GeminiPoolManager 
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.registry import Registry 

logger = get_logger(__name__)

class EnsembleClient:
    """
    管理 AI 智能体的池，并执行推理请求。
    """
    
    def __init__(
        self, 
        gemini_manager: GeminiPoolManager, 
        prompt_manager: PromptManager, 
        agent_registry: Dict[str, Any],
        global_registry: Registry 
    ):
        """
        初始化 EnsembleClient。
        
        参数:
            gemini_manager (GeminiPoolManager): 用于管理 Gemini 客户端和速率限制。
            prompt_manager (PromptManager): 用于管理和渲染提示。
            agent_registry (Dict[str, Any]): V1 智能体的配置 (来自 config)。
            global_registry (Registry): V2 智能体的依赖注入容器。
        """
        self.gemini_manager = gemini_manager 
        self.prompt_manager = prompt_manager
        self.agent_registry = agent_registry 
        self.global_registry = global_registry 
        self.max_workers = agent_registry.get("config", {}).get("max_parallel_agents", 5)
        self.log_prefix = "EnsembleClient:"

    @property
    def api_gateway(self):
        """
        [Compatibility] Expose self as the 'api_gateway' for Retriever.
        Retriever expects an object with 'generate_text', which EnsembleClient now implements.
        """
        return self

    async def generate_text(self, prompt: Union[str, List[Any]], model_id: Optional[str] = None, use_json_mode: bool = False, tools: Optional[List[Any]] = None) -> Any:
        """
        [Fix IV.1] Generates text using the GeminiPoolManager.
        [Task V Fix] Added support for 'tools' and robust response access.
        
        Returns full response dict to expose function_call.
        """
        target_model = model_id if model_id else "gemini-1.5-flash" # Default
        
        # [Task 8] Remove broad exception handling to allow errors (Network, Timeout, Quota) to bubble up
        # This enables the Orchestrator/LoopManager to handle retries or circuit breaking.
        async with self.gemini_manager.get_client(target_model) as client:
            # Prepare contents
            contents = prompt if isinstance(prompt, list) else [prompt]
            generation_config = {"response_mime_type": "application/json"} if use_json_mode else None
            
            return await client.generate_content_async(
                contents=contents,
                generation_config=generation_config,
                tools=tools
            )

    async def run_llm_task(self, agent_prompt_name: str, context_map: Dict[str, Any]) -> Optional[str]:
        """
        [Task V Fix] Wrapper for FusionAgent to render prompt and generate text.
        """
        # Render the prompt using PromptManager
        prompt_text = self.prompt_manager.render_prompt(agent_prompt_name, **context_map)
        
        if not prompt_text:
            logger.warning(f"{self.log_prefix} run_llm_task: Prompt rendering failed for {agent_prompt_name}")
            return None

        # [Task 8] Allow exceptions to propagate
        response = await self.generate_text(prompt=prompt_text, use_json_mode=True)
        return response.get("text") if isinstance(response, dict) else str(response)

    def _clean_json_string(self, text: str) -> str:
        """
        [Task 3.3] Robust JSON Extraction: Finds outer-most braces.
        Ignores markdown code blocks or conversational text wrapper.
        """
        text = text.strip()
        
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx : end_idx + 1]
            
        return text

    async def run_chain_structured(self, prompt: str, tools: Optional[List[Any]] = None, model_name: Optional[str] = None) -> Optional[Any]:
        """
        [Task V Fix] Wrapper for FactChecker to execute with tools and return structured data (JSON).
        
        Implements a ReAct loop.
        """
        messages = [{"role": "user", "parts": [{"text": prompt}]}]
        max_turns = 5

        for _ in range(max_turns):
            try:
                # [Task 8] Allow system exceptions to propagate, catch only JSON errors
                response = await self.generate_text(
                    prompt=messages, 
                    model_id=model_name, 
                    use_json_mode=True, # FactChecker expects JSON response
                    tools=tools
                )
                
                if not response:
                    return None
                    
                function_call = response.get("function_call")
                text_content = response.get("text", "")

                if function_call:
                    logger.info(f"{self.log_prefix} Tool call detected: {function_call}")
                    # TODO: Execute tool and append result.
                    # Placeholder: break to prevent infinite loop as we lack the executor here
                    break

                if text_content:
                    cleaned = self._clean_json_string(text_content)
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError as e:
                        logger.warning(f"{self.log_prefix} JSON parse failed: {e}. Retrying...")
                        continue

            except Exception as e:
                 logger.error(f"{self.log_prefix} run_chain_structured error: {e}")
                 return None
        
        return None

    # --- V2 逻辑 (异步执行) ---
    
    async def execute_ensemble_v2(self, state: PipelineState) -> List[AgentDecision]:
        """
        (V2) 执行基于 Python 类的智能体集成。
        [Refactor 1.3] Async implementation using asyncio.gather.
        """
        tasks = []
        
        # 仅从配置中获取 V2 分析师智能体的 *名称*
        analyst_agents = [
            agent for agent in self.agent_registry.get("agents", [])
            if agent.get("role") == "analyst" and agent.get("version", "v1") == "v2"
        ]

        logger.info(f"{self.log_prefix} Executing ensemble V2 with {len(analyst_agents)} agents...")

        for agent_config in analyst_agents:
            agent_name = agent_config["name"]
            tasks.append(
                self._run_agent_inference_v2(state, agent_name)
            )

        # Gather results with error handling
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for res in results_raw:
            if isinstance(res, Exception):
                logger.error(f"{self.log_prefix} Agent inference V2 failed: {res}")
            elif res:
                results.append(res)
                    
        logger.info(f"{self.log_prefix} Ensemble execution V2 finished. Received {len(results)} decisions.")
        return results

    async def _run_agent_inference_v2(self, state: PipelineState, agent_name: str) -> Optional[AgentDecision]:
        """ (V2) Run single python-based agent (Async Wrapper). """
        try:
            start_time = time.time()
            
            # --- (FIX 1.3) 移除内部导入，使用注入的 registry ---
            agent_instance = self.global_registry.get_component(agent_name)
            
            if not agent_instance:
                 logger.error(f"{self.log_prefix} Agent '{agent_name}' not found in V2 registry.")
                 return None

            # V2 智能体应该有一个 'run' 方法
            if not hasattr(agent_instance, 'run'):
                logger.error(f"{self.log_prefix} Agent '{agent_name}' (V2) has no 'run' method.")
                return None

            # Run in thread to avoid blocking event loop if sync
            decision: AgentDecision = await asyncio.to_thread(agent_instance.run, state)
            
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
            response = await self.generate_text(
                prompt=prompt,
                model_id=model_id,
                use_json_mode=True
            )
            
            if not response:
                logger.warning(f"{self.log_prefix} Agent {agent_name} (V1) returned no response.")
                return None
            
            # Handle response type (Dict or Object)
            text_content = response.get("text", "") if isinstance(response, dict) else str(response)
            
            # 验证和解析
            try:
                response_data = json.loads(text_content)
                decision = AgentDecision(
                    agent_name=agent_name,
                    timestamp=datetime.now(),
                    **response_data
                )
                duration = time.time() - start_time
                logger.info(f"{self.log_prefix} Agent '{agent_name}' (V1) completed in {duration:.2f}s")
                return decision
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"{self.log_prefix} Agent {agent_name} (V1) output validation failed: {e}. Payload: {text_content[:200]}...")
                return None
                
        except Exception as e:
            logger.error(f"{self.log_prefix} Agent '{agent_name}' (V1) inference failed: {e}", exc_info=True)
            return None
