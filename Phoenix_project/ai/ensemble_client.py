"""
AI 集成客户端
负责并行（或串行）调用多个 AI 智能体。
"""
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
# 恢复原有逻辑所需的新增导入
import json
from datetime import datetime
from pydantic import ValidationError

# FIX (E3): 导入 AgentDecision (这个保留)
from Phoenix_project.core.schemas.fusion_result import AgentDecision
from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.api.gateway import IAPIGateway
from Phoenix_project.monitor.logging import get_logger

# (新增) 导入 PipelineState
from Phoenix_project.core.pipeline_state import PipelineState

logger = get_logger(__name__)

class EnsembleClient:
    """
    管理 AI 智能体的池，并执行推理请求。
    """
    
    def __init__(self, api_gateway: IAPIGateway, prompt_manager: PromptManager, agent_registry: Dict[str, Any]):
        self.api_gateway = api_gateway
        self.prompt_manager = prompt_manager
        self.agent_registry = agent_registry # 来自 agents/registry.yaml
        self.max_workers = agent_registry.get("config", {}).get("max_parallel_agents", 5)
        self.log_prefix = "EnsembleClient:"

    # (新增) execute_ensemble 的 (PipleineState) 版本
    def execute_ensemble_v2(self, state: PipelineState) -> List[AgentDecision]:
        """
        (V2) 并行执行在 agent_registry 中定义的所有 "analyst" 角色的智能体。
        使用 PipelineState 作为上下文。
        """
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
                
                # (V2) 我们不再渲染提示，而是将状态传递给智能体
                # 智能体将自己负责从状态中提取数据并渲染提示
                
                tasks.append(
                    executor.submit(
                        self._run_agent_inference_v2,
                        state=state,
                        agent_name=agent_name
                        # (model_id 等其他配置可以传入)
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

    # (新增) _run_agent_inference 的 (PipleineState) 版本
    def _run_agent_inference_v2(self, state: PipelineState, agent_name: str) -> Optional[AgentDecision]:
        """
        (V2) 运行单个智能体。
        """
        try:
            start_time = time.time()
            
            # 1. (V2) 从注册表获取智能体实例 (假设已注册)
            from Phoenix_project.registry import registry # (循环导入风险？)
            agent_instance = registry.resolve(agent_name)
            
            if not agent_instance:
                 logger.error(f"{self.log_prefix} Agent '{agent_name}' not found in registry.")
                 return None

            # 2. (V2) 运行智能体
            # 智能体自己处理 API 调用和解析
            decision: AgentDecision = agent_instance.run(state)
            
            duration = time.time() - start_time
            logger.info(f"{self.log_prefix} Agent '{agent_name}' (V2) completed in {duration:.2f}s")
            
            return decision

        except Exception as e:
            logger.error(f"{self.log_prefix} Agent '{agent_name}' (V2) inference failed: {e}", exc_info=True)
            return None

    # --- (V1 - 旧版逻辑) ---

    def execute_ensemble(self, context: str, target_symbols: List[str]) -> List[AgentDecision]:
        """
        (V1) 并行执行在 agent_registry 中定义的所有 "analyst" 角色的智能体。
        """
        tasks = []
        results = []
        
        analyst_agents = [
            agent for agent in self.agent_registry.get("agents", [])
            if agent.get("role") == "analyst"
        ]

        logger.info(f"{self.log_prefix} Executing ensemble (V1) for {target_symbols} with {len(analyst_agents)} agents...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for agent_config in analyst_agents:
                agent_name = agent_config["name"]
                prompt_name = agent_config["prompt"]
                
                # 渲染该智能体特定的提示
                prompt_text = self.prompt_manager.render_prompt(
                    prompt_name,
                    context=context,
                    target_symbols=target_symbols
                )
                
                if prompt_text:
                    tasks.append(
                        executor.submit(
                            self._run_agent_inference,
                            agent_name=agent_name,
                            prompt=prompt_text,
                            model_id=agent_config.get("model_id") # (e.g., "gemini-1.5-pro")
                        )
                    )
                else:
                    logger.warning(f"{self.log_prefix} Skipping agent {agent_name}: Failed to render prompt {prompt_name}")

            for future in as_completed(tasks):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"{self.log_prefix} Agent inference (V1) failed: {e}", exc_info=True)
                    
        logger.info(f"{self.log_prefix} Ensemble execution (V1) finished. Received {len(results)} decisions.")
        return results

    def _run_agent_inference(self, agent_name: str, prompt: str, model_id: Optional[str] = None) -> Optional[AgentDecision]:
        """
        (V1) 调用 LLM API 并将结果解析为 AgentDecision。
        """
        try:
            start_time = time.time()
            
            # 1. 调用 API Gateway (假设返回文本)
            response_text = self.api_gateway.generate(
                prompt=prompt,
                model_id=model_id
            )
            
            duration = time.time() - start_time
            logger.info(f"{self.log_prefix} Agent '{agent_name}' (V1) completed in {duration:.2f}s")
            
            if not response_text:
                logger.error(f"{self.log_prefix} Agent '{agent_name}' (V1) returned empty response")
                return None

            # 2. 将响应文本 (JSON) 解析为字典
            try:
                # 尝试找到json代码块，以防模型返回额外的文本
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
                    agent_name=agent_name, # 覆盖 agent name
                    timestamp=datetime.utcnow(), # 添加时间戳
                    **response_data
                )
                return decision
            except ValidationError as e:
                logger.error(f"{self.log_prefix} Agent '{agent_name}' (V1) returned invalid JSON structure: {e}\nResponse: {response_data}")
                return None

        except Exception as e:
            logger.error(f"{self.log_prefix} Agent '{agent_name}' (V1) inference failed: {e}", exc_info=True)
            return None
