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
from core.schemas.fusion_result import AgentDecision
from ai.prompt_manager import PromptManager
from api.gateway import IAPIGateway
from monitor.logging import get_logger

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

    def execute_ensemble(self, context: str, target_symbols: List[str]) -> List[AgentDecision]:
        """
        并行执行在 agent_registry 中定义的所有 "analyst" 角色的智能体。
        """
        tasks = []
        results = []
        
        analyst_agents = [
            agent for agent in self.agent_registry.get("agents", [])
            if agent.get("role") == "analyst"
        ]

        logger.info(f"{self.log_prefix} Executing ensemble for {target_symbols} with {len(analyst_agents)} agents...")

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
                    logger.error(f"{self.log_prefix} Agent inference failed: {e}", exc_info=True)
                    
        logger.info(f"{self.log_prefix} Ensemble execution finished. Received {len(results)} decisions.")
        return results

    def _run_agent_inference(self, agent_name: str, prompt: str, model_id: Optional[str] = None) -> Optional[AgentDecision]:
        """
        调用 LLM API 并将结果解析为 AgentDecision。
        """
        try:
            start_time = time.time()
            
            # --- 恢复原有逻辑 ---
            # 1. 调用 API Gateway (假设返回文本)
            response_text = self.api_gateway.generate(
                prompt=prompt,
                model_id=model_id
            )
            
            duration = time.time() - start_time
            logger.info(f"{self.log_prefix} Agent '{agent_name}' completed in {duration:.2f}s")
            
            if not response_text:
                logger.error(f"{self.log_prefix} Agent '{agent_name}' returned empty response")
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
                logger.error(f"{self.log_prefix} Agent '{agent_name}' returned non-JSON response: {response_text}")
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
                logger.error(f"{self.log_prefix} Agent '{agent_name}' returned invalid JSON structure: {e}\nResponse: {response_data}")
                return None
            # --- 结束恢复原有逻辑 ---

        except Exception as e:
            logger.error(f"{self.log_prefix} Agent '{agent_name}' inference failed: {e}", exc_info=True)
            return None

