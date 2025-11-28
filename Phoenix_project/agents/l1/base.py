# Phoenix_project/agents/l1/base.py
# [主人喵的修复] 修复了 safe_run 返回字典的问题，强制返回 EvidenceItem 对象
# 确保错误能被上层系统感知，而不是被类型过滤器静默吞噬。

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import traceback
import asyncio
from pydantic import ValidationError
from Phoenix_project.core.pipeline_state import PipelineState
# [修复] 引入标准数据契约
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem, EvidenceType

logger = logging.getLogger(__name__)

class BaseL1Agent(ABC):
    """
    Abstract Base Class for all L1 agents.
    L1 agents are specialized LLM-driven experts that generate EvidenceItems.
    """
    
    def __init__(self, agent_id: str, llm_client: Any):
        """
        Initializes the L1 agent.
        
        Args:
            agent_id (str): The unique identifier for the agent (from registry.yaml).
            llm_client (Any): An instance of an LLM client (e.g., GeminiPoolManager).
        """
        self.agent_id = agent_id
        self.llm_client = llm_client

    @abstractmethod
    async def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> EvidenceItem:
        """
        The main execution method for the agent.
        Must return an EvidenceItem.
        """
        pass

    async def safe_run(self, state: PipelineState, dependencies: Dict[str, Any]) -> EvidenceItem:
        """
        [Beta FIX] A defensive wrapper around the abstract run method.
        Handles exceptions and returns a valid ERROR EvidenceItem if the agent crashes.
        """
        try:
            logger.info(f"Agent {self.agent_id} starting execution.")
            
            # 基础输入检查
            if not dependencies and state is None:
                 logger.warning(f"Agent {self.agent_id} received empty state and dependencies.")

            result = await self.run(state, dependencies)
            
            # [双重检查] 确保子类实现返回了正确的类型
            if not isinstance(result, EvidenceItem):
                # 尝试兼容字典返回，但最好强制对象
                if isinstance(result, dict):
                    try:
                        result = EvidenceItem.model_validate(result)
                    except:
                        raise ValueError(f"Agent {self.agent_id} returned invalid dict")
                else:
                    raise ValueError(f"Agent {self.agent_id} returned {type(result)} instead of EvidenceItem")
                
            return result

        except (ValueError, KeyError, TypeError, AttributeError, ValidationError) as e:
            # [Task 2.1] Catch deterministic Business Logic Errors only.
            # System/Network errors (RuntimeError, ConnectionError) will bubble up for Retry.
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Agent {self.agent_id} CRASHED: {error_msg}\n{stack_trace}")
            
            # [修复] 返回合法的 EvidenceItem 对象，而不是字典
            # 这样 CognitiveEngine 就能接收到错误信息，而不会因为类型检查而丢弃它
            return EvidenceItem(
                agent_id=self.agent_id,
                content=f"CRITICAL FAILURE: Agent crashed during execution. Error: {error_msg}",
                evidence_type=EvidenceType.GENERIC, # 或者可以定义一个 SYSTEM_ERROR 类型
                confidence=0.0, # 零置信度
                data_horizon="Immediate",
                symbols=[], # 无法确定涉及哪些代码，留空
                metadata={
                    "status": "CRASHED",
                    "error_type": type(e).__name__,
                    "stack_trace": stack_trace
                }
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id='{self.agent_id}')>"
