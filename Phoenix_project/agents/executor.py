import logging
from typing import List, Any, Dict, AsyncGenerator

from Phoenix_project.agents.registry import AgentRegistry
from Phoenix_project.core.schemas.task_schema import Task
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem
from Phoenix_project.agents.l1.base import L1Agent
from Phoenix_project.agents.l2.base import L2Agent
from Phoenix_project.agents.l3.base import L3Agent

logger = logging.getLogger(__name__)

class AgentExecutor:
    """
    负责执行单个智能体并处理其生命周期和错误。
    """
    def __init__(self, agent_registry: AgentRegistry):
        """
        初始化 AgentExecutor。
        
        参数:
            agent_registry (AgentRegistry): 存储所有已实例化智能体的注册表。
        """
        self.agent_registry = agent_registry
        logger.info("AgentExecutor initialized.")

    async def run_agent(self, agent_id: str, task: Task, context_data: List[Any]) -> List[EvidenceItem]:
        """
        异步执行单个智能体。
        此方法现在是异步的，以支持异步的 agent.run() 方法。
        
        参数:
            agent_id (str): 要执行的智能体的 ID。
            task (Task): 要传递给智能体的任务。
            context_data (List[Any]): 智能体所需的上下文数据。
            
        返回:
            List[EvidenceItem]: 从智能体运行中收集到的 EvidenceItem 列表。
        """
        logger.debug(f"AgentExecutor attempting to run agent: {agent_id} for task: {task.task_id}")
        
        agent = self.agent_registry.get(agent_id)
        
        if agent is None:
            logger.error(f"Agent with ID '{agent_id}' not found in registry.")
            return []

        if not isinstance(agent, (L1Agent, L2Agent, L3Agent)):
             logger.warning(f"Agent {agent_id} is not of a recognized base type (L1, L2, L3).")
             return []

        evidence_list = []
        try:
            # 关键变更：
            # 1. 使用 'async for' 来迭代异步生成器。
            # 2. 'agent.run' 现在被假定为一个 'async def' 方法。
            async for evidence in agent.run(task, context_data):
                if isinstance(evidence, EvidenceItem):
                    evidence.agent_id = agent.agent_id # 确保设置了 agent_id
                    evidence_list.append(evidence)
                    logger.debug(f"Agent {agent_id} produced evidence: {evidence.evidence_id}")
                else:
                    logger.warning(f"Agent {agent_id} yielded non-EvidenceItem: {type(evidence)}")
                    
            logger.info(f"Agent {agent_id} completed run for task {task.task_id}, produced {len(evidence_list)} items.")

        except NotImplementedError:
            logger.error(f"Agent {agent_id} (Type: {type(agent).__name__}) has not implemented the 'run' method.")
        except Exception as e:
            logger.error(f"Error executing agent {agent_id} for task {task.task_id}: {e}", exc_info=True)
            # 根据策略，这里可能需要重试或错误处理
            
        return evidence_list

    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        获取智能体的当前状态（模拟）。
        
        参数:
            agent_id (str): 智能体的 ID。
            
        返回:
            Dict[str, Any]: 包含智能体状态信息的字典。
        """
        agent = self.agent_registry.get(agent_id)
        if agent is None:
            return {"error": "Agent not found"}
        
        # TODO: 实现更丰富的状态跟踪
        return {
            "agent_id": agent_id,
            "class": type(agent).__name__,
            "status": "idle", # 假设状态，未来可以扩展
            "processed_tasks": 0 # 模拟
        }
