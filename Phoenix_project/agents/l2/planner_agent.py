"""
L2 Agent: Planner
Refactored from reasoning/planner.py.
Responsible for L1 Task Initialization as per the blueprint.
"""
from typing import Any, Dict, List
import json

from Phoenix_project.agents.l2.base import BaseL2Agent
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.task_schema import TaskGraph
from Phoenix_project.api.gateway import APIGateway
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class PlannerAgent(BaseL2Agent):
    """
    Implements the L2 Planner agent.
    Inherits from BaseL2Agent and implements the run method
    to decompose the main task using an LLM.
    """
    
    def __init__(self, agent_id: str, api_gateway: APIGateway):
        """
        Initializes the PlannerAgent.
        
        Args:
            agent_id (str): The unique identifier for the agent.
            api_gateway (APIGateway): The gateway for making LLM calls.
        """
        # 注意：我们调用 super().__init__ 并传递 api_gateway 作为 llm_client
        super().__init__(agent_id=agent_id, llm_client=api_gateway)
        self.api_gateway = api_gateway # 也可以选择本地保存
        logger.info(f"PlannerAgent (id='{self.agent_id}') initialized.")

    def _create_default_plan(self, ticker: str) -> TaskGraph:
        """
        回退方法：如果 LLM 规划失败，则生成一个简单的默认计划。
        """
        logger.warning(f"LLM planning failed. Falling back to default plan for {ticker}.")
        graph_dict = {
            "subgoals": [
                f"analyze fundamentals for {ticker}", 
                f"analyze technicals for {ticker}", 
                f"run adversary on {ticker}"
            ],
            "dependencies": {
                "fusion": ["analyze fundamentals", "analyze technicals", "run adversary"]
            }
        }
        return TaskGraph(**graph_dict)

    # 签名已更新：接受 dependencies=None 以保持一致性
    async def run(self, state: PipelineState, dependencies: Dict[str, Any] = None) -> TaskGraph:
        """
        分析 PipelineState 中的主任务，并使用 LLM 生成一个
        JSON 格式的多步骤执行图（subgoals 和 dependencies）。
        
        Args:
            state (PipelineState): The current pipeline state, containing the main task.
            dependencies (Dict[str, Any], optional): Not used by the planner, but required
                                                  for a consistent L2 interface. Defaults to None.
            
        Returns:
            TaskGraph: A Pydantic model defining the subgoals and dependencies.
        """
        
        # 1. 从 state 中提取主任务信息
        try:
            # 假设 state 有一个方法可以获取主任务描述
            task_query_data = state.get_main_task_query() 
            main_task_description = task_query_data.get("description", "Analyze the current market situation.")
            ticker = task_query_data.get("symbol", "UNKNOWN")
        except Exception as e:
            logger.error(f"Could not extract main task from state: {e}")
            return self._create_default_plan("UNKNOWN")

        # 2. 构造 Prompt
        prompt = f"""
        You are a high-level task planner for an AI financial analysis system.
        Your goal is to decompose a main task into a JSON graph of subgoals and their dependencies.
        The available L1 agents are: {state.get_available_l1_agents()}

        Main Task: "{main_task_description}"
        
        Respond ONLY with a valid JSON object adhering to the following schema:
        {{
          "subgoals": ["list of subgoal descriptions, e.g., 'analyze fundamentals for {ticker}'"],
          "dependencies": {{
            "some_task": ["dependency_1", "dependency_2"],
            "fusion": ["list of all final subgoals that must complete before fusion"]
          }}
        }}
        """
        
        try:
            # 3. 调用 APIGateway (self.llm_client)
            # 注意：BaseL2Agent 将 api_gateway 存储为 self.llm_client
            # 我们假设 send_request 是一个异步方法
            response_str = await self.llm_client.send_request(
                model_name="gemini-1.5-flash", # 使用一个快速的模型进行规划
                prompt=prompt,
                temperature=0.1,
                max_tokens=1024
            )
            
            # 4. 解析返回的 JSON 字符串
            
            # 清理可能的 markdown 标记
            if "```json" in response_str:
                json_str = response_str.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_str.strip()
                
            graph_dict = json.loads(json_str)
            
            # 5. 实例化 TaskGraph 对象
            task_graph = TaskGraph(**graph_dict)
            logger.info(f"Successfully generated task graph for {ticker} with {len(task_graph.subgoals)} subgoals.")
            return task_graph

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from planner LLM: {e}\nResponse: {response_str[:200]}...")
            return self._create_default_plan(ticker)
        except Exception as e:
            logger.error(f"Error during LLM task planning: {e}", exc_info=True)
            return self._create_default_plan(ticker)

    def __repr__(self) -> str:
        return f"<PlannerAgent(id='{self.agent_id}')>"
