"""
L2 Agent: Metacognitive Agent
Refactored from ai/metacognitive_agent.py.
Responsible for "Supervision" of L1/L2 agent reasoning (CoT).
"""
from typing import Any, List

from Phoenix_project.agents.l2.base import BaseL2Agent
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem
from Phoenix_project.core.schemas.supervision_result import SupervisionResult
from Phoenix_project.api.gateway import APIGateway
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class MetacognitiveAgent(BaseL2Agent):
    """
    Implements the L2 Metacognitive agent.
    This agent monitors the CoT of other agents to identify
    divergence and potential hallucinations using an LLM.
    """
    
    def __init__(self, agent_id: str, api_gateway: APIGateway):
        """
        Initializes the MetacognitiveAgent.
        
        Args:
            agent_id (str): The unique identifier for the agent.
            api_gateway (APIGateway): The gateway for making LLM calls.
        """
        super().__init__(agent_id=agent_id, llm_client=api_gateway)
        self.api_gateway = api_gateway
        logger.info(f"MetacognitiveAgent (id='{self.agent_id}') initialized.")


    async def run(self, state: PipelineState, evidence_items: List[EvidenceItem]) -> SupervisionResult:
        """
        Monitors the CoT (reasoning) of L1 EvidenceItems.
        
        Args:
            state (PipelineState): The current state, used to access CoT traces.
            evidence_items (List[EvidenceItem]): The collected list of L1 outputs.
            
        Returns:
            SupervisionResult: A single result object summarizing the findings.
        """
        
        target_agents = [item.agent_id for item in evidence_items]
        
        if not evidence_items:
            return SupervisionResult(
                agent_id=self.agent_id,
                analysis_summary="No evidence items provided to supervise.",
                target_agent_ids=target_agents,
                flags=["NO_EVIDENCE"]
            )

        # 1. 遍历传入的 evidence_items，提取 reasoning
        reasoning_context = ""
        for i, item in enumerate(evidence_items):
            # L1 EvidenceItem 将其 reasoning 存储在 'content' 字段中
            reasoning_context += f"--- Evidence {i+1} (Agent: {item.agent_id}, Confidence: {item.confidence:.2f}) ---\n"
            reasoning_context += f"{item.content}\n\n"

        # 2. 构造一个 Prompt
        prompt = f"""
        You are a Metacognitive Supervisor AI. Your task is to analyze the reasoning (Chain of Thought) from a group of financial analyst AIs to identify flaws.
        
        Review the following evidence items provided by different L1 agents:
        
        {reasoning_context}
        
        Based on this input, perform the following two tasks:
        1. Identify any direct contradictions or logical conflicts between the agents.
        2. Assess if any agent's reasoning seems unsubstantiated, biased, or like a potential "hallucination" (making claims without sufficient evidence).
        
        Provide a concise, single-paragraph summary of your findings.
        If you find specific issues, list the flags (e.g., "HALLUCINATION", "DIVERGENCE").
        
        Respond in the following format:
        SUMMARY: [Your one-paragraph analysis]
        FLAGS: [COMMA_SEPARATED_LIST_OF_FLAGS_OR_NONE]
        """

        try:
            # 3. 调用 self.api_gateway.send_request(...)
            response_str = await self.api_gateway.send_request(
                model_name="gemini-1.5-pro", # 需要一个强大的模型来进行元认知
                prompt=prompt,
                temperature=0.3,
                max_tokens=512
            )
            
            # 4. 将 LLM 的文本回复封装到 SupervisionResult 对象中
            summary = "Failed to parse LLM response."
            flags = ["PARSING_ERROR"]
            
            # 解析 LLM 回复
            for line in response_str.split('\n'):
                if line.upper().startswith("SUMMARY:"):
                    summary = line[len("SUMMARY:"):].strip()
                elif line.upper().startswith("FLAGS:"):
                    flags_str = line[len("FLAGS:"):].strip()
                    if flags_str.upper() != "NONE":
                        flags = [f.strip() for f in flags_str.split(',')]
                    else:
                        flags = []
                        
            return SupervisionResult(
                agent_id=self.agent_id,
                analysis_summary=summary,
                target_agent_ids=target_agents,
                flags=flags
            )

        except Exception as e:
            logger.error(f"MetacognitiveAgent failed LLM call: {e}", exc_info=True)
            return SupervisionResult(
                agent_id=self.agent_id,
                analysis_summary=f"LLM call failed: {e}",
                target_agent_ids=target_agents,
                flags=["LLM_ERROR"]
            )

    def __repr__(self) -> str:
        return f"<MetacognitiveAgent(id='{self.agent_id}')>"
