# Phoenix_project/agents/executor.py
import asyncio
from typing import List
from monitor.metrics import L1_LAT # 来自 Task 25 的更正后导入
import time


async def _mock_gemini_call(agent_name: str, plan: dict, rag_context: str) -> dict:
    """模拟函数以模拟并行的 Gemini API 调用。"""
    start_time = time.time()
    await asyncio.sleep(0.1)  # 模拟网络延迟
    latency = time.time() - start_time
    L1_LAT.observe(latency) # 使用导入的指标
    
    return {
        "agent": agent_name,
        "cot": [
            f"{agent_name} CoT: Analyzed plan step 1.",
            f"{agent_name} CoT: Used RAG context: {str(rag_context)[:20]}...", # Added str() for safety
            f"{agent_name} CoT: Reached a conclusion."
        ],
        "result": "Buy" if agent_name == "technical" else "Hold",
        "confidence": 0.78 if agent_name == "technical" else 0.65
    }


async def run_agents(plan: dict, rag_context: str) -> list[dict]:
    """Invokes Gemini in parallel, returns structured results."""
    
    # We now dynamically get the agent IDs from the plan's keys.
    agent_ids_to_run = list(plan.keys())
    
    tasks = []
    for agent_id in agent_ids_to_run:
        tasks.append(_mock_gemini_call(agent_id, plan, rag_context))
        
    results = await asyncio.gather(*tasks)
    
    return results

