# Phoenix_project/controller/loop_manager.py
import asyncio
from typing import Dict, Any
from controller.orchestrator import Orchestrator
from fusion.uncertainty_guard import guard

# TODO: Load this from config (Task 20, 'auto_retry')
MAX_LOOPS = 3 

async def control_loop(task: dict) -> dict:
    """
    Manages re-reasoning and early stopping.
    Automatically decides whether to continue based on uncertainty/consistency.
    """
    
    orchestrator = Orchestrator()
    final_result = {}
    
    for i in range(MAX_LOOPS):
        # 1. 运行主管道
        result = await orchestrator.run_pipeline(task)
        
        # 2. 检查不确定性
        guarded_result = guard(result) # 我们将使用默认阈值
        
        if guarded_result.get("status") != "RE_REASONING_TRIGGERED":
            final_result = guarded_result
            break # 提前停止，不确定性足够低
        
        # TODO: 添加逻辑来修改 'task' 以进行重新推理
        task["re_reasoning_prompt"] = f"Loop {i+1}: Previous analysis was uncertain. Please re-evaluate."
        final_result = guarded_result # 存储最后的不确定结果，以防达到最大循环次数
        
    return final_result
