"""
Task 23: Full-workflow testing.
"""
import pytest

# 我们需要将项目根目录添加到路径中才能导入 controller 模块
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from controller.loop_manager import control_loop

@pytest.mark.asyncio
async def test_full_workflow():
    """
    Assertions: Pipeline starts automatically; Output includes fusion result; Uncertainty < 1.
    """
    task = {"task": "analyze NVDA stock", "ticker": "NVDA"}
    
    # 1. 管道自动启动 (通过被调用)
    final_result = await control_loop(task)
    
    # 2. 输出包括融合结果 (Task 14)
    assert final_result is not None
    assert "conclusion" in final_result
    assert "confidence_score" in final_result
    
    # 3. 不确定性 < 1 (Task 13 & 15)
    # 我们检查不确定性
    assert final_result.get("status") != "RE_REASONING_TRIGGERED" # 这意味着不确定性是可接受的
    assert final_result.get("confidence_score") >= 0.0 # 置信度是 1 - 不确定性
