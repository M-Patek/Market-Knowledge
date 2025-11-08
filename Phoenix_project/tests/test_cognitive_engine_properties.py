# tests/test_cognitive_engine_properties.py
import pytest
import asyncio # 任务 3: 导入 asyncio
from unittest.mock import MagicMock, AsyncMock # 任务 3: 需要 AsyncMock

# --- [修复] ---
# 修复：将 'cognitive.engine' 转换为 'Phoenix_project.cognitive.engine'
from Phoenix_project.cognitive.engine import CognitiveEngine, CognitiveError # 导入 CognitiveError
# 修复：将 'core.pipeline_state' 转换为 'Phoenix_project.core.pipeline_state'
from Phoenix_project.core.pipeline_state import PipelineState
# 修复：将 'core.schemas.fusion_result' 转换为 'Phoenix_project.core.schemas.fusion_result'
from Phoenix_project.core.schemas.fusion_result import FusionResult
# [任务 3] 导入 (模拟) 依赖项
from Phoenix_project.ai.reasoning_ensemble import ReasoningEnsemble
from Phoenix_project.evaluation.fact_checker import FactChecker
from Phoenix_project.fusion.uncertainty_guard import UncertaintyGuard
from Phoenix_project.evaluation.voter import Voter

# 标记所有测试为 asyncio
pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_dependencies():
    """Mocks all async dependencies for CognitiveEngine."""
    mock_reasoning_ensemble = MagicMock(spec=ReasoningEnsemble)
    mock_fact_checker = MagicMock(spec=FactChecker)
    mock_uncertainty_guard = MagicMock(spec=UncertaintyGuard)
    mock_voter = MagicMock(spec=Voter)
    
    # [任务 3] 模拟 async 方法
    mock_reasoning_ensemble.reason = AsyncMock()
    mock_fact_checker.check_facts = AsyncMock()
    # uncertainty_guard.apply_guardrail 是同步的 (根据 engine.py)
    mock_uncertainty_guard.apply_guardrail = MagicMock()
    
    return {
        "reasoning_ensemble": mock_reasoning_ensemble,
        "fact_checker": mock_fact_checker,
        "uncertainty_guard": mock_uncertainty_guard,
        "voter": mock_voter
    }

@pytest.fixture
def cognitive_engine(mock_dependencies):
    """Fixture to create a CognitiveEngine with mocked dependencies."""
    # 基于 cognitive/engine.py 的 __init__
    config = {
        "fact_check_threshold": 0.7,
        "uncertainty_threshold": 0.6,
        "fc_confidence_boost": 0.1,
        "fc_confidence_penalty": 0.2
    }
    
    engine = CognitiveEngine(
        config=config,
        **mock_dependencies
    )
    return engine

# 任务 3: 重构测试 1
async def test_cognitive_engine_runs_cycle(cognitive_engine, mock_dependencies):
    """
    Tests the main `process_cognitive_cycle` workflow.
    (替换 test_cognitive_engine_generates_signals)
    """
    # 1. 创建一个 PipelineState
    state = PipelineState(initial_state={"main_task_query": {"description": "Test"}}, max_history=10)
    
    # 2. 模拟依赖项的返回值
    mock_fusion_result = MagicMock(spec=FusionResult)
    mock_fusion_result.confidence = 0.9 # 高于 0.7 的 fact_check_threshold
    mock_fusion_result.reasoning = "Test reasoning"
    mock_fusion_result.final_decision = "BUY"
    
    mock_dependencies["reasoning_ensemble"].reason.return_value = mock_fusion_result
    
    # 模拟 FactChecker
    mock_fact_check_report = {"overall_support": "Supported"}
    mock_dependencies["fact_checker"].check_facts.return_value = mock_fact_check_report
    
    # 模拟 UncertaintyGuard (它返回修改后的 fusion_result)
    # 在这个测试中，我们让 guard 返回一个新对象，以确认它被调用了
    mock_guarded_result = MagicMock(spec=FusionResult)
    mock_guarded_result.final_decision = "BUY" # 未更改
    mock_dependencies["uncertainty_guard"].apply_guardrail.return_value = mock_guarded_result

    # 3. 运行引擎的认知周期
    cognitive_result = await cognitive_engine.process_cognitive_cycle(state)

    # 4. 验证结果
    
    # 检查是否调用了 reason
    mock_dependencies["reasoning_ensemble"].reason.assert_called_once_with(state)
    
    # 检查是否调用了 fact_checker (因为 confidence > 0.7)
    mock_dependencies["fact_checker"].check_facts.assert_called_once_with("Test reasoning")
    
    # 检查是否调用了 uncertainty_guard
    mock_dependencies["uncertainty_guard"].apply_guardrail.assert_called_once()
    
    # 检查最终返回的字典
    assert isinstance(cognitive_result, dict)
    assert cognitive_result["final_decision"] == mock_guarded_result
    assert cognitive_result["fact_check_report"] == mock_fact_check_report
    assert state.get_value("last_fusion_result") == mock_fusion_result

# 任务 3: 重构测试 2
async def test_cognitive_engine_handles_reasoning_failure(cognitive_engine, mock_dependencies):
    """
    Tests that the engine raises a CognitiveError if reasoning fails.
    (替换 test_cognitive_engine_handles_no_candidates)
    """
    state = PipelineState(initial_state={"main_task_query": {"description": "Test"}}, max_history=10)
    
    # 模拟 reasoning_ensemble.reason 抛出异常
    mock_dependencies["reasoning_ensemble"].reason.side_effect = ValueError("LLM API failed")
    
    # 运行并断言
    with pytest.raises(CognitiveError, match="ReasoningEnsemble failed: LLM API failed"):
        await cognitive_engine.process_cognitive_cycle(state)
    
    # 确保 fact_checker 和 guard 未被调用
    mock_dependencies["fact_checker"].check_facts.assert_not_called()
    mock_dependencies["uncertainty_guard"].apply_guardrail.assert_not_called()
