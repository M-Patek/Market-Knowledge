# tests/test_pipeline_state.py
import pytest
from datetime import datetime

# --- [修复] ---
# 修复：将 'core.pipeline_state' 转换为 'Phoenix_project.core.pipeline_state'
from Phoenix_project.core.pipeline_state import PipelineState
# 修复：将 'core.schemas.data_schema' 转换为 'Phoenix_project.core.schemas.data_schema'
from Phoenix_project.core.schemas.data_schema import MarketEvent, Signal
# 修复：将 'core.schemas.evidence_schema' 转换为 'Phoenix_project.core.schemas.evidence_schema'
from Phoenix_project.core.schemas.evidence_schema import Evidence
# 修复：将 'core.schemas.fusion_result' 转换为 'Phoenix_project.core.schemas.fusion_result'
from Phoenix_project.core.schemas.fusion_result import FusionResult
# 修复：将 'core.schemas.risk_schema' 转换为 'Phoenix_project.core.schemas.risk_schema'
from Phoenix_project.core.schemas.risk_schema import RiskAssessment
# --- [修复结束] ---


@pytest.fixture
def state():
    """Provides a clean PipelineState for each test."""
    return PipelineState(event_id="evt_001")

def test_pipeline_state_initialization(state):
    """Tests the default values upon initialization."""
    assert state.event_id == "evt_001"
    assert state.start_time is not None
    assert state.current_step == "INITIALIZED"
    assert state.is_successful is False
    assert state.error_message is None
    
    # Check that all data fields are empty
    assert state.raw_event is None
    assert state.context_data == {}
    assert state.analysis_results == []
    assert state.fusion_result is None
    assert state.risk_assessment is None
    assert state.signals == []

def test_add_step_and_timing(state):
    """Tests the logging of pipeline steps and their timings."""
    state.add_step("DATA_INGESTION", 10.5)
    state.add_step("ANALYSIS", 150.2)
    
    assert state.current_step == "ANALYSIS"
    assert "DATA_INGESTION" in state.timings
    assert state.timings["DATA_INGESTION"] == 10.5
    assert "ANALYSIS" in state.timings
    assert state.timings["ANALYSIS"] == 150.2

def test_set_raw_event(state):
    """Tests attaching the raw event."""
    event = MarketEvent(
        id="evt_001",
        source="test",
        timestamp=datetime.now(),
        content="Test event"
    )
    state.set_raw_event(event)
    assert state.raw_event == event

def test_add_analysis_result(state):
    """Tests appending analysis results (Evidence)."""
    evidence1 = Evidence(
        source="agent_1",
        content="Evidence 1",
        timestamp=datetime.now(),
        confidence=0.8
    )
    evidence2 = Evidence(
        source="agent_2",
        content="Evidence 2",
        timestamp=datetime.now(),
        confidence=0.9
    )
    
    state.add_analysis_result(evidence1)
    state.add_analysis_result(evidence2)
    
    assert len(state.analysis_results) == 2
    assert state.analysis_results[0] == evidence1
    assert state.analysis_results[1] == evidence2

def test_set_fusion_result(state):
    """Tests setting the final FusionResult."""
    fusion = FusionResult(
        event_id="evt_001",
        timestamp=datetime.now(),
        assessment="Fused assessment",
        trade_candidates=[{"ticker": "AAPL"}]
    )
    state.set_fusion_result(fusion)
    assert state.fusion_result == fusion

def test_set_risk_assessment(state):
    """Tests setting the RiskAssessment."""
    risk = RiskAssessment(
        allows_execution=False,
        reason="Market volatility too high",
        confidence=0.95
    )
    state.set_risk_assessment(risk)
    assert state.risk_assessment == risk
    
def test_set_signals(state):
    """Tests setting the final list of Signals."""
    signal1 = Signal(
        symbol="AAPL",
        signal_type="BUY",
        strength=0.5
    )
    state.set_signals([signal1])
    assert len(state.signals) == 1
    assert state.signals[0] == signal1

def test_mark_failure(state):
    """Tests marking the pipeline as failed."""
    state.mark_failure("ANALYSIS", "Something went wrong")
    
    assert state.is_successful is False
    assert state.current_step == "ANALYSIS"
    assert state.error_message == "Something went wrong"

def test_mark_success(state):
    """Tests marking the pipeline as successful."""
    state.mark_success("SIGNAL_GENERATION")
    
    assert state.is_successful is True
    assert state.current_step == "SIGNAL_GENERATION"
    assert state.error_message is None
    assert state.end_time is not None
    assert state.total_duration_ms > 0
