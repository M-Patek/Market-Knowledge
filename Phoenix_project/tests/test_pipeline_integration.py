"""
Integration test for the CognitiveEngine and PipelineOrchestrator (Layer 8).
"""
import pytest
from unittest.mock import MagicMock, patch
from Phoenix_project.cognitive.engine import CognitiveEngine
from Phoenix_project.pipeline_orchestrator import PipelineOrchestrator
from Phoenix_project.pipeline_state import PipelineState

def test_cognitive_engine_runs_pipeline_orchestrator(sample_nvda_event):
    """
    Tests that the CognitiveEngine correctly uses the sample_nvda_event
    and calls the PipelineOrchestrator.
    """
    # 1. Mock the DataManager to return our single sample event
    mock_data_manager = MagicMock()
    mock_data_manager.stream_data.return_value = [sample_nvda_event]
    
    # 2. Mock the PipelineOrchestrator to intercept the call
    mock_orchestrator_instance = MagicMock()
    # Configure its run_pipeline method to return a dummy PipelineState
    mock_orchestrator_instance.run_pipeline.return_value = PipelineState(ticker="NVDA")

    # 3. Patch the *constructor* of PipelineOrchestrator to return our mock instance
    with patch('Phoenix_project.cognitive.engine.PipelineOrchestrator', return_value=mock_orchestrator_instance) as mock_orchestrator_constructor:
        
        # 4. Initialize the CognitiveEngine with the mock DataManager
        engine = CognitiveEngine(data_manager=mock_data_manager)
        
        # 5. Run the simulation
        engine.run_simulation()

    # 6. Assert that the orchestrator's run_pipeline method was called
    #    exactly once with our sample event.
    mock_orchestrator_instance.run_pipeline.assert_called_once_with(sample_nvda_event)
