import pytest
import yaml
import os  # Ensure os is imported
from datetime import datetime

# Add project root to path to find 'core' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.pipeline_state import PipelineState

# --- Helper Function ---

def load_config(config_path: str):
    """Loads a YAML config file."""
    absolute_path = os.path.abspath(config_path)
    if not os.path.exists(absolute_path):
        pytest.fail(f"Config file not found at absolute path: {absolute_path}")
    with open(absolute_path, 'r') as f:
        return yaml.safe_load(f)

# --- Fixtures ---

@pytest.fixture
def config():
    """
    Provides the main system configuration, loaded from the correct path.
    """
    # FIX: Make path relative to this file, then go up to project root
    # __file__ -> tests/test_pipeline_state.py
    # os.path.dirname(__file__) -> tests/
    # os.path.dirname(os.path.dirname(__file__)) -> Phoenix_project/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # FIX: Corrected path from 'Phoenix_project/config.yaml' to 'config/system.yaml'
    config_path = os.path.join(base_dir, 'config', 'system.yaml')
    
    return load_config(config_path)

@pytest.fixture
def state():
    """Provides a clean instance of PipelineState for each test."""
    return PipelineState()

# --- Test Cases ---

def test_load_config(config):
    """Tests that the config fixture loads without errors."""
    assert config is not None
    logger.info(f"Config loaded, keys: {config.keys()}")

def test_config_structure(config):
    """Tests the basic structure of the loaded config."""
    assert "services" in config
    assert "llm" in config
    assert "event_stream" in config
    assert "data_management" in config
    assert "api_server" in config

def test_pipeline_state_initialization(state):
    """Tests the initial default values of the PipelineState."""
    assert state.get_state('system_status') == 'initializing'
    assert state.get_state('last_processed_event_id') is None
    assert isinstance(state.get_state('last_decision_cycle_time'), datetime)
    assert state.get_state('current_analysis_task') is None
    assert state.get_state('active_circuit_breakers') == []
    assert state.get_full_state()['metrics']['total_events_processed'] == 0

def test_set_and_get_state(state):
    """Tests basic set/get functionality."""
    state.set_state('system_status', 'running')
    assert state.get_state('system_status') == 'running'
    
    state.set_state('current_analysis_task', 'Analyze AAPL')
    assert state.get_state('current_analysis_task') == 'Analyze AAPL'

def test_set_non_existent_key(state):
    """Tests that setting an unknown key raises a KeyError."""
    with pytest.raises(KeyError):
        state.set_state('non_existent_key', 'some_value')

def test_get_non_existent_key(state):
    """Tests that getting an unknown key returns None."""
    assert state.get_state('another_non_existent_key') is None

def test_increment_metric(state):
    """Tests the metric incrementing utility."""
    assert state.get_full_state()['metrics']['total_events_processed'] == 0
    state.increment_metric('total_events_processed')
    assert state.get_full_state()['metrics']['total_events_processed'] == 1
    state.increment_metric('total_events_processed', 5)
    assert state.get_full_state()['metrics']['total_events_processed'] == 6

def test_increment_non_existent_metric(state):
    """Tests incrementing a metric that doesn't exist."""
    with pytest.raises(KeyError):
        state.increment_metric('non_existent_metric')

def test_update_last_processed_event(state):
    """Tests the specific helper for updating event ID and time."""
    event_id = "evt_12345"
    state.update_last_processed_event(event_id)
    
    assert state.get_state('last_processed_event_id') == event_id
    assert state.get_full_state()['metrics']['total_events_processed'] == 1
    # Check if the timestamp was updated (within a reasonable delta)
    time_diff = datetime.now() - state.get_state('last_event_processed_time')
    assert time_diff.total_seconds() < 1.0

def test_start_and_end_decision_cycle(state):
    """Tests the helpers for managing the decision cycle state."""
    task = "Analyze GOOGL"
    state.start_decision_cycle(task)
    
    assert state.get_state('system_status') == 'processing'
    assert state.get_state('current_analysis_task') == task
    
    state.end_decision_cycle()
    
    assert state.get_state('system_status') == 'idle'
    assert state.get_state('current_analysis_task') is None
    assert state.get_full_state()['metrics']['total_decision_cycles'] == 1
    time_diff = datetime.now() - state.get_state('last_decision_cycle_time')
    assert time_diff.total_seconds() < 1.0

def test_circuit_breaker_tracking(state):
    """Tests adding and removing circuit breaker states."""
    breaker_name = "llm_api"
    state.add_circuit_breaker(breaker_name)
    assert state.get_state('active_circuit_breakers') == ["llm_api"]
    
    # Test adding duplicates (should have no effect)
    state.add_circuit_breaker(breaker_name)
    assert state.get_state('active_circuit_breakers') == ["llm_api"]
    
    state.add_circuit_breaker("db_connection")
    assert "llm_api" in state.get_state('active_circuit_breakers')
    assert "db_connection" in state.get_state('active_circuit_breakers')
    
    state.remove_circuit_breaker(breaker_name)
    assert state.get_state('active_circuit_breakers') == ["db_connection"]
    
    # Test removing non-existent (should not fail)
    state.remove_circuit_breaker("non_existent")
    assert state.get_state('active_circuit_breakers') == ["db_connection"]
