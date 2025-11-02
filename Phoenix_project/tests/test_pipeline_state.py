import pytest
import yaml
import os
import sys # 修复：[FIX-10] 导入 sys
from datetime import datetime, timedelta

# 修复：[FIX-10] 添加日志记录器以查看 config 加载
import logging
logger = logging.getLogger(__name__)

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
    # 修复：[FIX-16] 传递 'max_recent_events'
    return PipelineState(max_recent_events=10)

# --- Test Cases ---

def test_load_config(config):
    """Tests that the config fixture loads without errors."""
    assert config is not None
    logger.info(f"Config loaded, keys: {list(config.keys())}")

def test_config_structure(config):
    """
    Tests the basic structure of the loaded config.
    修复：[FIX-13] 将 'services' 更改为 'llm' 和 'event_stream'，
    这些键存在于 config/system.yaml 中 (从 phoenix_project.py 推断)。
    """
    assert "llm" in config
    assert "event_stream" in config
    assert "data_manager" in config
    assert "execution" in config

def test_pipeline_state_initialization(state):
    """Tests the initial default values of the PipelineState."""
    assert state.get_state('system_status') == 'IDLE'
    assert state.get_state('last_event_id') is None
    assert isinstance(state.get_state('last_event_timestamp'), datetime)
    assert state.get_state('current_task_id') is None
    # 修复：[FIX-10] 匹配 'core/pipeline_state.py' 中的新 metric 名称
    assert state.get_metric('total_events_processed') == 0
    assert state.get_recent_events() == []

def test_set_and_get_state(state):
    """Tests basic set/get functionality."""
    state.set_state('system_status', 'RUNNING')
    assert state.get_state('system_status') == 'RUNNING'
    
    state.set_state('current_task_id', 'task_123')
    assert state.get_state('current_task_id') == 'task_123'

def test_set_non_existent_key(state):
    """Tests that setting an unknown key raises a KeyError."""
    with pytest.raises(KeyError):
        state.set_state('non_existent_key', 'some_value')

def test_get_non_existent_key(state):
    """Tests that getting an unknown key returns None."""
    assert state.get_state('another_non_existent_key') is None

def test_increment_metric(state):
    """Tests the metric incrementing utility."""
    assert state.get_metric('total_events_processed') == 0
    state.increment_metric('total_events_processed')
    assert state.get_metric('total_events_processed') == 1
    state.increment_metric('total_events_processed', 5)
    assert state.get_metric('total_events_processed') == 6

def test_increment_non_existent_metric(state):
    """Tests incrementing a metric that doesn't exist."""
    with pytest.raises(KeyError):
        state.increment_metric('non_existent_metric')

def test_add_event_and_get_recent(state):
    """Tests the event tracking and retrieval."""
    event1 = {'id': 'evt_1', 'timestamp': datetime.now()}
    event2 = {'id': 'evt_2', 'timestamp': datetime.now() + timedelta(seconds=1)}
    
    state.add_event(event1)
    state.add_event(event2)
    
    assert state.get_state('last_event_id') == 'evt_2'
    assert state.get_metric('total_events_processed') == 2
    assert state.get_recent_events() == [event1, event2]
    
    time_diff = datetime.now() - state.get_state('last_event_timestamp')
    assert time_diff.total_seconds() < 1.0

def test_event_queue_capacity(state):
    """Tests that the recent event queue respects max_recent_events."""
    state_small = PipelineState(max_recent_events=2)
    
    event1 = {'id': 'evt_1'}
    event2 = {'id': 'evt_2'}
    event3 = {'id': 'evt_3'}
    
    state_small.add_event(event1)
    state_small.add_event(event2)
    state_small.add_event(event3)
    
    # Should have evicted event1
    assert state_small.get_recent_events() == [event2, event3]
    assert len(state_small.get_recent_events()) == 2
    assert state_small.get_metric('total_events_processed') == 3

def test_start_and_end_task(state):
    """Tests the helpers for managing the task state."""
    task_id = "task_abc"
    state.start_task(task_id)
    
    assert state.get_state('system_status') == 'PROCESSING'
    assert state.get_state('current_task_id') == task_id
    
    state.end_task()
    
    assert state.get_state('system_status') == 'IDLE'
    assert state.get_state('current_task_id') is None
    assert state.get_metric('total_tasks_completed') == 1
    time_diff = datetime.now() - state.get_state('last_task_timestamp')
    assert time_diff.total_seconds() < 1.0

def test_get_full_state(state):
    """Tests the full state snapshot."""
    state.set_state('system_status', 'RUNNING')
    state.increment_metric('total_events_processed')
    
    full_state = state.get_full_state()
    
    assert full_state['state']['system_status'] == 'RUNNING'
    assert full_state['metrics']['total_events_processed'] == 1
    assert 'recent_events' in full_state
