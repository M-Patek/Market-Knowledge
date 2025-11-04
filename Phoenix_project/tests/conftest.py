# tests/conftest.py
import pytest
import os
import sys
from unittest.mock import MagicMock

# --- [修复] ---
# 修复：将项目根目录 (Phoenix_project) 添加到 sys.path
# 这允许测试以 'pytest' 的方式运行
# 并正确解析 'from controller...' 或 'from core...'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- [修复结束] ---

# Fixtures for AI components
@pytest.fixture
def mock_embedding_client():
    """Mocks the EmbeddingClient."""
    # 修复：将 'ai.embedding_client' 转换为 'Phoenix_project.ai.embedding_client'
    from Phoenix_project.ai.embedding_client import EmbeddingClient
    mock = MagicMock(spec=EmbeddingClient)
    # 模拟 get_embeddings 的异步行为
    mock.get_embeddings.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
    return mock

# ... (other fixtures)

# Fixtures for core components
@pytest.fixture
def sample_pipeline_state():
    """Provides a default PipelineState."""
    # 修复：将 'core.pipeline_state' 转换为 'Phoenix_project.core.pipeline_state'
    from Phoenix_project.core.pipeline_state import PipelineState
    return PipelineState(event_id="test_event_001")

# Fixtures for DataManager
@pytest.fixture
def mock_data_manager():
    """Mocks the DataManager."""
    # 修复：将 'data_manager' 转换为 'Phoenix_project.data_manager'
    from Phoenix_project.data_manager import DataManager
    mock = MagicMock(spec=DataManager)
    return mock

# Fixtures for Cognitive Engine components
@pytest.fixture
def mock_portfolio_constructor():
    """Mocks the PortfolioConstructor."""
    # 修复：将 'cognitive.portfolio_constructor' 转换为 'Phoenix_project.cognitive.portfolio_constructor'
    from Phoenix_project.cognitive.portfolio_constructor import PortfolioConstructor
    mock = MagicMock(spec=PortfolioConstructor)
    return mock

@pytest.fixture
def mock_risk_manager():
    """Mocks the RiskManager."""
    # 修复：将 'cognitive.risk_manager' 转换为 'Phoenix_project.cognitive.risk_manager'
    from Phoenix_project.cognitive.risk_manager import RiskManager
    mock = MagicMock(spec=RiskManager)
    return mock
