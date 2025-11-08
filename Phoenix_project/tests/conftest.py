# tests/conftest.py
import pytest
from unittest.mock import MagicMock
import asyncio

# 修复 (第 3 阶段): 此文件没有引用 MarketEvent，因此无需更改。

# Fixtures for AI components
@pytest.fixture
def mock_embedding_client():
    """Mocks the EmbeddingClient."""
    # 修复：将 'ai.embedding_client' 转换为 'Phoenix_project.ai.embedding_client'
    from Phoenix_project.ai.embedding_client import EmbeddingClient
    mock = MagicMock(spec=EmbeddingClient)
    # 模拟 get_embeddings 的异步行为
    # mock.get_embeddings.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
    
    # [任务 3 修复] 确保 mock 的方法是 async
    async def _mock_get_embeddings(texts):
        await asyncio.sleep(0) # 模拟 async
        return [[0.1] * 768 for _ in texts]
    
    mock.get_embeddings = _mock_get_embeddings
    return mock

# ... (other fixtures)

# Fixtures for core components
@pytest.fixture
def sample_pipeline_state():
    """Provides a default PipelineState."""
    # 修复：将 'core.pipeline_state' 转换为 'Phoenix_project.core.pipeline_state'
    from Phoenix_project.core.pipeline_state import PipelineState
    # [任务 3 修复] 传入一个最小的 initial_state 以匹配 engine.py 的期望
    return PipelineState(initial_state={"main_task_query": {"description": "Test"}}, max_history=10)

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
