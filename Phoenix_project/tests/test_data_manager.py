"""
测试 DataManager (已更新)
"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime

from data_manager import DataManager
from config.loader import ConfigLoader

# FIX (E10): 重写测试以使用模拟的 ConfigLoader

@pytest.fixture
def mock_config_loader() -> ConfigLoader:
    """
    模拟一个 ConfigLoader。
    """
    loader = MagicMock(spec=ConfigLoader)
    # (E5 Fix) 模拟 DataManager 所需的配置
    loader.get_system_config.return_value = {
        "data_store": {
            "local_base_path": "/test/data"
        }
    }
    return loader

@pytest.fixture
def sample_data_catalog() -> dict:
    """
    返回一个示例数据目录。
    """
    return {
        "market_data_AAPL": {
            "path": "market/aapl.parquet",
            "format": "parquet",
            "timestamp_col": "timestamp"
        },
        "news_events": {
            "path": "news/events.csv",
            "format": "csv",
            "timestamp_col": "date"
        }
    }

@pytest.fixture
def data_manager(mock_config_loader: ConfigLoader, sample_data_catalog: dict) -> DataManager:
    """
    返回一个使用模拟依赖项的 DataManager 实例。
    """
    # (E5 Fix) 使用正确的构造函数
    return DataManager(config_loader=mock_config_loader, data_catalog=sample_data_catalog)

# --- 模拟 Pandas 读取 ---
@pytest.fixture
def mock_parquet_data() -> pd.DataFrame:
    """
    模拟从 Parquet 文件读取的 DataFrame。
    """
    return pd.DataFrame({
        'open': [150], 'high': [151], 'low': [149], 'close': [150.5], 'volume': [1000]
    }, index=[pd.to_datetime("2023-01-01 10:00:00", utc=True)])

@pytest.fixture
def mock_csv_data() -> pd.DataFrame:
    """
    模拟从 CSV 文件读取的 DataFrame。
    """
    return pd.DataFrame({
        'id': ['news1'], 'source': ['Reuters'], 'content': ['Test'], 'date': [pd.to_datetime("2023-01-01 09:00:00", utc=True)]
    })

# 使用 patch 来模拟 os.path.exists 和 pandas I/O
@patch('os.path.exists', return_value=True)
@patch('pandas.read_parquet')
def test_load_market_data(mock_read_parquet, mock_exists, data_manager: DataManager, mock_parquet_data: pd.DataFrame):
    """
    测试 get_market_data 是否正确加载、缓存和过滤数据。
    """
    mock_read_parquet.return_value = mock_parquet_data
    
    start_date = datetime(2023, 1, 1, 0, 0, 0)
    end_date = datetime(2023, 1, 2, 0, 0, 0)
    
    # 1. 第一次调用（加载）
    dfs = data_manager.get_market_data(["AAPL"], start_date, end_date)
    
    assert "AAPL" in dfs
    assert len(dfs["AAPL"]) == 1
    assert dfs["AAPL"].iloc[0]['close'] == 150.5
    
    # 验证 pd.read_parquet 被调用
    mock_read_parquet.assert_called_once_with("/test/data/market/aapl.parquet")
    
    # 2. 第二次调用（从缓存）
    mock_read_parquet.reset_mock()
    dfs_cached = data_manager.get_market_data(["AAPL"], start_date, end_date)
    
    # 验证 pd.read_parquet *没有* 被再次调用
    mock_read_parquet.assert_not_called()
    assert len(dfs_cached["AAPL"]) == 1

@patch('os.path.exists', return_value=True)
@patch('pandas.read_csv')
def test_fetch_data_for_batch(mock_read_csv, mock_exists, data_manager: DataManager, mock_csv_data: pd.DataFrame):
    """
    测试 fetch_data_for_batch 是否正确转换数据为 Pydantic 模式。
    (我们只测试 NewsData，因为 MarketData 在上一个测试中已覆盖)
    """
    mock_read_csv.return_value = mock_csv_data
    
    start_date = datetime(2023, 1, 1, 0, 0, 0)
    end_date = datetime(2023, 1, 2, 0, 0, 0)
    
    # (模拟 DataManager，使其不加载 MarketData)
    data_manager.data_catalog = {"news_events": data_manager.data_catalog["news_events"]}
    
    batch = data_manager.fetch_data_for_batch(start_date, end_date, [])
    
    assert len(batch["market_data"]) == 0
    assert len(batch["news_data"]) == 1
    
    news_item = batch["news_data"][0]
    # (E1 Fix) 验证它是否是 NewsData
    assert news_item.id == "news1"
    assert news_item.source == "Reuters"
