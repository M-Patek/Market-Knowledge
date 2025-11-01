from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# 此 schema 基于 tests/test_data_manager.py 中的模拟 config
# 它是修复该测试文件所必需的。

class DataSourceProvider(BaseModel):
    api_key_env_var: str

class DataSourcesNetwork(BaseModel):
    user_agent: str
    proxy: Dict[str, Any]
    request_timeout: int
    retry_attempts: int
    retry_backoff_factor: int

class DataSourcesHealthProbes(BaseModel):
    failure_threshold: int
    cooldown_minutes: int

class DataSources(BaseModel):
    priority: List[str]
    providers: Dict[str, DataSourceProvider]
    network: DataSourcesNetwork
    health_probes: DataSourcesHealthProbes

class AIEnsembleConfig(BaseModel):
    enable: bool
    config_file_path: str
    
class ExecutionModel(BaseModel):
    impact_coefficient: float
    max_volume_share: float
    min_trade_notional: int
    
class PositionSizer(BaseModel):
    method: str
    parameters: Dict[str, Any]
    
class Optimizer(BaseModel):
    study_name: str
    n_trials: int
    parameters: Dict[str, Any]
    
class Observability(BaseModel):
    metrics_port: int
    
class Audit(BaseModel):
    s3_bucket_name: str

class StrategyConfig(BaseModel):
    """
    用于验证 test_data_manager.py 中模拟配置的 Pydantic 模型。
    """
    start_date: str
    end_date: str
    asset_universe: List[str]
    market_breadth_tickers: List[str]
    log_level: str
    data_sources: DataSources
    sma_period: int
    rsi_period: int
    rsi_overbought_threshold: int
    opportunity_score_threshold: int
    vix_high_threshold: int
    vix_low_threshold: int
    capital_modifier_high_vix: float
    capital_modifier_normal_vix: float
    capital_modifier_low_vix: float
    initial_cash: int
    commission_rate: float
    max_total_allocation: float
    ai_ensemble_config: AIEnsembleConfig
    ai_mode: str
    walk_forward: Dict[str, Any]
    execution_model: ExecutionModel
    position_sizer: PositionSizer
    optimizer: Optimizer
    observability: Observability
    audit: Audit
    data_manager: Optional[Dict[str, Any]] = None # 为 data_manager.cache_dir 添加

