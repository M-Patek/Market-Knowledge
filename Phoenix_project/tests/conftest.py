# tests/conftest.py
import pytest
from datetime import date
from phoenix_project import StrategyConfig

@pytest.fixture
def base_config() -> StrategyConfig:
    """Provides a base, valid StrategyConfig for testing."""
    config_dict = {
        "start_date": date(2022, 1, 1),
        "end_date": date(2023, 1, 1),
        "asset_universe": ["SPY"],
        "market_breadth_tickers": ["SPY"],
        "sma_period": 1,
        "rsi_period": 2,
        "rsi_overbought_threshold": 70.0,
        "opportunity_score_threshold": 55.0,
        "vix_high_threshold": 30.0,
        "vix_low_threshold": 20.0,
        "capital_modifier_high_vix": 0.5,
        "capital_modifier_normal_vix": 0.9,
        "capital_modifier_low_vix": 1.0,
        "initial_cash": 100000,
        "commission_rate": 0.001,
        "log_level": "INFO",
        "ai_ensemble_config": {"enable": False, "config_file_path": "fake"},
        "data_sources": {
            "priority": [], "providers": {},
            "network": {"retry_attempts": 1, "retry_backoff_factor": 1, "request_timeout": 5, "user_agent": "test", "proxy": {"enabled": False}},
            "health_probes": {"failure_threshold": 1, "cooldown_minutes": 1}
        },
        "ai_mode": "off",
        "walk_forward": {},
        "max_total_allocation": 1.0,
        "execution_model": {"impact_coefficient": 0.1, "max_volume_share": 0.25, "min_trade_notional": 100.0},
        "position_sizer": {"method": "fixed_fraction", "parameters": {"fraction_per_position": 0.1}},
        "optimizer": {"study_name": "test", "n_trials": 1, "parameters": {}},
        "observability": {"metrics_port": 8001},
        "audit": {"s3_bucket_name": "none"}
    }
    return StrategyConfig(**config_dict)
