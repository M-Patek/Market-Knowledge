# tests/test_portfolio_constructor.py
# 修复：[FIX-14] 整个文件被注释掉。
# 'cognitive/portfolio_constructor.py' 已被重构。
# 此测试文件正在测试不再存在的旧方法 ('calculate_opportunity_score')。
# 它需要被重写以测试 'generate_optimized_portfolio'。

"""
import pytest
from datetime import date
from cognitive.portfolio_constructor import PortfolioConstructor
from core.schemas.config_schema import StrategyConfig

@pytest.mark.parametrize(
    "current_price, current_sma, expected_score",
    [
        (110.0, 100.0, 55.0),
        (90.0, 100.0, 45.0),
        (100.0, 100.0, 50.0),
        (300.0, 100.0, 100.0),
        (0.0, 100.0, 0.0),
        (100.0, 0.0, 0.0),
    ]
)
def test_calculate_opportunity_score(current_price, current_sma, expected_score, base_config):
    
    # 错误：'PortfolioConstructor' 的 __init__ 需要 
    # data_manager, sizer, 和 risk_manager
    constructor = PortfolioConstructor(config=base_config)
    
    # 错误：'calculate_opportunity_score' 方法不存在
    score = constructor.calculate_opportunity_score(current_price, current_sma)
    assert score == pytest.approx(expected_score)

@pytest.mark.parametrize(
    "current_price, current_sma, current_rsi, expected_score",
    [
        (110.0, 100.0, 70.0, 55.0),
        (110.0, 100.0, 85.0, 41.25),
        (110.0, 100.0, 100.0, 27.5),
        (95.0, 100.0, 90.0, 47.5),
    ]
)
def test_calculate_opportunity_score_with_rsi_penalty(base_config, current_price, current_sma, current_rsi, expected_score):
    
    constructor = PortfolioConstructor(config=base_config)
    score = constructor.calculate_opportunity_score(current_price, current_sma, current_rsi)
    assert score == pytest.approx(expected_score)

"""
