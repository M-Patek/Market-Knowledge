import pytest

# 修正：[FIX-ImportError]
# 将所有 `..` 相对导入更改为从项目根目录开始的绝对导入，
# 以匹配 `conftest.py` 设置的 sys.path 约定。
from cognitive.portfolio_constructor import PortfolioConstructor, Portfolio
from cognitive.risk_manager import RiskManager
from sizing.base import BaseSizer
from core.pipeline_state import PipelineState
from core.schemas.fusion_result import FusionResult, FinalDecision

# --- Mock Classes ---

class MockSizer(BaseSizer):
    """Mock sizer that returns a fixed size."""
    def calculate_size(self, state, symbol, decision, uncertainty):
        if symbol == "AAPL":
            return 0.5 # Allocate 50%
        return 0.0

@pytest.fixture
def risk_manager():
    """Fixture for a simple RiskManager."""
    config = {"risk_manager": {"min_capital_modifier": 0.1}}
    return RiskManager(config)

@pytest.fixture
def sizer():
    """Fixture for the mock sizer."""
    return MockSizer()

@pytest.fixture
def portfolio_constructor(risk_manager, sizer):
    """Fixture for the PortfolioConstructor."""
    return PortfolioConstructor(risk_manager=risk_manager, position_sizer=sizer)

@pytest.fixture
def state():
    """Fixture for a basic PipelineState."""
    s = PipelineState()
    s.update_portfolio(total_value=100000.0, cash=100000.0, positions={})
    return s

@pytest.fixture
def fusion_result():
    """Fixture for a basic FusionResult."""
    return FusionResult(
        event_id="test_event",
        final_decision=FinalDecision(
            decision="BUY",
            confidence=0.8,
            analysis="Mock analysis"
        ),
        cognitive_uncertainty=0.2, # Low uncertainty
        supporting_evidence=[]
    )

# --- Tests ---

def test_portfolio_constructor_init(portfolio_constructor):
    """Tests initialization."""
    assert portfolio_constructor is not None
    assert isinstance(portfolio_constructor.risk_manager, RiskManager)
    assert isinstance(portfolio_constructor.position_sizer, MockSizer)

def test_generate_portfolio_low_uncertainty(
    portfolio_constructor, 
    state, 
    fusion_result
):
    """
    Tests portfolio generation with LOW uncertainty (0.2).
    RiskManager should apply a high capital modifier (e.g., ~1.0).
    Sizer allocates 50% to AAPL.
    Final weight = 50% * 1.0 = 50%
    """
    
    # We need to tell the sizer which symbol the decision is for
    fusion_result.metadata = {"target_symbol": "AAPL"}
    fusion_result.cognitive_uncertainty = 0.2
    
    portfolio = portfolio_constructor.generate_optimized_portfolio(
        state, 
        fusion_result
    )
    
    assert portfolio is not None
    assert isinstance(portfolio, Portfolio)
    
    # Check capital modifier (low uncertainty -> high modifier)
    # With uncertainty=0.2, modifier = 1.0 - (0.2 / 1.0) * (1.0 - 0.1) = 1.0 - 0.18 = 0.82
    expected_modifier = 0.82 
    assert portfolio.metadata["capital_modifier"] == pytest.approx(expected_modifier)
    
    # Check weights
    # Base size = 0.5 (from MockSizer)
    # Final weight = 0.5 * 0.82 = 0.41
    assert portfolio.weights["AAPL"] == pytest.approx(0.5 * expected_modifier)
    assert portfolio.weights["CASH"] == pytest.approx(1.0 - (0.5 * expected_modifier))

def test_generate_portfolio_high_uncertainty(
    portfolio_constructor, 
    state, 
    fusion_result
):
    """
    Tests portfolio generation with HIGH uncertainty (0.9).
    RiskManager should apply a low capital modifier (e.g., ~0.1).
    Sizer allocates 50% to AAPL.
    Final weight = 50% * 0.1 = 5%
    """
    
    fusion_result.metadata = {"target_symbol": "AAPL"}
    fusion_result.cognitive_uncertainty = 0.9
    
    portfolio = portfolio_constructor.generate_optimized_portfolio(
        state, 
        fusion_result
    )
    
    # Check capital modifier (high uncertainty -> low modifier)
    # With uncertainty=0.9, modifier = 1.0 - (0.9 / 1.0) * (1.0 - 0.1) = 1.0 - 0.81 = 0.19
    expected_modifier = 0.19
    assert portfolio.metadata["capital_modifier"] == pytest.approx(expected_modifier)
    
    # Check weights
    # Base size = 0.5 (from MockSizer)
    # Final weight = 0.5 * 0.19 = 0.095
    assert portfolio.weights["AAPL"] == pytest.approx(0.5 * expected_modifier)
    assert portfolio.weights["CASH"] == pytest.approx(1.0 - (0.5 * expected_modifier))

def test_generate_portfolio_no_target_symbol(
    portfolio_constructor, 
    state, 
    fusion_result
):
    """
    Tests that if the FusionResult has no 'target_symbol',
    the portfolio remains 100% cash.
    """
    # fusion_result.metadata["target_symbol"] is NOT set
    
    portfolio = portfolio_constructor.generate_optimized_portfolio(
        state, 
        fusion_result
    )
    
    assert portfolio.weights == {"CASH": 1.0}
    assert portfolio.metadata["reason"] == "No target_symbol in FusionResult metadata."

def test_generate_portfolio_hold_decision(
    portfolio_constructor, 
    state, 
    fusion_result
):
    """
    Tests that a "HOLD" or "NEUTRAL" decision results in 100% cash
    (i.e., no new position is taken).
    """
    fusion_result.metadata = {"target_symbol": "AAPL"}
    fusion_result.final_decision.decision = "HOLD"
    
    portfolio = portfolio_constructor.generate_optimized_portfolio(
        state, 
        fusion_result
    )
    
    assert portfolio.weights == {"CASH": 1.0}
    assert portfolio.metadata["reason"] == "Decision is 'HOLD', taking no action."
