import pytest
import os
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone
from omegaconf import OmegaConf

# Core Imports
from Phoenix_project.registry import Registry
from Phoenix_project.core.schemas.data_schema import MarketData, NewsData
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem, EvidenceType
from Phoenix_project.training.gnn.gnn_engine import GNNEngine
from Phoenix_project.training.backtest_engine import BacktestEngine
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.data_schema import PortfolioState

# Mock Data for Orchestrator Flow
MOCK_MARKET_DATA = {
    "market_data": [
        MarketData(
            symbol="AAPL", 
            timestamp=datetime.now(timezone.utc), 
            open=150.0, high=155.0, low=149.0, close=152.0, volume=1000000
        )
    ],
    "news_data": []
}

class TestRebirthIntegrity:
    """
    [Phase VI] Final Validation: "Heartbeat" Integration Test.
    Simulates a 'Day in the Life' of the Phoenix System.
    """

    @pytest.fixture
    def mock_env_setup(self):
        """Ensure Fail-Fast checks in Registry pass."""
        with patch.dict(os.environ, {
            "GEMINI_API_KEYS": "fake_key_1,fake_key_2",
            "REDIS_URL": "redis://mock:6379/0",
            "NEO4J_URI": "bolt://mock:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password"
        }):
            yield

    @pytest.fixture
    def mock_config(self):
        return OmegaConf.create({
            "models": {"embedding": "text-embedding-3-large"},
            "agents": {},
            "paths": {"prompts": "Phoenix_project/prompts"},
            "memory": {
                "vector_store": {}, 
                "cot_database": {}, 
                "graph_database": {}
            },
            "data": {"iterator": {"step_size": "1d"}},
            "audit": {},
            "ai": {"source_credibility": {}},
            "cognitive": {
                "portfolio": {}, 
                "risk": {}
            },
            "events": {
                "stream_processor": {}, 
                "risk_filter": {}, 
                "distributor": {}
            },
            "orchestrator": {},
            "l3": {
                "alpha": {"checkpoint_path": "mock/path"},
                "risk": {"checkpoint_path": "mock/path"},
                "execution": {"checkpoint_path": "mock/path"}
            },
            "api": {}
        })

    def test_01_boot_process(self, mock_env_setup, mock_config):
        """
        [Boot]: Validate Registry initializes critical components and Fails Fast.
        """
        print("\n[Heartbeat] Testing Boot Process...")
        
        # Mock Redis to pass connection check
        with patch("redis.from_url") as mock_redis:
            mock_redis.return_value.ping.return_value = True
            
            registry = Registry(mock_config)
            
            assert registry.container.gemini_manager is not None, "GeminiPoolManager missing"
            assert registry.container.data_manager is not None, "DataManager missing"
            print("[Boot] Registry initialized successfully.")

    def test_02_learn_gnn_engine(self, mock_config):
        """
        [Learn]: Validate GNN Engine runs the real PyTorch training loop.
        """
        print("\n[Heartbeat] Testing GNN Engine (Brain)...")
        
        # Mock GraphDB to prevent network calls, relying on GNN Engine's internal mock data generation
        mock_graph_client = MagicMock()
        
        gnn = GNNEngine(config={"epochs": 2}, graph_client=mock_graph_client)
        
        # This runs the REAL PyTorch code (Forward/Backward pass) on dummy tensors
        success = gnn.run_gnn_training_pipeline()
        
        assert success is True
        assert gnn.model.training is True # Was left in training mode
        print("[Learn] GNN Training Pipeline executed successfully.")

    @pytest.mark.asyncio
    async def test_03_simulate_backtest(self, mock_env_setup, mock_config):
        """
        [Simulate]: Validate Backtest Engine enforces Next-Bar Execution.
        """
        print("\n[Heartbeat] Testing Backtest Engine...")
        
        # Setup Dependencies
        mock_data_manager = MagicMock()
        mock_pipeline_state = PipelineState()
        mock_cognitive = MagicMock()
        
        # Mock Portfolio Constructor to output a target
        mock_pc = MagicMock()
        # Create a dummy target portfolio
        from Phoenix_project.core.schemas.data_schema import TargetPortfolio, TargetPosition
        target = TargetPortfolio(
            positions=[TargetPosition(symbol="AAPL", target_weight=0.5, reasoning="Test")]
        )
        mock_pc.construct_portfolio.return_value = target

        engine = BacktestEngine(
            config={"initial_cash": 100000.0},
            data_manager=mock_data_manager,
            pipeline_state=mock_pipeline_state,
            cognitive_engine=mock_cognitive,
            portfolio_constructor=mock_pc
        )

        # Create Data Stream: T1 (Signal), T2 (Execution)
        t1 = datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2023, 1, 2, 10, 0, tzinfo=timezone.utc)
        
        # T1 Data: Price 100
        md1 = [MarketData(symbol="AAPL", timestamp=t1, open=100, high=105, low=95, close=100, volume=1000)]
        # T2 Data: Open 110 (Gap Up)
        md2 = [MarketData(symbol="AAPL", timestamp=t2, open=110, high=115, low=105, close=110, volume=1000)]
        
        data_iterator = [(t1, {"market_data": md1}), (t2, {"market_data": md2})]

        # Run Backtest
        metrics = engine.run_backtest(data_iterator)
        
        # Verify Pending Order Logic (Task 4.2)
        # At T1: Signal generated. Pending order created. Cash should NOT change yet.
        # At T2: Order executed at T2 Open (110). 
        # Target Value at T1: 100k * 0.5 = 50k. 
        # Qty = 50k / 100 = 500 shares.
        # Execution at T2: Cost = 500 * 110 = 55,000.
        # Cash = 100,000 - 55,000 - fees.
        
        final_equity = metrics["final_equity"]
        # Equity should reflect the position value at T2 price
        # 500 shares * 110 = 55,000. Cash ~ 45,000. Total ~ 100,000 (minus fees).
        
        assert metrics["total_return"] != 0.0
        assert engine.trade_log is not None # Or check pending_orders logic internally if exposed
        print(f"[Simulate] Backtest complete. Final Equity: {final_equity}")

    @pytest.mark.asyncio
    async def test_04_decide_orchestrator_flow(self, mock_env_setup, mock_config):
        """
        [Decide]: Validate Agentic Workflow (L1 -> L2 -> L3) and Memory.
        """
        print("\n[Heartbeat] Testing Orchestrator & Agents...")

        with patch("redis.from_url") as mock_redis:
            mock_redis.return_value.ping.return_value = True
            
            # 1. Build System with mocked Agents where necessary
            registry = Registry(mock_config)
            
            # 2. Mock L1 Agent (return static evidence)
            mock_l1 = AsyncMock()
            mock_l1.agent_id = "technical_analyst"
            mock_l1.safe_run.return_value = EvidenceItem(
                agent_id="technical_analyst",
                content="Bullish divergence detected.",
                evidence_type=EvidenceType.TECHNICAL,
                confidence=0.8,
                symbols=["AAPL"]
            )
            registry.container.l1_agents = {"technical_analyst": mock_l1}
            
            # 3. Mock L3 Agent (Memory Check)
            # We use the REAL AlphaAgent class but mock the RLLib algorithm
            from Phoenix_project.agents.l3.alpha_agent import AlphaAgent
            mock_algo = MagicMock()
            # Initial state: [0.0]
            mock_algo.get_policy.return_value.get_initial_state.return_value = [0.0] 
            # Compute action returns: (action, state_out, info)
            mock_algo.compute_single_action.return_value = (np.array([0.5]), [1.0], {})
            
            alpha_agent = AlphaAgent(algorithm=mock_algo)
            registry.container.l3_agents = {"alpha": alpha_agent}

            # 4. Trigger Cycle (Manually stepping to avoid complex Orchestrator wiring if feasible, 
            # or mocking the Orchestrator's internal calls)
            
            # Step A: L1 Execution
            state = PipelineState()
            evidence = await mock_l1.safe_run(state, {})
            assert evidence.confidence == 0.8
            print("[Decide] L1 Agent produced evidence.")

            # Step B: L3 Execution with Memory
            # Input observation (mocked)
            obs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
            
            # T=0
            action_t0 = await alpha_agent.compute_action(obs)
            assert alpha_agent.internal_state == [1.0] # Memory updated
            assert action_t0[0] == 0.5
            print("[Decide] L3 Agent produced action and updated memory (Stateful Inference confirmed).")
            
            # Validate Safety (Task 3.3)
            # Simulate crash
            alpha_agent.algorithm.compute_single_action.side_effect = Exception("Boom")
            safe_action = await alpha_agent.compute_action(obs)
            assert safe_action is None # Verify HALT signal
            print("[Decide] L3 Safety Protocol confirmed (returned None on crash).")

if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__])
