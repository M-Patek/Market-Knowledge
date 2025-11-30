import unittest
import numpy as np
import asyncio
from unittest.mock import MagicMock, patch

# Import System Components
from Phoenix_project.agents.l3.risk_agent import RiskAgent
from Phoenix_project.agents.l3.alpha_agent import AlphaAgent
from Phoenix_project.agents.l3.execution_agent import ExecutionAgent
from Phoenix_project.core.schemas.fusion_result import FusionResult

# Mocking Ray RLLib to run tests without full RL environment
# We create a dummy class to satisfy the type check in BaseDRLAgent
class MockAlgorithm:
    def get_policy(self):
        policy = MagicMock()
        policy.get_initial_state.return_value = []
        return policy
    
    def compute_single_action(self, *args, **kwargs):
        # Default behavior: return random action or zero
        return np.array([0.5], dtype=np.float32)

# Patching the import in the base module to allow instantiation with MockAlgorithm
with patch('ray.rllib.algorithms.algorithm.Algorithm', MockAlgorithm):
    # Re-import to apply patch if modules were already loaded (safeguard)
    pass

class TestSafetyMechanisms(unittest.TestCase):
    """
    [Task 4.2] Global Circuit Breaker & Safety Tests.
    Verifies that L3 agents handle edge cases and override signals correctly.
    """

    def setUp(self):
        # Inject the MockAlgorithm which mimics RLLib
        # Note: We patch 'isinstance' check implicitly by ensuring MockAlgorithm matches expected spec if needed, 
        # but since we can't easily patch built-in isinstance, we assume the user's environment 
        # allows us to mock the class definition or we rely on duck typing if Python allowed.
        # Given the strict type check in BaseDRLAgent, we rely on `unittest.mock.patch` on the class itself
        # during the test method execution.
        self.mock_algo = MockAlgorithm()

    def test_risk_hard_stop(self):
        """
        [Safety] Verify RiskAgent bypasses NN when L2 triggers HALT.
        """
        # We patch the base class import of Algorithm to allow our Mock
        with patch('Phoenix_project.agents.l3.base.Algorithm', MockAlgorithm):
            agent = RiskAgent(algorithm=self.mock_algo)
            
            # 1. Create Normal Observation
            obs = np.zeros(7, dtype=np.float32)
            
            # 2. Create Mock FusionResult with HALT signal
            # Using MagicMock to avoid Pydantic validation complexity in test
            mock_fusion = MagicMock(spec=FusionResult)
            mock_fusion.decision = "HALT"
            mock_fusion.confidence = 0.9
            
            # 3. Execute Action (Async)
            loop = asyncio.new_event_loop()
            try:
                action = loop.run_until_complete(
                    agent.compute_action(obs, fusion_result=mock_fusion)
                )
            finally:
                loop.close()
            
            # 4. Assert Hard Stop
            print(f"Risk Action: {action}")
            # Should be exactly [1.0]
            np.testing.assert_array_equal(
                action, 
                np.array([1.0], dtype=np.float32), 
                "RiskAgent did not return Hard Halt [1.0] on 'HALT' signal."
            )

    def test_alpha_math_safety(self):
        """
        [Safety] Verify AlphaAgent sanitizes bad input (NaN/Inf protection).
        """
        with patch('Phoenix_project.agents.l3.base.Algorithm', MockAlgorithm):
            agent = AlphaAgent(algorithm=self.mock_algo)
            
            # 1. Corrupted Data Injection
            bad_state_data = {
                "balance": 1000.0,
                "initial_balance": 1000.0,
                "position_weight": 0.5,
                "price": 0.0,          # Singular Point
                "prev_price": 0.0,     # Singular Point
                "volume": -100.0       # Physical Impossibility
            }
            
            # 2. Format Observation
            obs = agent._format_obs(bad_state_data, None)
            
            print(f"Alpha Obs (Sanitized): {obs}")
            
            # 3. Assertions
            # Check for NaN / Inf
            self.assertFalse(np.any(np.isnan(obs)), "Observation contains NaN (Math Shield Failed)")
            self.assertFalse(np.any(np.isinf(obs)), "Observation contains Inf (Math Shield Failed)")
            
            # Check Logic: LogReturn (idx 2) and LogVolume (idx 3) should be 0.0
            self.assertEqual(obs[2], 0.0, "LogReturn not zeroed for price=0.0")
            self.assertEqual(obs[3], 0.0, "LogVolume not zeroed for negative volume")

    def test_execution_pass_through(self):
        """
        [Safety] Verify ExecutionAgent is dormant (Pass-Through).
        """
        with patch('Phoenix_project.agents.l3.base.Algorithm', MockAlgorithm):
            agent = ExecutionAgent(algorithm=self.mock_algo)
            obs = np.zeros(9, dtype=np.float32)
            
            loop = asyncio.new_event_loop()
            try:
                action = loop.run_until_complete(agent.compute_action(obs))
            finally:
                loop.close()
            
            print(f"Execution Action: {action}")
            # Should be neutral [0.0]
            np.testing.assert_array_equal(
                action, 
                np.array([0.0], dtype=np.float32), 
                "ExecutionAgent is not passive."
            )

if __name__ == '__main__':
    unittest.main()
