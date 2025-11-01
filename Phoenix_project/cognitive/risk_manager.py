import numpy as np
import pandas as pd
from typing import Dict, Any

from ..core.pipeline_state import PipelineState
from ..data_manager import DataManager

class RiskManager:
    """
    Dynamically adjusts capital allocation based on AI uncertainty and market volatility.
    It provides a "capital modifier" (0.0 to 1.0) that scales the
    portfolio's overall exposure.
    """

    def __init__(self, config, data_manager: DataManager):
        """
        Initializes the RiskManager.
        
        Args:
            config: The main strategy configuration object.
            data_manager: Client to access historical market data.
        """
        self.config = config
        self.risk_config = config.get('risk_manager', {})
        self.data_manager = data_manager
        
        # Uncertainty scaling parameters
        self.uncertainty_sensitivity = self.risk_config.get('uncertainty_sensitivity', 1.5)
        self.min_capital_modifier = self.risk_config.get('min_capital_modifier', 0.2)
        
        # Volatility target parameters
        self.use_vol_target = self.risk_config.get('use_volatility_target', True)
        self.target_annual_vol = self.risk_config.get('target_annual_volatility', 0.15)
        self.vol_lookback_days = self.risk_config.get('volatility_lookback_days', 60)
        self.portfolio_index_ticker = self.risk_config.get('portfolio_index_ticker', 'SPY')

        
    def calculate_capital_modifier(
        self, 
        cognitive_uncertainty: float, 
        state: PipelineState
    ) -> float:
        """
        Calculates the final capital modifier by combining uncertainty and volatility.

        Args:
            cognitive_uncertainty (float): The uncertainty score from the AI (0.0 to 1.0).
            state: The current PipelineState.

        Returns:
            float: The capital modifier (e.g., 0.75), bounded between
                   min_capital_modifier and 1.0.
        """
        
        # 1. Calculate modifier from AI Uncertainty
        # We use an exponential decay function.
        # High uncertainty -> low modifier
        # uncertainty = 0.0 -> modifier = 1.0
        # uncertainty = 1.0 -> modifier = e^(-sensitivity)
        uncertainty_modifier = np.exp(-self.uncertainty_sensitivity * cognitive_uncertainty)
        
        # 2. Calculate modifier from Market Volatility
        if self.use_vol_target:
            vol_modifier = self._calculate_volatility_modifier(state)
        else:
            vol_modifier = 1.0
            
        # 3. Combine modifiers (e.g., by taking the minimum)
        # This ensures we always respect the most conservative constraint.
        combined_modifier = min(uncertainty_modifier, vol_modifier)
        
        # 4. Enforce floor
        final_modifier = max(self.min_capital_modifier, combined_modifier)
        
        return final_modifier

    def _calculate_volatility_modifier(self, state: PipelineState) -> float:
        """
        Calculates a modifier based on a portfolio-level volatility target.
        Modifier = Target Vol / Realized Vol
        If Realized Vol is high, modifier is < 1.0
        If Realized Vol is low, modifier is > 1.0 (capped at 1.0)
        """
        try:
            # Get historical data for the portfolio index (e.g., SPY)
            end_date = state.timestamp
            start_date = end_date - pd.Timedelta(days=self.vol_lookback_days * 2) # Get extra data for rolling calc
            
            index_data = self.data_manager.get_historical_data(
                self.portfolio_index_ticker,
                start_date,
                end_date
            )
            
            if index_data.empty or len(index_data) < self.vol_lookback_days:
                # Not enough data, return neutral modifier
                return 1.0

            # Calculate realized volatility
            returns = index_data['close'].pct_change()
            # Annualized std dev of returns
            realized_vol = returns.rolling(window=self.vol_lookback_days).std().iloc[-1] * np.sqrt(252)
            
            if realized_vol == 0:
                return 1.0 # Avoid division by zero
                
            # Calculate modifier
            vol_modifier = self.target_annual_vol / realized_vol
            
            # Cap the modifier at 1.0 (we don't "lever up" if vol is low)
            return min(vol_modifier, 1.0)
            
        except Exception as e:
            # In case of data error, fail safe
            print(f"[RiskManager] Error calculating volatility modifier: {e}")
            return 1.0 # Return neutral modifier
