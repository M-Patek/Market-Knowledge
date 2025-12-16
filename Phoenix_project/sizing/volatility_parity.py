import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class VolatilityParitySizer:
    """
    Sizing strategy based on Volatility Parity (Equal Risk Contribution approximation).
    Allocates weights proportional to inverse volatility: w_i ~ 1/sigma_i.
    Preserves the direction (Buy/Sell) from the input signal.
    [Task P2-BASE-02] Robust Data Handling & Inverse Volatility Weighting.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Default volatility if data is missing (e.g., 2% daily)
        self.default_volatility = self.config.get("default_volatility", 0.02)

    def size_positions(
        self, 
        candidates: List[Dict[str, Any]], 
        max_total_allocation: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Calculates position sizes ensuring risk parity among candidates.
        
        Args:
            candidates: List of dicts with 'ticker', 'weight' (signal), and optional 'volatility'.
            max_total_allocation: Total leverage allowed (default 1.0).
            
        Returns:
            List of candidates with updated 'capital_allocation_pct'.
        """
        if not candidates:
            return []

        # [Task P2-BASE-02] Data Integrity Check & Preparation
        processed_candidates = []
        inv_vol_sum = 0.0
        
        for cand in candidates:
            symbol = cand.get("ticker")
            signal_weight = cand.get("weight", 0.0)
            
            # Skip zero-weight signals (no action)
            if abs(signal_weight) < 1e-6:
                continue

            # 1. Use injected volatility or fallback
            vol = cand.get("volatility")
            
            if vol is None or not isinstance(vol, (int, float)) or vol <= 0:
                # Log warning only if it's a significant active position
                logger.warning(
                    f"Volatility data missing/invalid for {symbol} (val={vol}). "
                    f"Using default: {self.default_volatility:.2%}"
                )
                vol = self.default_volatility
            
            # 2. Calculate Inverse Volatility Score
            # We use 1/vol as the raw score for allocation magnitude
            inv_vol = 1.0 / vol
            inv_vol_sum += inv_vol
            
            # Preserve signal direction
            direction = 1.0 if signal_weight > 0 else -1.0
            
            processed_candidates.append({
                "original_candidate": cand,
                "direction": direction,
                "inv_vol": inv_vol
            })

        if inv_vol_sum == 0:
            logger.warning("No valid candidates for Volatility Parity sizing.")
            return []

        # 3. Normalize Weights
        results = []
        for item in processed_candidates:
            # Weight = (1/vol) / sum(1/vol) * Total_Alloc * Direction
            raw_alloc = (item["inv_vol"] / inv_vol_sum) * max_total_allocation
            final_alloc = raw_alloc * item["direction"]
            
            cand = item["original_candidate"]
            cand["capital_allocation_pct"] = float(final_alloc)
            
            # Store metadata for debugging
            cand["meta_sizing_method"] = "volatility_parity"
            cand["meta_volatility_used"] = 1.0 / item["inv_vol"]
            
            results.append(cand)
            
        return results
