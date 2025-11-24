"""
Phoenix Project - Critical Fix Validation Script
Run this script to verify Phase 0-3 fixes:
1. TLM "Anti-Money" PnL Logic
2. Risk Manager Annualized Volatility
3. Order Manager State Machine Integrity
"""
import sys
import os
import asyncio
import logging
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Phoenix_project.execution.trade_lifecycle_manager import TradeLifecycleManager
from Phoenix_project.cognitive.risk_manager import RiskManager
from Phoenix_project.execution.order_manager import OrderManager
from Phoenix_project.core.schemas.data_schema import Fill, Position, Order, OrderStatus, MarketData

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Validation")

async def test_tlm_pnl_fix():
    """
    [Task 0.1 Verification]
    Scenario: Short 10 @ 100, Buy 10 @ 90.
    Old Bug: (90 - 100) * 10 = -100 (Loss).
    Expected Fix: (100 - 90) * 10 = +100 (Profit).
    """
    logger.info("--- Testing TLM PnL Fix (Short Cover) ---")
    
    # Setup
    tlm = TradeLifecycleManager(initial_cash=10000.0, tabular_db=None) # DB None for in-memory
    
    # 1. Establish Short Position (Manually inject state to simulate previous fills)
    # Short 10 units @ $100
    symbol = "TEST_ASSET"
    tlm.positions[symbol] = Position(
        symbol=symbol,
        quantity=-10.0,
        average_price=100.0,
        market_value=-1000.0,
        unrealized_pnl=0.0
    )
    tlm.cash = 11000.0 # Received 1000 cash from short sale
    
    # 2. Execute Buy to Cover
    # Buy 10 units @ $90
    fill = Fill(
        id="fill_1",
        order_id="order_1",
        symbol=symbol,
        quantity=10.0, # Positive for Buy
        price=90.0,
        timestamp=datetime.now(),
        commission=0.0
    )
    
    await tlm.on_fill(fill)
    
    # 3. Assertions
    expected_pnl = (100.0 - 90.0) * 10.0 # 100 Profit
    
    if abs(tlm.realized_pnl - expected_pnl) < 1e-6:
        logger.info(f"✅ PASSED: Realized PnL is {tlm.realized_pnl} (Expected {expected_pnl})")
    else:
        logger.error(f"❌ FAILED: Realized PnL is {tlm.realized_pnl} (Expected {expected_pnl})")
        raise AssertionError("TLM PnL Logic Failed")

async def test_risk_volatility_fix():
    """
    [Task 3.3 Verification]
    Scenario: Calculate volatility of a known price sequence.
    Old Bug: Returns raw daily std dev.
    Expected Fix: Returns std dev * sqrt(frequency_factor).
    """
    logger.info("--- Testing Risk Manager Volatility Fix ---")
    
    # Setup
    config = {
        "risk_manager": {
            "volatility_window": 5,
            "frequency_factor": 252 # Daily annualization
        }
    }
    mock_redis = AsyncMock()
    rm = RiskManager(config, mock_redis)
    await rm.initialize([]) # Init state
    
    # Inject Price History directly to control the calculation
    # Sequence of prices: 100, 101, 102, 101, 100
    # Log returns: ln(1.01), ln(1.0099), ln(0.9901), ln(0.9900)
    prices = [100, 101, 102, 101, 100]
    rm.price_history["TEST"] = getattr(rm, "price_history").get("TEST", [])
    # Access deque directly or use internal structure
    from collections import deque
    rm.price_history["TEST"] = deque(prices, maxlen=5)
    
    # Trigger check with a new price (dummy) just to run logic, 
    # but we care about the math on the history.
    # Actually, check_volatility takes MarketData.
    
    # Calculate Expected Manually
    arr = np.array(prices)
    log_returns = np.log(arr[1:] / arr[:-1])
    raw_std = np.std(log_returns)
    expected_vol = raw_std * np.sqrt(252)
    
    # Run Method
    # We pass the *last* price again to trigger the check. 
    # Note: check_volatility appends the price passed to history. 
    # So we should pass the next price in sequence or re-pass the last one.
    # Let's create a new RM instance to be clean.
    rm = RiskManager(config, mock_redis)
    # Feed first 4 prices manually
    rm.price_history["TEST"] = deque(prices[:-1], maxlen=5)
    
    # Feed 5th price via method
    md = MarketData(symbol="TEST", price=prices[-1], timestamp=datetime.now())
    
    # The method returns a signal only if threshold is exceeded. 
    # We can inspect the log or temporarily lower threshold to 0. 
    # Or better, we can invoke the internal logic if accessible, 
    # but let's trust the method return if we set threshold low.
    rm.volatility_threshold = 0.0 # Force trigger
    
    signal = rm.check_volatility(md)
    
    if signal:
        calculated_vol = signal.current_volatility
        logger.info(f"Calculated Volatility: {calculated_vol:.6f}")
        logger.info(f"Expected Volatility:   {expected_vol:.6f}")
        
        if abs(calculated_vol - expected_vol) < 1e-6:
             logger.info("✅ PASSED: Volatility matches annualized formula.")
        else:
             logger.error(f"❌ FAILED: Volatility {calculated_vol} != Expected {expected_vol}")
             raise AssertionError("Risk Volatility Math Failed")
    else:
        # Should not happen with threshold 0.0
        logger.error("❌ FAILED: No volatility signal generated.")

async def test_order_state_fix():
    """
    [Task 3.2 Verification]
    Scenario: Order is FILLED. Update arrives saying PENDING.
    Old Bug: Status changes to PENDING.
    Expected Fix: Update rejected, status remains FILLED.
    """
    logger.info("--- Testing Order State Machine Fix ---")
    
    # Setup
    mock_broker = MagicMock()
    mock_broker.subscribe_fills = MagicMock()
    mock_broker.subscribe_order_status = MagicMock()
    
    mock_tlm = AsyncMock()
    mock_dm = AsyncMock()
    
    om = OrderManager(mock_broker, mock_tlm, mock_dm)
    
    # 1. Create a FILLED order in active_orders (simulating a completed trade)
    order_id = "ord_123"
    filled_order = Order(
        id=order_id,
        symbol="AAPL",
        quantity=10,
        price=150.0,
        status=OrderStatus.FILLED,
        order_type="LIMIT"
    )
    om.active_orders[order_id] = filled_order
    
    # 2. Attempt Invalid Transition: FILLED -> PENDING
    invalid_update = Order(
        id=order_id,
        symbol="AAPL",
        quantity=10,
        price=150.0,
        status=OrderStatus.PENDING, # Zombie update from laggy broker
        order_type="LIMIT"
    )
    
    await om._on_order_status_update(invalid_update)
    
    # 3. Assertions
    # Order should have been removed from active_orders if FILLED (normal logic),
    # BUT if we strictly simulate the state check:
    # The OrderManager deletes FILLED orders from active_orders. 
    # So if it was there, and we sent PENDING, it should NOT update it to PENDING.
    # Let's check if it remains as the original object or is removed.
    
    # Actually, in _on_order_status_update:
    # 1. It fetches current order.
    # 2. Checks _can_transition.
    # 3. If invalid, returns (ignoring update).
    
    # So the order in active_orders should remain the *original* object (FILLED).
    # Wait, the code removes FILLED orders at the end of the method. 
    # So if we manually put it there to test the transition check:
    
    # Re-setup: Put it as PARTIALLY_FILLED first to keep it active.
    om.active_orders.clear()
    partial_order = Order(
        id=order_id,
        symbol="AAPL",
        quantity=10,
        price=150.0,
        status=OrderStatus.PARTIALLY_FILLED,
        order_type="LIMIT"
    )
    om.active_orders[order_id] = partial_order
    
    # Try to move PARTIALLY_FILLED -> PENDING (Invalid: can only go to FILLED/CANCELLED)
    # (Based on the transition map added in Task 3.2)
    
    await om._on_order_status_update(invalid_update) # invalid_update is PENDING
    
    current_status = om.active_orders[order_id].status
    
    if current_status == OrderStatus.PARTIALLY_FILLED:
        logger.info(f"✅ PASSED: Transition rejected. Status is still {current_status}")
    else:
        logger.error(f"❌ FAILED: Invalid transition allowed. Status became {current_status}")
        raise AssertionError("Order State Machine Failed")

async def main():
    logger.info("Starting Validation Suite...")
    try:
        await test_tlm_pnl_fix()
        await test_risk_volatility_fix()
        await test_order_state_fix()
        logger.info("\n🎉 ALL SYSTEMS GO. CODE RED VERIFIED.")
    except Exception as e:
        logger.critical(f"\n💀 VALIDATION FAILED: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
