# execution/trade_lifecycle_manager.py

from collections import defaultdict
import logging
from typing import Dict, List

# Assuming a trade event structure, which we can refine.
# from models.trade_event import TradeEvent
from audit_manager import AuditManager

logger = logging.getLogger(__name__)


class TradeLifecycleManager:
    """
    Subscribes to executed trade events, tracks the lifecycle of trades associated
    with a specific decision_id, and calculates the final P&L once a position
    is fully closed.
    """

    def __init__(self, audit_manager: AuditManager):
        self.open_positions: Dict[str, List] = defaultdict(list)
        self.audit_manager = audit_manager

    def process_trade_event(self, trade_event):
        """
        Main entry point to handle a new executed trade event from the OrderManager.
        """
        decision_id = trade_event.get("decision_id")
        if not decision_id:
            logger.warning("Received trade event without a decision_id. Skipping.")
            return

        self.open_positions[decision_id].append(trade_event)

        # Check if the position for this decision_id is now closed
        if self._is_position_closed(decision_id):
            self._calculate_and_report_pnl(decision_id)

    def _is_position_closed(self, decision_id: str) -> bool:
        # Placeholder logic: Sum of quantities for all trades under this ID is zero.
        # This assumes long trades have positive quantity, short trades have negative.
        total_quantity = sum(trade.get("quantity", 0) for trade in self.open_positions[decision_id])
        return total_quantity == 0

    def _calculate_and_report_pnl(self, decision_id: str):
        # Placeholder logic: Calculate net P&L from all trades.
        trades = self.open_positions.pop(decision_id) # Remove from open positions
        pnl = sum(trade.get("price", 0) * -trade.get("quantity", 0) for trade in trades) # Simplified P&L
        logger.info(f"Position for decision_id {decision_id} closed. Net P&L: {pnl}")

        self.audit_manager.update_pnl_for_decision(decision_id, pnl)
