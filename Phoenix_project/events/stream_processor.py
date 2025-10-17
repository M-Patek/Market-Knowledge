# events/stream_processor.py
from collections import deque
import math
import numpy as np

# Assuming an observability module exists for alerts
# from observability import alert_manager

class RealTimeDQM:
    """
    Performs real-time Data Quality Management on a stream of market data
    by calculating a rolling Z-Score on percentage price changes.
    """
    def __init__(self, window_size=100, z_score_threshold=5.0):
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        # Use a dictionary to store the state for each ticker
        self.ticker_states = {}

    def check_anomaly(self, event):
        """
        Checks an event for price anomalies using a rolling Z-Score.
        Returns True if an anomaly is detected, False otherwise.
        """
        if event.type != 'MARKET_DATA':
            return False

        ticker = event.ticker
        price = event.price
        state = self.ticker_states.get(ticker)

        if state is None:
            # Initialize state for a new ticker
            self.ticker_states[ticker] = {
                'history': deque(maxlen=self.window_size),
                'last_price': price,
                'sum': 0.0,
                'sum_sq': 0.0
            }
            return False # Cannot calculate change for the first data point

        last_price = state['last_price']
        pct_change = (price - last_price) / last_price if last_price != 0 else 0

        history = state['history']
        old_value = 0.0
        if len(history) == self.window_size:
            old_value = history[0] # The oldest value that will be pushed out

        # Incrementally update sums
        state['sum'] += pct_change - old_value
        state['sum_sq'] += pct_change**2 - old_value**2
        history.append(pct_change)
        state['last_price'] = price

        # Defer statistical judgment until the window is sufficiently full
        n = len(history)
        if n < self.window_size / 2: # Wait for at least half the window
            return False

        # Calculate mean and standard deviation from the running sums
        rolling_mean = state['sum'] / n
        # Use the more numerically stable formula for variance
        variance = (state['sum_sq'] / n) - (rolling_mean**2)
        if variance < 0: variance = 0 # Handle potential floating point inaccuracies
        rolling_std = math.sqrt(variance)

        if rolling_std == 0:
            return False # Avoid division by zero if prices are flat

        z_score = (pct_change - rolling_mean) / rolling_std

        if abs(z_score) > self.z_score_threshold:
            # alert_manager.trigger_alert('price_anomaly', {'ticker': ticker, 'z_score': z_score})
            print(f"ALERT: Price anomaly detected for {ticker}. Z-Score: {z_score:.2f}")
            return True
        
        return False

class StreamProcessor:
    def __init__(self, event_queue, dqm_enabled=True):
        self.event_queue = event_queue
        self.dqm_validator = RealTimeDQM() if dqm_enabled else None

    def process_event(self, event):
        if self.dqm_validator:
            if self.dqm_validator.check_anomaly(event):
                # If an anomaly is detected, we might choose to drop the event
                # or forward it to a separate "quarantine" queue.
                print(f"Quarantining anomalous event: {event}")
                return # Stop processing this contaminated event
        
        # If event is valid, continue with existing logic
        print(f"Processing valid event: {event}")
        # ... forward to event distributor etc.
