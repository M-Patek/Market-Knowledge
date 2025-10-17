# events/stream_processor.py
from collections import deque
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
        self.pct_change_history = {} # Stores deque of recent pct_changes
        self.last_price = {}         # Stores the last seen price for each ticker

    def check_anomaly(self, event):
        """
        Checks an event for price anomalies using a rolling Z-Score.
        Returns True if an anomaly is detected, False otherwise.
        """
        if event.type != 'MARKET_DATA':
            return False

        ticker = event.ticker
        price = event.price

        if ticker not in self.last_price:
            self.last_price[ticker] = price
            self.pct_change_history[ticker] = deque(maxlen=self.window_size)
            return False # Cannot calculate change for the first data point

        last_price = self.last_price[ticker]
        pct_change = (price - last_price) / last_price if last_price != 0 else 0
        
        history = self.pct_change_history[ticker]

        # Defer statistical judgment until the window is full
        if len(history) < self.window_size:
            history.append(pct_change)
            self.last_price[ticker] = price
            return False 

        # Now, history is a deque of percentage changes, making this more efficient
        rolling_mean = np.mean(history)
        rolling_std = np.std(history)

        if rolling_std == 0:
            history.append(pct_change)
            self.last_price[ticker] = price
            return False # Avoid division by zero if prices are flat
            
        z_score = (pct_change - rolling_mean) / rolling_std
        
        history.append(pct_change)
        self.last_price[ticker] = price

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
