import time
import threading
import os
import logging
from collections import deque

# --- Imports for Task 5.2 ---
import asyncio
import google.generativeai as genai
from typing import Dict, Any

class GeminiPoolManager:
    """
    Manages a pool of Gemini API keys, handling rate limits, cooldowns,
    and automatic retries with circuit breaker logic.
    """
    def __init__(self, api_keys, cooldown_time=60, max_failures=5, failure_window=300, logger=None):
        """
        Initializes the pool manager.

        Args:
            api_keys (list): A list of Gemini API key strings.
            cooldown_time (int): Seconds a key remains inactive after tripping.
            max_failures (int): Number of failures within failure_window to trip the key.
            failure_window (int): Time window in seconds to track failures.
            logger (logging.Logger, optional): Logger instance.
        """
        if not api_keys:
            raise ValueError("API keys list cannot be empty.")
            
        self.api_keys = api_keys
        self.cooldown_time = cooldown_time
        self.max_failures = max_failures
        self.failure_window = failure_window
        self._lock = threading.Lock()
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Internal state for each key
        self._key_states = {}
        for key in api_keys:
            self._key_states[key] = {
                'active': True,
                'cooldown_until': 0,
                'tripped': False,
                'metrics': {
                    'latency_ms': 0.0,
                    'error_rate': 0.0,
                    'daily_cost': 0.0,
                    'request_count': 0,
                    'error_count': 0
                },
                'failure_count': 0, # Kept for simple auto-trip logic
                'last_failure_time': 0 # Kept for simple auto-trip logic
            }
        
        # Start the cooldown resetter thread
        self._stop_event = threading.Event()
        self._reset_thread = threading.Thread(target=self._cooldown_resetter, daemon=True)
        self._reset_thread.start()

    def _cooldown_resetter(self):
        """
        Background thread to periodically reactivate keys after their cooldown.
        """
        while not self._stop_event.is_set():
            with self._lock:
                current_time = time.time()
                for key, state in self._key_states.items():
                    # Only reactivate if it was auto-tripped (not manually tripped)
                    if not state['active'] and not state['tripped'] and current_time >= state['cooldown_until']:
                        state['active'] = True
                        state['cooldown_until'] = 0
                        state['failure_count'] = 0 # Reset failure count on recovery
                        state['last_failure_time'] = 0
                        self.logger.info(f"Key {key[:4]}... automatically restored from cooldown.")
            
            time.sleep(1) # Check every second

    def report_failure(self, key):
        """
        Reports a failure for a specific API key.
        If failures exceed max_failures within failure_window, trips the key.
        """
        with self._lock:
            if key in self._key_states:
                # Update metrics
                metrics = self._key_states[key]['metrics']
                metrics['request_count'] += 1  # A failure is still a request
                metrics['error_count'] += 1
                
                if metrics['request_count'] > 0:
                    metrics['error_rate'] = (metrics['error_count'] / metrics['request_count']) * 100
                else:
                    metrics['error_rate'] = 0.0 # Should not happen if request_count > 0, but good for safety

                # Existing failure logic
                current_time = time.time()
                # Simple failure window logic (can be improved to sliding window)
                if current_time - self._key_states[key].get('last_failure_time', 0) > self.failure_window:
                    self._key_states[key]['failure_count'] = 1
                else:
                    self._key_states[key]['failure_count'] += 1
                
                self._key_states[key]['last_failure_time'] = current_time

                # Check if the key should be tripped (and not already manually tripped)
                if self._key_states[key]['failure_count'] >= self.max_failures and not self._key_states[key]['tripped']:
                    self._key_states[key]['active'] = False
                    self._key_states[key]['cooldown_until'] = current_time + self.cooldown_time
                    self.logger.warning(f"Key {key[:4]}... automatically tripped. Cooling down for {self.cooldown_time}s.")

                self.logger.info(f"Key {key} failure count incremented to {self._key_states[key]['failure_count']}")

    def report_success(self, key, latency_ms, cost):
        """
        Reports a successful API call and updates the key's metrics.

        Args:
            key (str): The API key used for the call.
            latency_ms (float): The latency of the successful call in milliseconds.
            cost (float): The cost associated with the successful call.
        """
        with self._lock:
            if key in self._key_states:
                metrics = self._key_states[key]['metrics']
                
                # Update metrics
                metrics['request_count'] += 1
                metrics['daily_cost'] += cost
                
                # Recalculate average latency (using cumulative moving average)
                # Ensure request_count is at least 1 to avoid division by zero, though it should be > 0 here
                metrics['latency_ms'] = ((metrics['latency_ms'] * (metrics['request_count'] - 1)) + latency_ms) / max(1, metrics['request_count'])
                
                # Recalculate error rate
                metrics['error_rate'] = (metrics['error_count'] / max(1, metrics['request_count'])) * 100

                self.logger.debug(f"Key {key} success reported. Latency: {latency_ms}ms, Cost: {cost}")

    def get_all_key_states(self):
        """
        Returns a copy of the current states of all API keys.

        Returns:
            dict: A copy of the _key_states dictionary.
        """
        with self._lock:
            # Return a copy to prevent modification outside the class
            return self._key_states.copy()

    def manual_trip(self, key):
        """
        Manually trips a key, setting it to inactive and tripped.

        Args:
            key (str): The API key to trip.
        """
        with self._lock:
            if key in self._key_states:
                self._key_states[key]['active'] = False
                self._key_states[key]['tripped'] = True
                self._key_states[key]['cooldown_until'] = 0 # Manual trip doesn't need automatic cooldown
                self.logger.warning(f"Key {key} has been manually tripped by user.")
                return True
            return False

    def manual_restore(self, key):
        """
        Manually restores a key, setting it to active.
        This clears any tripped status and failure counts.

        Args:
            key (str): The API key to restore.
        """
        with self._lock:
            if key in self._key_states:
                self._key_states[key]['active'] = True
                self._key_states[key]['tripped'] = False
                self._key_states[key]['cooldown_until'] = 0
                self._key_states[key]['failure_count'] = 0
                self._key_states[key]['last_failure_time'] = 0
                self.logger.warning(f"Key {key} has been manually restored by user.")
                return True
            return False

    def get_key(self):
        """
        Gets the next available active API key.
        Cycles through keys in a round-robin fashion.
        """
        with self._lock:
            # Find the next active key
            active_keys = [key for key, state in self._key_states.items() if state['active']]
            if not active_keys:
                self.logger.error("No active API keys available in the pool.")
                return None
            
            # Simple round-robin: pick the first active key and move it to the end
            # This isn't true round-robin, but ensures we don't always pick the same key
            # A deque would be better for true round-robin
            key_to_use = active_keys[0]
            
            # Move key to end of dictionary (Python 3.7+ feature)
            self._key_states[key_to_use] # Access to "use" it
            
            self.logger.debug(f"Using API key {key_to_use[:4]}...")
            return key_to_use

    def stop(self):
        """
        Stops the background thread.
        """
        self.logger.info("Stopping GeminiPoolManager background thread.")
        self._stop_event.set()
        self._reset_thread.join()

# Example usage (if run as main)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    keys = ["key1_fake", "key2_fake", "key3_fake"]
    pool = GeminiPoolManager(api_keys=keys, cooldown_time=10, max_failures=2, logger=logger)
    
    try:
        # Simulate usage
        key1 = pool.get_key()
        logger.info(f"Got key: {key1}")
        
        pool.report_failure(key1)
        pool.report_success(key1, 150.5, 0.01)
        pool.report_failure(key1) # This should trip the key
        
        logger.info(f"Key state after failures: {pool.get_all_key_states().get(key1)}")
        
        key2 = pool.get_key()
        logger.info(f"Got new key: {key2}") # Should be key2_fake
        
        logger.info("Waiting for key1 to recover...")
        time.sleep(12)
        
        key3 = pool.get_key()
        logger.info(f"Got key after cooldown: {key3}") # Should be key1_fake again
        
        # Test manual controls
        pool.manual_trip(key3)
        logger.info(f"Key state after manual trip: {pool.get_all_key_states().get(key3)}")
        key4 = pool.get_key()
        logger.info(f"Got key after manual trip: {key4}") # Should not be key3
        
        pool.manual_restore(key3)
        logger.info(f"Key state after manual restore: {pool.get_all_key_states().get(key3)}")
        key5 = pool.get_key()
        logger.info(f"Got key after manual restore: {key5}") # Should be key3
        
    finally:
        pool.stop()

# --- Task 5.2: Tiered Model Strategy & Query Function ---

# Configure LLM tiers for different agents
# L3 Causal Inference uses the strongest model
# Fact-checking uses a strong model
# All other L1 agents use a fast, economical model
MODEL_TIER_CONFIG: Dict[str, str] = {
    "metacognitive_agent": "gemini-1.5-pro-latest",
    "fact_checker_adversary": "gemini-1.5-pro-latest",
    # "innovation_tracker": "gemini-1.5-flash-latest", # Example
    # "supply_chain_intel": "gemini-1.5-flash-latest", # Example
}
DEFAULT_MODEL = "gemini-1.5-flash-latest"

async def query_model(prompt: str, agent_name: str) -> str:
    """
    Task 0.3: The actual function to query the Gemini API.
    Task 5.2: Implements the Tiered Model Strategy.
    """
    logger = logging.getLogger("PhoenixProject.QueryModel")
    
    # Task 5.2 Logic: Select model based on agent name
    model_name = MODEL_TIER_CONFIG.get(agent_name, DEFAULT_MODEL)
    logger.info(f"Querying for agent '{agent_name}'. Using model: {model_name}")

    try:
        # Assumes genai.configure(api_key=...) has been called elsewhere
        model = genai.GenerativeModel(model_name=model_name)
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error during Gemini API call for {agent_name} (model {model_name}): {e}")
        return f'{{"error": "Failed to generate response: {e}"}}'
