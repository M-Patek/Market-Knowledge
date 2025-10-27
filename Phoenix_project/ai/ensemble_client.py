import time
import logging
from typing import Dict, Any, List
from . import llm_client
from observability import CircuitBreaker


class _SingleAIClient:
    """
    A wrapper for a single LLM client instance, incorporating reliability features
    like circuit breaking and request timing.
    """
    def __init__(self, client_name: str, config: Dict[str, Any]):
        self.client_name = client_name
        self.client = llm_client.LLMClient(config)
        self.logger = logging.getLogger(f"PhoenixProject.AIClient.{self.client_name}")
        # [Sub-Task 2.3.1] Use the standardized circuit breaker
        self.circuit_breaker = CircuitBreaker(failure_threshold=config.get('failure_threshold', 3), recovery_timeout=config.get('circuit_open_duration_seconds', 60))
        self.logger.info(f"Initialized AI client '{self.client_name}' with model '{config.get('model_name', 'N/A')}'.")

    def update_client_config(self, new_config: Dict[str, Any]):
        """Hot-reloads the configuration for the underlying LLM client."""
        self.client.update_config(new_config)
        self.logger.info(f"Updated config for AI client '{self.client_name}'.")

    def execute_llm_call(self, prompt: str, temperature: float) -> Dict[str, Any]:
        """Executes a call to the LLM, wrapped in the circuit breaker logic."""
        try:
            self.logger.info(f"Executing LLM call via client '{self.client_name}'.")
            # Wrap the external call with the circuit breaker
            response = self.circuit_breaker.call(self.client.generate_text, prompt, temperature)
            return {
                "response": response,
                "client_name": self.client_name,
                "status": "success"
            }
        except Exception as e:
            self.logger.error(f"API call to '{self.client_name}' failed: {e}", exc_info=True)
            # The circuit breaker handles the failure counting and state changes internally
            return {"error": str(e), "client_name": self.client_name}


class AIEnsembleClient:
    """
    Manages an ensemble of multiple AI clients, distributing requests and
    handling failures gracefully.
    """
    def __init__(self, ensemble_config: Dict[str, Any]):
        self.logger = logging.getLogger("PhoenixProject.AIEnsembleClient")
        self.clients: Dict[str, _SingleAIClient] = {}
        
        if 'clients' not in ensemble_config:
            self.logger.error("No 'clients' defined in ensemble configuration.")
            return

        for client_name, client_config in ensemble_config.get('clients', {}).items():
            self.clients[client_name] = _SingleAIClient(client_name, client_config)
            
        self.logger.info(f"AI Ensemble Client initialized with {len(self.clients)} clients: {list(self.clients.keys())}")

    def update_client_configs(self, new_ensemble_config: Dict[str, Any]):
        """
        Updates the configurations for all managed clients.
        """
        self.logger.info("Starting hot-reload of AI client configurations...")
        for client_name, new_config in new_ensemble_config.get('clients', {}).items():
            if client_name in self.clients:
                self.clients[client_name].update_client_config(new_config)
            else:
                self.logger.warning(f"Config provided for unknown client '{client_name}'. Ignoring.")
        self.logger.info("AI client configuration reload complete.")

    def execute_concurrent_calls(self, prompt: str, temperature: float = 0.5) -> List[Dict[str, Any]]:
        """
        Executes the same prompt across all healthy clients in the ensemble concurrently.
        
        (Note: The 'concurrent.futures' import was removed as the provided
         implementation runs calls sequentially. A true concurrent implementation
         would use a ThreadPoolExecutor.)
        """
        self.logger.info(f"Executing concurrent calls for prompt (length={len(prompt)}).")
        
        results = []
        
        # This is a sequential execution, not concurrent.
        # A true concurrent implementation would use ThreadPoolExecutor:
        # with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
        #     futures = {
        #         executor.submit(client.execute_llm_call, prompt, temperature): name
        #         for name, client in self.clients.items()
        #     }
        #     for future in concurrent.futures.as_completed(futures):
        #         results.append(future.result())
        
        # Sticking to the sequential logic from the original file for now
        for client_name, client in self.clients.items():
            result = client.execute_llm_call(prompt, temperature)
            results.append(result)

        successful_results = [r for r in results if 'error' not in r]
        failed_clients = [r['client_name'] for r in results if 'error' in r]
        
        self.logger.info(f"Concurrent calls complete. {len(successful_results)} successful, {len(failed_clients)} failed.")
        if failed_clients:
            self.logger.warning(f"Failed clients: {failed_clients}")
            
        return results
