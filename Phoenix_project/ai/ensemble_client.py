# ai/ensemble_client.py
"""
Orchestrates a "jury" of multiple AI models to get a diverse set of analyses
for a single asset, improving robustness and surfacing disagreement.

This client is responsible for:
- Loading ensemble configuration.
- Managing the lifecycle and health (circuit breaker) of individual AI clients.
- Rendering prompts using a robust, auditable renderer.
- Executing concurrent API calls.
- Validating responses against Pydantic schemas and retrieved documents.
- Writing detailed audit logs for every transaction.
"""
import os
import json
import uuid
import asyncio
import logging
import yaml
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Assuming the new renderer and validation models are in these locations
from .prompt_renderer import render_prompt
from .validation import AssetAnalysisModel, validate_response_against_retrieved_docs, ValidationErrorWithContext
from observability import AI_CALL_LATENCY # Placeholder for observability metrics

# --- Internal Client for a Single Model ---

class _SingleAIClient:
    """
    Manages the connection, state, and API calls for a single AI model.
    Internal helper class for the EnsembleAIClient.
    """
    def __init__(self, client_id: str, config: Dict[str, Any]):
        self.client_id = client_id
        self.config = config
        self.logger = logging.getLogger(f"PhoenixProject.SingleAIClient.{self.client_id}")
        
        # State for Circuit Breaker
        self.failures = 0
        self.cooldown_until: Optional[datetime] = None

        # API Setup
        api_key = os.getenv(self.config['api_key_env_var'])
        if not api_key:
            raise ValueError(f"API key env var '{self.config['api_key_env_var']}' not set for client '{self.client_id}'.")
        
        # NOTE: In a real multi-provider system, you would abstract this part.
        # For now, we assume all models are from the same provider (e.g., Google GenAI).
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.config['model_name'])
        self.semaphore = asyncio.Semaphore(self.config['max_concurrent'])
        self.generation_config = {"response_mime_type": "application/json"}
        self.runtime_settings = {
            "temperature": self.config.get('temperature', 0.1),
            "top_p": self.config.get('top_p', 1.0)
        }

        self.logger.info(f"'{self.client_id}' initialized for model '{self.config['model_name']}' with temp={self.runtime_settings['temperature']}.")

    def is_healthy(self) -> bool:
        if self.cooldown_until and datetime.utcnow() < self.cooldown_until:
            self.logger.warning(f"Circuit breaker for '{self.client_id}' is open. Skipping.")
            return False
        return True

    def record_failure(self):
        self.failures += 1
        probes_config = self.config['health_probes']
        if self.failures >= probes_config['failure_threshold']:
            cooldown = timedelta(minutes=probes_config['cooldown_minutes'])
            self.cooldown_until = datetime.utcnow() + cooldown
            self.logger.critical(f"'{self.client_id}' circuit breaker opened for {cooldown.total_seconds() / 60} mins.")

    def record_success(self):
        self.failures = 0
        self.cooldown_until = None

    @retry(
        wait=wait_random_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(2), # 1 initial call + 1 retry
        reraise=True
    )
    async def get_analysis(self, prompt_details: dict, audit_id: str, run_id: str, ticker: str) -> Optional[AssetAnalysisModel]:
        if not self.is_healthy():
            return None

        raw_text = ""
        prompt_meta = prompt_details.get("meta", {})
        
        try:
            async with self.semaphore:
                with AI_CALL_LATENCY.time():
                    response = await asyncio.wait_for(
                        self.model.generate_content_async(
                            prompt_details["final_prompt"],
                            generation_config={**self.generation_config, **self.runtime_settings}
                        ),
                        timeout=self.config['timeout_seconds']
                    )
                    raw_text = response.text
            
            model_data = json.loads(raw_text)
            
            # --- System-side enrichment and validation ---
            model_data['audit_id'] = audit_id
            model_data['model_version'] = self.config['model_name']
            model_data['ticker'] = ticker
            
            validated_model = AssetAnalysisModel.model_validate(model_data)
            validate_response_against_retrieved_docs(validated_model, prompt_meta.get("retrieved_doc_ids", []))
            
            self._write_audit(audit_id, run_id, ticker, prompt_details, raw_text, validated_model.dict(), success=True)
            self.record_success()
            return validated_model

        except (json.JSONDecodeError, ValidationErrorWithContext) as e:
            self.logger.error(f"Validation failed for '{self.client_id}': {e}. Raw Text: '{raw_text}'")
            self._write_audit(audit_id, run_id, ticker, prompt_details, raw_text, success=False, error=str(e))
            return None
        except Exception as e:
            self.logger.error(f"API call failed for '{self.client_id}': {e}")
            self.record_failure()
            self._write_audit(audit_id, run_id, ticker, prompt_details, raw_text, success=False, error=str(e))
            raise

    def _write_audit(self, audit_id: str, run_id: str, ticker: str, prompt_details: dict, raw_response: str, parsed_response: Optional[dict] = None, success: bool = False, error: Optional[str] = None):
        # This function should implement atomic, structured logging as per the plan
        # For brevity, this is a simplified version.
        log_dir = os.path.join("ai_audit_logs", datetime.utcnow().strftime('%Y-%m-%d'))
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, f"{audit_id}.json")
        
        record = {
            "audit_id": audit_id, "run_id": run_id, "client_id": self.client_id,
            "ticker": ticker, "timestamp_utc": datetime.utcnow().isoformat(),
            "success": success, "error": error,
            "prompt_details": prompt_details,
            "raw_response": raw_response,
            "parsed_response": parsed_response,
            "model_config": {"name": self.config['model_name'], **self.runtime_settings}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

# --- Main Ensemble Client ---

class EnsembleAIClient:
    """ Orchestrates the AI jury to get a diverse set of analyses. """
    def __init__(self, config_file_path: str, run_id: str):
        self.logger = logging.getLogger("PhoenixProject.EnsembleAIClient")
        self.run_id = run_id
        
        with open(config_file_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.clients: Dict[str, _SingleAIClient] = {}
        for client_id, client_config in self.config.get('clients', {}).items():
            if client_config.get('enable', False):
                self.clients[client_id] = _SingleAIClient(client_id, client_config)
        
        if not self.clients:
            raise ValueError("No AI clients are enabled in the configuration file.")
        
        self.logger。info(f"EnsembleAIClient initialized with {len(self.clients)} active clients.")

    async def get_ensemble_asset_analysis(self, ticker: str, retrieved_docs: List[Dict[str, Any]]) -> List[AssetAnalysisModel]:
        """
        Calls all enabled AI clients concurrently to get a list of analyses.
        """
        call_uuid = uuid.uuid4()
        tasks = []

        for client_id, client 在 self.clients.items():
            try:
                # 1. Render the specific prompt for this client
                prompt_details = render_prompt(
                    template_path=client.config['prompt_template'],
                    ticker=ticker,
                    retrieved_docs=retrieved_docs
                )
                
                # 2. Create a unique audit ID for this specific transaction
                audit_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}-{self.run_id}-{ticker}-{client_id}-{call_uuid.hex[:8]}"
                
                # 3. Schedule the API call
                task = client.get_analysis(prompt_details, audit_id, self.run_id, ticker)
                tasks.append(task)
            except Exception as e:
                self.logger.error(f"Failed to prepare task for client '{client_id}': {e}")
        
        if not tasks:
            return []

        self.logger.info(f"Dispatching ensemble for '{ticker}' (call UUID: {call_uuid}).")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = []
        for res, client_id in zip(results, self.clients.keys()):
            if isinstance(res, AssetAnalysisModel):
                valid_responses.append(res)
                self.logger.info(f"Received valid response from '{client_id}' for '{ticker}'.")
            elif isinstance(res, Exception):
                self.logger.error(f"Ensemble member '{client_id}' for '{ticker}' failed after all retries: {res}")
        
        self.logger.info(f"Ensemble for '{ticker}' complete: {len(valid_responses)}/{len(self.clients)} valid responses.")
        return valid_responses
