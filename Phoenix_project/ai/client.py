# ai/client.py

import os
import json
import hashlib
import logging
import asyncio
import datetime
from typing import List, Dict, Any, Protocol, Tuple

import google.generativeai as genai
from tenacity import retry, wait_exponential, stop_after_attempt, wait_random_jitter
from pydantic import ValidationError

from ai.validation import AssetAnalysisModel, MarketSentimentModel
from observability import AI_CALL_LATENCY

# --- Custom Exception ---
class MalformedResponseError(Exception):
    """Raised when the AI response format is not as expected."""
    pass

class AIClient(Protocol):
    """A protocol defining the interface for AI interaction services."""

    async def get_market_sentiment(self, headlines: List[str]) -> tuple[Dict[str, Any], str | None]:
        ...

    async def get_batch_asset_analysis(self, tickers: List[str], date_obj: datetime.date) -> tuple[Dict[str, Dict[str, Any]], str | None]:
        ...

    async def generate_summary_report(self, metrics: Dict[str, Any]) -> tuple[str, str | None]:
        ...

class GeminiAIClient:
    """Production-ready client for the Google Gemini API."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("PhoenixProject.GeminiAIClient")
        api_key = os.getenv(self.config['api_key_env_var'])
        if not api_key:
            raise ValueError(f"API key env var '{self.config['api_key_env_var']}' not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.config['model_name'])
        self.generation_config = {"response_mime_type": "application/json"}
        self.timeout = self.config.get('request_timeout', 60)
        self.audit_log_dir = "ai_audit_logs"
        self.semaphore = asyncio.Semaphore(self.config.get('max_concurrent_requests', 5))
        self.retention_days = self.config.get('audit_log_retention_days', 30)
        os.makedirs(self.audit_log_dir, exist_ok=True)
        self.logger.info(f"GeminiAIClient initialized for model '{self.config['model_name']}'.")
        self._cleanup_old_audit_logs()

    async def _generate_with_timeout(self, *args, **kwargs):
        """Wraps the API call with the client's semaphore for concurrency control."""
        with AI_CALL_LATENCY.time():
            async with self.semaphore:
                return await asyncio.wait_for(
                    self.model.generate_content_async(*args, **kwargs),
                    timeout=self.timeout
                )

    def _cleanup_old_audit_logs(self):
        if self.retention_days <= 0:
            self.logger.info("Audit log retention is disabled. Skipping cleanup.")
            return
        
        self.logger.info(f"Cleaning up audit logs older than {self.retention_days} days...")
        cutoff_datetime = datetime.datetime.utcnow() - datetime.timedelta(days=self.retention_days)
        cleaned_count = 0
        for filename in os.listdir(self.audit_log_dir):
            try:
                # Use the full timestamp for precise comparison
                timestamp_str = filename.split('_')[0]
                file_datetime = datetime.datetime.strptime(timestamp_str, '%Y%m%d%H%M%S%f')
                if file_datetime < cutoff_datetime:
                    os.remove(os.path.join(self.audit_log_dir, filename))
                    cleaned_count += 1
            except (IndexError, ValueError):
                # Ignore files with incorrect name format
                continue
        self.logger.info(f"Removed {cleaned_count} old audit log files.")

    def _create_safe_audit_filename(self, call_type: str, context: Dict[str, Any]) -> str:
        timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        context_str = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
        context_hash = hashlib.sha256(context_str.encode('utf-8')).hexdigest()[:12]
        return f"{timestamp}_{call_type}_{context_hash}.json"

    def _save_audit_record(self, call_type: str, prompt: str, raw_response_text: str, context: Dict[str, Any]) -> str:
        try:
            filename = self._create_safe_audit_filename(call_type, context)
            filepath = os.path.join(self.audit_log_dir, filename)
            record = {
                "timestamp_utc": datetime.datetime.utcnow().isoformat(), "call_type": call_type,
                "context": context, "prompt": prompt, "raw_response": raw_response_text
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            
            # Set file permissions to read/write for owner only (600)
            if hasattr(os, 'chmod'): # os.chmod is not available on all OSes (e.g. Windows)
                os.chmod(filepath, 0o600)
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to save AI audit record for '{call_type}': {e}")
            return ""

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random_jitter(jitter=2),
        stop=stop_after_attempt(3), reraise=True
    )
    async def get_market_sentiment(self, headlines: List[str]) -> tuple[Dict[str, Any], str | None]:
        prompt = self.config['prompts']['market_sentiment'].format(headlines="\n- ".join(headlines))
        raw_text = ""
        try:
            response = await self._generate_with_timeout(prompt, generation_config=self.generation_config)
            raw_text = response.text
            audit_path = self._save_audit_record("market_sentiment", prompt, raw_text, context={"headlines": headlines})
            # --- Validation Firewall ---
            validated_data = MarketSentimentModel.model_validate_json(raw_text)
            return validated_data.model_dump(), audit_path
        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error(f"AI response validation failed for market sentiment. Error: {e}. Raw Text: '{raw_text}'. Falling back to neutral.")
            return {"sentiment_score": 0.0, "reasoning": "Fallback due to validation error."}, None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during sentiment analysis: {e}")
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random_jitter(jitter=2),
        stop=stop_after_attempt(3), reraise=True
    )
    async def get_batch_asset_analysis(self, tickers: List[str], date_obj: datetime.date) -> tuple[Dict[str, Dict[str, Any]], str | None]:
        prompt = self.config['prompts']['batch_asset_analysis'].format(tickers=", ".join(tickers))
        raw_text = "" # Initialize to empty string for error logging
        context = {"tickers": tickers, "date": str(date_obj)}
        try:
            response = await self._generate_with_timeout(prompt, generation_config=self.generation_config)
            raw_text = response.text
            audit_path = self._save_audit_record("batch_asset_analysis", prompt, raw_text, context=context)
            # --- Validation Firewall ---
            # The response is a dict where keys are tickers and values are the analysis objects
            raw_data = json.loads(raw_text)
            validated_data = {ticker: AssetAnalysisModel.model_validate(analysis) for ticker, analysis in raw_data.items()}
            return {ticker: model.model_dump() for ticker, model in validated_data.items()}, audit_path
        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error(f"AI response validation failed for batch asset analysis. Error: {e}. Raw Text: '{raw_text}'. Falling back to neutral for all tickers.")
            fallback_analysis = {"adjustment_factor": 1.0, "confidence": 0.0, "reasoning": "Fallback due to validation error."}
            return {ticker: fallback_analysis for ticker in tickers}, None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during batch analysis: {e}")
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random_jitter(jitter=2),
        stop=stop_after_attempt(3), reraise=True
    )
    async def generate_summary_report(self, metrics: Dict[str, Any]) -> tuple[str, str | None]:
        # This method remains largely the same, returning a string
        formatted_metrics = []
        for key, value 在 metrics.items():
            if isinstance(value, float):
                if "value" in key or "cash" in key: formatted_metrics.append(f"- {key.replace('_', ' ').title()}: ${value:,.2f}")
                elif "return" in key or "rate" in key: formatted_metrics.append(f"- {key.replace('_', ' ').title()}: {value:.2%}")
                else: formatted_metrics.append(f"- {key.replace('_', ' ').title()}: {value:.3f}")
            elif value is not None: formatted_metrics.append(f"- {key.replace('_', ' ').title()}: {value}")
        metrics_str = "\n".join(formatted_metrics)
        # Using a hash of metrics for context to avoid huge filenames
        context = {"metrics_hash": hashlib.sha256(metrics_str.encode())。hexdigest()[:12]}
        prompt = self.config['prompts']['summary_report'].format(metrics=metrics_str)
        response = await self._generate_with_timeout(prompt)
        audit_path = self._save_audit_record("summary_report", prompt, response.text, context=context)
        return response.text, audit_path


class MockAIClient:
    """A mock client for deterministic, offline testing."""

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("PhoenixProject.MockAIClient")
        self.logger.info("MockAIClient initialized.")

    async def get_market_sentiment(self, headlines: List[str]) -> tuple[Dict[str, Any], str | None]:
        self.logger.info("Mocking market sentiment request.")
        await asyncio.sleep(0.01)
        return {"sentiment_score": 0.1, "reasoning": "Mocked neutral sentiment."}, "mock_sentiment_audit.json"

    async def get_batch_asset_analysis(self, tickers: List[str], date_obj: datetime.date) -> tuple[Dict[str, Dict[str, Any]], str | None]:
        self.logger.info(f"Mocking batch asset analysis for {tickers}.")
        await asyncio.sleep(0.05)
        mock_factors = {'QQQ': 1.15, 'SPY': 1.05, 'IWM': 0.95, 'GLD': 0.90, 'TLT': 0.85}
        response: Dict[str, Dict[str, Any]] = {}
        for ticker in tickers:
            factor = mock_factors.get(ticker, 1.0)
            confidence = min(1.0, abs(factor - 1.0) * 2.5)
            response[ticker] = {
                "adjustment_factor": factor,
                "confidence": confidence,
                "reasoning": f"Mock analysis for {ticker}."
            }
        return response, f"mock_batch_analysis_{date_obj.isoformat()}.json"

    async def generate_summary_report(self, metrics: Dict[str, Any]) -> tuple[str, str | None]:
        self.logger.info("Mocking summary report generation.")
        await asyncio.sleep(0.01)
        return "## Mock Marshal's Report\n\nThis is a simulated report based on mock data. All systems performed nominally.", "mock_summary_report.json"
