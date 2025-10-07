# phoenix_project/gemini_service.py

import os
import json
import logging
import asyncio
import datetime
from typing import List, Dict, Any

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Custom Exception ---
class MalformedResponseError(Exception):
    """Custom exception for when Gemini's response is not in the expected format."""
    pass

class GeminiService:
    """
    A robust, production-ready service class to handle all interactions
    with the Google Gemini API. Features retries, timeouts, and response validation.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Gemini client using configuration.

        Args:
            config: A dictionary containing Gemini API settings from config.yaml.
        """
        self.config = config
        self.logger = logging.getLogger("PhoenixProject.GeminiService")

        api_key = os.getenv(self.config['api_key_env_var'])
        if not api_key:
            self.logger.critical(f"API key environment variable '{self.config['api_key_env_var']}' not set.")
            raise ValueError(f"API key environment variable '{self.config['api_key_env_var']}' not set.")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(self.config['model_name'])
        self.generation_config = {"response_mime_type": "application/json"}
        self.timeout = self.config.get('request_timeout', 60)
        self.audit_log_dir = "ai_audit_logs"
        os.makedirs(self.audit_log_dir, exist_ok=True)
        self.logger.info(f"GeminiService initialized for model '{self.config['model_name']}'.")

    async def _generate_with_timeout(self, *args, **kwargs):
        """Helper to run async generation with a timeout."""
        return await asyncio.wait_for(
            self.model.generate_content_async(*args, **kwargs),
            timeout=self.timeout
        )

    def _save_audit_record(self, call_type: str, prompt: str, raw_response_text: str, context: Dict[str, Any]) -> None:
        """Saves a record of the AI interaction for auditing purposes."""
        try:
            timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            context_str = "_".join(str(v) for v in context.values())
            filename = f"{timestamp_str}_{call_type}_{context_str}.json"
            filepath = os.path.join(self.audit_log_dir, filename)

            record = {
                "timestamp_utc": datetime.datetime.utcnow().isoformat(),
                "call_type": call_type,
                "context": context,
                "prompt": prompt,
                "raw_response": raw_response_text
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Failed to save AI audit record for '{call_type}': {e}")

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception), # Retry on any transient exception
        reraise=True
    )
    async def get_market_sentiment(self, headlines: List[str]) -> Dict[str, Any]:
        """
        Analyzes news headlines to determine market sentiment.
        Corresponds to "Suggestion 1".
        """
        self.logger.info("Requesting market sentiment analysis from Gemini...")
        prompt = self.config['prompts']['market_sentiment'].format(headlines="\n- ".join(headlines))

        try:
            response = await self._generate_with_timeout(prompt, generation_config=self.generation_config)
            raw_text = response.text
            self._save_audit_record("market_sentiment", prompt, raw_text, context={"date": datetime.date.today().isoformat()})
            response_data = json.loads(raw_text)

            if 'sentiment_score' not in response_data or 'reasoning' not in response_data:
                raise MalformedResponseError(f"Missing required keys in sentiment response: {response_data}")

            self.logger.info(f"Received valid market sentiment: score={response_data['sentiment_score']:.2f}")
            return response_data
        except (json.JSONDecodeError, MalformedResponseError) as e:
            self.logger.error(f"Failed to parse or validate sentiment response: {e}")
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during sentiment analysis: {e}")
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def get_asset_analysis(self, ticker: str, date_obj: datetime.date) -> Dict[str, Any]:
        """
        Performs a qualitative analysis on a single asset.
        Corresponds to "Suggestion 2".
        """
        # --- [Development Mock] ---
        # TODO: Replace with actual API call in production.
        # This mock logic provides deterministic results for testing.
        self.logger.info(f"Requesting qualitative analysis for ticker '{ticker}' from Gemini...")
        await asyncio.sleep(0.05) # Simulate network latency

        mock_factors = {
            'QQQ': 1.15, # Tech-heavy, generally positive outlook
            'SPY': 1.05, # Broad market, slightly positive
            'IWM': 0.95, # Small caps, slightly cautious
            'GLD': 0.90, # Gold, defensive, implies market uncertainty
            'TLT': 0.85, # Bonds, very defensive, implies risk-off
        }
        factor = mock_factors.get(ticker, 1.0)
        # Mock confidence based on how far the factor is from neutral 1.0
        confidence = min(1.0, abs(factor - 1.0) * 2.5)
        reasoning = f"Mock analysis for {ticker}: Based on its market segment, a factor of {factor} is assigned."

        mock_response = {
            "adjustment_factor": factor,
            "confidence": confidence,
            "reasoning": reasoning
        }
        # For auditing, we simulate the raw text that the API would have returned
        raw_mock_text = json.dumps(mock_response, indent=2)
        self._save_audit_record("asset_analysis"， "mock_prompt_for_dev", raw_mock_text, context={"ticker": ticker, "date": date_obj.isoformat()})
        self.logger。info(f"Received mock analysis for {ticker}: factor={mock_response['adjustment_factor']:.2f}, confidence={mock_response['confidence']:.2f}")
        return mock_response
        # --- [End of Development Mock] ---

        # --- [Production Code] ---
        # prompt = self.config['prompts']['asset_analysis'].format(ticker=ticker)
        # try:
        #     response = await self._generate_with_timeout(prompt, generation_config=self.generation_config)
        #     response_data = json.loads(response.text)
        #     if 'adjustment_factor' not in response_data or 'confidence' not in response_data or 'reasoning' not in response_data:
        #         raise MalformedResponseError(f"Missing required keys in asset analysis response: {response_data}")
        #     self.logger.info(f"Received valid asset analysis for {ticker}: factor={response_data['adjustment_factor']:.2f}, confidence={response_data['confidence']:.2f}")
        #     return response_data
        # except Exception as e:
        #     self.logger.error(f"An unexpected error occurred during asset analysis for {ticker}: {e}")
        #     raise

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def generate_summary_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generates a narrative "Marshal's Report" from backtest metrics.
        Corresponds to "Suggestion 3".
        """
        self.logger.info("Requesting 'Marshal's Report' generation from Gemini...")

        # Format metrics into a readable string for the prompt
        formatted_metrics = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if "value" in key or "cash" in key:
                    formatted_metrics.append(f"- {key.replace('_', ' ').title()}: ${value:,.2f}")
                elif "return" in key or "rate" in key:
                    formatted_metrics.append(f"- {key.replace('_', ' ').title()}: {value:.2%}")
                else:
                    formatted_metrics.append(f"- {key.replace('_', ' ').title()}: {value:.3f}")
            elif value is not None:
                formatted_metrics.append(f"- {key.replace('_', ' ').title()}: {value}")
        metrics_str = "\n".join(formatted_metrics)

        prompt = self.config['prompts']['summary_report'].format(metrics=metrics_str)
        try:
            # Note: We expect a text response, so we don't use the JSON generation_config
            response = await self._generate_with_timeout(prompt)
            raw_text = response.text
            self._save_audit_record("summary_report", prompt, raw_text, context={"metrics_hash": hash(metrics_str)})
            return raw_text
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during summary report generation: {e}")
            raise
