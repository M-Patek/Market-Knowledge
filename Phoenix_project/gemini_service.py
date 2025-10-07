# phoenix_project/gemini_service.py

import os
import json
import hashlib
import logging
import asyncio
import datetime
from typing import List, Dict, Any

from pydantic import BaseModel, ValidationError
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Custom Exception ---
class MalformedResponseError(Exception):
    """Custom exception for when Gemini's response is not in the expected format."""
    pass

# --- [Optimized] Pydantic models for response validation ---
class SentimentResponse(BaseModel):
    """Pydantic model for the sentiment analysis response."""
    sentiment_score: float
    reasoning: str

class AssetAnalysisResponse(BaseModel):
    """Pydantic model for the asset analysis response."""
    adjustment_factor: float
    confidence: float
    reasoning: str

class GeminiService:
    """
    A robust, production-ready service class to handle all interactions
    with the Google Gemini API. Features retries, timeouts, and response validation.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Gemini client using configuration.
        """
        self.config = config
        self.logger = logging.getLogger("PhoenixProject.GeminiService")

        api_key = os.getenv(self.config['api_key_env_var'])
        if not api_key:
            raise ValueError(f"API key environment variable '{self.config['api_key_env_var']}' not set.")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(self.config['model_name'])
        self.generation_config = {"response_mime_type": "application/json"}
        # [Optimized] Mode for separating mock/production logic
        self.mode = self.config.get('mode', 'production')
        self.timeout = self.config.get('request_timeout', 60)
        self.audit_log_dir = "ai_audit_logs"
        os.makedirs(self.audit_log_dir, exist_ok=True)
        self.logger.info(f"GeminiService initialized for model '{self.config['model_name']}'.")

    async def _generate_with_timeout(self, *args, **kwargs):
        return await asyncio.wait_for(
            self.model.generate_content_async(*args, **kwargs),
            timeout=self.timeout
        )

    # [Optimized] Creates a secure, unique filename for an audit record.
    def _create_safe_audit_filename(self, call_type: str, context: Dict[str, Any]) -> str:
        timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        context_str = json.dumps(context, sort_keys=True, ensure_ascii=False)
        context_hash = hashlib.sha256(context_str.encode('utf-8')).hexdigest()[:12]
        return f"{timestamp}_{call_type}_{context_hash}.json"

    def _save_audit_record(self, call_type: str, prompt: str, raw_response_text: str, context: Dict[str, Any]) -> None:
        try:
            # [Optimized] Use the safe filename generation method
            filename = self._create_safe_audit_filename(call_type, context)
            filepath = os.path.join(self.audit_log_dir, filename)
            record = {
                "timestamp_utc": datetime.datetime.utcnow().isoformat(), "call_type": call_type,
                "context": context, "prompt": prompt, "raw_response": raw_response_text
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save AI audit record for '{call_type}': {e}")

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def get_market_sentiment(self, headlines: List[str]) -> Dict[str, Any]:
        self.logger.info("Requesting market sentiment analysis from Gemini...")
        prompt = self.config['prompts']['market_sentiment'].format(headlines="\n- ".join(headlines))
        try:
            response = await self._generate_with_timeout(prompt, generation_config=self.generation_config)
            raw_text = response.text
            self._save_audit_record("market_sentiment", prompt, raw_text, context={"date": datetime.date.today().isoformat()})
            
            # [Optimized] Use Pydantic for validation
            response_data = json.loads(raw_text)
            validated_data = SentimentResponse(**response_data)
            
            self.logger.info(f"Received valid market sentiment: score={validated_data.sentiment_score:.2f}")
            return validated_data.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error(f"Failed to parse or validate sentiment response: {e}")
            raise MalformedResponseError(f"Failed to parse or validate sentiment response: {e}") from e
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during sentiment analysis: {e}")
            raise

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3), reraise=True)
    async def get_asset_analysis(self, ticker: str, date_obj: datetime.date) -> Dict[str, Any]:
        self.logger。info(f"Requesting qualitative analysis for '{ticker}' (Mode: {self.mode})...")
        
        # [Optimized] Logic is now split based on the configured mode
        if self.mode == 'mock':
            await asyncio.sleep(0.05)
            mock_factors = {'QQQ': 1.15, 'SPY': 1.05, 'IWM': 0.95, 'GLD': 0.90, 'TLT': 0.85}
            factor = mock_factors.get(ticker, 1.0)
            confidence = min(1.0， abs(factor - 1.0) * 2.5)
            reasoning = f"Mock analysis for {ticker}: Based on its segment, a factor of {factor} is assigned."
            mock_response_data = {"adjustment_factor": factor, "confidence": confidence, "reasoning": reasoning}
            
            # [Optimized] Also validate mock data to ensure consistency
            validated_data = AssetAnalysisResponse(**mock_response_data)
            raw_response_text = json.dumps(validated_data.model_dump(), indent=2)
            self._save_audit_record("asset_analysis"， "mock_prompt_for_dev", raw_response_text, context={"ticker": ticker, "date": date_obj.isoformat()})
            self.logger.info(f"Received mock analysis for {ticker}: factor={validated_data.adjustment_factor:.2f}, confidence={validated_data.confidence:.2f}")
            return validated_data.model_dump()
        else: # Production mode
            prompt = self.config['prompts']['asset_analysis'].format(ticker=ticker)
            try:
                response = await self._generate_with_timeout(prompt, generation_config=self.generation_config)
                raw_text = response.text
                self._save_audit_record("asset_analysis", prompt, raw_text, context={"ticker": ticker, "date": date_obj.isoformat()})
                
                # [Optimized] Use Pydantic for validation
                response_data = json.loads(raw_text)
                validated_data = AssetAnalysisResponse(**response_data)
                
                self.logger.info(f"Received valid asset analysis for {ticker}: factor={validated_data.adjustment_factor:.2f}, confidence={validated_data.confidence:.2f}")
                return validated_data.model_dump()
            except (json.JSONDecodeError, ValidationError) as e:
                self.logger.error(f"Failed to parse or validate asset analysis for {ticker}: {e}")
                raise MalformedResponseError(f"Failed to parse or validate asset analysis response for {ticker}: {e}") from e
            except Exception as e:
                self.logger.error(f"An unexpected error occurred during asset analysis for {ticker}: {e}")
                raise

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3), reraise=True)
    async def generate_summary_report(self, metrics: Dict[str, Any]) -> str:
        self.logger.info("Requesting 'Marshal's Report' generation from Gemini...")
        formatted_metrics = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if "value" in key or "cash" in key: formatted_metrics.append(f"- {key.replace('_', ' ').title()}: ${value:,.2f}")
                elif "return" in key or "rate" in key: formatted_metrics.append(f"- {key.replace('_', ' ').title()}: {value:.2%}")
                else: formatted_metrics.append(f"- {key.replace('_', ' ').title()}: {value:.3f}")
            elif value is not None: formatted_metrics.append(f"- {key.replace('_', ' ').title()}: {value}")
        metrics_str = "\n".join(formatted_metrics)
        prompt = self.config['prompts']['summary_report'].format(metrics=metrics_str)
        try:
            response = await self._generate_with_timeout(prompt)
            raw_text = response.text
            self._save_audit_record("summary_report", prompt, raw_text, context={"metrics_hash": hash(metrics_str)})
            return raw_text
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during summary report generation: {e}")
            raise
