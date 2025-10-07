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
    """当Gemini的响应格式不符合预期时抛出的自定义异常。"""
    pass

# --- Pydantic models for response validation ---
class SentimentResponse(BaseModel):
    """用于情绪分析响应的Pydantic模型。"""
    sentiment_score: float
    reasoning: str

class AssetAnalysisResponse(BaseModel):
    """用于资产分析响应的Pydantic模型。"""
    adjustment_factor: float
    confidence: float
    reasoning: str

class BatchAssetAnalysisResponse(BaseModel):
    """用于验证批量资产分析响应的Pydantic模型。"""
    analyses: Dict[str, AssetAnalysisResponse]

class GeminiService:
    """
    一个健壮的、生产就绪的服务类，用于处理与Google Gemini API的所有交互。
    具有重试、超时和响应验证功能。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        使用配置初始化Gemini客户端。
        """
        self.config = config
        self.logger = logging.getLogger("PhoenixProject.GeminiService")

        api_key = os.getenv(self.config['api_key_env_var'])
        if not api_key:
            raise ValueError(f"API密钥环境变量 '{self.config['api_key_env_var']}' 未设置。")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(self.config['model_name'])
        self.generation_config = {"response_mime_type": "application/json"}
        self.mode = self.config.get('mode', 'production')
        self.timeout = self.config.get('request_timeout', 60)
        self.audit_log_dir = "ai_audit_logs"
        os.makedirs(self.audit_log_dir, exist_ok=True)
        self.logger.info(f"GeminiService已为模型 '{self.config['model_name']}' 初始化。")

    async def _generate_with_timeout(self, *args, **kwargs):
        return await asyncio.wait_for(
            self.model.generate_content_async(*args, **kwargs),
            timeout=self.timeout
        )

    def _create_safe_audit_filename(self, call_type: str, context: Dict[str, Any]) -> str:
        timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        context_str = json.dumps(context, sort_keys=True, ensure_ascii=False)
        context_hash = hashlib.sha256(context_str.encode('utf-8')).hexdigest()[:12]
        return f"{timestamp}_{call_type}_{context_hash}.json"

    def _save_audit_record(self, call_type: str, prompt: str, raw_response_text: str, context: Dict[str, Any]) -> None:
        try:
            filename = self._create_safe_audit_filename(call_type, context)
            filepath = os.path.join(self.audit_log_dir, filename)
            record = {
                "timestamp_utc": datetime.datetime.utcnow().isoformat(), "call_type": call_type,
                "context": context, "prompt": prompt, "raw_response": raw_response_text
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"无法为 '{call_type}' 保存AI审计记录: {e}")

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def get_market_sentiment(self, headlines: List[str]) -> Dict[str, Any]:
        self.logger.info("正在从Gemini请求市场情绪分析...")
        prompt = self.config['prompts']['market_sentiment'].format(headlines="\n- ".join(headlines))
        try:
            response = await self._generate_with_timeout(prompt, generation_config=self.generation_config)
            raw_text = response.text
            self._save_audit_record("market_sentiment", prompt, raw_text, context={"date": datetime.date.today().isoformat()})
            
            response_data = json.loads(raw_text)
            validated_data = SentimentResponse(**response_data)
            
            self.logger.info(f"收到有效的市场情绪: score={validated_data.sentiment_score:.2f}")
            return validated_data.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error(f"解析或验证情绪响应失败: {e}")
            raise MalformedResponseError(f"解析或验证情绪响应失败: {e}") from e
        except Exception as e:
            self.logger.error(f"情绪分析期间发生意外错误: {e}")
            raise

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3), reraise=True)
    async def get_asset_analysis(self, ticker: str, date_obj: datetime.date) -> Dict[str, Any]:
        self.logger.info(f"正在为 '{ticker}' 请求定性分析 (模式: {self.mode})...")
        
        if self.mode == 'mock':
            await asyncio.sleep(0.05)
            mock_factors = {'QQQ': 1.15, 'SPY': 1.05, 'IWM': 0.95, 'GLD': 0.90, 'TLT': 0.85}
            factor = mock_factors.get(ticker, 1.0)
            confidence = min(1.0, abs(factor - 1.0) * 2.5)
            reasoning = f"对 {ticker} 的模拟分析：基于其板块，分配因子为 {factor}。"
            mock_response_data = {"adjustment_factor": factor, "confidence": confidence, "reasoning": reasoning}
            
            validated_data = AssetAnalysisResponse(**mock_response_data)
            raw_response_text = json.dumps(validated_data.model_dump(), indent=2)
            self._save_audit_record("asset_analysis", "mock_prompt_for_dev", raw_response_text, context={"ticker": ticker, "date": date_obj.isoformat()})
            self.logger.info(f"收到 {ticker} 的模拟分析: factor={validated_data.adjustment_factor:.2f}, confidence={validated_data.confidence:.2f}")
            return validated_data.model_dump()
        else: # Production mode
            prompt = self.config['prompts']['asset_analysis'].format(ticker=ticker)
            try:
                response = await self._generate_with_timeout(prompt, generation_config=self.generation_config)
                raw_text = response.text
                self._save_audit_record("asset_analysis", prompt, raw_text, context={"ticker": ticker, "date": date_obj.isoformat()})
                
                response_data = json.loads(raw_text)
                validated_data = AssetAnalysisResponse(**response_data)
                
                self.logger.info(f"收到 {ticker} 的有效资产分析: factor={validated_data.adjustment_factor:.2f}, confidence={validated_data.confidence:.2f}")
                return validated_data.model_dump()
            except (json.JSONDecodeError, ValidationError) as e:
                self.logger.error(f"解析或验证 {ticker} 的资产分析失败: {e}")
                raise MalformedResponseError(f"解析或验证 {ticker} 的资产分析响应失败: {e}") from e
            except Exception as e:
                self.logger.error(f"在为 {ticker} 进行资产分析时发生意外错误: {e}")
                raise

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3), reraise=True)
    async def get_batch_asset_analysis(self, tickers: List[str], date_obj: datetime.date) -> Dict[str, Any]:
        self.logger。info(f"正在为 {tickers} 请求批量定性分析 (模式: {self.mode})...")
        context = {"tickers": tickers, "date": date_obj.isoformat()}

        if self.mode == 'mock':
            await asyncio.sleep(0.1) # 模拟稍长的批量调用
            mock_factors = {'QQQ': 1.15， 'SPY': 1.05， 'IWM': 0.95， 'GLD': 0.90， 'TLT': 0.85}
            batch_response = {"analyses": {}}
            for ticker 在 tickers:
                factor = mock_factors.get(ticker, 1.0)
                confidence = min(1.0， abs(factor - 1.0) * 2.5)
                reasoning = f"对 {ticker} 的模拟分析：基于其板块，分配因子为 {factor}。"
                batch_response["analyses"][ticker] = {"adjustment_factor": factor, "confidence": confidence, "reasoning": reasoning}

            validated_data = BatchAssetAnalysisResponse(**batch_response)
            raw_response_text = json.dumps(validated_data.model_dump(), indent=2)
            self._save_audit_record("batch_asset_analysis", "mock_prompt_for_dev_batch", raw_response_text, context)
            self.logger.info(f"收到 {tickers} 的模拟批量分析。")
            return validated_data.model_dump()["analyses"]
        else: # Production mode
            prompt = self.config['prompts']['batch_asset_analysis'].format(tickers=", ".join(tickers))
            try:
                response = await self._generate_with_timeout(prompt, generation_config=self.generation_config)
                raw_text = response.text
                self._save_audit_record("batch_asset_analysis", prompt, raw_text, context)

                response_data = json.loads(raw_text)
                validated_data = BatchAssetAnalysisResponse(analyses=response_data)

                self.logger。info(f"收到 {tickers} 的有效批量资产分析。")
                return validated_data.model_dump()["analyses"]
            except (json.JSONDecodeError, ValidationError) as e:
                self.logger.error(f"解析或验证 {tickers} 的批量资产分析失败: {e}")
                raise MalformedResponseError(f"解析或验证 {tickers} 的批量资产分析响应失败: {e}") from e
            except Exception as e:
                self.logger.error(f"在为 {tickers} 进行批量资产分析时发生意外错误: {e}")
                raise

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3), reraise=True)
    async def generate_summary_report(self, metrics: Dict[str, Any]) -> str:
        self.logger.info("正在从Gemini请求生成'元帅报告'...")
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
            self.logger.error(f"在生成摘要报告时发生意外错误: {e}")
            raise
