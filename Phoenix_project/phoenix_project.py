# Phoenix Project - Final Corrected Version (Phoenix Resurrected)
# A collaborative masterpiece by Gemini & AI, guided by our Master.
# This version features a robust Pydantic configuration, resilient and concurrent data fetching
# with a multi-provider fallback system, an intelligent cache, professional logging, and a comprehensive HTML reporting system.

import time
import os
import uuid
from datetime import date, datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import asyncio
import yaml
import hashlib
import pyarrow
import json
import random
from io import StringIO
from typing import List, Dict, Optional, Any, Literal
from pathlib import Path

# --- Libraries for Data Sources ---
import requests
from alpha_vantage.timeseries import TimeSeries
from twelvedata import TDClient
from tiingo import TiingoClient
# ------------------------------------

from pydantic import BaseModel, Field, field_validator, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pythonjsonlogger import jsonlogger
from jinja2 import Environment, FileSystemLoader
from sizing.base import IPositionSizer
from sizing.fixed_fraction import FixedFractionSizer
from execution_model import VolumeShareSlippageModel
from audit_manager import archive_logs_to_s3
from observability import BACKTEST_DURATION, TRADES_EXECUTED, start_metrics_server, CACHE_HITS, CACHE_MISSES
from ai.client import AIClient, GeminiAIClient, MockAIClient, MalformedResponseError
import backtrader as bt
import pandas as pd
import yfinance as yf


# --- [Refactored] Configuration Models ---
class GeminiConfig(BaseModel):
    enable: bool = False
    mode: Literal["mock", "production"] = "mock"
    api_key_env_var: str
    model_name: str
    request_timeout: int = 90
    audit_log_retention_days: int = 30
    max_concurrent_requests: int = 5
    prompts: Dict[str, str]

class ExecutionConfig(BaseModel):
    impact_coefficient: float = Field(..., gt=0)
    max_volume_share: float = Field(..., gt=0, le=1.0)
    min_trade_notional: float = Field(default=1.0, gt=0)

class ObservabilityConfig(BaseModel):
    metrics_port: int = Field(..., gt=1023)

class AuditConfig(BaseModel):
    s3_bucket_name: str

class PositionSizerConfig(BaseModel):
    method: str
    parameters: Dict[str, Any] = {}

class OptimizerConfig(BaseModel):
    study_name: str
    n_trials: int
    parameters: Dict[str, Any]

# --- [新增] Data Source & Network Models ---
class ProviderConfig(BaseModel):
    api_key: Optional[str] = None

class ProxyConfig(BaseModel):
    enabled: bool = False
    http: Optional[str] = None
    https: Optional[str] = None

class NetworkConfig(BaseModel):
    user_agent: Optional[str] = None
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)

class DataSourcesConfig(BaseModel):
    priority: List[str]
    providers: Dict[str, ProviderConfig]
    network: NetworkConfig

class StrategyConfig(BaseModel):
    start_date: date
    end_date: date
    asset_universe: List[str] = Field(..., min_items=1)
    market_breadth_tickers: List[str] = Field(..., min_items=1)
    sma_period: int = Field(..., gt=0)
    rsi_period: int = Field(..., gt=0)
    rsi_overbought_threshold: float = Field(..., ge=0, le=100)
    opportunity_score_threshold: float = Field(..., ge=0, le=100)
    vix_high_threshold: float = Field(..., gt=0)
    vix_low_threshold: float = Field(..., gt=0)
    capital_modifier_high_vix: float
    capital_modifier_normal_vix: float
    capital_modifier_low_vix: float
    initial_cash: float = Field(..., gt=0)
    commission_rate: float = Field(..., ge=0)
    log_level: str
    gemini_config: GeminiConfig
    data_sources: DataSourcesConfig
    ai_mode: Literal["off", "raw", "processed"] = "processed"
    walk_forward: Dict[str, Any]
    execution_model: ExecutionConfig
    position_sizer: PositionSizerConfig
    audit: AuditConfig
    observability: ObservabilityConfig
    optimizer: OptimizerConfig
    max_total_allocation: float = Field(default=1.0, gt=0, le=1.0)

    @field_validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        if 'start_date' in values.data and v <= values.data['start_date']:
            raise ValueError('end_date must be strictly after start_date')
        return v

# --- Custom Exception for Data Layer ---
class DataFetchError(Exception):
    pass

# --- [重大重构] Data Management Layer ---
class DataManager:
    def __init__(self, config: StrategyConfig, cache_dir: str = "data_cache"):
        self.config = config
        self.ds_config = config.data_sources
        self.cache_dir = cache_dir
        self.logger = logging.getLogger("PhoenixProject.DataManager")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"DataManager initialized. Cache directory: '{self.cache_dir}'.")
        self.session = self._create_session()
        self.tiingo_client = self._create_tiingo_client()

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        network_config = self.ds_config.network
        if network_config.user_agent:
            session.headers['User-Agent'] = network_config.user_agent
            self.logger.info("Global User-Agent set for all requests.")
        if network_config.proxy.enabled:
            proxies = {'http': network_config.proxy.http, 'https': network_config.proxy.https}
            session.proxies = proxies
            self.logger.info("Global proxy enabled for all requests.")
        return session

    def _create_tiingo_client(self) -> Optional[TiingoClient]:
        api_key = self.ds_config.providers.get("tiingo", ProviderConfig()).api_key
        if not api_key or "YOUR_KEY" in api_key:
            return None
        return TiingoClient({'api_key': api_key, 'session': self.session})

    # --- Individual Data Fetcher Methods ---

    async def _fetch_from_alpha_vantage(self, tickers: List[str], start: date, end: date) -> Optional[pd.DataFrame]:
        api_key = self.ds_config.providers.get("alpha_vantage", ProviderConfig()).api_key
        if not api_key or "YOUR_KEY" in api_key:
            self.logger.warning("Alpha Vantage API key not provided. Skipping.")
            return None
            
        try:
            all_dfs = []
            # [修正]：移除了不支持的 session 参数
            ts = TimeSeries(key=api_key, output_format='pandas') 
            for ticker in tickers:
                self.logger.debug(f"Fetching {ticker} from Alpha Vantage...")
                data, _ = await asyncio.to_thread(ts.get_daily_adjusted, symbol=ticker, outputsize='full')
                data = data.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '6. volume': 'Volume'})
                data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
                data.index = pd.to_datetime(data.index)
                data = data[(data.index.date >= start) & (data.index.date <= end)]
                data.columns = pd.MultiIndex.from_product([data.columns, [ticker]])
                all_dfs.append(data)
                await asyncio.sleep(15)
            
            if not all_dfs: return None
            final_df = pd.concat(all_dfs, axis=1).sort_index()
            final_df.index.name = 'Date'
            return final_df
        except Exception as e:
            self.logger.error(f"Alpha Vantage fetch failed: {e}")
            return None

    async def _fetch_from_twelvedata(self, tickers: List[str], start: date, end: date) -> Optional[pd.DataFrame]:
        api_key = self.ds_config.providers.get("twelvedata", ProviderConfig()).api_key
        if not api_key or "YOUR_KEY" in api_key:
            self.logger.warning("Twelve Data API key not provided. Skipping.")
            return None

        try:
            # TDClient 不直接支持 session, 但其内部使用 requests, 会受全局代理影响(如果设置)
            td = TDClient(apikey=api_key)
            all_dfs = []
            for ticker in tickers:
                self.logger.debug(f"Fetching {ticker} from Twelve Data...")
                ts = await asyncio.to_thread(
                    td.time_series, symbol=ticker, interval="1day",
                    start_date=start.isoformat(), end_date=end.isoformat(), outputsize=5000
                )
                df = ts.as_pandas()
                df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
                all_dfs.append(df)
            
            if not all_dfs: return None
            final_df = pd.concat(all_dfs, axis=1).sort_index(ascending=False)
            final_df.index = pd.to_datetime(final_df.index)
            final_df.index.name = 'Date'
            return final_df
        except Exception as e:
            self.logger.error(f"Twelve Data fetch failed: {e}")
            return None

    async def _fetch_from_tiingo(self, tickers: List[str], start: date, end: date) -> Optional[pd.DataFrame]:
        if not self.tiingo_client:
            self.logger.warning("Tiingo client not initialized (API key missing). Skipping.")
            return None
        
        try:
            self.logger.debug(f"Fetching {len(tickers)} tickers from Tiingo one by one...")
            all_dfs = []
            for ticker in tickers:
                # [修正]: 逐个获取股票的 OHLCV 数据
                df = await asyncio.to_thread(
                    self.tiingo_client.get_dataframe, ticker,
                    frequency='daily',
                    startDate=start.isoformat(), endDate=end.isoformat()
                )
                df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
                all_dfs.append(df)

            if not all_dfs: return None
            final_df = pd.concat(all_dfs, axis=1).sort_index()
            final_df.index = pd.to_datetime(final_df.index)
            final_df.index.name = 'Date'
            return final_df
        except Exception as e:
            self.logger.error(f"Tiingo fetch failed: {e}")
            return None

    async def _fetch_from_yfinance(self, tickers: List[str], start: date, end: date) -> Optional[pd.DataFrame]:
        self.logger.info(f"Attempting fallback to yfinance for tickers: {tickers}...")
        try:
            df = await asyncio.to_thread(
                yf.download, tickers=tickers, start=start, end=end,
                auto_adjust=True, progress=False, group_by='ticker', session=self.session
            )
            if df.empty: return None
            
            if len(tickers) == 1:
                df.columns = pd.MultiIndex.from_product([df.columns, tickers])
            
            df.index.name = 'Date'
            return df.rename(columns=lambda c: c.capitalize(), level=0)
        except Exception as e:
            self.logger.error(f"yfinance fallback fetch failed: {e}")
            return None

    # --- Main Data Orchestration ---
    
    async def get_asset_universe_data(self) -> Optional[pd.DataFrame]:
        params = {'tickers': sorted(self.config.asset_universe), 'start': self.config.start_date, 'end': self.config.end_date}
        cache_filename = self._generate_cache_filename("asset_universe_data", params)
        cache_path = os.path.join(self.cache_dir, cache_filename)

        if os.path.exists(cache_path):
            self.logger.info(f"Loading asset universe data from cache: {cache_path}")
            CACHE_HITS.inc()
            return pd.read_parquet(cache_path).set_index('Date')

        self.logger.info("Asset universe cache not found. Fetching fresh data from providers...")
        CACHE_MISSES.inc()

        data = None
        for provider in self.ds_config.priority:
            self.logger.info(f"--- Attempting data fetch with provider: {provider.upper()} ---")
            if provider == "alpha_vantage": data = await self._fetch_from_alpha_vantage(**params)
            elif provider == "twelvedata": data = await self._fetch_from_twelvedata(**params)
            elif provider == "tiingo": data = await self._fetch_from_tiingo(**params)
            elif provider == "yfinance": data = await self._fetch_from_yfinance(**params)
            
            if data is not None and not data.empty:
                self.logger.info(f"Successfully fetched data from {provider.upper()}.")
                data.columns = pd.MultiIndex.from_tuples([(val[0], val[1].capitalize()) for val in data.columns])
                await self._write_cache_direct(cache_path, data, provider, params)
                return data
            else:
                self.logger.warning(f"Provider {provider.upper()} failed to return data. Trying next provider...")
        
        self.logger.critical("All data providers failed. Could not fetch asset universe data.")
        return None

    async def get_vix_data(self) -> Optional[pd.Series]:
        df = await self._fetch_from_yfinance(['^VIX'], self.config.start_date, self.config.end_date)
        return df['Close']['^VIX'] if df is not None and not df.empty else None

    async def get_treasury_yield_data(self) -> Optional[pd.Series]:
        df = await self._fetch_from_yfinance(['^TNX'], self.config.start_date, self.config.end_date)
        return df['Close']['^TNX'] if df is not None and not df.empty else None
    
    async def get_market_breadth_data(self) -> Optional[pd.Series]:
        params = {'tickers': sorted(self.config.market_breadth_tickers), 'start': self.config.start_date, 'end': self.config.end_date}
        all_prices_df_container = await self._fetch_from_yfinance(**params)
        
        if all_prices_df_container is None or 'Close' not in all_prices_df_container.columns.get_level_values(0):
            self.logger.warning("Could not fetch market breadth data.")
            return None
        
        all_prices_df = all_prices_df_container['Close']
        smas = all_prices_df.rolling(window=self.config.sma_period).mean()
        is_above_sma = all_prices_df > smas
        breadth_series = (is_above_sma.sum(axis=1) / all_prices_df.notna().sum(axis=1)).fillna(0)
        breadth_series.name = 'market_breadth_indicator'
        return breadth_series

    async def get_aligned_data(self) -> Optional[Dict[str, pd.DataFrame | pd.Series]]:
        self.logger.info("--- Starting data alignment and sanitization ---")
        asset_df = await self.get_asset_universe_data()
        if asset_df is None or asset_df.empty:
            self.logger.critical("Cannot create master index: Asset universe data is missing.")
            return None
        master_index = asset_df.index
        self.logger.info(f"Master trading index created with {len(master_index)} days.")
        self.logger.info("Fetching auxiliary data streams concurrently...")
        tasks = { "vix": self.get_vix_data(), "treasury_yield": self.get_treasury_yield_data(), "market_breadth": self.get_market_breadth_data() }
        results = await asyncio.gather(*tasks.values())
        data_streams = dict(zip(tasks.keys(), results))
        sanitized_streams = {}
        for name, series in data_streams.items():
            if series is not None and not series.empty:
                 aligned_series = series.reindex(master_index).ffill().bfill()
                 sanitized_streams[name] = aligned_series
                 if aligned_series.isnull().any(): self.logger.warning(f"{name} data still contains NaNs after sanitization.")
            else:
                 sanitized_streams[name] = pd.Series(0, index=master_index, dtype=float)
                 self.logger.warning(f"Data for '{name}' could not be fetched. Using a series of zeros.")

        self.logger.info("--- Data alignment and sanitization complete ---")
        return {"asset_universe_df": asset_df, **sanitized_streams}
    
    def _generate_cache_filename(self, prefix: str, params: Dict) -> str:
        param_string = json.dumps(params, sort_keys=True, default=str)
        param_hash = hashlib.sha256(param_string.encode()).hexdigest()[:16]
        return f"{prefix}_{param_hash}.parquet"

    async def _write_cache_direct(self, cache_path: str, data: pd.DataFrame | pd.Series, source: str, params: Dict):
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        try:
            data.reset_index().to_parquet(tmp_path, engine='pyarrow', index=False)
            os.replace(tmp_path, cache_path)
            self.logger.info(f"Saved fresh data to cache: {cache_path}")
            meta_path = Path(cache_path).with_suffix('.meta.json')
            param_hash = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:16]
            metadata = { "params_hash": param_hash, "created_utc": datetime.utcnow().isoformat(), "source": source, "params": {k: str(v) for k, v in params.items()} }
            with open(meta_path, 'w', encoding='utf-8') as f: json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to write cache file {cache_path}: {e}")
            if os.path.exists(tmp_path): os.remove(tmp_path)


# --- Cognitive Engine Layer ---
class CognitiveEngine:
    def __init__(self, config: StrategyConfig, sentiment_data: Optional[Dict[date, float]] = None, asset_analysis_data: Optional[Dict[date, Dict]] = None, ai_mode: str = "processed"):
        self.config = config
        self.logger = logging.getLogger("PhoenixProject.CognitiveEngine")
        self.risk_manager = RiskManager(config, sentiment_data)
        self.portfolio_constructor = PortfolioConstructor(config, asset_analysis_data, mode=config.ai_mode)
        self.position_sizer = self._create_sizer(config.position_sizer)

    def _create_sizer(self, sizer_config: PositionSizerConfig) -> IPositionSizer:
        method = sizer_config.method
        params = sizer_config.parameters
        self.logger.info(f"Initializing position sizer: '{method}' with params: {params}")
        if method == "fixed_fraction":
            return FixedFractionSizer(**params)
        else:
            raise ValueError(f"Unknown position sizer method: {method}")

    def determine_allocations(self, candidate_analysis: List[Dict], current_vix: float, current_date: date) -> List[Dict]:
        self.logger.info("--- [Cognitive Engine Call: Marshal Coordination] ---")
        capital_modifier = self.risk_manager.get_capital_modifier(current_vix, current_date)
        worthy_targets = self.portfolio_constructor.identify_opportunities(candidate_analysis, current_date)
        effective_max_allocation = self.config.max_total_allocation * capital_modifier
        battle_plan = self.position_sizer.size_positions(worthy_targets, effective_max_allocation)
        
        self.logger.info("--- [Cognitive Engine's Final Battle Plan] ---")
        final_total_allocation = sum(d['capital_allocation_pct'] for d in battle_plan)
        self.logger.info(f"Final planned capital deployment: {final_total_allocation:.2%}")
        for deployment in battle_plan: self.logger.info(f"- Asset: {deployment['ticker']}, Deploy Capital: {deployment['capital_allocation_pct']:.2%}")
        return battle_plan

    def calculate_opportunity_score(self, current_price: float, current_sma: float, current_rsi: float) -> float:
        return self.portfolio_constructor.calculate_opportunity_score(current_price, current_sma, current_rsi)

class RiskManager:
    def __init__(self, config: StrategyConfig, sentiment_data: Optional[Dict[date, float]] = None):
        self.config = config
        self.sentiment_data = sentiment_data if sentiment_data is not None else {}
        self.logger = logging.getLogger("PhoenixProject.RiskManager")
        if self.sentiment_data: self.logger.info(f"RiskManager initialized with {len(self.sentiment_data)} days of sentiment data.")

    def get_capital_modifier(self, current_vix: float, current_date: date) -> float:
        self.logger.info(f"Assessing risk for {current_date.isoformat()}. VIX: {current_vix:.2f}")
        if current_vix > self.config.vix_high_threshold:
            base_modifier = self.config.capital_modifier_high_vix
            self.logger.info(f"VIX indicates High Fear. Base modifier: {base_modifier:.2f}")
        elif current_vix < self.config.vix_low_threshold:
            base_modifier = self.config.capital_modifier_low_vix
            self.logger.info(f"VIX indicates Low Fear. Base modifier: {base_modifier:.2f}")
        else:
            base_modifier = self.config.capital_modifier_normal_vix
            self.logger.info(f"VIX indicates Normal Fear. Base modifier: {base_modifier:.2f}")
        if not self.sentiment_data: return base_modifier
        sentiment_score = self.sentiment_data.get(current_date, 0.0)
        sentiment_adjustment = 1.0 + (sentiment_score * 0.2)
        final_modifier = base_modifier * sentiment_adjustment
        final_modifier = max(0.0, min(1.1, final_modifier))
        self.logger.info(f"Gemini Sentiment Score: {sentiment_score:.2f}. Final Capital Modifier: {final_modifier:.2%}")
        return final_modifier

class PortfolioConstructor:
    def __init__(self, config: StrategyConfig, asset_analysis_data: Optional[Dict[date, Dict]] = None, mode: str = "processed"):
        self.config = config
        self.asset_analysis_data = asset_analysis_data or {}
        self.logger = logging.getLogger("PhoenixProject.PortfolioConstructor")
        self.mode = mode
        self._ema_state = {}
        self.ema_alpha = 0.2
        self.global_scale = 1.0

    def calculate_opportunity_score(self, current_price: float, current_sma: float, current_rsi: float) -> float:
        if current_sma <= 0: return 0.0
        momentum_score = 50 + 50 * ((current_price / current_sma) - 1)
        if momentum_score > 50 and current_rsi > self.config.rsi_overbought_threshold:
            overbought_intensity = (current_rsi - self.config.rsi_overbought_threshold) / (100 - self.config.rsi_overbought_threshold)
            penalty_factor = 1.0 - (overbought_intensity * 0.5)
            final_score = momentum_score * penalty_factor
        else:
            final_score = momentum_score
        return max(0.0, min(100.0, final_score))
        
    def _sanitize_ai_output(self, raw: Dict) -> tuple[float, float]:
        try:
            f = float(raw.get("adjustment_factor", 1.0))
            c = float(raw.get("confidence", 0.0))
        except (ValueError, TypeError): return 1.0, 0.0
        f = max(0.3, min(2.0, f))
        c = max(0.0, min(1.0, c))
        return f, c

    def _effective_factor(self, ticker: str, reported_factor: float, confidence: float) -> float:
        effective = 1.0 + confidence * (reported_factor - 1.0)
        prev = self._ema_state.get(ticker, 1.0)
        smoothed = prev * (1 - self.ema_alpha) + effective * self.ema_alpha
        self._ema_state[ticker] = smoothed
        final = smoothed * self.global_scale
        final = max(0.5, min(1.2, final))
        return final

    def identify_opportunities(self, candidate_analysis: List[Dict], current_date: date) -> List[Dict]:
        self.logger.info("PortfolioConstructor is identifying high-quality opportunities...")
        adjusted_candidates = []
        daily_asset_analysis = self.asset_analysis_data.get(current_date, {})
        for candidate in candidate_analysis:
            ticker = candidate["ticker"]
            original_score = candidate["opportunity_score"]
            final_factor = 1.0
            confidence = 0.0
            if self.mode in ["raw", "processed"]:
                raw_analysis = daily_asset_analysis.get(ticker, {})
                reported_factor, confidence = self._sanitize_ai_output(raw_analysis)
                if self.mode == "raw": final_factor = reported_factor
                else: final_factor = self._effective_factor(ticker, reported_factor, confidence)
            adjusted_score = candidate["opportunity_score"] * final_factor
            adjusted_candidates.append({**candidate, "adjusted_score": adjusted_score, "ai_factor": final_factor, "ai_confidence": confidence})
            if final_factor != 1.0 and self.mode != 'off': self.logger.info(f"AI Insight for {ticker} (Mode: {self.mode}): Conf={confidence:.2f}, FinalFactor={final_factor:.3f}. Score: {original_score:.2f} -> {adjusted_score:.2f}")
        
        worthy_targets = [res for res in adjusted_candidates if res["adjusted_score"] > self.config.opportunity_score_threshold]
        if not worthy_targets:
            self.logger.info("PortfolioConstructor: No opportunities met the threshold.")
        else:
            self.logger.info(f"PortfolioConstructor: Identified {len(worthy_targets)} high-quality opportunities.")
        return worthy_targets

# --- Strategy Execution Layer ---
class RomanLegionStrategy(bt.Strategy):
    params = (
        ('config', None), ('vix_data', None), ('treasury_yield_data', None), 
        ('market_breadth_data', None), ('sentiment_data', None), ('asset_analysis_data', None),
    )

    def __init__(self):
        self.logger = logging.getLogger("PhoenixProject.Strategy")
        if self.p.config is None: raise ValueError("StrategyConfig object not provided!")
        self.config = self.p.config
        self.execution_model = VolumeShareSlippageModel(
            impact_coefficient=self.config.execution_model.impact_coefficient,
            max_volume_share=self.config.execution_model.max_volume_share,
            min_trade_notional=self.config.execution_model.min_trade_notional
        )
        self.cognitive_engine = CognitiveEngine(self.config, self.p.sentiment_data, self.p.asset_analysis_data, ai_mode=self.config.ai_mode)
        self.data_map = {d._name: d for d in self.datas}
        self.sma_indicators = {d._name: bt.indicators.SimpleMovingAverage(d.close, period=self.config.sma_period) for d in self.datas}
        self.rsi_indicators = {d._name: bt.indicators.RSI(d.close, period=self.config.rsi_period) for d in self.datas}
        self.vix_lookup = {pd.Timestamp(k).date(): float(v) for k, v in self.p.vix_data.items()}
        self.yield_lookup = {pd.Timestamp(k).date(): float(v) for k, v in self.p.treasury_yield_data.items()}
        self.breadth_lookup = {pd.Timestamp(k).date(): float(v) for k, v in self.p.market_breadth_data.items()}

    def start(self):
        self.logger.info(f"{self.datas[0].datetime.date(0).isoformat()}: [Legion Commander]: Awaiting daily orders...")

    def next(self):
        if len(self.datas[0]) < self.config.sma_period: return
        current_date = self.datas[0].datetime.date(0)
        self.logger.info(f"--- {current_date.isoformat()}: Daily Rebalancing Briefing ---")
        current_vix = self.vix_lookup.get(current_date)
        current_yield = self.yield_lookup.get(current_date)
        current_breadth = self.breadth_lookup.get(current_date)
        if current_vix is None:
            self.logger.warning(f"Critical data VIX missing for {current_date}, halting for the day.")
            return
        self.logger.info(f"VIX Index: {current_vix:.2f}, 10Y Yield: {current_yield:.2f if current_yield else 'N/A'}%, Market Breadth: {current_breadth:.2% if current_breadth else 'N/A'}")
        candidate_analysis = [{
            "ticker": ticker,
            "opportunity_score": self.cognitive_engine.calculate_opportunity_score(
                d.close[0], self.sma_indicators[ticker][0], self.rsi_indicators[ticker][0])
        } for ticker, d in self.data_map.items()]

        battle_plan = self.cognitive_engine.determine_allocations(candidate_analysis, current_vix, current_date)
        self.execution_model.rebalance(self, battle_plan)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            TRADES_EXECUTED.inc()
            dt = self.datas[0].datetime.date(0).isoformat()
            if order.isbuy(): self.logger.info(f"{dt}: BUY EXECUTED, {order.data._name}, Size: {order.executed.size}, Price: {order.executed.price:.2f}")
            elif order.issell(): self.logger.info(f"{dt}: SELL EXECUTED, {order.data._name}, Size: {order.executed.size}, Price: {order.executed.price:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.logger.warning(f"{self.datas[0].datetime.date(0).isoformat()}: Order for {order.data._name} failed: {order.getstatusname()}")

# --- Reporting Engine ---
async def generate_ai_report(ai_client: Optional[AIClient], context: Dict) -> tuple[str | None, str | None]:
    if not ai_client: return None, None
    logger = logging.getLogger("PhoenixProject.ReportGenerator")
    logger.info("Generating AI Marshal's Report...")
    try:
        report_text, audit_path = await ai_client.generate_summary_report(context)
        logger.info("Successfully generated AI Marshal's Report.")
        return report_text, audit_path
    except Exception as e:
        logger.error(f"Failed to generate AI report: {e}")
        return "## Marshal's Debriefing Failed ##\n\nAn error occurred during communication with the AI Command. The quantitative report below remains accurate.", None

async def generate_html_report(cerebro: bt.Cerebro, strat: RomanLegionStrategy, ai_client: Optional[AIClient] = None, audit_files: List[str] = None, report_filename="phoenix_report.html"):
    logger = logging.getLogger("PhoenixProject.ReportGenerator")
    logger.info("Generating HTML after-action report...")
    trade_analyzer = getattr(strat.analyzers, 'trade_analyzer', None)
    ta = trade_analyzer.get_analysis() if trade_analyzer else {}
    total_trades = ta.get('total', {}).get('total', 0)
    winning_trades = ta.get('won', {}).get('total', 0)
    losing_trades = ta.get('lost', {}).get('total', 0)
    win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0
    sharpe_analyzer = getattr(strat.analyzers, 'sharpe_ratio', None)
    sharpe_analysis = sharpe_analyzer.get_analysis() if sharpe_analyzer else {}
    sharpe_ratio = sharpe_analysis.get('sharperatio', None)
    returns_analyzer = getattr(strat.analyzers, 'returns', None)
    returns_analysis = returns_analyzer.get_analysis() if returns_analyzer else {}
    drawdown_analyzer = getattr(strat.analyzers, 'drawdown', None)
    dd_analysis = drawdown_analyzer.get_analysis() if drawdown_analyzer else {}
    context = {
        "final_value": cerebro.broker.getvalue(), "total_return": returns_analysis.get('rtot', 0.0),
        "sharpe_ratio": sharpe_ratio, "max_drawdown": dd_analysis.get('max', {}).get('drawdown', 0.0),
        "total_trades": total_trades, "winning_trades": winning_trades, "losing_trades": losing_trades, "win_rate": win_rate, "report_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "plot_filename": "phoenix_plot.png"
    }
    context['ai_summary_report'], summary_audit_file = await generate_ai_report(ai_client, context)
    
    all_audit_files = audit_files or []
    if summary_audit_file: all_audit_files.append(os.path.basename(summary_audit_file))
    context['audit_files'] = sorted(list(set(all_audit_files)))
    try:
        plot_path = Path(report_filename).with_suffix('.png')
        logger.info(f"Saving backtest plot to {plot_path}...")
        figures = cerebro.plot(style='candlestick', barup='green', bardown='red', iplot=False)
        if figures: 
            figures[0][0].savefig(plot_path, dpi=300)
            context['plot_filename'] = plot_path.name
        else: raise RuntimeError("Cerebro plot returned no figures.")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
        context['plot_filename'] = None
    try:
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template("report_template.html")
        html_output = template.render(context)
        with open(report_filename, 'w', encoding='utf-8') as f: f.write(html_output)
        logger.info(f"Successfully generated HTML report: {report_filename}")
    except Exception as e: logger.error(f"Failed to generate HTML report: {e}")

# --- AI Pre-computation Helpers ---
def fetch_mock_news_for_date(date_obj: date) -> List[str]:
    base_headlines = ["Global markets show mixed signals...", "Tech sector rally continues...", "New geopolitical tensions...", "Federal Reserve hints at a more hawkish stance..."]
    if date_obj.weekday() == 0: base_headlines.append("Weekend uncertainty weighs on market open.")
    elif date_obj.weekday() == 4: base_headlines.append("Positive jobs report boosts investor confidence.")
    random.shuffle(base_headlines)
    return base_headlines[:3]

async def precompute_sentiments(ai_client: AIClient, dates: List[date]) -> tuple[Dict[date, float], List[str]]:
    logger = logging.getLogger("PhoenixProject.SentimentPrecomputer")
    cache_file = Path("data_cache") / "sentiment_cache.json"    
    cached_sentiments_str = {}
    if cache_file.exists():        
        logger.info(f"Found sentiment cache file: {cache_file}")
        try:
            with open(cache_file, 'r') as f: cached_sentiments_str = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Cache file {cache_file} is corrupted. Re-computing all sentiments.")
            cached_sentiments_str = {}

    cached_dates = {date.fromisoformat(k) for k in cached_sentiments_str.keys()}
    required_dates = set(dates)
    missing_dates = sorted(list(required_dates - cached_dates))
    all_audit_files = []
    if missing_dates:
        logger.info(f"Sentiment cache miss for {len(missing_dates)} days. Fetching incrementally...")
        semaphore = getattr(ai_client, 'semaphore', None)
        async def fetch_single_sentiment(date_obj: date, sem: asyncio.Semaphore | None):
            async with sem if sem else asyncio.Semaphore():
                try:
                    mock_headlines = fetch_mock_news_for_date(date_obj)
                    analysis, audit_path = await ai_client.get_market_sentiment(mock_headlines)
                    logger.info(f"Sentiment for {date_obj.isoformat()}: {analysis.get('sentiment_score', 0.0):.2f}")
                    return date_obj, analysis.get('sentiment_score', 0.0), audit_path
                except (MalformedResponseError, Exception) as e:
                    logger.error(f"Could not compute sentiment for {date_obj.isoformat()}: {e}. Defaulting to neutral (0.0).")
                    return date_obj, 0.0, None

        tasks = [fetch_single_sentiment(date_obj, semaphore) for date_obj in missing_dates]
        new_results = await asyncio.gather(*tasks)

        for d, score, audit_path in new_results:
            if d: cached_sentiments_str[d.isoformat()] = score
            if audit_path: all_audit_files.append(os.path.basename(audit_path))

        with open(cache_file, 'w') as f: json.dump(cached_sentiments_str, f, indent=2)
        logger.info(f"Updated and saved sentiment cache to: {cache_file}")
    else:
        logger.info("Sentiment cache is fully populated. No new data needed.")
    return {dt: cached_sentiments_str[dt.isoformat()] for dt in required_dates}, all_audit_files

async def precompute_asset_analyses(ai_client: AIClient, dates: List[date], asset_universe: List[str]) -> tuple[Dict[date, Dict], List[str]]:
    logger = logging.getLogger("PhoenixProject.AssetAnalysisPrecomputer")
    cache_file = Path("data_cache") / "asset_analysis_cache.json"
    cached_data_str = {}
    if cache_file.exists():
        logger.info(f"Found asset analysis cache file: {cache_file}")
        try:
            with open(cache_file, 'r') as f: cached_data_str = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Cache file {cache_file} is corrupted. Re-computing all asset analyses.")
            cached_data_str = {}
    cached_dates = {date.fromisoformat(k) for k in cached_data_str.keys()}
    required_dates = set(dates)
    missing_dates = sorted(list(required_dates - cached_dates))
    all_audit_files = []
    if missing_dates:
        logger.info(f"Asset analysis cache miss for {len(missing_dates)} days. Fetching incrementally...")
        semaphore = getattr(ai_client, 'semaphore', None)
        async def fetch_single_analysis(date_obj: date, sem: asyncio.Semaphore | None):
            async with sem if sem else asyncio.Semaphore():
                try:
                    daily_batch_analysis, audit_path = await ai_client.get_batch_asset_analysis(asset_universe, date_obj)
                    logger.info(f"Successfully computed batch analysis for {date_obj.isoformat()}.")
                    return date_obj, daily_batch_analysis, audit_path
                except Exception as e:
                    logger.error(f"Could not compute batch analysis for {date_obj.isoformat()}: {e}. Defaulting all tickers to neutral for this day.")
                    default_analysis = {ticker: {"adjustment_factor": 1.0, "confidence": 0.0, "reasoning": "Batch computation failed"} for ticker in asset_universe}
                    return date_obj, default_analysis, None

        tasks = [fetch_single_analysis(date_obj, semaphore) for date_obj in missing_dates]
        new_results = await asyncio.gather(*tasks)

        for d, analysis, audit_path in new_results:
            if d: cached_data_str[d.isoformat()] = analysis
            if audit_path: all_audit_files.append(os.path.basename(audit_path))

        with open(cache_file, 'w') as f: json.dump(cached_data_str, f)
        logger.info(f"Updated and saved asset analysis cache to: {cache_file}")
    else:
        logger.info("Asset analysis cache is fully populated. No new data needed.")
    return {dt: cached_data_str[dt.isoformat()] for dt in required_dates}, all_audit_files

# --- Main Execution Engine ---
async def run_single_backtest(config: StrategyConfig, all_aligned_data: Dict, ai_client: Optional[AIClient] = None):
    logger = logging.getLogger("PhoenixProject")
    logger.info(f"--- Launching 'Phoenix Project' in SINGLE BACKTEST mode (AI: {config.ai_mode.upper()}) ---")

    master_dates = [dt.to_pydatetime().date() for dt in all_aligned_data["asset_universe_df"].index]
    sentiment_lookup = {}
    asset_analysis_lookup = {}
    run_audit_files = []
    if ai_client and config.ai_mode != "off":
        sentiment_lookup, sentiment_audits = await precompute_sentiments(ai_client, master_dates)
        asset_analysis_lookup, asset_audits = await precompute_asset_analyses(ai_client, master_dates, config.asset_universe)
        run_audit_files.extend(sentiment_audits + asset_audits)

    cerebro = bt.Cerebro()
    try:
        all_data_df = all_aligned_data["asset_universe_df"]
        if all_data_df is None or all_data_df.empty: raise ValueError("Asset universe data is empty.")
        unique_tickers = all_data_df.columns.get_level_values(0).unique()
        for ticker in unique_tickers:
            ticker_df = all_data_df[ticker].copy()
            ticker_df.columns = [col.lower() for col in ticker_df.columns]
            ticker_df.dropna(inplace=True)
            if not ticker_df.empty:
                data_feed = bt.feeds.PandasData(dataname=ticker_df, name=ticker)
                cerebro.adddata(data_feed)
    except Exception as e:
        logger.critical(f"A critical error occurred during data loading: {e}. Aborting.")
        return

    if not cerebro.datas:
        logger.critical("Failed to load data for any asset. Aborting operation.")
        return
        
    cerebro.addstrategy(
        RomanLegionStrategy, config=config, vix_data=all_aligned_data["vix"],
        treasury_yield_data=all_aligned_data["treasury_yield"], market_breadth_data=all_aligned_data["market_breadth"],
        sentiment_data=sentiment_lookup, asset_analysis_data=asset_analysis_lookup
    )
    cerebro.broker.setcash(config.initial_cash)
    cerebro.broker.setcommission(commission=config.commission_rate)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days, compression=1, annualize=True, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    results = cerebro.run()
    strat = results[0]
    await generate_html_report(cerebro, strat, ai_client, run_audit_files, report_filename=f"phoenix_report_{config.ai_mode}_single.html")

async def main():
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config_params = yaml.safe_load(f)
        config = StrategyConfig(**config_params)
    except (FileNotFoundError, ValidationError) as e:
        print(f"CRITICAL: Error loading or validating 'config.yaml': {e}. Aborting.")
        return

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    class RunIdFilter(logging.Filter):
        def filter(self, record):
            record.run_id = run_id
            return True

    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logger = logging.getLogger("PhoenixProject")
    logger.setLevel(log_level)
    logger.handlers。clear()

    formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(run_id)s %(message)s')
    log_dir = "logs"; os.makedirs(log_dir, exist_ok=True)
    log_filename = f"phoenix_project_{datetime.now().strftime('%Y%m%d')}.log"
    stream_handler = logging.StreamHandler(); stream_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(os.path.join(log_dir, log_filename), maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'); file_handler.setFormatter(formatter)

    for handler in [stream_handler, file_handler]:
        handler.addFilter(RunIdFilter())
        logger.addHandler(handler)
    
    from optimizer import Optimizer
    logger.info("Phoenix Project logging system initialized in JSON format.", extra={'run_id': run_id})
    start_metrics_server(config.observability.metrics_port)

    ai_client: Optional[AIClient] = None
    if config.gemini_config。enable:
        try:
            gem_cfg = config.gemini_config.model_dump()
            ai_client = MockAIClient(gem_cfg) if gem_cfg['mode'] == 'mock' else GeminiAIClient(gem_cfg)
            logger.info(f"AI Client has been enabled and initialized in '{config.gemini_config。mode}' mode.")
        except Exception as e:
            logger.error(f"Failed to initialize AI Client: {e}. Continuing without AI features.")

    start_time = time.time()
    try:
        data_manager = DataManager(config)
        all_aligned_data = await data_manager.get_aligned_data()
        if not all_aligned_data:
            logger.critical("Failed to get aligned data. Aborting operation.")
            return

        if config.walk_forward.get('enabled', False):
            optimizer = Optimizer(config, all_aligned_data, ai_client)
            optimizer.run_optimization()
        else:
            await run_single_backtest(config, all_aligned_data, ai_client)
    finally:
        duration = time.time() - start_time
        BACKTEST_DURATION.set(duration)
        logger.info(f"Backtest completed in {duration:.2f} seconds.")

    if config.audit.s3_bucket_name and config.audit.s3_bucket_name != "your-phoenix-project-audit-logs-bucket":
        archive_logs_to_s3(source_dir="ai_audit_logs", bucket_name=config.audit.s3_bucket_name)
    
    logger.info("--- Operation Concluded ---")

if __name__ == '__main__':
    asyncio.run(main())
