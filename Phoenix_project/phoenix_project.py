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
from filelock import FileLock
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
from execution.order_manager import OrderManager
from execution.adapters import BacktraderBrokerAdapter
from audit_manager import archive_logs_to_s3
from ai.evidence_fusion import EvidenceFusionEngine, FusedResponseModel
from strategy_handler import StrategyDataHandler
from cognitive.engine import CognitiveEngine
from observability import (BACKTEST_DURATION, TRADES_EXECUTED, start_metrics_server, CACHE_HITS, CACHE_MISSES,
                           PROVIDER_REQUESTS_TOTAL, PROVIDER_ERRORS_TOTAL, PROVIDER_LATENCY_SECONDS)
from ai.ensemble_client import EnsembleAIClient
from ai.prompt_renderer import render_prompt
import backtrader as bt
import pandas as pd
import yfinance as yf


# --- [Refactored] Configuration Models ---
class AIEnsembleConfig(BaseModel):
    enable: bool
    config_file_path: str

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
    api_key_env_var: Optional[str] = None

class ProxyConfig(BaseModel):
    enabled: bool = False
    http: Optional[str] = None
    https: Optional[str] = None

class NetworkConfig(BaseModel):
    user_agent: Optional[str] = None
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_backoff_factor: int = 2

class HealthProbesConfig(BaseModel):
    failure_threshold: int
    cooldown_minutes: int

class DataSourcesConfig(BaseModel):
    priority: List[str]
    providers: Dict[str, ProviderConfig]
    network: NetworkConfig
    health_probes: HealthProbesConfig

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
    ai_ensemble_config: AIEnsembleConfig
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
        self._provider_health = {
            provider: {"failures": 0, "cooldown_until": None}
            for provider in self.ds_config.priority
        }
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
        env_var = self.ds_config.providers.get("tiingo", ProviderConfig()).api_key_env_var
        if not env_var: return None
        api_key = os.getenv(env_var)
        if not api_key:
            self.logger.warning(f"Tiingo API key env var '{env_var}' not set. Skipping.")
            return None
        return TiingoClient({'api_key': api_key, 'session': self.session})

    def _is_provider_healthy(self, provider: str) -> bool:
        health = self._provider_health.get(provider)
        if not health:
            return True # Assume healthy if not tracked
        if health["cooldown_until"] and datetime.utcnow() < health["cooldown_until"]:
            self.logger.warning(f"Provider '{provider}' is in cooldown until {health['cooldown_until'].isoformat()}Z. Skipping.")
            return False
        return True

    def _record_provider_failure(self, provider: str):
        health = self._provider_health[provider]
        health["failures"] += 1
        if health["failures"] >= self.ds_config.health_probes.failure_threshold:
            cooldown = timedelta(minutes=self.ds_config.health_probes.cooldown_minutes)
            health["cooldown_until"] = datetime.utcnow() + cooldown
            self.logger.critical(f"Provider '{provider}' exceeded failure threshold. Placing it in cooldown for {cooldown.total_seconds()/60} minutes.")

    def _record_provider_success(self, provider: str):
        self._provider_health[provider]["failures"] = 0
        self._provider_health[provider]["cooldown_until"] = None

    # --- Individual Data Fetcher Methods ---
    @retry(wait=wait_exponential(multiplier=2, min=2, max=30), stop=stop_after_attempt(3))
    async def _fetch_from_alpha_vantage(self, tickers: List[str], start: date, end: date) -> Optional[pd.DataFrame]:
        env_var = self.ds_config.providers.get("alpha_vantage", ProviderConfig()).api_key_env_var
        if not env_var: return None
        api_key = os.getenv(env_var)
        if not api_key:
            self.logger.warning(f"Alpha Vantage API key env var '{env_var}' not set. Skipping.")
            return None
            
        try:
            all_dfs = []
            ts = TimeSeries(key=api_key, output_format='pandas', timeout=self.ds_config.network.request_timeout) 
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

    @retry(wait=wait_exponential(multiplier=2, min=2, max=30), stop=stop_after_attempt(3))
    async def _fetch_from_twelvedata(self, tickers: List[str], start: date, end: date) -> Optional[pd.DataFrame]:
        env_var = self.ds_config.providers.get("twelvedata", ProviderConfig()).api_key_env_var
        if not env_var: return None
        api_key = os.getenv(env_var)
        if not api_key:
            self.logger.warning(f"Twelve Data API key env var '{env_var}' not set. Skipping.")
            return None

        try:
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

    @retry(wait=wait_exponential(multiplier=2, min=2, max=30), stop=stop_after_attempt(3))
    async def _fetch_from_tiingo(self, tickers: List[str], start: date, end: date) -> Optional[pd.DataFrame]:
        if not self.tiingo_client:
            self.logger.warning("Tiingo client not initialized (API key missing). Skipping.")
            return None
        
        try:
            self.logger.debug(f"Fetching {len(tickers)} tickers from Tiingo one by one...")
            all_dfs = []
            for ticker in tickers:
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

    @retry(wait=wait_exponential(multiplier=2, min=2, max=30), stop=stop_after_attempt(3))
    async def _fetch_from_yfinance(self, tickers: List[str], start: date, end: date) -> Optional[pd.DataFrame]:
        self.logger.info(f"Attempting fallback to yfinance for tickers: {tickers}...")
        try:
            df = await asyncio.to_thread(
                yf.download, tickers=tickers, start=start, end=end, timeout=self.ds_config.network.request_timeout,
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
    async def _fetch_and_cache_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Handles fetching, caching, and fallback for a single ticker."""
        params = {'ticker': ticker, 'start': self.config.start_date, 'end': self.config.end_date}
        cache_filename = self._generate_cache_filename(f"asset_data_{ticker}", params)
        cache_path = os.path.join(self.cache_dir, cache_filename)

        cached_df = None
        fetch_start_date = self.config.start_date

        if os.path.exists(cache_path):
            self.logger.debug(f"Reading cache for '{ticker}' to check for incremental update.")
            cached_df = pd.read_parquet(cache_path).set_index('Date')
            last_cached_date = cached_df.index.max().date()
            
            if last_cached_date >= self.config.end_date:
                self.logger.info(f"Cache for '{ticker}' is up-to-date. Loading from cache.")
                CACHE_HITS.inc()
                return cached_df
            
            self.logger.info(f"Cache for '{ticker}' is outdated (last date: {last_cached_date}). Fetching incrementally.")
            fetch_start_date = last_cached_date + timedelta(days=1)
        else:
            self.logger.info(f"Cache miss for '{ticker}'. Fetching fresh data...")
            CACHE_MISSES.inc()

        fetch_params = {'tickers': [ticker], 'start': fetch_start_date, 'end': self.config.end_date}

        for provider in [p for p in self.ds_config.priority if self._is_provider_healthy(p)]:
            start_time = time.time()
            PROVIDER_REQUESTS_TOTAL.labels(provider=provider).inc()
            self.logger.info(f"Attempting to fetch '{ticker}' with provider: {provider.upper()}")
            try:
                data = None
                if provider == "alpha_vantage": data = await self._fetch_from_alpha_vantage(**fetch_params)
                elif provider == "twelvedata": data = await self._fetch_from_twelvedata(**fetch_params)
                elif provider == "tiingo": data = await self._fetch_from_tiingo(**fetch_params)
                elif provider == "yfinance": data = await self._fetch_from_yfinance(**fetch_params)

                if data is not None and not data.empty:
                    PROVIDER_LATENCY_SECONDS.labels(provider=provider).observe(time.time() - start_time)
                    self.logger.info(f"Successfully fetched data for '{ticker}' from {provider.upper()}.")
                    self._record_provider_success(provider)
                    
                    combined_df = pd.concat([cached_df, data]) if cached_df is not None else data
                    await self._write_cache_direct(cache_path, combined_df, provider, params)
                    return combined_df
                else:
                    raise DataFetchError(f"Provider {provider.upper()} returned no data for {ticker}.")
            except Exception as e:
                self.logger.error(f"Provider {provider.upper()} failed for '{ticker}': {e}")
                PROVIDER_ERRORS_TOTAL.labels(provider=provider).inc()
                PROVIDER_LATENCY_SECONDS.labels(provider=provider).observe(time.time() - start_time)
                self._record_provider_failure(provider)
        
        if cached_df is not None:
            self.logger.warning(f"All providers failed for '{ticker}'. Using stale cached data.")
            return cached_df
            
        self.logger.critical(f"All providers failed for ticker '{ticker}' and no cache exists. It will be excluded.")
        return None

    async def get_asset_universe_data(self) -> Optional[pd.DataFrame]:
        self.logger.info(f"--- Starting CONCURRENT data acquisition for {len(self.config.asset_universe)} assets ---")
        tasks = [self._fetch_and_cache_ticker_data(ticker) for ticker in sorted(self.config.asset_universe)]
        results = await asyncio.gather(*tasks)
        
        all_ticker_dfs = [df for df in results if df is not None]
        if not all_ticker_dfs:
            self.logger.critical("Could not fetch data for ANY ticker in the universe.")
            return None
        
        final_df = pd.concat(all_ticker_dfs, axis=1).sort_index()
        self.logger.info("Successfully acquired and combined data for all available tickers.")
        return final_df

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
    
    def _generate_cache_filename(self, prefix: str, params: Dict[str, Any]) -> str:
        param_string = json.dumps(params, sort_keys=True, default=str)
        param_hash = hashlib.sha256(param_string.encode()).hexdigest()[:16]
        return f"{prefix}_{param_hash}.parquet"

    async def _write_cache_direct(self, cache_path: str, data: pd.DataFrame | pd.Series, source: str, params: Dict):
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        try:
            lock_path = f"{cache_path}.lock"
            lock = FileLock(lock_path)
            with lock:
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
        
        # Setup the new Execution Layer
        broker_adapter = BacktraderBrokerAdapter(self.broker)
        self.order_manager = OrderManager(
            broker_adapter=broker_adapter,
            impact_coefficient=self.config.execution_model.impact_coefficient,
            max_volume_share=self.config.execution_model.max_volume_share,
            min_trade_notional=self.config.execution_model.min_trade_notional
        )
        
        self.cognitive_engine = CognitiveEngine(self.config, self.p.asset_analysis_data, sentiment_data=self.p.sentiment_data, ai_mode=self.config.ai_mode)
        self.data_handler = StrategyDataHandler(
            self, self.config, self.p.vix_data, self.p.treasury_yield_data, self.p.market_breadth_data
        )

    def start(self):
        self.logger.info(f"{self.datas[0].datetime.date(0).isoformat()}: [Legion Commander]: Awaiting daily orders...")

    def next(self):
        if len(self.datas[0]) < self.config.sma_period: return

        daily_data = self.data_handler.get_daily_data_packet(self.cognitive_engine)
        if not daily_data:
            return
        
        self.logger.info(f"--- {daily_data.current_date.isoformat()}: Daily Rebalancing Briefing ---")
        battle_plan = self.cognitive_engine.determine_allocations(daily_data.candidate_analysis, daily_data.current_vix, daily_data.current_date)
        self.order_manager.rebalance(self, battle_plan)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            TRADES_EXECUTED.inc()
            dt = self.datas[0].datetime.date(0).isoformat()
            if order.isbuy(): self.logger.info(f"{dt}: BUY EXECUTED, {order.data._name}, Size: {order.executed.size}, Price: {order.executed.price:.2f}")
            elif order.issell(): self.logger.info(f"{dt}: SELL EXECUTED, {order.data._name}, Size: {order.executed.size}, Price: {order.executed.price:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.logger.warning(f"{self.datas[0].datetime.date(0).isoformat()}: Order for {order.data._name} failed: {order.getstatusname()}")

# --- Reporting Engine ---
async def generate_html_report(cerebro: bt.Cerebro, strat: RomanLegionStrategy, audit_files: List[str] = None, report_filename="phoenix_report.html"):
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
    context['ai_summary_report'] = "Ensemble analysis complete. Review audit logs for detailed AI interactions."
    
    all_audit_files = audit_files or []
    # if summary_audit_file: all_audit_files.append(os.path.basename(summary_audit_file))
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
def fetch_mock_docs_for_ticker(ticker: str) -> List[Dict[str, Any]]:
    # In a real system, this would be a sophisticated RAG pipeline
    return [
        {
            "id": f"NEWS-{ticker}-1",
            "content": f"'{ticker}' shows strong performance in recent quarter. Analyst consensus is 'Buy'."
        }
    ]

async def precompute_asset_analyses(
    ensemble_client: EnsembleAIClient,
    fusion_engine: EvidenceFusionEngine,
    dates: List[date],
    asset_universe: List[str]
) -> Dict[date, Dict]:
    logger = logging.getLogger("PhoenixProject.AssetAnalysisPrecomputer")
    cache_file = Path("data_cache") / "asset_analysis_cache.json"
    cached_data_str = {}
    if cache_file.exists():
        logger.info(f"Found asset analysis cache file: {cache_file}")
        try:
            with open(cache_file, 'r') as f:
                cached_data_str = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Cache file {cache_file} is corrupted. Re-computing all asset analyses.")
            cached_data_str = {}
    
    cached_dates = {date.fromisoformat(k) for k in cached_data_str.keys()}
    required_dates = set(dates)
    missing_dates = sorted(list(required_dates - cached_dates))
    
    if missing_dates:
        logger.info(f"Asset analysis cache miss for {len(missing_dates)} days. Fetching incrementally...")

        async def fetch_and_fuse_for_date(date_obj: date):
            daily_fused_results = {}
            for ticker in asset_universe:
                mock_docs = fetch_mock_docs_for_ticker(ticker)
                
                ensemble_responses = await ensemble_client.get_ensemble_asset_analysis(ticker, mock_docs)
                fused_response = fusion_engine.fuse(ensemble_responses)
                
                # Store a serializable version of the FusedResponse
                daily_fused_results[ticker] = fused_response.dict()
            
            return date_obj, daily_fused_results

        tasks = [fetch_and_fuse_for_date(d) for d in missing_dates]
        new_results = await asyncio.gather(*tasks)

        for d, analysis in new_results:
            if d:
                cached_data_str[d.isoformat()] = analysis

        with open(cache_file, 'w') as f: json.dump(cached_data_str, f, default=str) # Use default=str for datetime
        logger.info(f"Updated and saved asset analysis cache to: {cache_file}")
    else:
        logger.info("Asset analysis cache is fully populated. No new data needed.")

    return {date.fromisoformat(dt_str): analysis for dt_str, analysis in cached_data_str.items() if date.fromisoformat(dt_str) in required_dates}

# --- Main Execution Engine ---
async def run_single_backtest(config: StrategyConfig, all_aligned_data: Dict, ensemble_client: Optional[EnsembleAIClient] = None, fusion_engine: Optional[EvidenceFusionEngine] = None):
    logger = logging.getLogger("PhoenixProject")
    logger.info(f"--- Launching 'Phoenix Project' in SINGLE BACKTEST mode (AI: {config.ai_mode.upper()}) ---")

    master_dates = [dt.to_pydatetime().date() for dt in all_aligned_data["asset_universe_df"].index]
    asset_analysis_lookup = {}
    if ensemble_client and fusion_engine and config.ai_mode != "off":
        asset_analysis_lookup = await precompute_asset_analyses(ensemble_client, fusion_engine, master_dates, config.asset_universe)

    cerebro = bt.Cerebro()
    try:
        all_data_df = all_aligned_data["asset_universe_df"]
        if all_data_df is None or all_data_df.empty: raise ValueError("Asset universe data is empty.")
        unique_tickers = all_data_df.columns.get_level_values(1).unique()
        for ticker in unique_tickers:
            ticker_df = all_data_df.xs(ticker, level=1, axis=1).copy()
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
        sentiment_data={}, asset_analysis_data=asset_analysis_lookup
    )
    cerebro.broker.setcash(config.initial_cash)
    cerebro.broker.setcommission(commission=config.commission_rate)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days, compression=1, annualize=True, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    results = cerebro.run()
    strat = results[0]
    await generate_html_report(cerebro, strat, report_filename=f"phoenix_report_{config.ai_mode}_single.html")

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

    ensemble_client: Optional[EnsembleAIClient] = None
    fusion_engine: Optional[EvidenceFusionEngine] = None

    if config.ai_ensemble_config.enable:
        try:
            ensemble_config_path = config.ai_ensemble_config。config_file_path
            fusion_engine = EvidenceFusionEngine(config_file_path=ensemble_config_path)
            ensemble_client = EnsembleAIClient(config_file_path=ensemble_config_path, run_id=run_id)
        except Exception as e:
            logger.error(f"Failed to initialize AI Ensemble: {e}. Continuing without AI features.")

    start_time = time.time()
    try:
        data_manager = DataManager(config)
        all_aligned_data = await data_manager.get_aligned_data()
        if not all_aligned_data:
            logger.critical("Failed to get aligned data. Aborting operation.")
            return

        if config.walk_forward.get('enabled', False):
            optimizer = Optimizer(config, all_aligned_data, ai_client=ensemble_client, fusion_engine=fusion_engine)
            optimizer.run_optimization()
        else:
            await run_single_backtest(config, all_aligned_data, ensemble_client, fusion_engine)
    finally:
        duration = time.time() - start_time
        BACKTEST_DURATION.set(duration)
        logger.info(f"Backtest completed in {duration:.2f} seconds.")

    if config.audit.s3_bucket_name and config.audit.s3_bucket_name != "your-phoenix-project-audit-logs-bucket":
        archive_logs_to_s3(source_dir="ai_audit_logs", bucket_name=config.audit.s3_bucket_name)
    
    logger.info("--- Operation Concluded ---")

if __name__ == '__main__':
    asyncio.run(main())

