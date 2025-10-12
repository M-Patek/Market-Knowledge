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
import json
from io import StringIO
from filelock import FileLock
from typing import List, Dict, Optional, Any, Literal
import jsonschema
from pathlib import Path

# --- Libraries for Data Sources ---
import requests
from alpha_vantage.timeseries import TimeSeries
from twelvedata import TDClient
from tiingo import TiingoClient
# ------------------------------------

from pydantic import BaseModel, Field, field_validator, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential
from pythonjsonlogger import jsonlogger
from jinja2 import Environment, FileSystemLoader
from execution.order_manager import OrderManager
from execution.adapters import BacktraderBrokerAdapter
from audit_manager import archive_logs_to_s3
from snapshot_manager import SnapshotManager
from strategy_handler import StrategyDataHandler
from cognitive.engine import CognitiveEngine
from observability import (BACKTEST_DURATION, TRADES_EXECUTED, start_metrics_server, CACHE_HITS, CACHE_MISSES,
                           PROVIDER_REQUESTS_TOTAL, PROVIDER_ERRORS_TOTAL, PROVIDER_LATENCY_SECONDS, PROVIDER_DATA_FRESHNESS_SECONDS)
from ai.retriever import HybridRetriever
from ai.reasoning_ensemble import ReasoningEnsemble, BayesianReasoner, SymbolicRuleReasoner, LLMExplainerReasoner, CausalInferenceReasoner
from ai.bayesian_fusion_engine import BayesianFusionEngine
from ai.embedding_client import EmbeddingClient
from ai.vector_db_client import VectorDBClient
from ai.temporal_db_client import TemporalDBClient
from ai.tabular_db_client import TabularDBClient
from ai.ensemble_client import EnsembleAIClient
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
    replay_snapshot_id: Optional[str] = None
    start_date: date
    end_date: date
    asset_universe: List[str] = Field(..., min_length=1)
    market_breadth_tickers: List[str] = Field(..., min_length=1)
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
    def __init__(self, config: StrategyConfig, cache_dir: str = "data_cache", snapshot_id: Optional[str] = None):
        self.config = config
        self.ds_config = config.data_sources
        self.snapshot_id = snapshot_id

        if self.snapshot_id:
            self.cache_dir = os.path.join("snapshots", self.snapshot_id)
        else:
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

        # --- [NEW] Data Contract Loading ---
        self.data_catalog = None
        try:
            with open("data_catalog.json", "r", encoding="utf-8") as f:
                self.data_catalog = json.load(f)
            jsonschema.Draft7Validator.check_schema(self.data_catalog)
            self.logger.info("Successfully loaded and validated the data contract catalog.")
        except (FileNotFoundError, json.JSONDecodeError, jsonschema.SchemaError) as e:
            self.logger.error(f"Failed to load or validate data_catalog.json: {e}. Data contract enforcement will be disabled.")
            self.data_catalog = None

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

    def _validate_data_against_contract(self, df: pd.DataFrame, asset_id: str) -> bool:
        """Validates a DataFrame against the loaded JSON Schema data contract."""
        if not self.data_catalog:
            self.logger.warning("Data catalog not loaded. Skipping contract validation.")
            return True # Fail open if the catalog itself is broken

        asset_schema_ref = next((asset['schema'] for asset in self.data_catalog.get('data_assets', []) if asset['asset_id'] == asset_id), None)
        if not asset_schema_ref:
            self.logger.warning(f"No schema found for asset_id '{asset_id}' in data_catalog.json. Skipping validation.")
            return True

        # Resolve the $ref to get the actual schema object from definitions
        schema_name = asset_schema_ref['$ref'].split('/')[-1]
        schema = self.data_catalog.get('definitions', {}).get(schema_name)

        validator = jsonschema.Draft7Validator(schema)
        # Convert DataFrame to a list of dicts for validation
        records = df.to_dict('records')

        for i, record in enumerate(records):
            errors = sorted(validator.iter_errors(record), key=lambda e: e.path)
            if errors:
                for error in errors:
                    self.logger.error(f"Data contract violation for asset '{asset_id}' (row {i}): {error.message} in field '{'.'.join(map(str, error.path))}'")
                return False # Reject the entire batch on first error
        return True

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
        # Implementation...
        pass

    @retry(wait=wait_exponential(multiplier=2, min=2, max=30), stop=stop_after_attempt(3))
    async def _fetch_from_twelvedata(self, tickers: List[str], start: date, end: date) -> Optional[pd.DataFrame]:
        # Implementation...
        pass

    @retry(wait=wait_exponential(multiplier=2, min=2, max=30), stop=stop_after_attempt(3))
    async def _fetch_from_tiingo(self, tickers: List[str], start: date, end: date) -> Optional[pd.DataFrame]:
        # Implementation...
        pass

    @retry(wait=wait_exponential(multiplier=2, min=2, max=30), stop=stop_after_attempt(3))
    async def _fetch_from_yfinance(self, tickers: List[str], start: date, end: date) -> Optional[pd.DataFrame]:
        # Implementation...
        pass

    # --- Main Data Orchestration ---
    async def _fetch_and_cache_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Handles fetching, caching, and fallback for a single ticker."""
        # --- [NEW] Snapshot Replay Logic ---
        if self.snapshot_id:
            self.logger.info(f"REPLAY MODE: Loading '{ticker}' data from snapshot '{self.snapshot_id}'.")

        params = {'ticker': ticker, 'start': self.config.start_date, 'end': self.config.end_date}
        cache_filename = self._generate_cache_filename(f"asset_data_{ticker}", params)
        cache_path = os.path.join(self.cache_dir, cache_filename)

        cached_df = None
        fetch_start_date = self.config.start_date

        if os.path.exists(cache_path):
            self.logger.debug(f"Reading cache for '{ticker}' to check for incremental update.")
            cached_df = pd.read_parquet(cache_path)
            last_cached_date = cached_df['available_at'].max().date()

            if last_cached_date >= self.config.end_date:
                self.logger.info(f"Cache for '{ticker}' is up-to-date. Loading from cache.")
                CACHE_HITS.inc()
                return cached_df

            self.logger.info(f"Cache for '{ticker}' is outdated (last date: {last_cached_date}). Fetching incrementally.")
            fetch_start_date = last_cached_date + timedelta(days=1)
        else:
            self.logger.info(f"Cache miss for '{ticker}'. Fetching fresh data...")
            CACHE_MISSES.inc()

        # --- [MODIFIED] Skip network calls if in snapshot mode ---
        if self.snapshot_id:
            self.logger.error(f"Data for ticker '{ticker}' not found in snapshot '{self.snapshot_id}'. This should not happen in replay mode.")
            return None

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

                    # --- [NEW] Add Provenance Timestamps ---
                    data['observed_at'] = pd.to_datetime(datetime.utcnow(), utc=True)
                    data.reset_index(inplace=True)
                    data.rename(columns={'Date': 'available_at'}, inplace=True)

                    # --- [NEW] Enforce Data Contract ---
                    is_valid = self._validate_data_against_contract(data, 'asset_universe_daily_bars')
                    if not is_valid:
                        self.logger.critical(f"Data from provider '{provider.upper()}' for ticker '{ticker}' failed contract validation. Rejecting data.")
                        raise DataFetchError(f"Data from {provider.upper()} failed contract validation.")
                    
                    # --- [NEW] Calculate and Record Freshness Metric ---
                    if not data.empty:
                        latest_available_at = data['available_at'].max()
                        observed_at = data['observed_at'].iloc[0] # All rows have the same observation time
                        freshness_seconds = (observed_at - latest_available_at).total_seconds()
                        PROVIDER_DATA_FRESHNESS_SECONDS.labels(provider=provider).set(freshness_seconds)

                    combined_df = pd.concat([cached_df, data]) if cached_df is not None else data
                    # Set index back to available_at for internal processing
                    combined_df.set_index('available_at', inplace=True)
                    await self._write_cache_direct(cache_path, combined_df, provider, params)
                    return combined_df
                else:
                    raise DataFetchError(f"Provider {provider.upper()} returned no data for {ticker}.")
            except Exception as e:
                self.logger.error(f"Provider {provider.upper()} failed for '{ticker}': {e}")
                PROVIDER_ERRORS_TOTAL.labels(provider=provider).inc()
                PROVIDER_LATENCY_SECONDS.labels(provider=provider).observe(time.time() - start_time)
                self._record_provider_failure(provider)
        
        if cached_df is not None: cached_df.set_index('available_at', inplace=True)
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

    def get_required_cache_files(self) -> List[str]:
        """Gets a list of all cache files required for the current config."""
        files = []
        for ticker in self.config.asset_universe:
            files.append(self._generate_cache_filename(f"asset_data_{ticker}", {'ticker': ticker, 'start': self.config.start_date, 'end': self.config.end_date}))
        return files

    async def get_vix_data(self) -> Optional[pd.Series]:
        # Implementation...
        pass

    async def get_treasury_yield_data(self) -> Optional[pd.Series]:
        # Implementation...
        pass

    async def get_market_breadth_data(self) -> Optional[pd.Series]:
        # Implementation...
        pass

    async def get_aligned_data(self) -> Optional[Dict[str, Any]]:
        # Implementation...
        pass

    def _generate_cache_filename(self, prefix: str, params: Dict[str, Any]) -> str:
        param_string = json.dumps(params, sort_keys=True, default=str)
        param_hash = hashlib.sha256(param_string.encode()).hexdigest()[:16]
        return f"{prefix}_{param_hash}.parquet"

    async def _write_cache_direct(self, cache_path: str, data: pd.DataFrame, source: str, params: Dict):
        # Implementation...
        pass


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
        self.emergency_signal: Optional[float] = None # Flag for emergency event

    def start(self):
        self.logger.info(f"{self.datas[0].datetime.date(0).isoformat()}: [Legion Commander]: Awaiting daily orders...")

    def next(self):
        # --- [NEW] Explicit Temporal Consistency Check ---
        decision_time = self.datas[0].datetime.date(0)
        for d in self.datas:
            available_at_time = d.datetime.date(0)
            if available_at_time > decision_time:
                self.logger.critical(f"CRITICAL FAULT: Lookahead bias detected for {d._name}! "
                                     f"Data available at {available_at_time} is being processed at decision time {decision_time}. Halting.")
                self.env.runstop() # Stop the backtest immediately
                return

        if len(self.datas[0]) < self.config.sma_period: return

        # --- [CRITICAL] Prioritize checking for an injected emergency signal ---
        if self.emergency_signal is not None:
            self.logger.critical(f"--- {self.datas[0].datetime.date(0).isoformat()}: EMERGENCY REBALANCE TRIGGERED ---")
            battle_plan = self.cognitive_engine.determine_allocations(
                candidate_analysis=[], current_vix=0, current_date=self.datas[0].datetime.date(0),
                emergency_factor=self.emergency_signal
            )
            self.order_manager.rebalance(self, battle_plan)
            self.emergency_signal = None # Reset the signal after acting
            return # Skip the rest of the logic for this bar

        # --- Standard Daily Rebalancing Logic ---
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

    def inject_emergency_factor(self, factor: float):
        """
        External entry point for the EventDistributor to inject a high-priority signal.
        """
        self.logger.critical(f"Received an emergency signal with factor {factor:.3f}. "
                             f"Flagging for immediate action on the next bar.")
        self.emergency_signal = factor

# --- Reporting Engine ---
async def generate_html_report(cerebro: bt.Cerebro, strat: RomanLegionStrategy, report_filename="phoenix_report.html"):
    # Implementation...
    pass

# --- AI Pre-computation Helpers ---
async def precompute_asset_analyses(
    reasoning_ensemble: ReasoningEnsemble,
    retriever: HybridRetriever,
    dates: List[date],
    asset_universe: List[str]
) -> Dict[date, Dict]:
    """Pre-computes all AI analyses for all assets for all dates to prevent repeated API calls."""
    logger = logging.getLogger("PhoenixProject.AIPrecomputation")
    logger.info(f"Starting AI pre-computation for {len(dates)} dates and {len(asset_universe)} assets.")
    master_lookup = {d: {} for d in dates}

    for single_date in dates:
        for ticker in asset_universe:
            logger.debug(f"Analyzing {ticker} for date {single_date.isoformat()}...")
            # In a real system, the query would be more sophisticated, perhaps incorporating the date.
            query = f"What is the near-term outlook for {ticker}?"
            evidence_list = await retriever.retrieve(query=query, ticker=ticker)
            hypothesis = f"Assess the investment-worthiness of {ticker} for the near-term."

            analysis_result = await reasoning_ensemble.analyze(hypothesis, evidence_list)
            master_lookup[single_date][ticker] = analysis_result

    logger.info("AI pre-computation complete.")
    return master_lookup

# --- Main Execution Engine ---
async def run_single_backtest(config: StrategyConfig, all_aligned_data: Dict, reasoning_ensemble: Optional[ReasoningEnsemble] = None, retriever: Optional[HybridRetriever] = None):
    logger = logging.getLogger("PhoenixProject")
    logger.info(f"--- Launching 'Phoenix Project' in SINGLE BACKTEST mode (AI: {config.ai_mode.upper()}) ---")

    master_dates = [dt.to_pydatetime().date() for dt in all_aligned_data["asset_universe_df"].index]
    asset_analysis_lookup = {}
    if reasoning_ensemble and retriever and config.ai_mode != "off":
        asset_analysis_lookup = await precompute_asset_analyses(reasoning_ensemble, retriever, master_dates, config.asset_universe)

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
                # --- [NEW] Ensure 'available_at' is the datetime index for backtrader ---
                if 'available_at' in ticker_df.columns:
                    ticker_df.set_index('available_at', inplace=True)
                
                # Drop metadata columns that are not OHLCV
                ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                ticker_df = ticker_df[[col for col in ohlcv_cols if col in ticker_df.columns]]

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
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days, compression=1, annualize=True)
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
    logger.handlers.clear()

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

    # --- Full Cognitive Stack Initialization ---
    ensemble_client: Optional[EnsembleAIClient] = None
    bayesian_fusion_engine: Optional[BayesianFusionEngine] = None
    reasoning_ensemble: Optional[ReasoningEnsemble] = None
    retriever: Optional[HybridRetriever] = None

    if config.ai_ensemble_config.enable:
        try:
            ensemble_config_path = config.ai_ensemble_config.config_file_path
            embedding_client = EmbeddingClient()
            bayesian_fusion_engine = BayesianFusionEngine(embedding_client=embedding_client)
            
            # [NEW] Initialize all RAG components
            vector_db = VectorDBClient()
            temporal_db = TemporalDBClient()
            tabular_db = TabularDBClient()
            with open(ensemble_config_path, 'r') as f:
                ai_config = yaml.safe_load(f)
            retriever = HybridRetriever(vector_db, temporal_db, tabular_db, embedding_client, ai_config.get('retriever'， {}))

            # Initialize all reasoners
            bayesian_reasoner = BayesianReasoner(bayesian_fusion_engine)
            symbolic_reasoner = SymbolicRuleReasoner()
            llm_explainer = LLMExplainerReasoner()
            causal_reasoner = CausalInferenceReasoner()

            # Construct the final ensemble
            reasoning_ensemble = ReasoningEnsemble(
                bayesian_reasoner, symbolic_reasoner, llm_explainer, causal_reasoner
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize full cognitive stack: {e}. Continuing with limited AI.")

    start_time = time.time()
    try:
        # --- [NEW] Snapshot Logic Integration ---
        snapshot_id_to_replay = config.replay_snapshot_id

        data_manager = DataManager(config, snapshot_id=snapshot_id_to_replay)
        all_aligned_data = await data_manager.get_aligned_data()
        if not all_aligned_data:
            logger.critical("Failed to get aligned data. Aborting operation.")
            return

        if not snapshot_id_to_replay:
            # This is a live run, so we create a snapshot of the data we just used
            snapshot_manager = SnapshotManager()
            required_files = data_manager.get_required_cache_files()
            snapshot_manager.create_snapshot(run_id, required_files)

        if config.walk_forward.get('enabled', False):
            optimizer = Optimizer(config, all_aligned_data, reasoning_ensemble=reasoning_ensemble, retriever=retriever)
            optimizer.run_optimization()
        else:
            await run_single_backtest(config, all_aligned_data, reasoning_ensemble, retriever)
    finally:
        duration = time.time() - start_time
        BACKTEST_DURATION.set(duration)
        logger.info(f"Backtest completed in {duration:.2f} seconds.")

    if config.audit.s3_bucket_name and config.audit.s3_bucket_name != "your-phoenix-project-audit-logs-bucket":
        archive_logs_to_s3(source_dir="ai_audit_logs", bucket_name=config.audit.s3_bucket_name)

    logger.info("--- Operation Concluded ---")

if __name__ == '__main__':
    asyncio.run(main())
