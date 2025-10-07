# Phoenix Project - Final Optimized Version (Phoenix Resurrected)
# A collaborative masterpiece by Gemini & AI, guided by our Master.
# This version features a robust Pydantic configuration, resilient and concurrent data fetching,
# an intelligent auto-invalidating cache, professional logging, and a comprehensive HTML reporting system.

import os
import datetime
import logging
from logging.handlers import RotatingFileHandler
import asyncio
import yaml
import hashlib
import json
import random
from io import StringIO
from typing import List, Dict, Optional, Any, Literal
from pathlib import Path

from pydantic import BaseModel, Field, validator, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from jinja2 import Environment, FileSystemLoader
from gemini_service import GeminiService, MalformedResponseError
import httpx
import backtrader as bt
import pandas as pd


# --- [Optimized] Configuration Models ---
class GeminiConfig(BaseModel):
    """Configuration for the Gemini AI Advisor service."""
    enable: bool = False
    mode: Literal["mock", "production"] = "production"
    api_key_env_var: str
    model_name: str
    request_timeout: int = 90
    prompts: Dict[str, str]

class StrategyConfig(BaseModel):
    """
    Central configuration for the Phoenix Project.
    Uses Pydantic for robust, self-validating configuration.
    """
    start_date: str
    end_date: str
    asset_universe: List[str] = Field(..., min_items=1)
    market_breadth_tickers: List[str] = Field(..., min_items=1)
    sma_period: int = Field(..., gt=0)
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
    ai_mode: Literal["off", "raw", "processed"] = "processed"
    max_total_allocation: float = Field(default=1.0, gt=0, le=1.0)

    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

# --- [Optimized] Custom Exception for Data Layer ---
class DataFetchError(Exception):
    """Custom exception for non-retryable data fetching errors."""
    pass

# --- Data Management Layer ---
class DataManager:
    """
    Handles all data fetching, caching, and pre-processing with a resilient asynchronous engine.
    """
    def __init__(self, config: StrategyConfig, cache_dir: str = "data_cache"):
        self.config = config
        self.cache_dir = cache_dir
        self.logger = logging.getLogger("PhoenixProject.DataManager")
        self.semaphore = asyncio.Semaphore(8)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"DataManager initialized. Cache directory set to '{self.cache_dir}'.")

    def _generate_cache_filename(self, prefix: str, params: Dict) -> str:
        param_string = json.dumps(params, sort_keys=True)
        param_hash = hashlib.sha256(param_string.encode()).hexdigest()[:16]
        return f"{prefix}_{param_hash}.parquet"

    async def _fetch_with_cache_async(self, cache_filename: str, fetch_coro, *args, **kwargs) -> Optional[pd.DataFrame]:
        cache_path = os.path.join(self.cache_dir, cache_filename)
        if os.path.exists(cache_path):
            self.logger.info(f"Loading data from cache: {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                return df
            except Exception as e:
                self.logger.error(f"Failed to load from cache {cache_path}: {e}. Refetching.")

        self.logger.info(f"Cache not found for '{cache_filename}'. Fetching fresh data...")
        data = await fetch_coro(*args, **kwargs)

        if data is None or data.empty:
            self.logger.warning(f"Fetching function for '{cache_filename}' returned no data.")
            return None

        data.reset_index().to_parquet(cache_path)
        self.logger.info(f"Saved fresh data to cache: {cache_path}")
        return data

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True
    )
    async def _async_fetch_one_ticker(self, client: httpx.AsyncClient, ticker: str, start_ts: int, end_ts: int) -> pd.DataFrame:
        URL = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        params = {"period1": start_ts, "period2": end_ts, "interval": "1d", "events": "history", "includeAdjustedClose": "true"}
        async with self.semaphore:
            try:
                self.logger.info(f"Fetching data for {ticker}...")
                response = await client.get(URL, params=params, timeout=20)
                response.raise_for_status()
                df = pd.read_csv(StringIO(response.text), index_col='Date', parse_dates=True)
                df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
                return df
            except httpx.HTTPStatusError as e:
                self.logger.warning(f"HTTP error for {ticker}: {e.response.status_code}. Retrying if applicable...")
                raise
            except Exception as e:
                # [Optimized] Raise a specific exception instead of returning None
                self.logger.error(f"Non-retryable error for {ticker}: {e}")
                raise DataFetchError(f"Non-retryable error for {ticker}: {e}") from e

    async def async_get_yfinance_data(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        start_dt = datetime.datetime.strptime(start, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end, '%Y-%m-%d')
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())

        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        async with httpx.AsyncClient(limits=limits) as client:
            tasks = [self._async_fetch_one_ticker(client, ticker, start_ts, end_ts) for ticker in tickers]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # [Optimized] Explicitly log failures instead of silently filtering
        valid_dfs = []
        for i, res in enumerate(results):
            if isinstance(res, pd.DataFrame):
                valid_dfs.append(res)
            elif isinstance(res, Exception):
                self.logger.error(f"Data fetch failed for ticker '{tickers[i]}': {res}")

        if not valid_dfs:
            self.logger.critical(f"Async fetch returned no valid data for any of the requested tickers: {tickers}")
            return None

        combined_df = pd.concat(valid_dfs, axis=1)
        return combined_df.sort_index()

    async def get_vix_data(self) -> Optional[pd.Series]:
        params = {'tickers': ['^VIX'], 'start': self.config.start_date, 'end': self.config.end_date}
        cache_filename = self._generate_cache_filename("vix_data", params)
        df = await self._fetch_with_cache_async(cache_filename, self.async_get_yfinance_data, **params)
        return df['Close']['^VIX'] if df is not None and ('Close', '^VIX') in df.columns else None

    async def get_treasury_yield_data(self) -> Optional[pd.Series]:
        params = {'tickers': ['^TNX'], 'start': self.config.start_date, 'end': self.config.end_date}
        cache_filename = self._generate_cache_filename("treasury_yield_data", params)
        df = await self._fetch_with_cache_async(cache_filename, self.async_get_yfinance_data, **params)
        return df['Close']['^TNX'] if df is not None and ('Close', '^TNX') in df.columns else None

    async def get_asset_universe_data(self) -> Optional[pd.DataFrame]:
        params = {'tickers': sorted(self.config.asset_universe), 'start': self.config.start_date, 'end': self.config.end_date}
        cache_filename = self._generate_cache_filename("asset_universe_data", params)
        return await self._fetch_with_cache_async(cache_filename, self.async_get_yfinance_data, **params)

    async def get_market_breadth_data(self) -> Optional[pd.Series]:
        params = {'tickers': sorted(self.config.market_breadth_tickers), 'start': self.config.start_date, 'end': self.config.end_date, 'sma': self.config.sma_period}
        cache_filename = self._generate_cache_filename("market_breadth_indicator", params)
        cache_path = os.path.join(self.cache_dir, cache_filename)
        if os.path.exists(cache_path):
            self.logger.info(f"Loading final market breadth from cache: {cache_path}")
            df = pd.read_parquet(cache_path).set_index('Date')
            return df.squeeze()

        self.logger.info("Calculating market breadth...")
        price_params = {k: v for k, v in params.items() if k != 'sma'}
        price_cache_filename = self._generate_cache_filename("market_breadth_prices", price_params)
        all_prices_df_container = await self._fetch_with_cache_async(
            price_cache_filename, self.async_get_yfinance_data,
            tickers=self.config.market_breadth_tickers, start=self.config.start_date, end=self.config.end_date
        )

        if all_prices_df_container is None or 'Close' not in all_prices_df_container: return None
        all_prices_df = all_prices_df_container['Close']
        smas = all_prices_df.rolling(window=self.config.sma_period).mean()
        is_above_sma = all_prices_df > smas
        breadth_series = (is_above_sma.sum(axis=1) / all_prices_df.notna().sum(axis=1)).fillna(0)
        breadth_series.to_frame(name='breadth').reset_index().to_parquet(cache_path)
        self.logger.info(f"Saved final market breadth to cache: {cache_path}")
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
            aligned_series = series.reindex(master_index) if series is not None else pd.Series(index=master_index)
            sanitized_streams[name] = aligned_series.ffill().bfill()
            if sanitized_streams[name].isnull().any(): self.logger.warning(f"{name} data still contains NaNs after sanitization.")
        self.logger.info("--- Data alignment and sanitization complete ---")
        return {"asset_universe_df": asset_df, **sanitized_streams}

# --- Cognitive Engine Layer ---
class CognitiveEngine:
    def __init__(self, config: StrategyConfig, sentiment_data: Optional[Dict[datetime.date, float]] = None, asset_analysis_data: Optional[Dict[datetime.date, Dict]] = None, ai_mode: str = "processed"):
        self.config = config
        self.logger = logging.getLogger("PhoenixProject.CognitiveEngine")
        self.risk_manager = RiskManager(config, sentiment_data)
        self.portfolio_constructor = PortfolioConstructor(config, asset_analysis_data, mode=ai_mode)

    def determine_allocations(self, candidate_analysis: List[Dict], current_vix: float, current_date: datetime.date) -> List[Dict]:
        self.logger.info("--- [Cognitive Engine Call: Marshal Coordination] ---")
        capital_modifier = self.risk_manager.get_capital_modifier(current_vix, current_date)
        return self.portfolio_constructor.construct_portfolio(candidate_analysis, capital_modifier, current_date)

    def analyze_asset_momentum(self, current_price: float, current_sma: float) -> float:
        return self.portfolio_constructor.analyze_asset_momentum(current_price, current_sma)

class RiskManager:
    def __init__(self, config: StrategyConfig, sentiment_data: Optional[Dict[datetime.date, float]] = None):
        self.config = config
        self.sentiment_data = sentiment_data if sentiment_data is not None else {}
        self.logger = logging.getLogger("PhoenixProject.RiskManager")
        if self.sentiment_data: self.logger.info(f"RiskManager initialized with {len(self.sentiment_data)} days of sentiment data.")

    def get_capital_modifier(self, current_vix: float, current_date: datetime.date) -> float:
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
    def __init__(self, config: StrategyConfig, asset_analysis_data: Optional[Dict[datetime.date, Dict]] = None, mode: str = "processed"):
        self.config = config
        self.asset_analysis_data = asset_analysis_data or {}
        self.logger = logging.getLogger("PhoenixProject.PortfolioConstructor")
        self.mode = mode
        self._ema_state = {}
        self.ema_alpha = 0.2
        self.global_scale = 1.0

    @staticmethod
    def analyze_asset_momentum(current_price: float, current_sma: float) -> float:
        if current_sma <= 0: return 0.0
        score = 50 + 50 * ((current_price / current_sma) - 1)
        return max(0.0, min(100.0, score))

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

    def construct_portfolio(self, candidate_analysis: List[Dict], capital_modifier: float, current_date: datetime.date) -> List[Dict]:
        self.logger.info("PortfolioConstructor is analyzing candidates...")
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
            self.logger.info("PortfolioConstructor: No high-quality opportunities found. Standing down.")
            return []
        total_score = sum(t['adjusted_score'] for t in worthy_targets)
        battle_plan = []
        for target in worthy_targets:
            base_allocation = target['adjusted_score'] / total_score
            final_allocation = base_allocation * capital_modifier
            battle_plan.append({"ticker": target['ticker'], "capital_allocation_pct": final_allocation})
        
        self.logger.info("--- [PortfolioConstructor's Final Battle Plan] ---")
        total_planned_allocation = sum(d['capital_allocation_pct'] for d in battle_plan)
        if total_planned_allocation > self.config.max_total_allocation:
            self.logger.warning(f"Total planned allocation {total_planned_allocation:.2%} exceeds cap of {self.config.max_total_allocation:.2%}. Scaling down.")
            scale_factor = self.config.max_total_allocation / total_planned_allocation
            for deployment in battle_plan: deployment['capital_allocation_pct'] *= scale_factor
        
        final_total_allocation = sum(d['capital_allocation_pct'] for d in battle_plan)
        self.logger.info(f"Final planned capital deployment: {final_total_allocation:.2%}")
        for deployment in battle_plan: self.logger.info(f"- Asset: {deployment['ticker']}, Deploy Capital: {deployment['capital_allocation_pct']:.2%}")
        return battle_plan

# --- Strategy Execution Layer ---
class RomanLegionStrategy(bt.Strategy):
    params = (('config', None), ('vix_data', None), ('treasury_yield_data', None), ('market_breadth_data', None), ('sentiment_data', None), ('asset_analysis_data', None))

    def __init__(self):
        self.logger = logging.getLogger("PhoenixProject.Strategy")
        if self.p.config is None: raise ValueError("StrategyConfig object not provided!")
        self.config = self.p.config
        self.cognitive_engine = CognitiveEngine(self.config, self.p.sentiment_data, self.p.asset_analysis_data, ai_mode=self.config.ai_mode)
        self.data_map = {d._name: d for d in self.datas}
        self.sma_indicators = {d._name: bt.indicators.SimpleMovingAverage(d.close, period=self.config.sma_period) for d in self.datas}
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
        candidate_analysis = [{"ticker": ticker, "opportunity_score": self.cognitive_engine.analyze_asset_momentum(d.close[0], self.sma_indicators[ticker][0])} for ticker, d in self.data_map.items()]
        battle_plan = self.cognitive_engine.determine_allocations(candidate_analysis, current_vix, current_date)
        self.logger.info("--- Starting Unified Rebalancing Protocol ---")
        total_value = self.broker.getvalue()
        target_portfolio = {deployment['ticker']: total_value * deployment['capital_allocation_pct'] for deployment in battle_plan}
        current_positions = {d._name for d in self.datas if self.getposition(d).size != 0}
        target_tickers = set(target_portfolio.keys())
        all_tickers_in_play = current_positions.union(target_tickers)
        for ticker in all_tickers_in_play:
            target_value = target_portfolio.get(ticker, 0.0)
            data_feed = self.getdatabyname(ticker)
            if not data_feed:
                self.logger.warning(f"Could not find data feed for {ticker} during rebalance. Skipping.")
                continue
            self.logger.info(f"Rebalance SYNC: Aligning {ticker} to target value ${target_value:,.2f}.")
            self.order_target_value(data=data_feed, value=target_value)
        self.logger.info("--- Rebalancing Protocol Concluded ---")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            dt = self.datas[0].datetime.date(0).isoformat()
            if order.isbuy(): self.logger.info(f"{dt}: BUY EXECUTED, {order.data._name}, Size: {order.executed.size}, Price: {order.executed.price:.2f}")
            elif order.issell(): self.logger.info(f"{dt}: SELL EXECUTED, {order.data._name}, Size: {order.executed.size}, Price: {order.executed.price:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.logger.warning(f"{self.datas[0].datetime.date(0).isoformat()}: Order for {order.data._name} failed: {order.getstatusname()}")

# --- Reporting Engine ---
async def generate_ai_report(gemini_service: Optional[GeminiService], context: Dict) -> Optional[str]:
    if not gemini_service: return None
    logger = logging.getLogger("PhoenixProject.ReportGenerator")
    logger.info("Generating AI Marshal's Report...")
    try:
        report_text = await gemini_service.generate_summary_report(context)
        logger.info("Successfully generated AI Marshal's Report.")
        return report_text
    except Exception as e:
        logger.error(f"Failed to generate AI report: {e}")
        return "## Marshal's Debriefing Failed ##\n\nAn error occurred during communication with the AI Command. The quantitative report below remains accurate."

async def generate_html_report(cerebro: bt.Cerebro, strat: RomanLegionStrategy, gemini_service: Optional[GeminiService] = None, report_filename="phoenix_report.html"):
    logger = logging.getLogger("PhoenixProject.ReportGenerator")
    logger.info("Generating HTML after-action report...")
    trade_analysis = strat.analyzers.trade_analyzer.get_analysis()
    total_trades = trade_analysis.total.get('total', 0)
    win_rate = (trade_analysis.won.get('total', 0) / total_trades) if total_trades > 0 else 0
    sharpe_ratio_analysis = strat.analyzers.sharpe_ratio.get_analysis()
    context = {
        "final_value": cerebro.broker.getvalue(), "total_return": strat.analyzers.returns.get_analysis().get('rtot', 0.0),
        "sharpe_ratio": sharpe_ratio_analysis.get('sharperatio', None), "max_drawdown": strat.analyzers.drawdown.get_analysis().max.get('drawdown', 0.0),
        "total_trades": total_trades, "winning_trades": trade_analysis.won.get('total', 0), "losing_trades": trade_analysis.lost.get('total', 0), "win_rate": win_rate,
        "report_date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "plot_filename": "phoenix_plot.png"
    }
    context['ai_summary_report'] = await generate_ai_report(gemini_service, context)
    try:
        logger.info(f"Saving backtest plot to {context['plot_filename']}...")
        figures = cerebro.plot(style='candlestick', barup='green', bardown='red', iplot=False)
        if figures: figures[0][0].savefig(context['plot_filename'], dpi=300)
        else: raise RuntimeError("Cerebro plot returned no figures.")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
        context['plot_filename'] = None
    try:
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template("report_template.html.txt")
        html_output = template.render(context)
        with open(report_filename, 'w', encoding='utf-8') as f: f.write(html_output)
        logger.info(f"Successfully generated HTML report: {report_filename}")
    except Exception as e: logger.error(f"Failed to generate HTML report: {e}")

# --- AI Pre-computation Helpers ---
def fetch_mock_news_for_date(date_obj: datetime.date) -> List[str]:
    base_headlines = ["Global markets show mixed signals...", "Tech sector rally continues...", "New geopolitical tensions...", "Federal Reserve hints at a more hawkish stance..."]
    if date_obj.weekday() == 0: base_headlines.append("Weekend uncertainty weighs on market open.")
    elif date_obj.weekday() == 4: base_headlines.append("Positive jobs report boosts investor confidence.")
    random.shuffle(base_headlines)
    return base_headlines[:3]

async def precompute_sentiments(gemini_service: GeminiService, dates: List[datetime.date]) -> Dict[datetime.date, float]:
    logger = logging.getLogger("PhoenixProject.SentimentPrecomputer")
    cache_file = Path("data_cache") / "sentiment_cache.json"
    if cache_file.exists():
        logger.info(f"Found sentiment cache file: {cache_file}")
        with open(cache_file, 'r') as f: cached_data_str = json.load(f)
        cached_data = {datetime.date.fromisoformat(k): v for k, v in cached_data_str.items()}
        if set(dates).issubset(set(cached_data.keys())):
            logger.info("Sentiment cache is valid. Loading from cache.")
            return {d: cached_data[d] for d in dates}
        else: logger.warning("Sentiment cache is stale. Re-computing.")
    
    logger.info(f"Pre-computing sentiments for {len(dates)} trading days...")
    sentiment_lookup_str = {}
    for date_obj in dates:
        try:
            mock_headlines = fetch_mock_news_for_date(date_obj)
            analysis = await gemini_service.get_market_sentiment(mock_headlines)
            sentiment_lookup_str[date_obj.isoformat()] = analysis.get('sentiment_score', 0.0)
            logger.info(f"Sentiment for {date_obj.isoformat()}: {analysis.get('sentiment_score', 0.0):.2f}")
        except (MalformedResponseError, Exception) as e:
            logger.error(f"Could not compute sentiment for {date_obj.isoformat()}: {e}. Defaulting to neutral (0.0).")
            sentiment_lookup_str[date_obj.isoformat()] = 0.0
    
    with open(cache_file, 'w') as f: json.dump(sentiment_lookup_str, f, indent=2)
    logger.info(f"Saved computed sentiments to cache: {cache_file}")
    return {datetime.date.fromisoformat(k): v for k, v in sentiment_lookup_str.items()}

async def precompute_asset_analyses(gemini_service: GeminiService, dates: List[datetime.date], asset_universe: List[str]) -> Dict[datetime.date, Dict]:
    logger = logging.getLogger("PhoenixProject.AssetAnalysisPrecomputer")
    cache_file = Path("data_cache") / "asset_analysis_cache.json"
    if cache_file.exists():
        logger.info(f"Found asset analysis cache file: {cache_file}")
        with open(cache_file, 'r') as f: cached_data_str = json.load(f)
        try:
            required_dates_str = {d.isoformat() for d in dates}
            cached_dates_str = set(cached_data_str.keys())
            if required_dates_str.issubset(cached_dates_str) and all(set(asset_universe).issubset(set(cached_data_str[d].keys())) for d in required_dates_str):
                logger.info("Asset analysis cache is valid. Loading from cache.")
                return {datetime.date.fromisoformat(k): v for k, v in cached_data_str.items()}
        except Exception: logger.warning("Asset analysis cache is corrupted. Re-computing.")

    logger.info(f"Pre-computing asset analyses for {len(asset_universe)} assets over {len(dates)} days...")
    analysis_lookup_str = {d.isoformat(): {} for d in dates}
    for date_obj in dates:
        for ticker in asset_universe:
            try:
                analysis = await gemini_service.get_asset_analysis(ticker, date_obj)
                analysis_lookup_str[date_obj.isoformat()][ticker] = analysis
            except Exception as e:
                logger.error(f"Could not compute analysis for {ticker} on {date_obj.isoformat()}: {e}. Defaulting to neutral.")
                analysis_lookup_str[date_obj.isoformat()][ticker] = {"adjustment_factor": 1.0, "reasoning": "Error in computation"}
    
    with open(cache_file, 'w') as f: json.dump(analysis_lookup_str, f)
    logger.info(f"Saved computed asset analyses to cache: {cache_file}")
    return {datetime.date.fromisoformat(k): v for k, v in analysis_lookup_str.items()}

# --- Main Execution Engine ---
async def main():
    try:
        config_params = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        # [Optimized] Flatten nested execution_mode config safely
        config_params.update(config_params.pop('execution_mode', {}))
        config = StrategyConfig(**config_params)
    except (FileNotFoundError, ValidationError) as e:
        print(f"CRITICAL: Error loading or validating 'config.yaml': {e}. Aborting.")
        exit()

    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logger = logging.getLogger("PhoenixProject")
    logger.setLevel(log_level)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        log_dir = "logs"; os.makedirs(log_dir, exist_ok=True)
        log_filename = f"phoenix_project_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        stream_handler = logging.StreamHandler(); stream_handler.setFormatter(formatter); logger.addHandler(stream_handler)
        file_handler = RotatingFileHandler(os.path.join(log_dir, log_filename), maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
        file_handler.setFormatter(formatter); logger.addHandler(file_handler)
    logger.info("Phoenix Project Final Optimized Version - Logging System Initialized.")

    gemini_service = None
    if config.gemini_config.enable:
        try:
            # [Optimized] Pass the validated config as a dictionary
            gemini_service = GeminiService(config.gemini_config.model_dump())
            logger.info(f"Gemini Service has been enabled and initialized in '{config.gemini_config.mode}' mode.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Service: {e}. Continuing without AI features.")

    data_manager = DataManager(config)
    all_aligned_data = await data_manager.get_aligned_data()
    if not all_aligned_data:
        logger.critical("Failed to get aligned data. Aborting operation.")
        return

    master_dates = [dt.to_pydatetime().date() for dt in all_aligned_data["asset_universe_df"].index]
    sentiment_lookup = {}
    asset_analysis_lookup = {}
    if gemini_service 和 config.ai_mode != "off":
        sentiment_lookup = await precompute_sentiments(gemini_service, master_dates)
        asset_analysis_lookup = await precompute_asset_analyses(gemini_service, master_dates, config.asset_universe)

    cerebro = bt.Cerebro()
    try:
        all_data_df = all_aligned_data["asset_universe_df"]
        if all_data_df is None or all_data_df.empty: raise ValueError("Asset universe data is empty.")
        unique_tickers = all_data_df.columns。get_level_values(1)。unique()
        for ticker 在 unique_tickers:
            ticker_df = all_data_df.xs(ticker, level=1, axis=1).copy()
            ticker_df.columns = [col.lower() for col in ticker_df.columns]
            ticker_df.dropna(inplace=True)
            if not ticker_df.empty:
                logger.info(f"Adding data feed for {ticker} with {len(ticker_df)} bars.")
                cerebro.adddata(bt.feeds.PandasData(dataname=ticker_df, name=ticker))
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

    logger.info(f"--- Launching 'Phoenix Project' (Resurrected Version) with AI Mode: {config.ai_mode.upper()} ---")
    results = cerebro.run()
    logger.info("--- Operation Concluded ---")
    strat = results[0]
    await generate_html_report(cerebro, strat, gemini_service, report_filename=f"phoenix_report_{config.ai_mode}.html")

if __name__ == '__main__':
    asyncio.run(main())
