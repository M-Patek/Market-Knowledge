# Phoenix Project - Final Optimized Version (Phoenix Resurrected)
# A collaborative masterpiece by Gemini & AI, guided by our Master.
# This version features an externalized configuration, refined strategy logic,
# modular logging, high-performance asynchronous data fetching, and a professional HTML reporting system.

import os
import datetime
import logging
import asyncio
import yaml
from io import StringIO
from typing import List, Dict, Optional
from dataclasses import dataclass

from jinja2 import Environment, FileSystemLoader
import httpx
import backtrader as bt
import pandas as pd


# --- [Phase 1.1] Configuration Layer (The Command Deck) V3.0 ---
@dataclass(frozen=True)
class StrategyConfig:
    """
    Central configuration for the Phoenix Project.
    Immutable (frozen) and now loaded from an external YAML file.
    All fields are mandatory and have no default values.
    """
    # 1. Backtest Timeframe & Universe
    start_date: str
    end_date: str
    asset_universe: List[str]

    # 1.5. Market Breadth Tickers
    market_breadth_tickers: List[str]

    # 2. Strategy Parameters
    sma_period: int
    opportunity_score_threshold: float

    # 3. Risk Management Parameters (VIX-based)
    vix_high_threshold: float
    vix_low_threshold: float
    capital_modifier_high_vix: float
    capital_modifier_normal_vix: float
    capital_modifier_low_vix: float

    # 4. Cerebro Engine Settings
    initial_cash: float
    commission_rate: float
    log_level: str


# --- Section 1.5: Data Management Layer (The Data Hub) V3.0 ---
class DataManager:
    """
    [Phase 3.0] Handles all data fetching, caching, and pre-processing.
    Features a high-performance asynchronous I/O engine for data fetching.
    """
    def __init__(self, config: StrategyConfig, cache_dir: str = "data_cache"):
        self.config = config
        self.cache_dir = cache_dir
        self.logger = logging.getLogger("PhoenixProject.DataManager")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"DataManager initialized. Cache directory set to '{self.cache_dir}'.")

    def _fetch_with_cache(self, cache_filename: str, fetch_func, *args, **kwargs) -> Optional[pd.DataFrame]:
        """Universal caching wrapper for any data fetching function."""
        cache_path = os.path.join(self.cache_dir, cache_filename)
        if os.path.exists(cache_path):
            self.logger.info(f"Loading data from cache: {cache_path}")
            try:
                df = pd.read_feather(cache_path)
                # Ensure 'Date' column is the index if it exists
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                return df
            except Exception as e:
                self.logger.error(f"Failed to load from cache {cache_path}: {e}. Refetching.")

        self.logger.info(f"Cache not found for '{cache_filename}'. Fetching fresh data...")
        data = fetch_func(*args, **kwargs)
        
        if data is None or data.empty:
            self.logger.warning(f"Fetching function for '{cache_filename}' returned no data.")
            return None

        data.reset_index().to_feather(cache_path)
        self.logger.info(f"Saved fresh data to cache: {cache_path}")
        return data

    async def _async_fetch_one_ticker(self, client: httpx.AsyncClient, ticker: str, start_ts: int, end_ts: int) -> Optional[pd.DataFrame]:
        """Asynchronously fetches data for a single ticker from Yahoo Finance."""
        URL = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true"
        }
        try:
            response = await client.get(URL, params=params, timeout=10)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text), index_col='Date', parse_dates=True)
            df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
            return df
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error for {ticker}: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            self.logger.error(f"Failed to fetch or parse data for {ticker}: {e}")
        return None

    async def async_get_yfinance_data(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        """The new async core for fetching data concurrently."""
        start_dt = datetime.datetime.strptime(start, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end, '%Y-%m-%d')
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        
        async with httpx.AsyncClient() as client:
            tasks = [self._async_fetch_one_ticker(client, ticker, start_ts, end_ts) for ticker in tickers]
            results = await asyncio.gather(*tasks)
        
        valid_dfs = [df for df in results if df is not None]
        if not valid_dfs:
            self.logger.warning(f"Async fetch returned no valid data for tickers: {tickers}")
            return None
            
        combined_df = pd.concat(valid_dfs, axis=1)
        return combined_df.sort_index()

    def get_yfinance_data(self, tickers: List[str] or str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Synchronous wrapper for the new async data fetching engine."""
        if isinstance(tickers, str): tickers = [tickers]
        try:
            return asyncio.run(self.async_get_yfinance_data(tickers, start, end))
        except Exception as e:
            self.logger.critical(f"An unexpected error occurred during async data fetch for {tickers}: {e}", exc_info=True)
            return None

    async def get_vix_data(self) -> Optional[pd.Series]:
        df = self._fetch_with_cache("vix_data.feather", self.get_yfinance_data, tickers=['^VIX'], start=self.config.start_date, end=self.config.end_date)
        return df['Close']['^VIX'] if df is not None and ('Close', '^VIX') in df.columns else None

    async def get_treasury_yield_data(self) -> Optional[pd.Series]:
        df = self._fetch_with_cache("treasury_yield_data.feather", self.get_yfinance_data, tickers=['^TNX'], start=self.config.start_date, end=self.config.end_date)
        return df['Close']['^TNX'] if df is not None and ('Close', '^TNX') in df.columns else None
        
    def get_asset_universe_data(self) -> Optional[pd.DataFrame]:
        """Fetches data for the main asset universe defined in the config."""
        return self._fetch_with_cache("asset_universe_data.feather", self.get_yfinance_data, tickers=self.config.asset_universe, start=self.config.start_date, end=self.config.end_date)

    def get_market_breadth_data(self) -> Optional[pd.Series]:
        """Calculates and caches the market breadth indicator."""
        cache_filename = f"market_breadth_sma{self.config.sma_period}.feather"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        if os.path.exists(cache_path):
            self.logger.info(f"Loading final market breadth from cache: {cache_path}")
            df = pd.read_feather(cache_path).set_index('Date')
            return df.squeeze()

        self.logger.info("Calculating market breadth...")
        all_prices_df_container = self._fetch_with_cache("market_breadth_prices.feather", self.get_yfinance_data, tickers=self.config.market_breadth_tickers, start=self.config.start_date, end=self.config.end_date)
        
        if all_prices_df_container is None or 'Close' not in all_prices_df_container:
            return None
        all_prices_df = all_prices_df_container['Close']

        smas = all_prices_df.rolling(window=self.config.sma_period).mean()
        is_above_sma = all_prices_df > smas
        breadth_series = (is_above_sma.sum(axis=1) / all_prices_df.notna().sum(axis=1)).fillna(0)
        
        breadth_series.to_frame(name='breadth').reset_index().to_feather(cache_path)
        self.logger.info(f"Saved final market breadth to cache: {cache_path}")
        return breadth_series

    async def get_aligned_data(self) -> Optional[Dict[str, pd.DataFrame | pd.Series]]:
        """
        [Phase 3.1] The core data pre-processing hub.
        Gathers, aligns, and sanitizes all data before backtesting using async concurrency.
        """
        self.logger.info("--- Starting data alignment and sanitization ---")
        
        asset_df = self.get_asset_universe_data()
        if asset_df is None or asset_df.empty:
            self.logger.critical("Cannot create master index: Asset universe data is missing.")
            return None
        master_index = asset_df.index
        self.logger.info(f"Master trading index created with {len(master_index)} days.")

        self.logger.info("Fetching auxiliary data streams concurrently...")
        tasks = {
            "vix": self.get_vix_data(),
            "treasury_yield": self.get_treasury_yield_data(),
            "market_breadth": asyncio.to_thread(self.get_market_breadth_data) # Wrap sync function
        }
        results = await asyncio.gather(*tasks.values())
        data_streams = dict(zip(tasks.keys(), results))
        
        sanitized_streams = {}
        for name, series in data_streams.items():
            aligned_series = series.reindex(master_index) if series is not None else pd.Series(index=master_index)
            sanitized_streams[name] = aligned_series.ffill().bfill()
            if sanitized_streams[name].isnull().any():
                self.logger.warning(f"{name} data still contains NaNs after sanitization.")
        
        self.logger.info("--- Data alignment and sanitization complete ---")
        return {"asset_universe_df": asset_df, **sanitized_streams}

# --- Section 2: Gemini Cognitive Engine (The Marshal's Brain) V2.0 ---
class CognitiveEngine:
    """[Phase 3.3] The central coordinating brain of the strategy."""
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.logger = logging.getLogger("PhoenixProject.CognitiveEngine")
        self.risk_manager = RiskManager(config)
        self.portfolio_constructor = PortfolioConstructor(config)

    def determine_allocations(self, candidate_analysis: List[Dict], current_vix: float) -> List[Dict]:
        """The primary entry point for the engine's decision-making process."""
        self.logger.info("--- [Cognitive Engine Call: Marshal Coordination] ---")
        capital_modifier = self.risk_manager.get_capital_modifier(current_vix)
        return self.portfolio_constructor.construct_portfolio(candidate_analysis, capital_modifier)

    def analyze_asset_momentum(self, current_price: float, current_sma: float) -> float:
        """A proxy method to access the momentum analysis function."""
        return self.portfolio_constructor.analyze_asset_momentum(current_price, current_sma)

class RiskManager:
    """[Phase 3.1] Encapsulates all market risk assessment logic."""
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.logger = logging.getLogger("PhoenixProject.RiskManager")

    def get_capital_modifier(self, current_vix: float) -> float:
        """Returns a capital modifier based on the current VIX level."""
        self.logger.info(f"RiskManager is assessing VIX: {current_vix:.2f}")
        if current_vix > self.config.vix_high_threshold:
            self.logger.info("RiskManager Read: High fear. Defensive stance.")
            return self.config.capital_modifier_high_vix
        elif current_vix < self.config.vix_low_threshold:
            self.logger.info("RiskManager Read: Low fear. Aggressive stance.")
            return self.config.capital_modifier_low_vix
        else:
            self.logger.info("RiskManager Read: Normal fear. Standard operations.")
            return self.config.capital_modifier_normal_vix

class PortfolioConstructor:
    """[Phase 3.2] Encapsulates the logic for constructing the target portfolio."""
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.logger = logging.getLogger("PhoenixProject.PortfolioConstructor")

    @staticmethod
    def analyze_asset_momentum(current_price: float, current_sma: float) -> float:
        """
        ANALYST V2.0: Analyzes an asset's momentum using a continuous score.
        The score reflects how far the price is from its SMA, clamped between 0 and 100.
        """
        if current_sma <= 0:
            return 0.0
        score = 50 + 50 * ((current_price / current_sma) - 1)
        return max(0.0, min(100.0, score))

    def construct_portfolio(self, candidate_analysis: List[Dict], capital_modifier: float) -> List[Dict]:
        """Filters candidates and calculates final capital allocation."""
        self.logger.info("PortfolioConstructor is analyzing candidates...")
        worthy_targets = [res for res in candidate_analysis if res["opportunity_score"] > self.config.opportunity_score_threshold]
        if not worthy_targets:
            self.logger.info("PortfolioConstructor: No high-quality opportunities found. Standing down.")
            return []
        total_score = sum(t['opportunity_score'] for t in worthy_targets)
        battle_plan = []
        for target in worthy_targets:
            base_allocation = target['opportunity_score'] / total_score
            final_allocation = base_allocation * capital_modifier
            battle_plan.append({"ticker": target['ticker'], "capital_allocation_pct": final_allocation})
        
        self.logger.info("--- [PortfolioConstructor's Final Battle Plan] ---")
        total_planned_allocation = sum(d['capital_allocation_pct'] for d in battle_plan)
        self.logger.info(f"Total planned capital deployment today: {total_planned_allocation:.2%}")
        for deployment in battle_plan:
            self.logger.info(f"- Asset: {deployment['ticker']}, Deploy Capital: {deployment['capital_allocation_pct']:.2%}")
        return battle_plan

# --- Section 3: Strategy Execution Layer (The Roman Legion) ---
class RomanLegionStrategy(bt.Strategy):
    params = (('config', None), ('vix_data', None), ('treasury_yield_data', None), ('market_breadth_data', None))

    def __init__(self):
        self.logger = logging.getLogger("PhoenixProject.Strategy")
        if self.p.config is None: raise ValueError("StrategyConfig object not provided!")
        self.config = self.p.config
        self.cognitive_engine = CognitiveEngine(self.config)
        self.data_map = {d._name: d for d in self.datas}
        self.sma_indicators = {d._name: bt.indicators.SimpleMovingAverage(d.close, period=self.config.sma_period) for d in self.datas}
        
        # Convert pandas Series to a dictionary for fast lookups
        self.vix_lookup = self.p.vix_data.to_dict()
        self.yield_lookup = self.p.treasury_yield_data.to_dict()
        self.breadth_lookup = self.p.market_breadth_data.to_dict()

    def start(self):
        self.logger.info(f"{self.datas[0].datetime.date(0).isoformat()}: [Legion Commander]: Awaiting daily orders...")

    def next(self):
        if len(self.datas[0]) < self.config.sma_period: return
        current_date_dt = self.datas[0].datetime.datetime(0)
        current_date = self.datas[0].datetime.date(0)
        self.logger.info(f"--- {current_date.isoformat()}: Daily Rebalancing Briefing ---")
        
        # Use fast dictionary lookup
        current_vix = self.vix_lookup.get(current_date_dt)
        current_yield = self.yield_lookup.get(current_date_dt)
        current_breadth = self.breadth_lookup.get(current_date_dt)

        if current_vix is None:
            self.logger.warning(f"Critical data VIX missing for {current_date}, halting for the day.")
            return

        self.logger.info(f"VIX Index: {current_vix:.2f}, 10Y Yield: {current_yield:.2f if current_yield else 'N/A'}%, Market Breadth: {current_breadth:.2% if current_breadth else 'N/A'}")

        candidate_analysis = [
            {"ticker": ticker, "opportunity_score": self.cognitive_engine.analyze_asset_momentum(d.close[0], self.sma_indicators[ticker][0])}
            for ticker, d in self.data_map.items()
        ]
        battle_plan = self.cognitive_engine.determine_allocations(candidate_analysis, current_vix)

        self.logger.info("--- Starting Unified Rebalancing Protocol ---")
        total_value = self.broker.getvalue()
        target_portfolio = {deployment['ticker']: total_value * deployment['capital_allocation_pct'] for deployment in battle_plan}
        
        current_positions = {pos.data._name for pos in self.positions if self.getposition(pos).size != 0}
        target_tickers = set(target_portfolio.keys())
        all_tickers_in_play = current_positions.union(target_tickers)

        for ticker in all_tickers_in_play:
            target_value = target_portfolio.get(ticker, 0.0)
            self.logger.info(f"Rebalance SYNC: Aligning {ticker} to target value ${target_value:,.2f}.")
            self.order_target_value(target=target_value, data=self.getdatabyname(ticker))
        self.logger.info("--- Rebalancing Protocol Concluded ---")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            dt = self.datas[0].datetime.date(0).isoformat()
            if order.isbuy(): self.logger.info(f"{dt}: BUY EXECUTED, {order.data._name}, Size: {order.executed.size}, Price: {order.executed.price:.2f}")
            elif order.issell(): self.logger.info(f"{dt}: SELL EXECUTED, {order.data._name}, Size: {order.executed.size}, Price: {order.executed.price:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.logger.warning(f"{self.datas[0].datetime.date(0).isoformat()}: Order for {order.data._name} failed: {order.getstatusname()}")

# --- Section 5: Reporting Engine ---
def generate_html_report(cerebro, strat, report_filename="phoenix_report.html"):
    """
    Generates a professional HTML report from the backtest results.
    """
    logger = logging.getLogger("PhoenixProject.ReportGenerator")
    logger.info("Generating HTML after-action report...")

    trade_analysis = strat.analyzers.trade_analyzer.get_analysis()
    total_trades = trade_analysis.total.get('total', 0)
    win_rate = (trade_analysis.won.get('total', 0) / total_trades) if total_trades > 0 else 0
    
    sharpe_ratio_analysis = strat.analyzers.sharpe_ratio.get_analysis()
    
    context = {
        "final_value": cerebro.broker.getvalue(),
        "total_return": strat.analyzers.returns.get_analysis().get('rtot', 0.0),
        "sharpe_ratio": sharpe_ratio_analysis.get('sharperatio', None),
        "max_drawdown": strat.analyzers.drawdown.get_analysis().max.get('drawdown', 0.0),
        "total_trades": total_trades,
        "winning_trades": trade_analysis.won.get('total', 0),
        "losing_trades": trade_analysis.lost.get('total', 0),
        "win_rate": win_rate,
        "report_date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "plot_filename": "phoenix_plot.png"
    }

    try:
        logger.info(f"Saving backtest plot to {context['plot_filename']}...")
        figures = cerebro.plot(style='candlestick', barup='green', bardown='red', iplot=False)
        if figures:
            figures[0][0].savefig(context['plot_filename'], dpi=300)
        else:
            raise RuntimeError("Cerebro plot returned no figures.")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
        context['plot_filename'] = None

    try:
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template("report_template.html")
        html_output = template.render(context)
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(html_output)
        logger.info(f"Successfully generated HTML report: {report_filename}")
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")

# --- Section 4: Main Execution Engine (The High Command) ---
if __name__ == '__main__':
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        config = StrategyConfig(**params)
    except FileNotFoundError:
        print("CRITICAL: Configuration file 'config.yaml' not found. Aborting.")
        exit()
    except Exception as e:
        print(f"CRITICAL: Error parsing 'config.yaml': {e}. Aborting.")
        exit()

    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logger = logging.getLogger("PhoenixProject")
    logger.setLevel(log_level)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"phoenix_project_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.info("Phoenix Project Final Optimized Version - Logging System Initialized.")

    data_manager = DataManager(config)
    all_aligned_data = asyncio.run(data_manager.get_aligned_data())

    if not all_aligned_data:
        logger.critical("Failed to get aligned data. Aborting operation.")
    else:
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
                    logger.info(f"Adding data feed for {ticker} with {len(ticker_df)} bars.")
                    cerebro.adddata(bt.feeds.PandasData(dataname=ticker_df, name=ticker))
        except Exception as e:
            logger.critical(f"A critical error occurred during data loading: {e}. Aborting.")
            exit()

        if not cerebro.datas:
            logger.critical("Failed to load data for any asset. Aborting operation.")
        else:
            cerebro.addstrategy(
                RomanLegionStrategy, 
                config=config,
                vix_data=all_aligned_data["vix"],
                treasury_yield_data=all_aligned_data["treasury_yield"],
                market_breadth_data=all_aligned_data["market_breadth"]
            )
            cerebro.broker.setcash(config.initial_cash)
            cerebro.broker.setcommission(commission=config.commission_rate)
            
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

            logger.info('--- Launching "Phoenix Project" (Resurrected Version) ---')
            results = cerebro.run()
            
            logger.info("--- Operation Concluded ---")
            strat = results[0]
            
            generate_html_report(cerebro, strat)
