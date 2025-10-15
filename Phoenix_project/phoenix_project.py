import os
import logging
import asyncio
from datetime import date, datetime
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import yaml
import pandas as pd
import numpy as np
from tiingo import TiingoClient
import yfinance as yf

from strategy_handler import RomanLegionStrategy
from ai.retriever import HybridRetriever
from ai.reasoning_ensemble import ReasoningEnsemble
from ai.vector_db_client import VectorDBClient
from ai.temporal_db_client import TemporalDBClient
from ai.tabular_db_client import TabularDBClient
from ai.ensemble_client import EnsembleAIClient
from ai.market_state_predictor import MarketStatePredictor, generate_market_state_labels
import backtrader as bt

# --- Configuration Models ---
class ProviderConfig(BaseModel):
    api_key_env_var: Optional[str] = None
    priority: int

class DataSourcesConfig(BaseModel):
    start_date: date
    end_date: date
    cache_dir: str
    providers: Dict[str, ProviderConfig]

class PositionSizerConfig(BaseModel):
    method: str
    parameters: Dict[str, Any]

class ExecutionModelConfig(BaseModel):
    impact_coefficient: float
    max_volume_share: float
    min_trade_notional: float
    average_spread_bps: float

class RiskManagerConfig(BaseModel):
    max_uncertainty: float
    min_capital_modifier: float

class PortfolioConstructorConfig(BaseModel):
    score_weights: Dict[str, float]
    ema_span: int

class DataQualityMonitoringConfig(BaseModel):
    completeness_threshold: float
    staleness_days_threshold: int
    price_change_std_dev_threshold: float

class MarketStatePredictorConfig(BaseModel):
    long_term_ma: int
    threshold: float
    model_params: Dict[str, Any]

class StrategyConfig(BaseModel):
    project_name: str
    version: str
    log_level: str
    asset_universe: List[str]
    max_total_allocation: float
    commission_rate: float
    sma_period: int
    rsi_period: int
    rsi_overbought_threshold: float
    opportunity_score_threshold: float
    initial_cash: float
    data_sources: DataSourcesConfig
    position_sizer: PositionSizerConfig
    ai_mode: str
    ai_clients_config_path: str
    ai_analysis_snapshot_dir: Optional[str] = None
    execution_model: ExecutionModelConfig
    risk_manager: RiskManagerConfig
    portfolio_constructor: PortfolioConstructorConfig
    data_quality_monitoring: DataQualityMonitoringConfig
    market_state_predictor: Optional[MarketStatePredictorConfig] = None

    class Config:
        extra = 'allow'


# --- Data Management ---
class DataFetchError(Exception):
    pass

class DataManager:
    def __init__(self, config: StrategyConfig, cache_dir: str = "data_cache", snapshot_id: Optional[str] = None):
        self.config = config
        self.ds_config = config.data_sources
        self.dqm_config = config.data_quality_monitoring
        self.snapshot_id = snapshot_id

        if self.snapshot_id:
            self.cache_dir = os.path.join(self.ds_config.cache_dir, self.snapshot_id)
        else:
            self.cache_dir = self.ds_config.cache_dir
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger = logging.getLogger("PhoenixProject.DataManager")
        self.logger.info(f"DataManager initialized. Caching to directory: '{self.cache_dir}'")

        self.tiingo_client = self._create_tiingo_client()

    def _get_cache_path(self, ticker: str, data_type: str) -> str:
        return os.path.join(self.cache_dir, f"{ticker}_{data_type}.csv")

    def _validate_data_contract(self, df: pd.DataFrame, data_type: str) -> bool:
        """
        Validates the DataFrame against a predefined data contract.
        """
        if df.empty:
            self.logger.warning(f"Data validation for '{data_type}' received an empty DataFrame.")
            return False

        required_columns = {
            "asset": ["Open", "High", "Low", "Close", "Volume"],
            "vix": ["Close"],
            "treasury_yield": ["Close"],
            "market_breadth": ["Close"]
        }
        
        if data_type not in required_columns:
            self.logger.error(f"Unknown data type for contract validation: {data_type}")
            return False

        if not all(col in df.columns for col in required_columns[data_type]):
            self.logger.error(f"Data contract validation failed for '{data_type}'. Missing columns. Required: {required_columns[data_type]}, Got: {list(df.columns)}")
            return False

        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.error(f"Data contract validation failed for '{data_type}'. Index is not a DatetimeIndex.")
            return False

        # Check for null values in critical columns
        for col in required_columns[data_type]:
            if df[col].isnull().any():
                self.logger.warning(f"Data contract validation for '{data_type}' found null values in critical column '{col}'.")
                # This might be acceptable, so we don't return False here but just log it.
        
        return True

    def _run_dqm_checks(self, df: pd.DataFrame, ticker: str) -> bool:
        """
        Performs a series of statistical data quality checks on a DataFrame.
        Returns True if all checks pass, False otherwise.
        """
        # 1. Completeness Check (e.g., for 'Close' price)
        completeness = df['Close'].notna().mean()
        if completeness < self.dqm_config.completeness_threshold:
            self.logger.error(f"DQM FAILED for '{ticker}': Completeness check failed. "
                              f"Got {completeness:.2%}, required {self.dqm_config.completeness_threshold:.2%}.")
            return False

        # 2. Staleness Check
        if not df.empty:
            last_data_point = df.index.max().date()
            staleness = (date.today() - last_data_point).days
            if staleness > self.dqm_config.staleness_days_threshold:
                self.logger.error(f"DQM FAILED for '{ticker}': Staleness check failed. "
                                  f"Data is {staleness} days old, threshold is {self.dqm_config.staleness_days_threshold}.")
                return False

        # 3. Data Drift / Anomaly Check (on daily returns)
        daily_returns = df['Close'].pct_change().dropna()
        if not daily_returns.empty:
            mean_return = daily_returns.mean()
            std_return = daily_returns.std()
            threshold = self.dqm_config.price_change_std_dev_threshold

            anomalies = daily_returns[abs(daily_returns - mean_return) > threshold * std_return]
            if not anomalies.empty:
                self.logger.error(f"DQM FAILED for '{ticker}': Anomaly detected. "
                                  f"Price change on {anomalies.index[0].date()} was {anomalies.iloc[0]:.2%}, "
                                  f"which is > {threshold} std devs from the mean.")
                return False
        
        self.logger.info(f"DQM checks passed successfully for '{ticker}'.")
        return True

    def _create_tiingo_client(self) -> Optional[TiingoClient]:
        env_var = self.ds_config.providers.get("tiingo", ProviderConfig()).api_key_env_var
        if not env_var: return None
        
        api_key = os.getenv(env_var)
        if not api_key:
            self.logger.warning(f"Environment variable '{env_var}' not set. Tiingo client will not be available.")
            return None
        return TiingoClient({'api_key': api_key})

    def _fetch_from_tiingo(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        if not self.tiingo_client: raise DataFetchError("Tiingo client not initialized.")
        df = self.tiingo_client.get_dataframe(ticker, startDate=start_date, endDate=end_date)
        df.rename(columns={'adjClose': 'Close', 'adjHigh': 'High', 'adjLow': 'Low', 'adjOpen': 'Open', 'adjVolume': 'Volume'}, inplace=True)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def _fetch_from_yfinance(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    async def _fetch_and_cache_ticker_data(self, ticker: str, data_type: str) -> Optional[pd.DataFrame]:
        cache_path = self._get_cache_path(ticker, data_type)
        if os.path.exists(cache_path):
            self.logger.info(f"Loading cached data for '{ticker}' from '{cache_path}'.")
            df = pd.read_csv(cache_path, index_col='Date', parse_dates=True)
            if not self._validate_data_contract(df, data_type) or not self._run_dqm_checks(df, ticker):
                self.logger.warning(f"Cached data for '{ticker}' is invalid. Refetching.")
                os.remove(cache_path) # Invalidate cache
            else:
                return df

        sorted_providers = sorted(
            self.ds_config.providers.items(),
            key=lambda item: item[1].priority,
            reverse=True
        )

        provider_map = {
            "tiingo": self._fetch_from_tiingo,
            "yfinance": self._fetch_from_yfinance,
        }

        for provider_name, provider_config in sorted_providers:
            if provider_name not in provider_map:
                continue

            try:
                self.logger.info(f"Attempting to fetch data for '{ticker}' from '{provider_name.upper()}'.")
                fetch_func = provider_map[provider_name]
                data = fetch_func(ticker, self.ds_config.start_date, self.ds_config.end_date)
                
                if data.empty:
                    self.logger.warning(f"Provider '{provider_name.upper()}' returned no data for '{ticker}'.")
                    continue

                is_valid = self._validate_data_contract(data, data_type)
                if not is_valid:
                    self.logger.critical(f"Data from provider '{provider_name.upper()}' for ticker '{ticker}' failed contract validation. Rejecting data.")
                    raise DataFetchError(f"Data from {provider_name.upper()} failed contract validation.")

                # --- [NEW] Perform Statistical DQM Checks ---
                dqm_passed = self._run_dqm_checks(data, ticker)
                if not dqm_passed:
                    self.logger.critical(f"Data from provider '{provider_name.upper()}' for ticker '{ticker}' failed DQM checks. Rejecting data.")
                    raise DataFetchError(f"Data from {provider_name.upper()} failed DQM checks.")
                
                # --- [NEW] Calculate and Record Freshness Metric ---
                if not data.empty:
                    last_date = data.index.max().date()
                    freshness = (date.today() - last_date).days
                    self.logger.info(f"Data for '{ticker}' from '{provider_name.upper()}' is {freshness} days old.")

                data.to_csv(cache_path)
                self.logger.info(f"Successfully fetched and cached data for '{ticker}' from '{provider_name.upper()}'.")
                return data

            except Exception as e:
                self.logger.error(f"Failed to fetch from '{provider_name.upper()}' for '{ticker}': {e}. Trying next provider.")
                continue

        self.logger.error(f"Failed to fetch data for '{ticker}' from all available providers.")
        return None

    async def get_all_data(self) -> Dict[str, pd.DataFrame]:
        tasks = []
        all_tickers = self.config.asset_universe + ["SPY", "^VIX", "^TNX"]
        for ticker in all_tickers:
            data_type = "vix" if ticker == "^VIX" else "treasury_yield" if ticker == "^TNX" else "asset"
            tasks.append(self._fetch_and_cache_ticker_data(ticker, data_type))
        
        results = await asyncio.gather(*tasks)
        
        data_map = {}
        for ticker, df in zip(all_tickers, results):
            if df is not None:
                data_map[ticker] = df
        
        return data_map
        
# --- Main Application Logic ---
def setup_logging(log_level: str):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

async def precompute_asset_analyses(
    reasoning_ensemble: ReasoningEnsemble,
    retriever: HybridRetriever,
    dates: List[date],
    asset_universe: List[str],
    market_state_confidence_map: Dict[date, float]
) -> Dict[date, Dict]:
    """Pre-computes all AI analyses for all assets for all dates to prevent repeated API calls."""
    logger = logging.getLogger("PhoenixProject.AIPrecomputation")
    logger.info(f"Starting AI pre-computation for {len(dates)} dates and {len(asset_universe)} assets.")
    master_lookup = {d: {} for d in dates}

    for single_date in dates:
        # Get the pre-computed market state confidence for this specific day
        daily_market_confidence = market_state_confidence_map.get(single_date, 0.0)
        for ticker in asset_universe:
            logger.debug(f"Analyzing {ticker} for date {single_date.isoformat()}...")
            # In a real system, the query would be more sophisticated, perhaps incorporating the date.
            query = f"What is the near-term outlook for {ticker}?"
            evidence_list = await retriever.retrieve(query=query, ticker=ticker)
            hypothesis = f"Assess the investment-worthiness of {ticker} for the near-term."
            analysis_result = await reasoning_ensemble.analyze(hypothesis, evidence_list, market_state_confidence=daily_market_confidence)
            master_lookup[single_date][ticker] = analysis_result

    logger.info("AI pre-computation complete.")
    return master_lookup

async def run_single_backtest(config: StrategyConfig, all_aligned_data: Dict, reasoning_ensemble: Optional[ReasoningEnsemble] = None, retriever: Optional[HybridRetriever] = None):
    logger = logging.getLogger("PhoenixProject")
    logger.info(f"--- Launching 'Phoenix Project' in SINGLE BACKTEST mode (AI: {config.ai_mode.upper()}) ---")

    # --- [NEW] Train and prepare the Market State Predictor ---
    market_state_predictor = None
    market_state_confidence_map = {} # This will hold date -> confidence
    if config.market_state_predictor:
        market_state_predictor = MarketStatePredictor(config.market_state_predictor.dict())
        market_proxy_df = all_aligned_data["asset_universe_df"].xs('SPY', level=1, axis=1).copy()
        market_proxy_df.columns = [col.lower() for col in market_proxy_df.columns]
        labels = generate_market_state_labels(market_proxy_df, config.market_state_predictor.long_term_ma, config.market_state_predictor.threshold)
        
        macro_features_df = pd.concat([all_aligned_data["vix"], all_aligned_data["treasury_yield"], all_aligned_data["market_breadth"]], axis=1)
        macro_features_df.columns = ['vix', 'yield', 'breadth']
        
        aligned_data = macro_features_df.join(labels, how='inner').dropna()
        market_state_predictor.train(aligned_data[['vix', 'yield', 'breadth']], aligned_data['market_state'])
        
        # Pre-compute market state confidence for all days
        daily_predictions = market_state_predictor.model.predict_proba(macro_features_df.dropna())
        daily_confidence = np.max(daily_predictions, axis=1) # The confidence is the probability of the predicted class
        daily_confidence_series = pd.Series(daily_confidence, index=macro_features_df.dropna().index)
        market_state_confidence_map = {k.date(): v for k, v in daily_confidence_series.to_dict().items()}

    master_dates = [dt.to_pydatetime().date() for dt in all_aligned_data["asset_universe_df"].index]
    asset_analysis_lookup = {}
    if reasoning_ensemble and retriever and config.ai_mode != "off":
        asset_analysis_lookup = await precompute_asset_analyses(reasoning_ensemble, retriever, master_dates, config.asset_universe, market_state_confidence_map)

    cerebro = bt.Cerebro()
    try:
        for ticker in config.asset_universe:
            df = all_aligned_data["asset_universe_df"].xs(ticker, level=1, axis=1)
            df.columns = [col.lower() for col in df.columns] # Ensure lowercase column names
            data_feed = bt.feeds.PandasData(dataname=df, datetime=None, open=0, high=1, low=2, close=3, volume=4, openinterest=-1)
            cerebro.adddata(data_feed, name=ticker)
    except KeyError as e:
        logger.error(f"Data for ticker {e} not found in the aligned DataFrame. It might have been dropped due to inconsistencies.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while adding data feeds to Cerebro: {e}")
        return

    cerebro.addstrategy(
        RomanLegionStrategy, config=config, vix_data=all_aligned_data["vix"],
        treasury_yield_data=all_aligned_data["treasury_yield"], market_breadth_data=all_aligned_data["market_breadth"], market_state_predictor=market_state_predictor,
        sentiment_data={}, asset_analysis_data=asset_analysis_lookup
    )
    cerebro.broker.setcash(config.initial_cash)
    cerebro.broker.setcommission(commission=config.commission_rate)
    
    logger.info(f"Starting portfolio value: {cerebro.broker.getvalue():,.2f}")
    results = cerebro.run()
    logger.info(f"Final portfolio value: {cerebro.broker.getvalue():,.2f}")
    # cerebro.plot() # Optional plotting

def align_dataframes(data_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    # ... [Implementation remains the same] ...
    pass

async def main():
    load_dotenv()
    with open("config.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    config = StrategyConfig(**config_dict)

    setup_logging(config.log_level)
    logger = logging.getLogger("PhoenixProject")

    data_manager = DataManager(config)
    raw_data_map = await data_manager.get_all_data()

    if not raw_data_map or "SPY" not in raw_data_map:
        logger.critical("Could not load critical market data (SPY). Exiting.")
        return

    all_aligned_data = align_dataframes(raw_data_map)
    if not all_aligned_data:
        return

    with open(config.ai_clients_config_path, 'r') as f:
        ai_config = yaml.safe_load(f)

    vector_db_client = VectorDBClient(config=ai_config['vector_database'])
    # Initialize other clients as needed
    temporal_db_client = TemporalDBClient()
    tabular_db_client = TabularDBClient()
    
    retriever = HybridRetriever(
        vector_db_client=vector_db_client,
        temporal_db_client=temporal_db_client,
        tabular_db_client=tabular_db_client,
        rerank_config=ai_config['retriever']
    )
    
    ensemble_ai_client = EnsembleAIClient(config.ai_clients_config_path)
    reasoning_ensemble = ensemble_ai_client.ensemble

    await run_single_backtest(config, all_aligned_data, reasoning_ensemble, retriever)


if __name__ == "__main__":
    asyncio.run(main())
