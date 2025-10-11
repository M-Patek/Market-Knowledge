Model Card: Phoenix Roman Legion Strategy
Version: 1.0 (Example Backtest Run)
Last Updated: 2025-10-11
Model Owner: Phoenix Project Team

1. Model Details
a. Overview
The Phoenix Roman Legion Strategy is a quantitative tactical asset allocation model. It operates on a universe of liquid ETFs and aims to dynamically allocate capital based on a combination of technical momentum indicators and macroeconomic risk assessments.

b. Model Type
Primary: Momentum & Trend Following

Secondary: Risk-Managed (VIX-based regime switching)

Cognitive Layer (Optional): AI-driven analysis of external information to produce a dynamic adjustment factor.

c. Intended Use & Scope
Primary Use Case: Research and backtesting of tactical asset allocation hypotheses on historical data.

Secondary Use Case: Simulated paper trading to evaluate performance in near-real-time conditions.

Out of Scope: This model is NOT intended for live, automated trading without significant further validation, regulatory compliance checks, and a human-in-the-loop oversight system. It does not constitute financial advice.

2. Technical Specifications
a. Data Sources
The model relies on the following data inputs:

Primary (OHLCV): Daily price and volume data for the asset universe, fetched via a prioritized list of providers (e.g., Alpha Vantage, Twelve Data).

Risk Indicator: CBOE Volatility Index (^VIX) daily closing values.

Market Breadth: Percentage of tickers in a defined universe trading above their 50-day SMA.

Cognitive Input (Optional): Unstructured text data (news, filings) processed by the AI Ensemble to generate analysis.

b. Key Parameters
The strategy's behavior is primarily governed by the config.yaml file. Key tunable parameters from an example run include:

sma_period: 50

rsi_period: 14

rsi_overbought_threshold: 70.0

opportunity_score_threshold: 55.0

vix_high_threshold: 30.0

vix_low_threshold: 20.0

position_sizer: fixed_fraction at 0.10 per position.

3. Performance (Example Data)
Performance metrics should be populated from a specific backtest report (phoenix_report.html).

Metric

Value

Description

Total Return

XX.XX%

Cumulative return over the backtest period.

Sharpe Ratio

X.XX

Risk-adjusted return (annualized).

Max Drawdown

YY.YY%

The largest peak-to-trough decline in portfolio value.

Win Rate

ZZ.ZZ%

Percentage of trades that were profitable.

Total Trades

NNN

The total number of trades executed.

4. Limitations & Ethical Considerations
a. Known Limitations
Historical Bias: The model's performance is based on historical data and does not guarantee future results.

Regime Sensitivity: The strategy is sensitive to changes in market regimes (e.g., sustained low-volatility trending markets vs. high-volatility choppy markets). Its parameters are optimized for a specific historical window and may not be optimal for all conditions.

Execution Slippage: The backtest relies on a simulated slippage model (OrderManager). Real-world slippage may be higher, impacting performance.

Data Provider Dependency: The system's reliability is dependent on the uptime and data quality of external API providers.

b. Ethical Considerations & Responsible AI
No Financial Advice: The model's output is for informational and research purposes only and must not be interpreted as financial advice.

AI Transparency: When the AI mode is enabled, all AI interactions, prompts, and outputs are logged in the ai_audit_logs/ directory to ensure full traceability and allow for bias auditing.

Human Oversight: The system is designed to be a decision-support tool, not a fully autonomous agent. All final trading decisions in a live environment should be subject to human review and approval.
