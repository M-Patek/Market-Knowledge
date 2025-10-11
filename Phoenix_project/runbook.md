Phoenix Project - Operational Runbook
Version: 1.0
Last Updated: 2025-10-11
Contact: Phoenix Project Team

1. System Overview
The Phoenix Project is a comprehensive quantitative trading research platform. It automates the entire backtesting pipeline, including data acquisition, feature calculation, AI-enhanced analysis, strategy execution simulation, and reporting.

The system can be run in two primary modes as configured in config.yaml:

Single Backtest: A single run of the strategy with a fixed set of parameters.

Walk-Forward Optimization: A rigorous, scientific process that repeatedly trains parameters on one historical period and validates them on a subsequent, unseen period.

2. Prerequisites & Setup
a. Clone the Repository
git clone <repository_url>
cd Phoenix_project

b. Environment Setup
Python Version: The project requires Python 3.11.

Install Dependencies:

pip install -r requirements.txt

c. Configure Environment Variables
The system requires API keys for data providers and AI models.

Copy the example environment file:

cp env.example .env

Open the .env file with a text editor.

Fill in your actual API keys for the services you intend to use (e.g., ALPHA_VANTAGE_API_KEY, GEMINI_API_KEY). The variable names must match those specified in config.yaml.

3. Core Operations
a. Configure the Run
All operations are controlled by the central config.yaml file. Before running, open this file and configure:

start_date & end_date

asset_universe

walk_forward -> enabled: true or enabled: false

ai_mode: Set to off, raw, or processed.

b. Running a Single Backtest
Ensure walk_forward: enabled is set to false in config.yaml.

Run the script:

python phoenix_project.py

Expected Output:

The console will show structured JSON logs for the backtest process.

Upon completion, a phoenix_report_single.html file and a phoenix_plot.png will be generated in the root directory.

c. Running a Walk-Forward Optimization
Ensure walk_forward: enabled is set to true in config.yaml.

Run the script:

python phoenix_project.py

Expected Output:

The console will log the progress for each training and testing window.

A phoenix_walk_forward_study.db (SQLite file) will be created to store optimization results.

An HTML report (phoenix_report_wf_X.html) will be generated for each out-of-sample test window.

4. Validation & Monitoring
a. Review the HTML Report
Open the generated .html report in a web browser. Key items to validate:

Final Portfolio Value & Total Return: The primary performance indicators.

Sharpe Ratio & Max Drawdown: Key risk-adjusted return metrics.

Trade Statistics: Check the total number of trades and the win rate.

Equity Curve Plot: Visually inspect the equity curve for periods of high volatility or long drawdowns.

b. Check Prometheus Metrics
While the script is running, the system exposes a Prometheus metrics endpoint.

URL: http://localhost:8000 (or the port configured in config.yaml)

Key Metrics to Monitor:

phoenix_cache_hits_total / phoenix_cache_misses_total: Monitor cache efficiency.

phoenix_provider_requests_total{provider="..."}: See which data providers are being used.

phoenix_provider_errors_total{provider="..."}: CRITICAL ALARM. A high or rapidly increasing number of errors indicates a problem with a data provider.

phoenix_ai_call_latency_seconds: Monitor the performance of the AI model APIs.

5. Troubleshooting & Critical Alarms
Alarm: phoenix_provider_errors_total is high.

Diagnosis: An external data provider is likely down or the API key is invalid.

Response:

Check the JSON logs for specific error messages related to the failing provider.

Verify the corresponding API key in your .env file is correct.

Check the status page of the data provider's website.

Mitigation: The system is designed with a fallback and circuit breaker mechanism. It will automatically try the next provider in the priority list. If all providers fail for a ticker, it will be excluded from that day's run.

Issue: Pydantic validation error on startup.

Diagnosis: The config.yaml file has an invalid value or a missing required field.

Response:

Carefully read the error message in the console. It will specify which field is incorrect (e.g., end_date must be strictly after start_date).

Correct the specified field in config.yaml and re-run the script.

Issue: AI analysis is not working.

Diagnosis: There may be an issue with the AI model API key, the prompt templates, or the network connection.

Response:

Verify the GEMINI_API_KEY in your .env file.

Check the ai_audit_logs/ directory. Each API call is logged here. Look for recent .json files with "success": false and inspect the "error" field for details.
