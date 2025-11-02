Phoenix Project - Operational Runbook
Version: 2.0 (Updated)
Last Updated: 2025-10-17
Contact: Phoenix Project Team

System Overview
The Phoenix Project is a comprehensive quantitative trading research platform. It automates the entire backtesting pipeline, including data acquisition, feature calculation, AI-enhanced analysis, strategy execution simulation, and reporting.

The system can be run in two primary modes as configured in config/system.yaml:

Single Backtest: A single run of the strategy with a fixed set of parameters.

Walk-Forward Optimization: A rigorous, scientific process that repeatedly trains parameters on one historical period and validates them on a subsequent, unseen period.

Prerequisites & Setup
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

Fill in your actual API keys for the services you intend to use (e.g., ALPHA_VANTAGE_API_KEY, GEMINI_API_KEY). The variable names must match those specified in config/system.yaml.

Core Operations
a. Configure the Run
All operations are controlled by the central configuration file.

[CRITICAL UPDATE] The main config file is config/system.yaml, not config.yaml.

Before running, open config/system.yaml and configure:

start_date & end_date

asset_universe

walk_forward -> enabled: true or enabled: false

ai_mode: Set to off, raw, or processed.

b. Running the System
The main entry point is phoenix_project.py. This script starts the main asynchronous loop for the Orchestrator and EventDistributor.

Run the script:

python phoenix_project.py

Note: This script is designed to run as a persistent service. For backtesting or walk-forward analysis (as mentioned in the config/system.yaml), you will likely use a different entry point (e.g., scripts/run_backtest.py or similar, which may need to be invoked via run_cli.py).

Expected Output:

The console will show structured JSON logs from loguru as the system initializes.

The system will log messages from the Orchestrator (e.g., running decision loop) and EventDistributor (e.g., consuming events).

c. Running Data Validation
To ensure data quality before ingestion, use the validation script:

python scripts/validate_dataset.py /path/to/your/data.jsonl --type market_event

Validation & Monitoring
a. Review the HTML Report
(For Backtesting/Walk-Forward runs) Open the generated .html report in a web browser. Key items to validate:

Final Portfolio Value & Total Return.

Sharpe Ratio & Max Drawdown.

Trade Statistics: Total trades and win rate.

Equity Curve Plot.

b. Check Prometheus Metrics
While the script is running, the system exposes a Prometheus metrics endpoint.

URL: http://localhost:8000 (or the port configured in config/system.yaml)

Key Metrics to Monitor:

phoenix_cache_hits_total / phoenix_cache_misses_total: Monitor cache efficiency.

phoenix_provider_requests_total{provider="..."}: See which data providers are being used.

phoenix_provider_errors_total{provider="..."}: CRITICAL ALARM. A high or rapidly increasing number of errors indicates a problem with a data provider.

phoenix_ai_call_latency_seconds: Monitor the performance of the AI model APIs (e.g., Gemini).

Troubleshooting & Critical Alarms
Alarm: phoenix_provider_errors_total is high.

Diagnosis: An external data provider is likely down or the API key is invalid.

Response:

Check the JSON logs for specific error messages related to the failing provider.

Verify the corresponding API key in your .env file is correct.

Check the status page of the data provider's website.

Mitigation: The system is designed with a fallback and circuit breaker mechanism. It will automatically try the next provider in the priority list.

Issue: Pydantic validation error on startup.

Diagnosis: The config/system.yaml file has an invalid value or a missing required field.

Response:

Carefully read the error message in the console. It will specify which field is incorrect (e.g., end_date must be strictly after start_date).

Correct the specified field in config/system.yaml and re-run the script.

Issue: AI analysis is not working (e.g., gemini_pool errors).

Diagnosis: There may be an issue with the AI model API key, the prompt templates, or the network connection.

Response:

Verify the GEMINI_API_KEY in your .env file.

Check the ai_audit_logs/ directory. Each API call is logged here. Look for recent .json files with "success": false and inspect the "error" field for details.
