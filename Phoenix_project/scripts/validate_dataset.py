import yaml
import pandas as pd
import sys
from datetime import datetime, timedelta

def run_dqm_checks():
    """
    Runs a suite of Data Quality Monitoring (DQM) checks against a dataset.
    Exits with a non-zero status code on failure.
    """
    try:
        print("--- Running Data Quality Monitoring (DQM) Checks ---")

        # 1. Load Configuration
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        dqm_config = config.get('data_quality_monitoring', {})
        completeness_threshold = dqm_config.get('completeness_threshold', 0.98)
        staleness_days = dqm_config.get('staleness_days_threshold', 3)

        print(f"DQM Config: Completeness >= {completeness_threshold:.2%}, Staleness <= {staleness_days} days")

        # 2. Load a representative dataset (e.g., SPY data)
        # In a real CI job, we'd fetch this from a designated test data store.
        # Here, we'll create a dummy dataframe for a robust demonstration.
        dates = pd.to_datetime(pd.date_range(end=datetime.today() - timedelta(days=2), periods=100))
        data = pd.DataFrame({
            'Close': range(100)
        }, index=dates)
        # Introduce some missing data to test the check
        data.iloc[10:15, 0] = None
        print(f"Loaded sample dataset with {len(data)} rows, from {data.index.min().date()} to {data.index.max().date()}")


        # 3. Perform Checks
        # a) Completeness Check
        completeness = data['Close'].notna().mean()
        print(f"Checking completeness... Found: {completeness:.2%}")
        if completeness < completeness_threshold:
            print(f"❌ DQM FAILED: Data completeness ({completeness:.2%}) is below the threshold of {completeness_threshold:.2%}.")
            sys.exit(1)
        print("✅ Completeness check passed.")

        # b) Staleness Check
        days_since_last_point = (datetime.now() - data.index.max()).days
        print(f"Checking staleness... Last data point is {days_since_last_point} days old.")
        if days_since_last_point > staleness_days:
            print(f"❌ DQM FAILED: Data is stale ({days_since_last_point} days old), exceeding the threshold of {staleness_days} days.")
            sys.exit(1)
        print("✅ Staleness check passed.")

        print("\n--- ✅ All DQM checks passed successfully! ---")
        sys.exit(0)

    except Exception as e:
        print(f"❌ An unexpected error occurred during DQM checks: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_dqm_checks()

