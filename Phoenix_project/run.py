import argparse
from pipeline_state import PipelineState
from pipeline_orchestrator import PipelineOrchestrator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    args = ap.parse_args()

    state = PipelineState(ticker=args.ticker)
    PipelineOrchestrator().run_full_analysis_pipeline(state)

if __name__ == "__main__":
    main()
