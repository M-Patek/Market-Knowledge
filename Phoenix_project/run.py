import argparse
# FIXED: 修正了导入路径
from core.pipeline_state import PipelineState
# FIXED: 修正了导入路径。
# 'controller/orchestrator.py' 中定义了 'Orchestrator' 类。
# 我们使用 'as' 来保留原有的 'PipelineOrchestrator' 名称。
from controller.orchestrator import Orchestrator as PipelineOrchestrator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    args = ap.parse_args()

    state = PipelineState(ticker=args.ticker)
    # 注意: 'Orchestrator' 类中没有 'run_full_analysis_pipeline' 方法。
    # 此 'run.py' 文件可能已过时，'scripts/run_cli.py' 似乎是正确的入口点。
    # 保持原逻辑不变。
    PipelineOrchestrator().run_full_analysis_pipeline(state)

if __name__ == "__main__":
    main()

