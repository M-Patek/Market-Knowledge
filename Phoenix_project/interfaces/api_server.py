# Phoenix_project/interfaces/api_server.py
import logging
from typing import Dict, Any
from fastapi import FastAPI
import uvicorn
from controller.loop_manager import control_loop # 我们的主管道入口 (Task 18)

app = FastAPI(
    title="Phoenix Project API",
    description="API server to trigger analysis (Task 21)"
)

logger = logging.getLogger("PhoenixProject.APIServer")


@app.post("/analyze")
async def trigger_analysis(task: dict) -> dict:
    """
    Route: POST /analyze { "ticker": "NVDA" }
    Request must return a complete JSON report.
    """
    logger.info(f"Received analysis request for: {task.get('ticker', 'UNKNOWN')}")
    # 这个调用会触发整个管道: 
    # loop -> orchestrator -> plan -> execute -> evaluate -> fuse -> guard
    final_report = await control_loop(task)
    return final_report


if __name__ == "__main__":
    # 这允许直接运行服务器进行测试。
    uvicorn.run(app, host="0.0.0.0", port=8000)
