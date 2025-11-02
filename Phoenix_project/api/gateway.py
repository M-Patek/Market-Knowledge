"""
API Gateway (FastAPI Server)

This module provides the external-facing REST API for the Phoenix project.
It allows:
- Health checks.
- Manually injecting events (e.g., from a webhook).
- Querying the system state (e.g., current portfolio).
- Triggering specific tasks (e.g., a manual rebalance).
"""
import logging
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Body
from typing import Dict, Any

# 修复：添加 pandas 导入
import pandas as pd

from ..core.schemas.data_schema import MarketEvent
from ..controller.orchestrator import Orchestrator
from ..core.pipeline_state import PipelineState
# A dependency injection system would be better here
# For now, we assume a global or passed 'orchestrator' instance

logger = logging.getLogger(__name__)

# --- Globals (to be replaced by proper dependency injection) ---
# This is simplified for the example. In a real app,
# the orchestrator would be initialized and passed properly.
orchestrator: Orchestrator = None
pipeline_state: PipelineState = None

def get_orchestrator() -> Orchestrator:
    """Dependency injector for the Orchestrator."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator is not initialized.")
    return orchestrator

def get_pipeline_state() -> PipelineState:
    """Dependency injector for the PipelineState."""
    if pipeline_state is None:
        raise HTTPException(status_code=503, detail="PipelineState is not initialized.")
    return pipeline_state

# --- FastAPI App ---
app = FastAPI(
    title="Phoenix Project API Gateway",
    description="API for interacting with the Phoenix cognitive trading system."
)

@app.on_event("startup")
async def startup_event():
    """
    Handles application startup.
    In a real app, this is where you'd initialize the main components
    (Orchestrator, DB connections, etc.) and assign them to the globals.
    """
    logger.info("API Gateway starting up...")
    # --- This is where the main app components would be built ---
    # global orchestrator, pipeline_state
    # config = ConfigLoader(...)
    # pipeline_state = PipelineState(...)
    # ... (build all dependencies)
    # orchestrator = Orchestrator(...)
    logger.warning("API running in 'stub' mode. Orchestrator/State are not initialized.")
    # For testing, we can create dummy objects
    if pipeline_state is None:
        global pipeline_state
        pipeline_state = PipelineState(initial_capital=100000)
    logger.info("Dummy PipelineState created for API testing.")


@app.get("/health", tags=["System"])
async def get_health():
    """Returns a health check response."""
    return {
        "status": "ok",
        "timestamp": pd.Timestamp.now().isoformat(),
        "service": "Phoenix API Gateway"
    }

@app.post("/inject/market_event", tags=["Events"])
async def inject_market_event(
    event: MarketEvent,
    orch: Orchestrator = Depends(get_orchestrator)
):
    """
    Manually injects a MarketEvent into the system.
    This bypasses the StreamProcessor.
    """
    try:
        logger.info(f"API: Injecting event_id: {event.event_id}")
        # The orchestrator handles this asynchronously
        await orch.on_event(event)
        return {"status": "success", "message": "Event received and queued."}
    except Exception as e:
        logger.error(f"API: Error injecting event: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state/portfolio", tags=["State"])
async def get_portfolio_state(
    state: PipelineState = Depends(get_pipeline_state)
):
    """
    Returns the current portfolio state (positions, cash, value).
    """
    try:
        return {
            "timestamp": state.get_current_time(),
            "total_value": state.get_total_portfolio_value(),
            "positions": state.get_all_positions(),
            "cash": state.get_cash()
        }
    except Exception as e:
        logger.error(f"API: Error getting portfolio state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/actions/trigger_rebalance", tags=["Actions"])
async def trigger_rebalance(
    orch: Orchestrator = Depends(get_orchestrator)
):
    """
    Manually triggers a scheduled task (e.g., "daily_rebalance").
    """
    try:
        task_name = "manual_rebalance_trigger"
        trigger_time = pd.Timestamp.now().to_pydatetime()
        logger.info(f"API: Manually triggering task: {task_name}")
        
        # This will dispatch the task via the orchestrator
        await orch.on_scheduled_task(task_name, trigger_time)
        
        return {"status": "success", "message": f"Task '{task_name}' dispatched."}
    except Exception as e:
        logger.error(f"API: Error triggering rebalance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    """
    Main entry point to run the API server directly.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("Starting API Gateway server...")
    
    # This assumes the main components (Orchestrator, etc.)
    # are initialized elsewhere and passed to this app.
    # For standalone running, the `startup_event` will log warnings.
    
    uvicorn.run(
        "api.gateway:app", # Points to this file (gateway.py) and the app object
        host="0.0.0.0",
        port=8000,
        reload=True, # Enable reload for development
        log_level="info"
    )
