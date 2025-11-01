from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List
import uvicorn
import asyncio

from ..monitor.logging import get_logger
from ..controller.orchestrator import Orchestrator
from ..context_bus import ContextBus
from ..core.schemas.data_schema import MarketEvent

logger = get_logger(__name__)

class APIGateway:
    """
    Provides an external REST API interface for the Phoenix project.
    Allows injecting events and querying system state.
    """
    
    def __init__(self, orchestrator: Orchestrator, context_bus: ContextBus, config: Dict[str, Any]):
        self.app = FastAPI(
            title="Phoenix Project API Gateway",
            description="API for event injection and system control.",
            version="2.0.0"
        )
        self.orchestrator = orchestrator
        self.context_bus = context_bus
        self.config = config.get('api_gateway', {})
        self.host = self.config.get('host', '0.0.0.0')
        self.port = self.config.get('port', 8000)
        
        self._setup_routes()
        logger.info(f"API Gateway initialized. Routes registered.")

    def _setup_routes(self):
        """Binds API endpoints to their handler functions."""
        
        @self.app.post("/v1/events/inject", status_code=202)
        async def inject_event(event: MarketEvent):
            """
            Asynchronously injects a new MarketEvent into the system.
            This simulates an event coming from a data stream.
            """
            try:
                # Don't block the API response; schedule the processing
                # The orchestrator is expected to handle this asynchronously
                asyncio.create_task(self.orchestrator.process_event(event))
                
                logger.info(f"Accepted event for injection: {event.event_id}")
                return {"status": "accepted", "event_id": event.event_id}
            
            except Exception as e:
                logger.error(f"Failed to accept event {event.event_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

        @self.app.get("/v1/system/status")
        async def get_system_status(self) -> Dict[str, Any]:
            """
Example:
            {"status": "running", "active_tasks": 5, "mode": "paper_trading"}
            """
            try:
                status = self.orchestrator.get_status()
                return status
            except Exception as e:
                logger.error(f"Failed to retrieve system status: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

        @self.app.get("/v1/context/state")
        async def get_context_state(self) -> Dict[str, Any]:
            """
            Retrieves the current state snapshot from the ContextBus.
            """
            try:
                state = self.context_bus.get_current_state()
                # We need to serialize this, as it might contain complex objects
                # For now, just return the dict representation
                # A proper implementation would use Pydantic models
                return {
                    "last_update": state.get('timestamp'),
                    "active_symbols": list(state.get('market_data', {}).keys()),
                    "recent_events_count": len(state.get('recent_events', []))
                }
            except Exception as e:
                logger.error(f"Failed to retrieve context state: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
                
        @self.app.get("/health")
        async def health_check():
            """Basic health check endpoint."""
            return {"status": "ok"}

    def run(self):
        """
        Starts the FastAPI server using uvicorn.
        This is a blocking call.
        """
        logger.info(f"Starting API Gateway server on {self.host}:{self.port}")
        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
        except Exception as e:
            logger.error(f"API Gateway server failed to start: {e}", exc_info=True)
            raise

# Example usage (if run as main)
if __name__ == "__main__":
    # This is for testing purposes only.
    # In production, this would be launched by the main phoenix_project.py
    
    logger.info("Running API Gateway in standalone test mode.")
    
    # Mock dependencies
    class MockOrchestrator:
        async def process_event(self, event):
            logger.info(f"[Mock] Processing event: {event.event_id}")
            await asyncio.sleep(0.1) # Simulate async work
        
        def get_status(self):
            return {"status": "running_mock", "active_tasks": 0, "mode": "mock"}
            
    class MockContextBus:
        def get_current_state(self):
            return {"timestamp": pd.Timestamp.now(), "market_data": {"AAPL": {}}, "recent_events": []}

    mock_config = {"api_gateway": {"host": "127.0.0.1", "port": 8000}}
    
    gateway = APIGateway(
        orchestrator=MockOrchestrator(),
        context_bus=MockContextBus(),
        config=mock_config
    )
    gateway.run()
