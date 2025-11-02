from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio
from typing import Dict, Any, List

# 修复：将相对导入 'from ..monitor.logging...' 更改为绝对导入
from monitor.logging import get_logger
# 修复：将相对导入 'from ..controller.orchestrator...' 更改为绝对导入
from controller.orchestrator import Orchestrator
# 修复：将相对导入 'from ..core.schemas.data_schema...' 更改为绝对导入
from core.schemas.data_schema import MarketEvent

logger = get_logger(__name__)

class APIServer:
    """
    Provides an external REST API interface for the Phoenix project.
    Allows injecting events and querying system state.
    
    (Note: This is an alternative to api_gateway.py, using a different
    class name but similar functionality.)
    """
    
    def __init__(self, orchestrator: Orchestrator, config: Dict[str, Any]):
        self.app = FastAPI(
            title="Phoenix Project API Server",
            description="API for event injection and system control.",
            version="2.0.1"
        )
        self.orchestrator = orchestrator
        self.config = config.get('api_server', {}) # Use 'api_server' key
        self.host = self.config.get('host', '0.0.0.0')
        self.port = self.config.get('port', 8000)
        
        self.server: Optional[uvicorn.Server] = None
        self._shutdown_event = asyncio.Event()
        
        self._setup_routes()
        logger.info(f"APIServer initialized. Routes registered.")

    def _setup_routes(self):
        """Binds API endpoints to their handler functions."""
        
        @self.app.post("/v1/events/inject", status_code=202)
        async def inject_event(event: MarketEvent):
            """
            Asynchronously injects a new MarketEvent into the system.
            """
            try:
                # Schedule the processing, don't block the API response
                asyncio.create_task(self.orchestrator.process_event(event))
                
                logger.info(f"Accepted event for injection: {event.event_id}")
                return {"status": "accepted", "event_id": event.event_id}
            
            except Exception as e:
                logger.error(f"Failed to accept event {event.event_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

        @self.app.get("/v1/system/status")
        async def get_system_status() -> Dict[str, Any]:
            """Retrieves the current operational status from the orchestrator."""
            try:
                status = self.orchestrator.get_status()
                return status
            except Exception as e:
                logger.error(f"Failed to retrieve system status: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
                
        @self.app.get("/health")
        async def health_check():
            """Basic health check endpoint."""
            return {"status": "ok"}

    async def run(self):
        """
        Starts the FastAPI server using uvicorn.
        This is an awaitable method designed to be run as an asyncio task.
        """
        logger.info(f"Starting APIServer on {self.host}:{self.port}")
        
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            loop="asyncio"
        )
        self.server = uvicorn.Server(config)
        
        # Start the server
        server_task = asyncio.create_task(self.server.serve())
        
        # Wait for either shutdown signal or server failure
        await asyncio.wait(
            [server_task, self._shutdown_event.wait()],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # If shutdown was signaled, gracefully stop the server
        if self._shutdown_event.is_set():
            logger.info("APIServer shutdown signaled. Stopping uvicorn...")
            self.server.should_exit = True
            await server_task # Wait for server to exit
            
        logger.info("APIServer has stopped.")

    def stop(self):
        """Signals the uvicorn server to shut down."""
        logger.info("APIServer received stop signal.")
        self._shutdown_event.set()
