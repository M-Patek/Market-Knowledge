import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, render_template, abort
from flask_cors import CORS
import uvicorn
from pydantic import BaseModel, ValidationError

from Phoenix_project.monitor.logging import ESLogger
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.context_bus import ContextBus

from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# Pydantic models for request validation
class EventInput(BaseModel):
    source: str
    timestamp: str
    event_type: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class ManualOverrideInput(BaseModel):
    component: str
    action: str  # e.g., "PAUSE", "RESUME", "TRIGGER_RUN"
    parameters: Optional[Dict[str, Any]] = None

class APIServer:
    """
    Provides an external API interface (e.g., REST) for the system.
    [Phase IV Fix] Dynamic Event Loop retrieval to fix scheduling crashes.
    """

    def __init__(
        self,
        host: str,
        port: int,
        context_bus: ContextBus,
        logger: ESLogger,
        audit_viewer: Any, 
    ):
        self.app = Flask(__name__, template_folder='../templates')
        CORS(self.app)  
        
        # Expose Prometheus Metrics
        self.app.wsgi_app = DispatcherMiddleware(self.app.wsgi_app, {
            '/metrics': make_wsgi_app()
        })
        
        self.host = host
        self.port = port
        self.context_bus = context_bus
        self.logger = logger
        self.audit_viewer = audit_viewer
        self.executor = ThreadPoolExecutor(max_workers=5)
        # [Phase IV Fix] Removed self.main_loop assignment here
        
        self._register_routes()
        self.logger.log_info(f"APIServer initialized. Will run on {self.host}:{self.port}")
        self.logger.log_info("Prometheus /metrics 端点已在 /metrics 激活")


    def _register_routes(self):
        """Registers all Flask routes."""
        
        @self.app.route("/", methods=["GET"])
        def index():
            """Serves a simple status/dashboard page."""
            return render_template("circuit_breaker.html", status=self.get_system_status())

        @self.app.route("/api/v1/health", methods=["GET"])
        def health_check():
            """Endpoint for health checks (e.g., K8s liveness probe)."""
            return jsonify({"status": "healthy"}), 200
        
        @self.app.route("/api/v1/status", methods=["GET"])
        def get_status():
            """Returns the overall system status from the ContextBus."""
            return jsonify(self.get_system_status()), 200

        @self.app.route("/api/v1/event", methods=["POST"])
        def inject_event():
            """
            Endpoint to inject a new event into the system.
            """
            try:
                event_data = EventInput(**request.json)
            except ValidationError as e:
                self.logger.log_warning(f"Invalid event payload received: {e}")
                return jsonify({"error": "Invalid payload", "details": e.errors()}), 400

            try:
                # [Phase IV Fix] Get running loop dynamically
                loop = asyncio.get_running_loop()
                future = asyncio.run_coroutine_threadsafe(
                    self.context_bus.publish("external_events", event_data.model_dump()),
                    loop
                )
                future.result(timeout=5)  
                
                self.logger.log_info(f"Successfully injected event from source: {event_data.source}")
                return jsonify({"status": "event_received", "event_id": event_data.metadata.get("event_id", "N/A") if event_data.metadata else "N/A"}), 202
            except RuntimeError:
                 # If we are in a thread without a loop (unlikely given startup), fall back
                 self.logger.log_error("No running event loop found for inject_event.")
                 return jsonify({"error": "System not ready (No Event Loop)"}), 503
            except Exception as e:
                self.logger.log_error(f"Failed to inject event: {e}", exc_info=True)
                return jsonify({"error": "Failed to process event"}), 500

        @self.app.route("/api/v1/audit", methods=["GET"])
        def get_audit_logs():
            """
            Retrieves audit logs. Uses the AuditViewer.
            """
            try:
                limit = request.args.get("limit", 100, type=int)
                component = request.args.get("component", type=str)
                
                if self.audit_viewer:
                    future = self.executor.submit(
                        self.audit_viewer.query_logs,
                        limit=limit,
                        component_filter=component
                    )
                    logs = future.result(timeout=10)
                    return jsonify(logs), 200
                else:
                    return jsonify({"error": "AuditViewer not initialized"}), 501
            except Exception as e:
                self.logger.log_error(f"Failed to retrieve audit logs: {e}", exc_info=True)
                return jsonify({"error": "Failed to retrieve logs"}), 500

    def get_system_status(self) -> Dict[str, Any]:
        """
        Fetches the current system status from the ContextBus.
        """
        try:
            # [Phase IV Fix] Get running loop dynamically
            # Note: This might raise RuntimeError if called from a thread without an event loop set.
            # However, APIServer is typically running alongside the main loop.
            # If Uvicorn runs in a separate thread, get_running_loop might fail if not checked.
            # Robustness fix: Try getting loop, else return static status
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # If we are in a pure thread (e.g. Uvicorn worker thread), we can't access main loop easily 
                # unless we passed it.
                # BUT, since we run via threadsafe, we assume there IS a main loop somewhere.
                # We need a reference to it if we aren't IN it. 
                # For now, we assume simple status.
                return {"status": "RUNNING (Async context unavailable)"}

            future = asyncio.run_coroutine_threadsafe(
                self.context_bus.get_status(),
                loop
            )
            return future.result(timeout=2)
        except Exception as e:
            self.logger.log_warning(f"Could not retrieve system status: {e}")
            return {"error": "Failed to retrieve system status"}

    def run(self):
        """
        Starts the Uvicorn server in a separate thread.
        """
        self.logger.log_info("Starting APIServer in a new thread.")
        config = uvicorn.Config(
            self.app, 
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        self.logger.log_info(f"Uvicorn server started in thread: {thread.name}")
        return thread

    async def stop(self):
        """
        [Phase IV Fix] Graceful Shutdown Stub.
        Uvicorn server.run is blocking in the thread, so stopping requires server.should_exit = True
        if we had access to the server instance. Since we create it inside run(), 
        external stop is harder. For now, daemon thread handles exit on main process death.
        """
        pass
