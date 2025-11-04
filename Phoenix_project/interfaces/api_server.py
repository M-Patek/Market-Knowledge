import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, render_template, abort
from flask_cors import CORS
import uvicorn
from pydantic import BaseModel, ValidationError

# 修正：将 'monitor.logging...' 转换为 'Phoenix_project.monitor.logging...'
from Phoenix_project.monitor.logging import ESLogger
# 修正：将 'core.pipeline_state...' 转换为 'Phoenix_project.core.pipeline_state...'
from Phoenix_project.core.pipeline_state import PipelineState
# 修正：将 'context_bus' 转换为 'Phoenix_project.context_bus'
from Phoenix_project.context_bus import ContextBus

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

    This server allows external systems to:
    1. Inject new data/events into the system.
    2. Query the current state of the system or specific components.
    3. Manually override or control system behavior (e.g., pause, resume).
    4. View system audit logs or performance metrics.
    """

    def __init__(
        self,
        host: str,
        port: int,
        context_bus: ContextBus,
        logger: ESLogger,
        audit_viewer: Any,  # Replace with actual AuditViewer class
    ):
        self.app = Flask(__name__, template_folder='../templates')
        CORS(self.app)  # Enable CORS for all routes
        self.host = host
        self.port = port
        self.context_bus = context_bus
        self.logger = logger
        self.audit_viewer = audit_viewer
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.main_loop = asyncio.get_event_loop()
        
        self._register_routes()
        self.logger.log_info(f"APIServer initialized. Will run on {self.host}:{self.port}")

    def _register_routes(self):
        """Registers all Flask routes."""
        
        @self.app.route("/", methods=["GET"])
        def index():
            """Serves a simple status/dashboard page."""
            # This could be expanded to a rich dashboard
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
            The event is published to the 'external_events' topic.
            """
            try:
                event_data = EventInput(**request.json)
            except ValidationError as e:
                self.logger.log_warning(f"Invalid event payload received: {e}")
                return jsonify({"error": "Invalid payload", "details": e.errors()}), 400

            try:
                # Run the async publish in the main event loop
                future = asyncio.run_coroutine_threadsafe(
                    self.context_bus.publish("external_events", event_data.model_dump()),
                    self.main_loop
                )
                future.result(timeout=5)  # Wait for confirmation
                
                self.logger.log_info(f"Successfully injected event from source: {event_data.source}")
                return jsonify({"status": "event_received", "event_id": event_data.metadata.get("event_id", "N/A") if event_data.metadata else "N/A"}), 202
            except Exception as e:
                self.logger.log_error(f"Failed to inject event: {e}", exc_info=True)
                return jsonify({"error": "Failed to process event"}), 500

        @self.app.route("/api/v1/control", methods=["POST"])
        def manual_override():
            """
            Endpoint for manual control/override of system components.
            Publishes a control message to the ContextBus.
            """
            try:
                control_data = ManualOverrideInput(**request.json)
            except ValidationError as e:
                self.logger.log_warning(f"Invalid control payload received: {e}")
                return jsonify({"error": "Invalid payload", "details": e.errors()}), 400

            try:
                topic = f"control_{control_data.component}"
                message = {
                    "action": control_data.action,
                    "parameters": control_data.parameters
                }
                
                # Run the async publish in the main event loop
                future = asyncio.run_coroutine_threadsafe(
                    self.context_bus.publish(topic, message),
                    self.main_loop
                )
                future.result(timeout=5)

                self.logger.log_info(f"Manual control command '{control_data.action}' sent to '{control_data.component}'")
                return jsonify({"status": "command_sent", "topic": topic, "action": control_data.action}), 200
            except Exception as e:
                self.logger.log_error(f"Failed to send control command: {e}", exc_info=True)
                return jsonify({"error": "Failed to process command"}), 500

        @self.app.route("/api/v1/audit", methods=["GET"])
        def get_audit_logs():
            """
            Retrieves audit logs. Uses the AuditViewer.
            """
            try:
                limit = request.args.get("limit", 100, type=int)
                component = request.args.get("component", type=str)
                
                # Run the sync audit_viewer call in a separate thread
                future = self.executor.submit(
                    self.audit_viewer.query_logs,
                    limit=limit,
                    component_filter=component
                )
                logs = future.result(timeout=10)
                
                return jsonify(logs), 200
            except Exception as e:
                self.logger.log_error(f"Failed to retrieve audit logs: {e}", exc_info=True)
                return jsonify({"error": "Failed to retrieve logs"}), 500

    def get_system_status(self) -> Dict[str, Any]:
        """
        Fetches the current system status from the ContextBus.
        This is a synchronous helper.
        """
        try:
            # We need to run the async get_status in the main loop
            future = asyncio.run_coroutine_threadsafe(
                self.context_bus.get_status(),
                self.main_loop
            )
            return future.result(timeout=2)
        except Exception as e:
            self.logger.log_warning(f"Could not retrieve system status: {e}")
            return {"error": "Failed to retrieve system status"}

    def run(self):
        """
        Starts the Uvicorn server in a separate thread.
        This allows the APIServer to run alongside the main async application.
        """
        self.logger.log_info("Starting APIServer in a new thread.")
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        
        # Run the server in a separate thread
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        self.logger.log_info(f"Uvicorn server started in thread: {thread.name}")
        return thread

if __name__ == "__main__":
    # Example usage (for testing)
    
    # Mock components
    class MockContextBus:
        async def publish(self, topic, message):
            print(f"MockPublish to {topic}: {message}")
            return True
        
        async def get_status(self):
            print("MockGetStatus called")
            return {"main_loop": "RUNNING", "last_event_time": "2023-01-01T12:00:00Z"}
            
    class MockLogger:
        def log_info(self, msg): print(f"INFO: {msg}")
        def log_warning(self, msg): print(f"WARNING: {msg}")
        def log_error(self, msg, exc_info=None): print(f"ERROR: {msg}")

    class MockAuditViewer:
        def query_logs(self, limit, component_filter):
            print(f"MockQueryLogs: limit={limit}, component={component_filter}")
            return [{"timestamp": "2023-01-01T12:00:00Z", "component": "test", "message": "Test log"}]

    # Set up a new event loop for the main thread
    main_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(main_loop)

    mock_bus = MockContextBus()
    mock_logger = MockLogger()
    mock_audit = MockAuditViewer()
    
    api_server = APIServer(
        host="0.0.0.0",
        port=8080,
        context_bus=mock_bus,
        logger=mock_logger,
        audit_viewer=mock_audit
    )
    
    # Run the server in a thread (as it would be in the main app)
    api_server.run()
    
    print("API Server is running in the background.")
    print("Access http://0.0.0.0:8080/ or http://0.0.0.0:8080/api/v1/health")
    
    try:
        # Keep the main thread alive to let the server thread run
        main_loop.run_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
        main_loop.stop()
