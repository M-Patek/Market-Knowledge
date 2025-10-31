from flask import Flask, request, jsonify
from flask_cors import CORS
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from audit_manager import AuditManager
from cognitive.engine import CognitiveEngine
from data_manager import DataManager
from api.gemini_pool_manager import GeminiPoolManager
from registry import registry
from observability import get_logger
from observability.metrics import REQ_TOTAL
from pipeline_state import PipelineState
from pipeline_orchestrator import PipelineOrchestrator
from schemas.fusion_result import FusionResult
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry import trace
from opentelemetry.instrumentation.flask import FlaskInstrumentor

# Configure logger for this module (Layer 12)
logger = get_logger("api_gateway", "api_gateway.log")

def create_app():
    """Application factory."""
    app = Flask(__name__)

    # Initialize OpenTelemetry
    provider = TracerProvider(resource=Resource.create({"service.name": "phoenix"}))
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    FlaskInstrumentor().instrument_app(app)
    
    CORS(app)

    # Initialize major components (Layer 1)
    engine = CognitiveEngine()
    data_manager = DataManager()
    audit_manager = AuditManager()
    gemini_pool = GeminiPoolManager()

    # Populate the registry (Layer 2)
    registry.set('cognitive_engine', engine)
    registry.set('data_manager', data_manager)
    registry.set('audit_manager', audit_manager)
    registry.set('gemini_pool', gemini_pool)
    
    logger.info("Phoenix API Gateway created and components registered.")

    @app.route('/analyze', methods=['GET'])
    def analyze_ticker():
        REQ_TOTAL.labels(path="/analyze").inc()
        ticker = request.args.get('ticker')
        logger.info(f"Received /analyze request for ticker: {ticker}")

        if not ticker:
            logger.warning("No ticker provided in /analyze request.")
            return jsonify({"error": "Ticker symbol is required"}), 400
        
        # TODO: This is where the full pipeline orchestration begins.
        # For now, we'll use a simplified flow or mock data.
        
        # Example of running the pipeline:
        state = PipelineState(ticker=ticker)
        
        # The Orchestrator uses the registered components to run the pipeline
        result: FusionResult = PipelineOrchestrator().run_full_analysis_pipeline(state)
        
        return jsonify(result.model_dump())

    @app.route('/circuit_breaker', methods=['GET'])
    def circuit_breaker_status():
        """Provides the status of the Gemini API circuit breaker."""
        status = registry.get('gemini_pool').get_breaker_status()
        
        # Example mock response
        mock_response = {
            "state": status.get("state", "UNKNOWN"),
            "failure_count": status.get("failures", 0),
            "success_count": status.get("successes", 0),
            "total_requests": status.get("total", 0)
        }
        
        return jsonify(mock_response), 200

    @app.get("/metrics")
    def metrics():
        return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=True)
