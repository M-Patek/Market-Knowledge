from flask import Flask, request, jsonify
from flask_cors import CORS
from audit_manager import AuditManager
from cognitive.engine import CognitiveEngine
from data_manager import DataManager
from api.gemini_pool_manager import GeminiPoolManager
from registry import registry
from observability import get_logger
from pipeline_state import PipelineState

# Configure logger for this module (Layer 12)
logger = get_logger(__name__)


def create_app():
    """
    Factory function to create and configure the Flask app.
    """
    app = Flask(__name__)
    CORS(app)

    # Resolve dependencies from the central registry (Layer 11)
    audit_manager: AuditManager = registry.resolve("audit_manager")
    data_manager: DataManager = registry.resolve("data_manager")
    cognitive_engine: CognitiveEngine = registry.resolve("cognitive_engine")
    gemini_pool: GeminiPoolManager = registry.resolve("gemini_pool")

    @app.route('/analyze', methods=['GET'])
    def analyze_ticker():
        ticker = request.args.get('ticker')
        logger.info(f"Received /analyze request for ticker: {ticker}")

        if not ticker:
            logger.warning("Request received without ticker parameter.")
            return jsonify({"error": "Ticker parameter is required"}), 400

        # Create a mock data event for the ticker
        data_event = {
            "ticker": ticker,
            "type": "api_request",
            "source": "gateway"
        }

        # Call the full pipeline via the cognitive engine (Layer 9)
        result_state: PipelineState = cognitive_engine.run_single_event(data_event)

        # Mock output as FusionResult and StrategySignal aren't defined yet
        mock_response = {
            "FusionResult": f"Mock fusion result for {ticker}",
            "StrategySignal": f"Mock strategy signal for {ticker}",
            "state_ticker": result_state.ticker
        }

        return jsonify(mock_response), 200

    return app
