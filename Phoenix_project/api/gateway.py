import os
from flask import Flask, request, jsonify, send_from_directory, g, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from logging.handlers import RotatingFileHandler
import yaml
import time
from functools import wraps

# Import project-specific modules
from ..cognitive.engine import CognitiveEngine
from ..data_manager import DataManager
from ..pipeline_orchestrator import PipelineOrchestrator
from ..observability import setup_logging, get_logger
from ..audit_manager import AuditManager
from ..api.gemini_pool_manager import GeminiPoolManager

# --- Global Variables ---
app = None
cognitive_engine = None
data_manager = None
pipeline_orchestrator = None
audit_manager = None
gemini_pool_manager = None

# --- Utility Functions ---

def load_config():
    """Loads the main configuration file."""
    # Assuming config.yaml is in the parent directory of 'api'
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    if not os.path.exists(config_path):
        # Fallback for different structures (e.g., running from root)
        config_path = 'config.yaml'
        if not os.path.exists(config_path):
            raise FileNotFoundError("config.yaml not found in expected locations.")
            
    with open(config_path, 'r') as f:
        return yaml.safe_load(config)

def setup_app_logging(app, config):
    """Sets up logging for the Flask app."""
    log_config = config.get('logging', {})
    log_file = log_config.get('file', 'logs/gateway.log')
    log_level = log_config.get('level', 'INFO')
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5)
    handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    app.logger.addHandler(handler)
    app.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Also configure the root logger for module-level logging
    setup_logging(config)

# --- Flask App Factory ---

def create_app():
    """Factory function to create the Flask application."""
    global app, cognitive_engine, data_manager, pipeline_orchestrator, audit_manager, gemini_pool_manager
    
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    CORS(app) # Enable CORS for all routes
    
    # Load config
    config = load_config()
    app.config['UPLOAD_FOLDER'] = config.get('data_upload_path', 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB upload limit
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Setup logging
    setup_app_logging(app, config)
    
    # Initialize core components
    # (Assuming these classes can be initialized)
    # Note: In a real app, you might pass configs to these
    audit_manager = AuditManager()
    data_manager = DataManager(config=config)
    cognitive_engine = CognitiveEngine(config=config, data_manager=data_manager, audit_manager=audit_manager)
    
    # Initialize GeminiPoolManager
    gemini_keys = os.getenv('GEMINI_API_KEYS')
    if not gemini_keys:
        app.logger.warning("GEMINI_API_KEYS environment variable not set. Using fallback keys from config.")
        gemini_keys = config.get('api_keys', {}).get('gemini', [])
    else:
        gemini_keys = gemini_keys.split(',')
        
    if not gemini_keys:
        app.logger.error("No Gemini API keys found in env or config. GeminiPoolManager will fail.")
        gemini_keys = []

    gemini_pool_config = config.get('gemini_pool_manager', {})
    gemini_pool_manager = GeminiPoolManager(
        api_keys=gemini_keys,
        cooldown_time=gemini_pool_config.get('cooldown_time', 60),
        max_failures=gemini_pool_config.get('max_failures', 5),
        failure_window=gemini_pool_config.get('failure_window', 300),
        logger=app.logger
    )
    
    # --- Request Hooks ---
    
    @app.before_request
    def before_request_logging():
        app.logger.info(f"Request: {request.method} {request.path}")
        g.start_time = time.time()

    @app.after_request
    def after_request_logging(response):
        latency = (time.time() - g.start_time) * 1000 # in ms
        app.logger.info(f"Response: {request.method} {request.path} {response.status_code} - Latency: {latency:.2f}ms")
        return response
        
    # --- Error Handlers ---
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not Found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Internal Server Error: {error}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500

    # --- API Routes ---

    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy"}), 200

    @app.route('/admin/circuit-breaker', methods=['GET'])
    def circuit_breaker_admin():
        """
        Serves the admin interface for the API Circuit Breaker.
        """
        return render_template('circuit_breaker.html')

    @app.route('/api/v1/chat', methods=['POST'])
    def chat():
        """
        Main endpoint for chat interactions.
        ---
        """
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
            
        user_message = data['message']
        session_id = data.get('session_id', 'default_session')
        
        try:
            # This is a mock response. Replace with actual cognitive_engine call
            # response_message = cognitive_engine.process_message(user_message, session_id)
            
            # Mock implementation for GeminiPoolManager integration
            key = gemini_pool_manager.get_key()
            if not key:
                 return jsonify({"error": "All API keys are currently unavailable."}), 503
            
            # Simulate API call
            start_time = time.time()
            time.sleep(0.5) # Simulate work
            latency = (time.time() - start_time) * 1000
            
            # Simulate success/failure
            import random
            if random.random() < 0.1: # 10% failure rate
                gemini_pool_manager.report_failure(key)
                response_data = {"error": "Simulated API failure"}
                status_code = 500
            else:
                cost = random.uniform(0.001, 0.005) # Simulate cost
                gemini_pool_manager.report_success(key, latency, cost)
                response_data = {"response": f"Processed using key {key[:4]}...", "session_id": session_id}
                status_code = 200
            
            return jsonify(response_data), status_code
            
        except Exception as e:
            app.logger.error(f"Error in /api/v1/chat: {e}", exc_info=True)
            return jsonify({"error": f"An error occurred: {e}"}), 500

    @app.route('/api/circuit-breaker/status', methods=['GET'])
    def get_circuit_breaker_status():
        """
        Gets the status of all API keys from the GeminiPoolManager.
        ---
        """
        app.logger.info("Received request for /api/circuit-breaker/status")
        all_states = gemini_pool_manager.get_all_key_states()
        return jsonify(all_states), 200

    @app.route('/api/circuit-breaker/trip', methods=['POST'])
    def manual_trip_key():
        """
        Manually trips a specific API key.
        Expects JSON: {"key": "api_key_string"}
        """
        data = request.json
        key = data.get('key')
        if not key:
            return jsonify({"error": "Missing 'key' in request body"}), 400
        
        app.logger.info(f"Received manual trip request for key: {key}")
        success = gemini_pool_manager.manual_trip(key)
        if success:
            return jsonify({"status": "success", "message": f"Key {key} manually tripped."}), 200
        else:
            return jsonify({"status": "error", "message": f"Key {key} not found."}), 404

    @app.route('/api/circuit-breaker/restore', methods=['POST'])
    def manual_restore_key():
        """
        Manually restores a specific API key.
        Expects JSON: {"key": "api_key_string"}
        """
        data = request.json
        key = data.get('key')
        if not key:
            return jsonify({"error": "Missing 'key' in request body"}), 400

        app.logger.info(f"Received manual restore request for key: {key}")
        success = gemini_pool_manager.manual_restore(key)
        if success:
            return jsonify({"status": "success", "message": f"Key {key} manually restored."}), 200
        else:
            return jsonify({"status": "error", "message": f"Key {key} not found."}), 404

    @app.route('/api/v1/projects/<project_id>/files', methods=['POST'])
    def upload_file(project_id):
        """
        Endpoint for uploading files.
        """
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file:
            filename = secure_filename(file.filename)
            project_dir = os.path.join(app.config['UPLOAD_FOLDER'], project_id)
            os.makedirs(project_dir, exist_ok=True)
            file_path = os.path.join(project_dir, filename)
            file.save(file_path)
            
            # Notify DataManager
            try:
                data_manager.add_data_source(file_path, {'type': 'local_upload', 'project_id': project_id})
                return jsonify({"message": "File uploaded successfully", "filename": filename}), 201
            except Exception as e:
                app.logger.error(f"Error processing uploaded file: {e}", exc_info=True)
                return jsonify({"error": f"Failed to process file: {e}"}), 500

    # --- Static File Serving (for documentation, etc.) ---
    
    @app.route('/docs/<path:path>')
    def serve_docs(path):
        """Serves static documentation files."""
        return send_from_directory('docs', path)

    return app

# --- Main Entry Point ---
if __name__ == '__main__':
    # This is for development run only
    # For production, use a WSGI server like Gunicorn
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
