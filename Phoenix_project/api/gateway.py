import os
import json
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import asyncio

# FIX: Changed import from 'observability' to 'monitor.logging'
from monitor.logging import get_logger
from controller.orchestrator import Orchestrator
from data_manager import DataManager
from config.system import load_config
from core.pipeline_state import PipelineState
from api.gemini_pool_manager import GeminiPoolManager

# --- Configuration & Initialization ---

# Load system configuration
config = load_config('config/system.yaml')
logger = get_logger('APIGateway')

app = Flask(__name__, template_folder='../templates')
app.config['UPLOAD_FOLDER'] = config.get('api_server', {}).get('upload_folder', 'uploads/')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Global Components ---
# Initialize core components
pipeline_state = PipelineState()
data_manager = DataManager(config, pipeline_state)

# Initialize the Gemini API pool
try:
    gemini_pool = GeminiPoolManager(
        api_key=os.environ.get("GEMINI_API_KEY"),
        pool_size=config.get('llm', {}).get('gemini_pool_size', 5)
    )
    logger.info(f"GeminiPoolManager initialized with size {config.get('llm', {}).get('gemini_pool_size', 5)}")
except Exception as e:
    logger.error(f"Failed to initialize GeminiPoolManager: {e}", exc_info=True)
    gemini_pool = None

# Initialize the main orchestrator
# Pass the shared Gemini pool to the orchestrator
orchestrator = Orchestrator(config, data_manager, pipeline_state, gemini_pool)

# --- Utility Functions ---

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'json', 'md'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API Endpoints ---

@app.route('/')
def index():
    """Serves the main dashboard/control interface."""
    logger.debug("Serving index page.")
    # TODO: Pass dynamic data to the template (e.g., system status)
    return render_template('index.html', system_status="Operational")

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Provides a simple health check endpoint."""
    logger.debug("Health check requested.")
    # TODO: Add deeper checks (e.g., DB connectivity, LLM API status)
    return jsonify({"status": "healthy", "version": "2.0.0"}), 200

@app.route('/api/v1/analyze', methods=['POST'])
async def analyze_task():
    """
    Asynchronously triggers a full analysis pipeline for a given task (e.g., asset).
    """
    data = request.json
    task_description = data.get('task')
    
    if not task_description:
        logger.warning("Analysis request failed: 'task' field missing.")
        return jsonify({"error": "Missing 'task' field in request body"}), 400
        
    if not orchestrator:
        logger.error("Orchestrator not initialized. Cannot process task.")
        return jsonify({"error": "System not ready"}), 503

    logger.info(f"Received analysis task: {task_description}")

    try:
        # Asynchronously run the main processing loop
        analysis_result = await orchestrator.run_main_loop(task_description)
        
        # TODO: Standardize the output format
        return jsonify({
            "status": "success",
            "task": task_description,
            "result": analysis_result 
        }), 200
        
    except Exception as e:
        logger.error(f"Error during analysis task '{task_description}': {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during analysis"}), 500

@app.route('/api/v1/inject/data', methods=['POST'])
async def inject_data():
    """
    Handles data injection from external sources (e.g., webhook).
    """
    data = request.json
    source = data.get('source')
    content = data.get('content')
    
    if not source or not content:
        logger.warning("Data injection failed: 'source' or 'content' missing.")
        return jsonify({"error": "Missing 'source' or 'content' field"}), 400

    logger.info(f"Injecting data from source: {source}")

    try:
        # Use the knowledge_injector component via the orchestrator
        # (Assuming orchestrator exposes a method for this)
        # This is an asynchronous operation
        ingestion_id = await orchestrator.inject_external_data(source, content)
        
        return jsonify({
            "status": "pending", 
            "message": "Data queued for ingestion",
            "ingestion_id": ingestion_id
        }), 202
        
    except Exception as e:
        logger.error(f"Error during data injection from '{source}': {e}", exc_info=True)
        return jsonify({"error": "Failed to queue data for ingestion"}), 500

@app.route('/api/v1/inject/file', methods=['POST'])
async def upload_file():
    """
    Allows uploading files (PDF, CSV, TXT) for knowledge ingestion.
    """
    if 'file' not in request.files:
        logger.warning("File upload failed: No 'file' part in request.")
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("File upload failed: No file selected.")
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"File '{filename}' uploaded successfully. Queuing for ingestion.")

        try:
            # Trigger ingestion process for the uploaded file
            # This is an asynchronous operation
            ingestion_id = await orchestrator.inject_file(filepath, filename)
            
            return jsonify({
                "status": "pending",
                "message": "File uploaded and queued for ingestion",
                "filename": filename,
                "ingestion_id": ingestion_id
            }), 202
            
        except Exception as e:
            logger.error(f"Error processing uploaded file '{filename}': {e}", exc_info=True)
            return jsonify({"error": "Failed to process uploaded file"}), 500
    else:
        logger.warning(f"File upload failed: File type not allowed ('{file.filename}')")
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/api/v1/status', methods=['GET'])
def get_system_status():
    """
    Returns the current state of the processing pipeline.
    """
    logger.debug("System status requested.")
    try:
        state_data = pipeline_state.get_full_state()
        return jsonify({
            "status": "success",
            "pipeline_state": state_data
        }), 200
    except Exception as e:
        logger.error(f"Error retrieving pipeline state: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve system state"}), 500

@app.route('/api/v1/config/reload', methods=['POST'])
def reload_config():
    """
    Triggers a reload of the system configuration.
    (Requires careful implementation to avoid race conditions)
    """
    logger.warning("Configuration reload triggered via API.")
    try:
        # This is a complex operation. The orchestrator needs to handle
        # this gracefully, potentially pausing and re-initializing components.
        success = orchestrator.reload_configuration('config/system.yaml')
        if success:
            return jsonify({"status": "success", "message": "Configuration reload initiated."}), 200
        else:
            return jsonify({"error": "Configuration reload failed."}), 500
    except Exception as e:
        logger.error(f"Error during configuration reload: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during reload."}), 500

# --- Circuit Breaker UI ---

@app.route('/circuit-breaker', methods=['GET'])
def circuit_breaker_ui():
    """
    Serves a simple UI to monitor and control circuit breakers.
    """
    logger.debug("Serving Circuit Breaker UI.")
    # The orchestrator's error handler would manage the state
    breakers = orchestrator.error_handler.get_all_breaker_states()
    return render_template('circuit_breaker.html', breakers=breakers)

@app.route('/api/v1/circuit-breaker/trip', methods=['POST'])
def trip_breaker():
    """Manually trips a circuit breaker."""
    data = request.json
    breaker_name = data.get('name')
    if not breaker_name:
        return jsonify({"error": "Missing 'name' field"}), 400
    
    logger.warning(f"Manual trip requested for circuit breaker: {breaker_name}")
    try:
        orchestrator.error_handler.manually_trip_breaker(breaker_name)
        return jsonify({"status": "success", "name": breaker_name, "state": "open"}), 200
    except KeyError:
        return jsonify({"error": f"Breaker '{breaker_name}' not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/circuit-breaker/reset', methods=['POST'])
def reset_breaker():
    """Manually resets a circuit breaker."""
    data = request.json
    breaker_name = data.get('name')
    if not breaker_name:
        return jsonify({"error": "Missing 'name' field"}), 400
    
    logger.info(f"Manual reset requested for circuit breaker: {breaker_name}")
    try:
        orchestrator.error_handler.manually_reset_breaker(breaker_name)
        return jsonify({"status": "success", "name": breaker_name, "state": "closed"}), 200
    except KeyError:
        return jsonify({"error": f"Breaker '{breaker_name}' not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Main Application Runner ---

def run_server():
    """Starts the Flask server."""
    server_config = config.get('api_server', {})
    host = server_config.get('host', '127.0.0.1')
    port = server_config.get('port', 5000)
    debug = server_config.get('debug', False)
    
    logger.info(f"Starting API Gateway on http://{host}:{port} (Debug: {debug})")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_server()
