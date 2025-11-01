import os
import sys
import asyncio
from dotenv import load_dotenv

# Add project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load environment variables from .env file
load_dotenv()

from config.system import load_config
from monitor.logging import get_logger
from controller.orchestrator import Orchestrator
from data_manager import DataManager
from core.pipeline_state import PipelineState
from strategy_handler import RomanLegionStrategy
from api.gemini_pool_manager import GeminiPoolManager
from events.stream_processor import StreamProcessor
from events.event_distributor import EventDistributor
from execution.order_manager import OrderManager

# --- Global Components ---
logger = get_logger('PhoenixMain')
config = None
orchestrator = None
gemini_pool = None
stream_processor = None
event_distributor = None

async def initialize_system():
    """
    Initializes all core components of the Phoenix system.
    """
    global config, orchestrator, gemini_pool, stream_processor, event_distributor, logger
    
    logger.info("--- PHOENIX PROJECT V2.0 INITIALIZATION START ---")
    
    try:
        # 1. Load Configuration
        config_path = os.getenv('CONFIG_PATH', 'config/system.yaml')
        config = load_config(config_path)
        if config is None:
            logger.critical("Failed to load system.yaml. Exiting.")
            sys.exit(1)
        logger.info(f"Configuration loaded from {config_path}")

        # 2. Initialize Gemini API Pool
        # This pool will be shared across all components
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY not set. LLM features will be disabled.")
        
        gemini_pool = GeminiPoolManager(
            api_key=gemini_api_key,
            pool_size=config.get('llm', {}).get('gemini_pool_size', 5)
        )
        logger.info(f"GeminiPoolManager initialized with size {config.get('llm', {}).get('gemini_pool_size', 5)}")

        # 3. Initialize Core State and Data Management
        pipeline_state = PipelineState()
        data_manager = DataManager(config, pipeline_state)
        logger.info("PipelineState and DataManager initialized.")

        # 4. Initialize Order Manager (Execution Layer)
        order_manager = OrderManager(config.get('execution', {}))
        logger.info("OrderManager initialized.")

        # 5. Initialize Strategy Handler (RomanLegion)
        # FIX: Removed redundant data loading (asset_analysis_data, sentiment_data)
        # FIX: StrategyHandler __init__ signature changed to just (config, data_manager)
        logger.info("Loading strategy data...")
        # The DataManager now handles loading data internally as needed by components.
        # We no longer pass explicit dataframes to the strategy.
        
        strategy = RomanLegionStrategy(
            config=config,
            data_manager=data_manager
        )
        logger.info("RomanLegionStrategy initialized.")

        # 6. Initialize Orchestrator (The "Brain")
        # Pass all shared components to the orchestrator
        orchestrator = Orchestrator(
            config=config,
            data_manager=data_manager,
            pipeline_state=pipeline_state,
            gemini_pool=gemini_pool,
            strategy_handler=strategy,
            order_manager=order_manager
        )
        logger.info("Main Orchestrator initialized.")
        
        # 7. Initialize Event Stream and Distributor
        stream_processor = StreamProcessor(config.get('event_stream', {}))
        event_distributor = EventDistributor(
            stream_processor=stream_processor,
            orchestrator=orchestrator,
            config=config.get('event_distributor', {})
        )
        logger.info("EventStreamProcessor and EventDistributor initialized.")

        logger.info("--- PHOENIX SYSTEM INITIALIZATION COMPLETE ---")
        return True

    except Exception as e:
        logger.critical(f"Fatal error during system initialization: {e}", exc_info=True)
        return False

async def run_system():
    """
    Starts the main asynchronous loops for the system.
    """
    global orchestrator, event_distributor, logger
    
    if not orchestrator or not event_distributor:
        logger.critical("System not initialized. Cannot run.")
        return

    logger.info("--- PHOENIX SYSTEM STARTING MAIN LOOPS ---")
    
    try:
        # 1. Start the orchestrator's main decision loop (e.g., runs every 5 min)
        orchestrator_task = asyncio.create_task(orchestrator.start_decision_loop())
        
        # 2. Start the event distributor (consumes from stream processor)
        event_distributor_task = asyncio.create_task(event_distributor.start_consuming())
        
        # 3. (Optional) Start a simulation task if in backtesting mode
        # sim_task = asyncio.create_task(run_simulation_if_enabled())
        
        logger.info("Orchestrator and Event Distributor loops are running.")
        
        # Wait for tasks to complete (or run forever)
        # In a real app, you'd have graceful shutdown logic
        await asyncio.gather(
            orchestrator_task,
            event_distributor_task
            # sim_task
        )
        
    except asyncio.CancelledError:
        logger.info("Main system loops cancelled.")
    except Exception as e:
        logger.error(f"An error occurred in the main system run loop: {e}", exc_info=True)
    finally:
        logger.info("--- PHOENIX SYSTEM SHUTTING DOWN ---")
        await shutdown_system()

async def shutdown_system():
    """
    Gracefully shuts down all system components.
    """
    global orchestrator, event_distributor, gemini_pool, logger
    
    logger.info("Initiating graceful shutdown...")
    if event_distributor:
        await event_distributor.stop_consuming()
        logger.info("Event Distributor stopped.")
        
    if orchestrator:
        await orchestrator.stop_decision_loop()
        logger.info("Orchestrator loop stopped.")
        
    if gemini_pool:
        await gemini_pool.close()
        logger.info("Gemini API Pool closed.")
        
    # TODO: Add shutdown for DataManager (e.g., close DB connections)
    
    logger.info("--- PHOENIX SHUTDOWN COMPLETE ---")

async def main():
    """
    Main entry point for the application.
    """
    if await initialize_system():
        # Handle graceful shutdown on SIGINT/SIGTERM
        loop = asyncio.get_running_loop()
        try:
            # TODO: Add signal handlers for graceful shutdown
            # loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(shutdown_system()))
            # loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(shutdown_system()))
            await run_system()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down.")
            await shutdown_system()
    else:
        sys.exit(1)

if __name__ == "__main__":
    # Check for CLI arguments (e.g., run backtest, run data validation)
    if len(sys.argv) > 1:
        command = sys.argv[1]
        logger.info(f"CLI command detected: {command}")
        # TODO: Implement CLI argument handling
        # e.g., if command == 'backtest': run_backtest()
        # e.g., if command == 'validate': run_validation()
        print(f"CLI command '{command}' not yet implemented. Starting main system.")
    
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Application failed to run: {e}", exc_info=True)
        sys.exit(1)
