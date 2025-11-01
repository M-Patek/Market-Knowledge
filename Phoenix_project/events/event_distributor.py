import asyncio
from asyncio import Queue
from typing import Dict, Any

from ..monitor.logging import get_logger
from ..controller.orchestrator import Orchestrator
from ..core.schemas.data_schema import MarketEvent

logger = get_logger(__name__)

class EventDistributor:
    """
    Manages the flow of *high-value* events from the real-time
    filters (e.g., RiskFilter) to the core processing pipeline (Orchestrator).
    
    It uses an internal asyncio.Queue to decouple ingestion from processing,
    allowing the system to handle bursts of events.
    """

    def __init__(self, config: Dict[str, Any], orchestrator: Orchestrator):
        """
        Initializes the EventDistributor.
        
        Args:
            config: The main system configuration.
            orchestrator: The Orchestrator to send events to.
        """
        self.config = config.get('event_distributor', {})
        self.orchestrator = orchestrator
        
        self.queue_max_size = self.config.get('queue_max_size', 1000)
        self.event_queue: Queue[MarketEvent] = Queue(maxsize=self.queue_max_size)
        
        self.consumer_task: asyncio.Task = None
        self._is_running = False
        logger.info(f"EventDistributor initialized with queue size: {self.queue_max_size}")

    async def start(self):
        """Starts the queue consumer task."""
        if self._is_running:
            logger.warning("EventDistributor consumer task is already running.")
            return
            
        logger.info("Starting EventDistributor consumer...")
        self._is_running = True
        self.consumer_task = asyncio.create_task(self._consume_events())

    async def stop(self):
        """Stops the queue consumer task gracefully."""
        if not self._is_running:
            logger.warning("EventDistributor consumer task is not running.")
            return
            
        logger.info("Stopping EventDistributor consumer...")
        self._is_running = False
        
        # Add a sentinel value (None) to unblock the queue.get()
        try:
            await self.event_queue.put(None)
        except asyncio.QueueFull:
            logger.error("Cannot add sentinel to a full queue during shutdown.")
            
        if self.consumer_task:
            try:
                await asyncio.wait_for(self.consumer_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("EventDistributor consumer task did not stop gracefully. Cancelling.")
                self.consumer_task.cancel()
        
        logger.info("EventDistributor consumer stopped.")

    async def publish(self, event: MarketEvent):
        """
        Publishes a new, high-value event to the internal queue
        to be processed by the Orchestrator.
        
        This is called by components like StreamProcessor or RiskFilter.
        
        Args:
            event (MarketEvent): The event to process.
        """
        if not self._is_running:
            logger.warning(f"EventDistributor is not running. Discarding event: {event.event_id}")
            return

        try:
            # Use put_nowait to avoid blocking the caller (e.g., the
            # real-time stream). If the queue is full, we drop the event.
            self.event_queue.put_nowait(event)
            logger.debug(f"Event enqueued for processing: {event.event_id}")
            
        except asyncio.QueueFull:
            logger.error(f"Event processing queue is FULL (Size: {self.queue_max_size}). "
                         f"Event {event.event_id} is being DROPPED.")
            # TODO: Add metric for dropped events

    async def _consume_events(self):
        """
        The main consumer loop.
        Pulls events from the queue and sends them to the Orchestrator
        one by one (serially).
        """
        while self._is_running:
            try:
                # Wait for an event
                event = await self.event_queue.get()
                
                # Check for shutdown sentinel
                if event is None:
                    logger.debug("Consumer received shutdown sentinel.")
                    break
                
                logger.info(f"Distributing event {event.event_id} to Orchestrator.")
                
                # --- OpenTelemetry Trace Span ---
                # A real implementation would start a trace span here
                # tracer.start_as_current_span("process_event")
                # ---------------------------------
                
                # Process the event using the Orchestrator
                # This is an 'await' because the Orchestrator's
                # processing is asynchronous and we want to ensure
                # events are processed *serially* (FIFO).
                await self.orchestrator.process_event(event)
                
                # Mark the task as done
                self.event_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Event consumer task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in event consumer loop: {e}", exc_info=True)
                # Avoid hammering on repeated errors
                await asyncio.sleep(1)
