import asyncio
from typing import Dict, List, Callable, Coroutine, Any
from core.pipeline_state import PipelineState
from monitor.logging import get_logger

logger = get_logger(__name__)

class EventDistributor:
    """
    A simple asynchronous pub/sub event bus.
    Components can subscribe to events and publish new ones.
    """

    def __init__(self, pipeline_state: PipelineState):
        self.pipeline_state = pipeline_state
        self._subscribers: Dict[str, List[Callable[..., Coroutine]]] = {}
        self._event_queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        logger.info("EventDistributor initialized.")

    async def _event_worker(self):
        """Worker task that processes events from the queue."""
        logger.info("Event worker started.")
        while True:
            try:
                event_name, kwargs = await self._event_queue.get()
                if event_name == "__STOP__":
                    logger.info("Event worker received stop signal.")
                    break
                    
                logger.debug(f"Processing event: {event_name}")
                
                # Global subscribers
                if "*" in self._subscribers:
                    for callback in self._subscribers["*"]:
                        try:
                            await callback(event_name, self.pipeline_state, **kwargs)
                        except Exception as e:
                            logger.error(f"Error in global subscriber for event {event_name}: {e}", exc_info=True)

                # Specific subscribers
                if event_name in self._subscribers:
                    for callback in self._subscribers[event_name]:
                        try:
                            await callback(self.pipeline_state, **kwargs)
                        except Exception as e:
                            logger.error(f"Error in subscriber for event {event_name}: {e}", exc_info=True)
                
                self._event_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("Event worker cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in event worker loop: {e}", exc_info=True)
                # Avoid busy-looping
                await asyncio.sleep(1)

    def start(self):
        """Starts the event processing worker."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._event_worker())
            logger.info("Event worker task created.")
        else:
            logger.warning("Event worker task is already running.")

    async def stop(self):
        """Stops the event processing worker."""
        if self._worker_task and not self._worker_task.done():
            logger.info("Stopping event worker...")
            await self._event_queue.put(("__STOP__", {}))
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Event worker did not stop gracefully. Cancelling.")
                self._worker_task.cancel()
            except asyncio.CancelledError:
                pass # Expected
            logger.info("Event worker stopped.")
        self._worker_task = None

    async def subscribe(self, event_name: str, callback: Callable[..., Coroutine]):
        """
        Subscribe a coroutine to a specific event.
        
        Callback signature should be:
        async def my_callback(pipeline_state: PipelineState, **kwargs)
        
        For global ("*") subscriptions:
        async def my_global_callback(event_name: str, pipeline_state: PipelineState, **kwargs)
        """
        if not asyncio.iscoroutinefunction(callback):
            raise ValueError("Event callback must be a coroutine function (async def).")
            
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(callback)
        logger.info(f"New subscription for event: {event_name}")

    async def publish(self, event_name: str, **kwargs):
        """
        Publish an event to the queue.
        This method is non-blocking.
        
        Args:
            event_name (str): The name of the event.
            **kwargs: Arbitrary data to pass to subscribers.
        """
        if self._worker_task is None or self._worker_task.done():
            logger.warning(f"Event '{event_name}' published, but event worker is not running.")
            return
            
        await self._event_queue.put((event_name, kwargs))
        logger.debug(f"Event '{event_name}' published to queue.")
