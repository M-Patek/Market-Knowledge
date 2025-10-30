from typing import Any, Callable, Dict, List
from collections import defaultdict

class ContextBus:
    """
    (L0) A simple publish/subscribe event bus to decouple system components.
    """
    def __init__(self):
        # A dictionary mapping event types (strings) to lists of callbacks
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: Callable):
        """
        Subscribe a callback function to a specific event type.
        
        Args:
            event_type: The name of the event to listen for (e.g., "L1_ANALYSIS_COMPLETE").
            callback: The function to call when the event is published.
        """
        self.subscribers[event_type].append(callback)

    def publish(self, event_type: str, *args, **kwargs):
        """
        Publish an event, calling all subscribed callbacks.
        
        Args:
            event_type: The name of the event being published.
            *args, **kwargs: Arguments to pass to the subscribed callbacks.
        """
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                # We might want to add error handling here later (e.g., try/except)
                callback(*args, **kwargs)
