"""
Asynchronous retry decorators for the Phoenix Project.
[Task 4A]
"""

import asyncio
import logging
import functools
from typing import Callable, Any, TypeVar, Coroutine

# Define a generic type variable for the wrapped function's return value
T = TypeVar('T')

logger = logging.getLogger(__name__)

def retry_with_exponential_backoff(
    max_retries: int = 5,
    initial_backoff: float = 1.0,
    exceptions_to_retry: tuple = (ConnectionError, asyncio.TimeoutError)
):
    """
    A decorator factory that creates an asynchronous retry decorator.
    Retries on specific exceptions with exponential backoff.
    
    Args:
        max_retries (int): Maximum number of retries (e.g., 5 means 1 initial try + 5 retries).
        initial_backoff (float): The initial delay in seconds for the first retry.
        exceptions_to_retry (tuple): A tuple of exception classes to catch and retry on.
    """
    
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        """The actual decorator."""
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            """The async wrapper that implements the retry logic."""
            
            # Loop for 1 initial attempt + max_retries
            for attempt in range(max_retries + 1):
                try:
                    # Await the coroutine
                    return await func(*args, **kwargs)
                except exceptions_to_retry as e:
                    if attempt == max_retries:
                        logger.error(
                            f"[Retry Failure] Giving up on {func.__name__} after {max_retries} retries. "
                            f"Final error: {e}", exc_info=True
                        )
                        raise e

                    delay = initial_backoff * (2 ** attempt)
                    logger.warning(
                        f"[Retry Attempt] {func.__name__} failed with {type(e).__name__}. "
                        f"Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})",
                        exc_info=False # Don't flood logs with stack traces on retries
                    )
                    await asyncio.sleep(delay)
            raise Exception("Retry logic exited loop unexpectedly.") # Should be unreachable
        return wrapper
    return decorator
