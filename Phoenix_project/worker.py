"""
Celery Worker Definition (Layer 15)

Defines the Celery application and any asynchronous tasks.
This file is the entry point for the 'worker' service in docker-compose.yml.
"""

import os
from celery import Celery
from kombu import Queue, Exchange

from monitor.logging import get_logger

# Configure logger for this module (Layer 12)
logger = get_logger(__name__)

# In a real app, this URL would come from config (e.g., os.environ.get('CELERY_BROKER_URL'))
app = Celery('phoenix_worker', broker='redis://localhost:6379/0')

@app.task
def example_task(x, y):
    """A placeholder task for the worker to execute."""
    logger.info(f"Executing example_task with args: {x}, {y}")
    return x + y

logger.info("Celery worker app defined.")
