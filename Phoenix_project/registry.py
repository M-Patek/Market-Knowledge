"""
Centralized Dependency Injection Registry.

Provides a singleton `Registry` class to register and resolve application-wide services,
aligning with the requirements of Layer 11.
"""

from monitor.logging import get_logger
# Configure logger for this module (Layer 12)
logger = get_logger(__name__)

class Registry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Registry, cls).__new__(cls)
            cls._instance._dependencies = {}
        return cls._instance

    def register(self, name: str, obj: any):
        logger.debug(f"Registering service: '{name}'")
        self._dependencies[name] = obj

    def resolve(self, name: str) -> any:
        logger.debug(f"Resolving service: '{name}'")
        try:
            return self._dependencies[name]
        except KeyError:
            raise AttributeError(f"Service '{name}' not found in registry.")

# Global instance for easy access
registry = Registry()
