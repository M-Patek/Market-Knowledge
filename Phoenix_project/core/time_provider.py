"""
Phoenix_project/core/time_provider.py
[Task FIX-HIGH-003] Time Abstraction Layer.
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

class TimeProvider(ABC):
    """
    Abstract base class for time provision.
    """
    @abstractmethod
    def get_current_time(self) -> datetime:
        """Returns the current time (UTC)."""
        pass
    
    @abstractmethod
    def set_simulation_time(self, sim_time: datetime):
        """Sets the simulation time (for backtesting/replay)."""
        pass

    @abstractmethod
    def clear_simulation_time(self):
        """Reverts to system clock."""
        pass

class SystemTimeProvider(TimeProvider):
    """
    Default Time Provider. 
    By default returns system time (UTC).
    If simulation_time is set (Time Machine), returns that instead.
    """
    def __init__(self):
        self._simulation_time: Optional[datetime] = None

    def get_current_time(self) -> datetime:
        if self._simulation_time:
            return self._simulation_time
        return datetime.now(timezone.utc)

    def set_simulation_time(self, sim_time: datetime):
        if sim_time.tzinfo is None:
            sim_time = sim_time.replace(tzinfo=timezone.utc)
        self._simulation_time = sim_time

    def clear_simulation_time(self):
        self._simulation_time = None
