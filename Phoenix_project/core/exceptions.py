"""
Custom exceptions for the Phoenix Project.
"""

class PhoenixError(Exception):
    """Base exception class for all project-specific errors."""
    pass

class CognitiveError(PhoenixError):
    """Raised when a fatal error occurs within the CognitiveEngine."""
    pass
