"""
Global fixtures for pytest
"""

import pytest

@pytest.fixture(scope="session")
def sample_nvda_event():
    """
    Provides a sample data event for NVDA (Layer 8, Task 2).
    This combines market data and text events.
    """
    return {
        "ticker": "NVDA",
        "type": "combined_event",
        "timestamp": "2025-10-27T14:30:00Z",
        "market_data": {
            "open": 900.00,
            "high": 910.50,
            "low": 895.00,
            "close": 905.75,
            "volume": 25000000
        },
        "text_events": [
            {
                "source": "Major News Outlet",
                "headline": "NVDA announces breakthrough in quantum computing, stock surges."
            },
            {
                "source": "SEC Filing",
                "summary": "Form 4 filing shows insider selling by a minor executive."
            }
        ]
    }
