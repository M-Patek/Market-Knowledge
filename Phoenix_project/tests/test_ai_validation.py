# tests/test_ai_validation.py
import pytest
from pydantic import ValidationError

# --- [修复] ---
# 修复：将 'core.schemas.evidence_schema' 转换为 'Phoenix_project.core.schemas.evidence_schema'
from Phoenix_project.core.schemas.evidence_schema import Evidence, MarketImpactAssessment
# --- [修复结束] ---


def test_evidence_schema_valid():
    """Tests that a valid Evidence object passes validation."""
    data = {
        "source": "test_source",
        "content": "Test content",
        "timestamp": "2023-01-01T12:00:00Z",
        "confidence": 0.8,
        "assessment": {
            "direction": "UP",
            "magnitude": "HIGH",
            "confidence": 0.7,
            "rationale": "Because tests said so."
        }
    }
    try:
        evidence = Evidence(**data)
        assert evidence.source == "test_source"
        assert evidence.assessment.direction == "UP"
    except ValidationError as e:
        pytest.fail(f"Valid data failed validation: {e}")

def test_evidence_schema_invalid_direction():
    """Tests that an invalid direction fails validation."""
    data = {
        "source": "test_source",
        "content": "Test content",
        "timestamp": "2023-01-01T12:00:00Z",
        "confidence": 0.8,
        "assessment": {
            "direction": "INVALID_DIRECTION", # Invalid
            "magnitude": "HIGH",
            "confidence": 0.7
        }
    }
    with pytest.raises(ValidationError):
        Evidence(**data)

def test_evidence_schema_invalid_confidence():
    """Tests that an out-of-range confidence fails validation."""
    data = {
        "source": "test_source",
        "content": "Test content",
        "timestamp": "2023-01-01T12:00:00Z",
        "confidence": 1.5, # Invalid (must be <= 1.0)
        "assessment": {
            "direction": "DOWN",
            "magnitude": "LOW",
            "confidence": 0.5
        }
    }
    with pytest.raises(ValidationError):
        Evidence(**data)
