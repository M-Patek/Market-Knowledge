# ai/validation.py
"""
Pydantic models and validation helpers for Evidence-First Ensemble.
Includes:
- EvidenceItem
- AssetAnalysisModel (formerly AIAssetResponse)
- MacroSignalModel
- ExecAdviceModel
- Validation helpers to assert evidence.doc_id is from retrieved docs

This file also includes a small CLI demo for quick local checks.
"""
from __future__ import annotations
from pydantic import BaseModel, Field, HttpUrl, ValidationError, root_validator, validator
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime
import uuid


class EvidenceItem(BaseModel):
    type: Literal['news', 'sec_filing', 'analyst_rating', 'market_data', 'research', 'other']
    source: str
    finding: str
    score: float = Field(..., ge=0.0, le=1.0)
    provenance_confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: Optional[datetime] = None # When the event occurred
    url: Optional[HttpUrl] = None

    # --- [NEW] Fine-grained Provenance Metadata ---
    source_id: Optional[str] = Field(None, description="The unique identifier of the source document or API record.")
    fetch_time: Optional[datetime] = Field(None, description="The UTC timestamp when the evidence was retrieved.")
    license: Optional[str] = Field(None, description="The license or usage terms of the source data.")
    retrieval_rank: Optional[int] = Field(None, description="The rank of this item in the retrieval results.")
    data_fingerprint: Optional[str] = Field(None, description="SHA256 hash of the source content for immutable traceability.")

    class Config:
        schema_extra = {
            'example': {
                'type': 'sec_filing',
                'source': 'SEC EDGAR',
                'finding': 'Company reported 12% YoY subscription revenue growth in Q3',
                'score': 1.0,
                'provenance_confidence': 1.0,
                'timestamp': '2025-10-01T12:00:00Z',
                'source_id': 'doc-xyz-123',
                'fetch_time': '2025-10-12T12:00:00Z',
                'license': 'Public Domain',
                'retrieval_rank': 1,
                'url': 'https://example.com/sec/abc-q3',
                'data_fingerprint': 'a1b2c3d4...'
            }
        }


class AssetAnalysisModel(BaseModel):
    """ The primary model for asset analysis, compatible with the ensemble system. """
    ticker: str
    adjustment_factor: float = Field(..., ge=0.5, le=1.5)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: List[EvidenceItem]
    reasoning: Optional[str] = None
    audit_id: Optional[str] = None
    model_version: Optional[str] = None

    @validator('evidence')
    def evidence_must_be_list(cls, v):
        if v is None:
            return []
        return v

    @root_validator(pre=False, skip_on_failure=True)
    def no_evidence_requires_neutral_or_low_confidence(cls, values):
        evidence = values.get('evidence') or []
        confidence = values.get('confidence')
        adjustment_factor = values.get('adjustment_factor')
        # If there is no evidence, require conservative output per policy
        if not evidence:
            if confidence is None:
                raise ValueError('confidence is required')
            # If there's no evidence but the model indicates non-neutral factor or high confidence, reject
            if (adjustment_factor is not None and abs(adjustment_factor - 1.0) > 1e-9) and confidence > 0.1:
                raise ValueError('Responses without evidence must not claim high confidence or non-neutral adjustment_factor')
        return values


class MacroSignalResponse(BaseModel):
    ticker: str
    adjustment_factor: float = Field(..., ge=0.5, le=1.5)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: List[EvidenceItem]
    reasoning: Optional[str] = None
    audit_id: Optional[str] = None
    model_version: Optional[str] = None


class ExecAdviceResponse(BaseModel):
    ticker: str
    execution_plan: Dict[str, Any]  # freeform but should be validated by higher-level logic
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: List[EvidenceItem]
    reasoning: Optional[str] = None
    audit_id: Optional[str] = None
    model_version: Optional[str] = None


# Helper validators / utilities
class ValidationErrorWithContext(Exception):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


def validate_response_against_retrieved_docs(response: AssetAnalysisModel, retrieved_doc_ids: List[str]):
    """
    Ensure every evidence.source_id in response exists in retrieved_doc_ids.
    Also enforce additional business rules.
    Raises ValidationErrorWithContext on failure.
    """
    missing = []
    for e in response.evidence:
        if e.source_id:
            if e.source_id not in retrieved_doc_ids:
                missing.append(e.source_id)
        else:
            if not e.url:
                missing.append(None)
    if missing:
        raise ValidationErrorWithContext('Some evidence.source_id values are not in retrieved_docs', {'missing_doc_ids': missing})
    
    for e in response.evidence:
        if e.score not in (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValidationErrorWithContext('Evidence.score must be one of the discrete values (0.0,0.25,0.5,0.75,1.0)', {'value': e.score, 'evidence': e.dict()})
    return True


# Small CLI/demo utilities
def make_neutral_response(ticker: str) -> AssetAnalysisModel:
    return AssetAnalysisModel(
        ticker=ticker,
        adjustment_factor=1.0,
        confidence=0.0,
        evidence=[],
        reasoning='NO_EVIDENCE: no verifiable documents provided',
        audit_id=str(uuid.uuid4()),
        model_version='local-demo'
    )


if __name__ == '__main__':
    # Quick demo of model validation
    import json
    example = {
        'ticker': 'ABC',
        'adjustment_factor': 1.08,
        'confidence': 0.75,
        'evidence': [
            {
                'type': 'sec_filing',
                'source': 'SEC',
                'finding': 'Subscription revenue up 12% YoY',
                'score': 1.0,
                'provenance_confidence': 1.0,
                'timestamp': '2025-10-01T12:00:00Z',
                'source_id': 'DOC-2025-10-01-1'
            }
        ],
        'reasoning': 'Subscription growth supports a modest up-weight.',
        'audit_id': str(uuid.uuid4()),
        'model_version': 'demo-1'
    }
    try:
        resp = AssetAnalysisModel.model_validate(example)
        print('Parsed response OK:')
        print(resp.json(indent=2))
        # Validate against retrieved doc ids
        validate_response_against_retrieved_docs(resp, ['DOC-2025-10-01-1', 'DOC-2025-10-02-1'])
        print('Validation against retrieved_docs passed')
    except ValidationError as e:
        print('Pydantic validation error:', e)
    except ValidationErrorWithContext as e:
        print('Business validation error:', e, e.details)
