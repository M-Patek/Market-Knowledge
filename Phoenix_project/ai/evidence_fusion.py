# ai/evidence_fusion.py
"""
Synthesizes multiple AI analyses into a single, evidence-weighted, and robust decision.
Acts as the "judge" for the AI "jury", incorporating production-grade engineering practices.
"""
import logging
import math
import yaml
import statistics
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Assuming the validation models are in this location
from .validation import AssetAnalysisModel, EvidenceItem

# --- Prometheus Metrics (placeholders, to be initialized in observability.py) ---
# from prometheus_client import Counter, Gauge
# FUSION_CALLS = Counter('phoenix_ai_fusion_calls_total', 'Total number of fusion invocations')
# FUSION_CONTROVERSIAL = Counter('phoenix_ai_fusion_controversial_total', 'Total number of controversial outcomes')
# FUSION_AVG_EVIDENCE = Gauge('phoenix_ai_fusion_avg_evidence_score', 'Average evidence score from the best model in a fusion')
# FUSION_DISPERSION = Gauge('phoenix_ai_fusion_dispersion', 'Dispersion value in a fusion')


# --- Pydantic Models for Structured & Auditable Output ---
class ModelContribution(BaseModel):
    """Details the contribution of a single model to the final fusion."""
    model_id: Optional[str] = Field(description="Identifier of the model version.")
    factor: float = Field(description="The adjustment_factor proposed by this model.")
    avg_evidence_score: float = Field(description="The average quality score of evidence provided by this model.")
    weight: float = Field(description="The final calculated weight of this model's opinion in the fusion.")
    weighted_contribution: float = Field(description="The portion of the final factor contributed by this model.")

class FusedResponseModel(BaseModel):
    """The structured, machine-readable output of the fusion process."""
    final_factor: float
    is_controversial: bool
    contributing_models: int
    dispersion: float = Field(description="Weighted standard deviation of adjustment factors, a measure of disagreement.")
    avg_evidence_score: float = Field(description="The average evidence score from the highest-quality response.")
    supporting_evidence_count: int
    contributions: List[ModelContribution]
    reasoning: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# --- Helper for Weighted Statistics ---
def _weighted_stats(values: List[float], weights: List[float]) -> tuple[float, float]:
    """Calculates weighted mean and weighted variance."""
    total_w = sum(weights)
    if total_w < 1e-9: # Return 0 if total weight is negligible
        return 0.0, 0.0
    
    mean = sum(v * w for v, w in zip(values, weights)) / total_w
    # Weighted variance
    var = sum(w * ((v - mean) ** 2) for v, w in zip(values, weights)) / total_w
    return mean, var

# --- Main Fusion Engine ---
class EvidenceFusionEngine:
    """
    Synthesizes multiple AI analyses into a single, evidence-weighted decision.
    """
    def __init__(self, config_file_path: str):
        self.logger = logging.getLogger("PhoenixProject.EvidenceFusionEngine")
        
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            self.config = config.get('fusion_engine', {})

        # Load thresholds and parameters from config, with sane defaults
        self.dispersion_threshold = self.config.get('dispersion_threshold', 0.15)
        self.min_avg_evidence_score = self.config.get('min_avg_evidence_score', 0.35)
        self.clamp_min = self.config.get('clamp_min', 0.7)
        self.clamp_max = self.config.get('clamp_max', 1.3)
        self.weight_floor = self.config.get('weight_floor', 0.1)
        
        self.type_weights = self.config.get('type_weights', {
            "sec_filing": 1.0, "analyst_rating": 0.8, "news": 0.7,
            "market_data": 0.9, "research": 0.9, "other": 0.5,
        })
        self.source_reputation = self.config.get('source_reputation', {
            "SEC": 1.0, "WSJ": 0.9, "Reuters": 0.9, "Bloomberg": 0.9, "Fed Reserve": 1.0,
        })
        
        self.logger.info("EvidenceFusionEngine initialized.")

    def _get_age_decay_factor(self, evidence_timestamp: Optional[datetime]) -> float:
        if not evidence_timestamp: return 0.7
        if evidence_timestamp.tzinfo is None: evidence_timestamp = evidence_timestamp.replace(tzinfo=timezone.utc)
        age_delta = datetime.now(timezone.utc) - evidence_timestamp
        age_in_days = age_delta.total_seconds() / 86400
        decay_factor = 0.92 ** age_in_days
        return max(0.1, decay_factor)

    def _score_evidence_item(self, item: EvidenceItem) -> float:
        type_weight = self.type_weights.get(item.type, 0.5)
        age_decay = self._get_age_decay_factor(item.timestamp)
        source_rep = self.source_reputation.get(item.source, 0.7)
        
        raw_score = type_weight * age_decay * source_rep * (item.score or 0.5) * (item.provenance_confidence or 0.6)
        return max(0.0, min(1.0, raw_score)) # Clamp score to [0, 1]

    def _parse_and_score_response(self, response: AssetAnalysisModel) -> Dict[str, Any]:
        if not response.evidence:
            return {"avg_evidence_score": 0.0, "weight": 0.0, "factor": response.adjustment_factor, "model_id": response.model_version}

        avg_evidence_score = statistics.mean([self._score_evidence_item(e) for e in response.evidence])
        weight = response.confidence * (self.weight_floor + avg_evidence_score)
        
        return {
            "avg_evidence_score": avg_evidence_score,
            "weight": weight,
            "factor": response.adjustment_factor,
            "model_id": response.model_version
        }

    def fuse(self, responses: List[AssetAnalysisModel]) -> FusedResponseModel:
        # FUSION_CALLS.inc()
        if not responses:
            # FUSION_CONTROVERSIAL.inc()
            return FusedResponseModel(
                final_factor=self.config.get('conservative_fallback_factor', 1.0),
                is_controversial=True, contributing_models=0, dispersion=0.0,
                avg_evidence_score=0.0, supporting_evidence_count=0, contributions=[],
                reasoning="NO_VALID_RESPONSES: Using conservative fallback factor."
            )

        parsed = [self._parse_and_score_response(r) for r in responses]
        
        weights = [p['weight'] for p in parsed]
        factors = [p['factor'] for p in parsed]
        avg_scores = [p['avg_evidence_score'] for p in parsed]

        mean_factor, var_factor = _weighted_stats(factors, weights)
        dispersion = math.sqrt(var_factor) if var_factor > 0 else 0.0
        
        max_avg_evidence = max(avg_scores) if avg_scores else 0.0
        is_controversial = (dispersion > self.dispersion_threshold) or (max_avg_evidence < self.min_avg_evidence_score)

        fused_factor = max(self.clamp_min, min(self.clamp_max, mean_factor))

        contributions = []
        total_weight = sum(weights)
        if total_weight > 1e-9:
            for p, w 在 zip(parsed, weights):
                contributions.append(ModelContribution(
                    model_id=p.get('model_id')，
                    factor=p['factor']，
                    avg_evidence_score=p['avg_evidence_score']，
                    weight=w,
                    weighted_contribution=(p['factor'] * w) / total_weight
                ))
        
        # --- Metrics & Auditing ---
        # FUSION_AVG_EVIDENCE.set(max_avg_evidence)
        # FUSION_DISPERSION.set(dispersion)
        # if is_controversial:
        #     FUSION_CONTROVERSIAL.inc()

        reasoning = (f"Fused from {len(responses)} models. WeightedMeanFactor={mean_factor:.3f}, "
                     f"Dispersion(WeightedStdev)={dispersion:.3f}, Controversial={is_controversial}.")
        self.logger.info(f"Fusion result: FinalFactor={fused_factor:.3f}. {reasoning}")

        return FusedResponseModel(
            final_factor=fused_factor,
            is_controversial=is_controversial,
            contributing_models=len(responses),
            dispersion=dispersion,
            avg_evidence_score=max_avg_evidence,
            supporting_evidence_count=sum(len(r.evidence) for r in responses),
            contributions=contributions,
            reasoning=reasoning
        )

