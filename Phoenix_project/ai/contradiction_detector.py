# Phoenix_project/ai/contradiction_detector.py
from typing import List
from observability.metrics import FUSION_CONFLICTS
from schemas.data_schema import MarketEvent, AnalystOpinion

class ContradictionDetector:
    """
    L2 (Level 2) service that scans structured data for contradictions.
    Example: Detects if a "BUY" rating is issued alongside a negative "MarketEvent".
    """
    
    def detect(self, events: List[MarketEvent], opinions: List[AnalystOpinion]) -> List[str]:
        contradictions = []
        
        # Mock Logic: Check for simple conflicts
        # A real implementation would be far more complex, using temporal logic,
        # entity linking, and possibly semantic models.
        
        has_negative_event = any(e.impact_score is not None and e.impact_score < -0.5 for e in events)
        
        for opinion in opinions:
            if opinion.rating == "BUY" and has_negative_event:
                # Find the specific negative event(s) for a better message
                for event in events:
                    if event.impact_score is not None and event.impact_score < -0.5:
                        contradiction_msg = (
                            f"Contradiction Detected: Analyst BUY rating from {opinion.analyst} "
                            f"conflicts with negative MarketEvent '{event.description}' "
                            f"(Impact: {event.impact_score})."
                        )
                        FUSION_CONFLICTS.inc()
                        contradictions.append(contradiction_msg)

            if opinion.rating == "SELL" and not has_negative_event:
                # This is a weaker form of contradiction, maybe a "divergence"
                pass 

        return contradictions
