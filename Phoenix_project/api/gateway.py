from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import sys
import os

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from observability import ShadowMonitor
from audit_manager import AuditManager
from prediction_server import PredictionServer
from pipeline_orchestrator import PipelineOrchestrator

app = FastAPI(title="Phoenix Bridge API Gateway")

# --- Instantiate Backend Services ---
# In a real application, these would be managed as singletons.
audit_manager = AuditManager()
shadow_monitor = ShadowMonitor(audit_manager)
# Using a placeholder config for the prediction server
mock_server_config = {"champion_model_path": "path/to/mock_champion"}
mock_pipeline_config_path = "config.yaml" # Assuming a config file exists
prediction_server = PredictionServer(mock_server_config)
pipeline_orchestrator = PipelineOrchestrator(mock_pipeline_config_path)

# --- API Models ---
class PromoteRequest(BaseModel):
    model_id: str

class ApproveRetrainingRequest(BaseModel):
    recommendation_id: str

@app.get("/api/v1/models/promotion_candidates")
async def get_promotion_candidates():
    """[Sub-Task 1.1.4] Returns all models in the AWAITING_PROMOTION queue."""
    return shadow_monitor.get_promotion_candidates_report()

@app.post("/api/v1/models/promote")
async def promote_model(request: PromoteRequest):
    """[Sub-Task 1.1.4] Promotes the shadow model to be the new champion."""
    # In a real system, we might check if request.model_id matches the current shadow model
    prediction_server.promote_shadow_to_champion()
    return {"status": "success", "message": f"Model '{request.model_id}' promotion process initiated."}

@app.get("/api/v1/retraining/recommendations")
async def get_retraining_recommendations():
    """[Sub-Task 1.2.2] Returns all retraining recommendations awaiting approval."""
    # This data would normally be read from the RetrainingRecommendations database table.
    mock_recommendation = {
        "recommendation_id": "rec_sharpe_decay_20231027",
        "trigger_reason": "SHARPE_DECAY",
        "supporting_data": {"rolling_sharpe_ratio": [0.9, 0.85, 0.7, 0.6]},
        "estimated_api_cost": 150.75
    }
    return {"recommendations": [mock_recommendation]}

@app.post("/api/v1/retraining/approve")
async def approve_retraining(request: ApproveRetrainingRequest):
    """[Sub-Task 1.2.3] Triggers a new training pipeline run."""
    pipeline_orchestrator.run()
    return {"status": "success", "message": f"Retraining approved for '{request.recommendation_id}'. Pipeline started."}

@app.get("/api/v1/decisions")
async def get_decisions(page: int = 1, size: int = Query(10, gt=0, le=100)):
    """[Sub-Task 3.2.1] Returns a paginated list of the latest decisions."""
    # This would normally query a document database with pagination.
    mock_decision_list = [
        {"decision_id": f"dec_id_00{i}", "timestamp": "2023-10-27T10:00:00Z", "final_order": "BUY SPY"}
        for i in range(100)
    ]
    start = (page - 1) * size
    end = start + size
    return {"decisions": mock_decision_list[start:end], "total": len(mock_decision_list), "page": page, "size": size}

@app.get("/api/v1/decisions/{decision_id}")
async def get_decision_details(decision_id: str):
    """[Sub-Task 3.2.1] Returns the complete audit JSON for a specific decision."""
    # This would fetch the full JSON object from a document database like Elasticsearch.
    mock_full_lineage = {
        "decision_id": decision_id,
        "timestamp": "2023-10-27T10:00:00Z",
        "triggering_event": {"type": "news", "headline": "Fed signals pause in rate hikes"},
        "rag_retrieval_results": [{"source": "doc_123", "text": "The Federal Reserve is likely to hold rates steady..."}],
        "l1_jury_debates": [{"model": "Claude 3.5 Sonnet", "output": "Bullish sentiment detected due to dovish Fed stance."}],
        "contradiction_analysis": {"contradiction_found": False, "details": "..."},
        "bayesian_fusion_summary": {"final_factor": "US_MONETARY_POLICY", "confidence": 0.85},
        "risk_management_inputs": {"volatility": 0.15, "max_drawdown": -0.05},
        "final_order": "BUY SPY"
    }
    return mock_full_lineage
