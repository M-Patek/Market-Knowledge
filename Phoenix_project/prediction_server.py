import logging
from typing import Dict, Any

from cognitive.engine import CognitiveEngine
from ai.reasoning_ensemble import ReasoningEnsemble # 假设这是模型的基类或类型
from audit_manager import AuditManager

class PredictionServer:
    """
    一个常驻服务，用于加载冠军模型和影子模型，
    处理传入的预测请求，并记录决策以供比较。
    """
    def __init__(self, champion_model: ReasoningEnsemble, audit_manager: AuditManager):
        self.logger = logging.getLogger("PhoenixProject.PredictionServer")
        self.champion_model = champion_model
        self.shadow_model: ReasoningEnsemble = None
        self.audit_manager = audit_manager
        self.logger.info("PredictionServer initialized with a Champion model.")

    def load_shadow_model(self, shadow_model: ReasoningEnsemble):
        """
        [Sub-Task 1.1.2] 加载一个新的（或候选的）影子模型进行并行评估。
        """
        self.shadow_model = shadow_model
        self.logger.info("A new Shadow model has been loaded for parallel evaluation.")

    def handle_prediction_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个预测请求。
        它同时运行冠军模型和影子模型（如果存在），
        但只返回冠军模型的决策。
        """
        # 1. 生成冠军模型的决策（生产决策）
        try:
            champion_decision = self.champion_model.make_decision(request_data)
        except Exception as e:
            self.logger.error(f"Champion model failed to make a decision: {e}", exc_info=True)
            return {"error": "Champion model failure", "decision": None}

        # 2. 如果影子模型存在，则在“影子”模式下运行它
        shadow_decision = None
        if self.shadow_model:
            try:
                shadow_decision = self.shadow_model.make_decision(request_data)
            except Exception as e:
                self.logger.error(f"Shadow model failed to make a decision: {e}", exc_info=True)
                shadow_decision = {"error": str(e)}
        
        # 3. 记录两个决策以进行离线比较 (Task 1.1.3)
        self.audit_manager.log_shadow_decision(
            champion_decision=champion_decision,
            shadow_decision=shadow_decision
        )

        # 4. 只返回冠军模型的决策
        return {
            "status": "success",
            "decision": champion_decision,
            "model_version_champion": self.champion_model.version, # 假设模型有版本
            "model_version_shadow": self.shadow_model.version if self.shadow_model else None
        }

    def promote_shadow_to_champion(self) -> Dict[str, Any]:
        """
        将当前加载的 '影子' 模型提升为 '冠军'。
        此方法由 API 网关在手动批准后调用。
        (Task 3.2 - Secure Promotion)
        """
        self.logger.info("Received request to promote shadow model to champion...")
        
        if not self.shadow_model:
            self.logger.error("Promotion failed: No shadow model is loaded.")
            return {"status": "error", "message": "No shadow model loaded."}

        # 这是原子性的提升步骤
        self.champion_model = self.shadow_model
        self.shadow_model = None # 影子槽现在为空
        
        self.logger.warning("PROMOTION COMPLETE: Shadow model is now the new Champion.")
        return {"status": "success", "message": "Shadow model promoted to champion."}
