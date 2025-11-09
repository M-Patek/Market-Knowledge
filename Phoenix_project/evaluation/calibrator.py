from ..core.schemas.fusion_result import FusionResult
from ..monitor.logging import get_logger
import math

logger = get_logger(__name__)

class Calibrator:
    """
    Calibrator (校准器) 负责调整融合结果的置信度分数。
    
    在 L1 融合之后、L2 监督之前应用。
    目标是纠正模型固有的过度自信或信心不足，
    使置信度分数更接近真实的基本事实概率。
    """

    def __init__(self, temperature: float = 1.5):
        """
        初始化校准器。
        
        Args:
            temperature (float): 用于温度缩放 (Temperature Scaling) 的温度值。
                T > 1 会使概率分布更“柔和”（即降低高置信度，提高低置信度），
                用于减少过度自信。
                0 < T < 1 会使分布更“尖锐”（增加信心）。
                T = 1 不执行任何操作。
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
            
        self.temperature = temperature
        logger.info(f"Calibrator initialized with temperature scaling T={self.temperature}")

    def calibrate(self, fusion_result: FusionResult) -> FusionResult:
        """
        对 FusionResult 的置信度分数应用校准。
        
        我们使用一种简化的温度缩放形式，应用于 [0, 1] 范围内的置信度分数。
        
        注意：真正的温度缩放通常应用于 softmax 之前的 logits。
        由于我们只有最终的置信度（概率），我们使用一种
        启发式方法：new_confidence = confidence^(1/T)
        这具有在 T > 1 时“拉低”高置信度（例如 0.9 -> 0.85）
        和“拉高”低置信度（例如 0.1 -> 0.15）的效果，
        从而使整体分布更接近 0.5。
        
        Args:
            fusion_result: 来自 FusionAgent 的原始结果。

        Returns:
            具有校准后置信度分数的 FusionResult。
        """
        
        if self.temperature == 1.0:
            # 如果 T=1，则无需校准
            return fusion_result

        original_confidence = fusion_result.confidence
        
        # 防止在 0 或 1 处出现数学问题
        # 我们假设置信度代表了“正确”情感的概率
        epsilon = 1e-9
        p = max(epsilon, min(1.0 - epsilon, original_confidence))

        try:
            # 将置信度（概率）转换为 logit
            # logit = log(p / (1 - p))
            logit = math.log(p / (1.0 - p))
            
            # 应用温度缩放
            scaled_logit = logit / self.temperature
            
            # 将缩放后的 logit 转换回概率 (置信度)
            # scaled_p = 1 / (1 + exp(-scaled_logit))
            scaled_confidence = 1.0 / (1.0 + math.exp(-scaled_logit))
            
            # 确保值在有效范围内
            scaled_confidence = max(0.0, min(1.0, scaled_confidence))

            logger.info(
                f"Calibrated confidence for {fusion_result.symbol} from "
                f"{original_confidence:.4f} to {scaled_confidence:.4f} "
                f"using temperature T={self.temperature}"
            )
            
            # 创建一个新的 FusionResult 或修改现有的（取决于实现）
            # 为了安全起见，我们创建一个新的实例
            
            calibrated_result = fusion_result.model_copy()
            calibrated_result.confidence = scaled_confidence
            
            return calibrated_result

        except Exception as e:
            logger.error(
                f"Failed to calibrate confidence {original_confidence} "
                f"with temperature {self.temperature}: {e}. "
                f"Returning original result.",
                exc_info=True
            )
            # 如果校准失败，返回原始结果
            return fusion_result
