# Phoenix_project/evaluation/calibrator.py
"""
Implements a service for calibrating uncertainty based on
consistency and critic feedback.
"""
import logging
from typing import List, Optional

from monitor.metrics import PROBABILITY_CALIBRATION_BRIER_SCORE # 更正后的导入


def compute_uncertainty(consistency: float, critic_issues: int) -> float:
    """
    Returns 0-1 confidence interval (uncertainty).
    High divergence (low consistency) -> high uncertainty.
    """
    # 模拟逻辑：基本不确定性是 (1 - consistency)
    base_uncertainty = 1.0 - consistency
    
    # 为 critic issues 增加惩罚
    # 每个 issue 增加 0.1 不确定性，最高为 1.0
    issue_penalty = critic_issues * 0.1
    
    final_uncertainty = min(1.0, base_uncertainty + issue_penalty)
    
    # TODO: PROBABILITY_CALIBRATION_BRIER_SCORE 指标来自旧文件。
    # 我们应该为这个 `final_uncertainty` 使用一个新的指标
    # 例如 UNCERTAINTY.set(final_uncertainty)
    
    return final_uncertainty
