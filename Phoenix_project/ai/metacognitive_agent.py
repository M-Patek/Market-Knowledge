# ai/metacognitive_agent.py
import logging
from typing import Dict, Any, List, Optional
import json
import yaml

from api.gemini_pool_manager import GeminiPoolManager
from audit_manager import AuditManager # 如 Task 2.3 所述


class MetaCognitiveAgent:
    """
    实现 L3 元认知代理 (Task 2.2)。
    分析最近的决策日志，以发现并输出基于因果模式的新的、透明的
    启发式规则。
    """

    def __init__(self, pool_manager: GeminiPoolManager, audit_manager: AuditManager):
        self.logger = logging.getLogger("PhoenixProject.MetaCognitiveAgent")
        self.pool_manager = pool_manager
        self.audit_manager = audit_manager
        self.model_name = "gemini-1.5-pro-latest" # 用于分析的高能力模型
        self.logger.info("MetaCognitiveAgent initialized.")

    async def run_analysis(self, days_to_analyze: int = 30) -> Optional[Dict[str, Any]]:
        """
        执行一次高频（例如 3-7 天）分析运行。
        专注于最近 'days_to_analyze' 周期的日志。
        """
        self.logger.info(f"Starting meta-cognitive analysis for the last {days_to_analyze} days.")

        # 1. 获取带有 P&L 基本事实的最近决策日志
        logs = await self.audit_manager.fetch_logs_with_pnl(days=days_to_analyze)
        if not logs:
            self.logger.warning("No logs found for the specified period. Ending analysis run.")
            return None

        # 2. 使用 LLM 发现因果模式
        discovered_rules_text = await self._discover_causal_patterns_with_llm(logs)

        if not discovered_rules_text:
            self.logger.warning("LLM analysis did not return any rules.")
            return None

        # 3. 将输出格式化为透明的 L3 规则 (JSON/YAML)
        try:
            parsed_rules = yaml.safe_load(discovered_rules_text)
            self.logger.info(f"Successfully discovered {len(parsed_rules)} new heuristic rules.")
            return {"status": "analysis_complete", "rules": parsed_rules}
        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse YAML output from LLM: {e}")
            self.logger.debug(f"Raw LLM output:\n{discovered_rules_text}")
            return None

    async def _discover_causal_patterns_with_llm(self, logs: List[Dict[str, Any]]) -> Optional[str]:
        """
        使用强大的 LLM 分析日志并提取因果的、人类可读的规则。
        返回 YAML 规则的原始文本块。
        """
        # 目前，我们将日志序列化为 JSON 发送给模型。
        logs_json_string = json.dumps(logs, indent=2, default=str) # 添加 default=str 以处理 datetime

        prompt = f"""
你是一位专攻因果推断的专家级金融分析师。
请分析以下 JSON 数据，其中包含决策日志及相应的盈亏（P&L）结果。
请识别决策参数与财务结果之间强烈的、非显而易见的因果模式。

决策日志:
{logs_json_string}

根据你的分析，以有效的 YAML 格式提出 1-3 条新的启发式规则。
每条规则必须是透明的、人类可读的，并包含一个 0.0 到 1.0 之间的 'confidence_score'。
你的回答必须 *仅仅* 是 YAML 代码块，以 '---' 或列表开头。
"""
        contents = [{"parts": [{"text": prompt}]}]
        try:
            response = await self.pool_manager.generate_content(
                model_name=self.model_name, contents=contents
            )
            return response.text
        except Exception as e:
            self.logger.error(f"LLM call for causal discovery failed: {e}")
            return None
