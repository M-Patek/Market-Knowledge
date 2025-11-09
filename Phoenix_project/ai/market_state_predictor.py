import re # [FIX] Import regex
from typing import Dict, Any, Optional
from Phoenix_project.api.gateway import APIGateway
from Phoenix_project.ai.prompt_manager import PromptManager

class MarketStatePredictor:
    """
    (AI) 使用 LLM (通过 APIGateway) 预测市场状态。
    这是一个用于 L2/L3 智能体 (例如 AlphaAgent) 的 AI 组件。
    """
    
    def __init__(self, llm_client: APIGateway, prompt_manager: PromptManager):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        # (假设我们有一个用于此任务的特定提示)
        # [FIX] We need to handle prompt loading better, maybe in PromptManager
        # For now, let's assume 'analyst.json' is the placeholder if 'market_state_predictor' is missing
        try:
             self.prompt_template = self.prompt_manager.get_prompt("market_state_predictor")
        except KeyError:
             self.prompt_template = self.prompt_manager.get_prompt("analyst") # Fallback

    async def predict(self, context_summary: str) -> Dict[str, Any]:
        """
        根据提供的上下文摘要预测市场状态。
        
        Args:
            context_summary (str): 来自 L2 Fusion Agent 或 Context Bus 的融合上下文。
            
        Returns:
            Dict[str, Any]: 包含预测状态、置信度和推理的字典。
        """
        
        if not self.prompt_template:
            return self._error_response("Prompt 'market_state_predictor' or fallback 'analyst' not found.")
            
        prompt = self.prompt_template.render(context_summary=context_summary)
        
        try:
            response_text = await self.llm_client.generate(prompt)
            return self._parse_prediction_response(response_text)
            
        except Exception as e:
            return self._error_response(str(e))

    def _parse_prediction_response(self, response: str) -> Dict[str, Any]:
        """
        解析来自 LLM 的响应。
        
        Assumes the prompt guides the LLM to return JSON or a simple format:
        STATE: [STATE] (e.g., Bullish, Bearish, Volatile, Neutral)
        CONFIDENCE: [0.0 - 1.0]
        REASONING: [Text]
        """
        try:
            # Simple key-value parsing (less robust than JSON) # [FIX] Using regex
            state = "Neutral"
            confidence = 0.5
            reasoning = response # Default

            # [FIX] More robust regex-based parsing
            # Extract State
            state_match = re.search(r"^STATE:\s*(.*)$", response, re.IGNORECASE | re.MULTILINE)
            if state_match:
                state = state_match.group(1).strip()

            # Extract Confidence
            conf_match = re.search(r"^CONFIDENCE:\s*([0-9.]+)", response, re.IGNORECASE | re.MULTILINE)
            if conf_match:
                confidence = float(conf_match.group(1).strip())
            
            # Extract Reasoning
            reason_match = re.search(r"^REASONING:\s*(.*)$", response, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if reason_match:
                reasoning = reason_match.group(1).strip()
            else:
                # Fallback if no REASONING: tag is found but others are
                if state_match or conf_match:
                     reasoning_lines = []
                     for line in response.split('\n'):
                         if not line.upper().startswith("STATE:") and not line.upper().startswith("CONFIDENCE:"):
                             reasoning_lines.append(line)
                     reasoning = "\n".join(reasoning_lines).strip()
            
            # Basic validation
            if state not in ["Bullish", "Bearish", "Volatile", "Neutral", "Sideways"]:
                state = "Neutral"
            
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                "status": "success",
                "state": state,
                "confidence": confidence,
                "reasoning": reasoning
            }

        except Exception as e:
            return self._error_response(f"Failed to parse LLM response: {e}. Raw: {response}")

    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """
        返回一个标准化的错误响应。
        """
        return {
            "status": "error",
            "state": "Neutral",
            "confidence": 0.0,
            "reasoning": f"Prediction failed: {error_msg}"
        }
