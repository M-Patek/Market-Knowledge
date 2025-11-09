from typing import Dict, Any, List
from Phoenix_project.agents.l1.base import L1BaseAgent
from Phoenix_project.core.schemas.data_schema import MarketEvent
from Phoenix_project.ai.reasoning_ensemble import ReasoningEnsemble
from Phoenix_project.ai.prompt_renderer import PromptRenderer
from Phoenix_project.reasoning.compressor import ContextCompressor
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class FundamentalAnalystAgent(L1BaseAgent):
    """
    L1 Agent specializing in fundamental analysis based on news,
    filings, and earnings reports.
    """

    def __init__(
        self,
        agent_id: str,
        config: Dict[str, Any],
        reasoning_ensemble: ReasoningEnsemble,
        prompt_renderer: PromptRenderer,
        compressor: ContextCompressor
    ):
        """
        Initializes the FundamentalAnalystAgent.
        """
        super().__init__(agent_id, config, reasoning_ensemble, prompt_renderer)
        self.compressor = compressor
        self.prompt_name = "l1_fundamental_analyst"
        logger.info(f"FundamentalAnalystAgent (ID: {agent_id}) initialized.")

    async def run(self, event: MarketEvent, context_window: List[MarketEvent]) -> Dict[str, Any]:
        """
        Processes a single market event (e.g., news) and generates
        a fundamental analysis.
        
        [任务 C.2] TODO: Optimize context compression.
        """
        
        symbol = event.get('symbol', 'N/A')
        logger.debug(f"{self.agent_id} processing event for {symbol}...")
        
        # [任务 C.2] 已实现：优化上下文压缩
        try:
            # 1. 将上下文窗口转换为单个文本块
            context_text = " ".join([
                f"Event: {evt.get('summary', str(evt.get('data', '')))} @ {evt.get('timestamp')}" 
                for evt in context_window
                if evt.get('summary') or evt.get('data') # 确保有内容
            ])
            
            if context_text:
                # 2. 压缩文本块
                compressed_context = self.compressor.compress_text(context_text, max_tokens=1000)
            else:
                compressed_context = "No relevant historical context found."
                
        except Exception as e:
            logger.warning(f"Failed to compress context window: {e}. Using raw event data.")
            # 回退：仅使用当前事件的摘要
            compressed_context = event.get('summary') or str(event.get('data', ''))

        # 2. 准备 Prompt 上下文
        render_context = {
            "symbol": symbol,
            "summary": event.get('summary', 'No summary provided.'),
            "source": event.get('source', 'Unknown source'),
            "historical_context": compressed_context # [任务 C.2] 使用压缩后的上下文
        }
        
        # 3. 渲染 Prompt
        try:
            rendered_prompt = self.prompt_renderer.render(
                self.prompt_name,
                render_context
            )
        except Exception as e:
            logger.error(f"Failed to render prompt {self.prompt_name}: {e}", exc_info=True)
            return self.generate_error_response(f"Prompt rendering failed: {e}")
            
        # 4. 调用推理
        try:
            # 假设 reasoning_ensemble.invoke 接受渲染后的 prompt
            analysis = await self.reasoning_ensemble.invoke(
                agent_id=self.agent_id,
                prompt=rendered_prompt
            )
            
            # 5. 格式化输出
            analysis['agent_name'] = self.agent_id
            analysis['symbol'] = symbol
            analysis['source_event_id'] = event.get('id')
            
            logger.info(f"{self.agent_id} generated analysis for {symbol}.")
            return analysis

        except Exception as e:
            logger.error(f"{self.agent_id} failed during reasoning: {e}", exc_info=True)
            return self.generate_error_response(f"Reasoning ensemble failed: {e}")

    def generate_error_response(self, error_msg: str) -> Dict[str, Any]:
        """
        Creates a standardized error response.
        """
        return {
            "agent_name": self.agent_id,
            "error": error_msg,
            "sentiment_label": "NEUTRAL",
            "sentiment_score": 0.5,
            "confidence": 0.0,
            "key_drivers": [],
            "key_risks": []
        }
