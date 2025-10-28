import logging
from typing import Dict, Any, List
import re

class PromptRenderer:
    """
    处理提示模板的渲染。
    现在还包括用于 Map-Reduce RAG 的分块和融合逻辑 (Task 1.3)。
    """
    def __init__(self):
        self.logger = logging.getLogger("PhoenixProject.PromptRenderer")
        # 在实际场景中，您可能会从文件加载模板
        self.templates = {}

    def _count_tokens(self, text: str) -> int:
        """
        一个简单的Token计数启发式方法。
        注意：这是一个近似值。为了精确计数，
        会使用像 tiktoken 这样的库，但我们使用基于单词的估计以避免新的依赖。
        """
        return len(re.findall(r'\w+', text))

    def chunk_context(self, long_text: str, max_tokens_per_chunk: int = 90000) -> List[str]:
        """
        根据Token限制将长文本分割成块。
        (Task 1.3 - Chunking Logic)
        """
        words = long_text.split()
        if not words:
            return []
        
        chunks = []
        current_chunk = []
        
        for word in words:
            # 检查添加下一个词是否会超过限制
            if self._count_tokens(" ".join(current_chunk + [word])) > max_tokens_per_chunk:
                # 完成当前块
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word] # 开始新块
            else:
                current_chunk.append(word)
        
        # 添加最后一个剩余的块
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        self.logger.info(f"Split long text into {len(chunks)} chunks.")
        return chunks

    def render_reducer_prompt(self, original_query: str, sub_conclusions: List[str]) -> str:
        """
        创建一个最终提示，将子结论合成为一个单一的答案。
        (Task 1.3 - Fusion Logic)
        """
        formatted_conclusions = "\n".join(f"- {conclusion}" for conclusion in sub_conclusions)

        reducer_template = """
原始查询是："{query}"

多个AI代理分析了一个大文档的各个区块，并提供了以下子结论：
{conclusions}

请将以上所有信息合成为一个单一、连贯、全面的最终答案，以回应原始查询。
"""
        return reducer_template.format(query=original_query, conclusions=formatted_conclusions)

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        使用命名模板和上下文词典渲染提示。
        """
        # 这是一个占位符。一个真实的实现会从 self.templates 中查找
        # 并使用 str.format() 或 jinja2 这样的模板引擎。
        if template_name == "example_prompt":
            return f"Analyzing {context.get('ticker')} with data point {context.get('data')}"
        
        self.logger.warning(f"Unknown prompt template '{template_name}'. Returning raw context.")
        return str(context)
