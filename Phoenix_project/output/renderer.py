# Phoenix_project/output/renderer.py
from typing import Dict, Any, Optional
from pathlib import Path
import logging # (主人喵的清洁计划 5.3) [新]
from jinja2 import Template # (主人喵的清洁计划 5.3) [新]

# (主人喵的清洁计划 5.3) [新]
logger = logging.getLogger(__name__)

# 我们知道这个模板存在于根目录中，根据我们的文件分析。
# 也许我们稍后应该将它移动到 `output/`，但现在我们遵循它的位置。
TEMPLATE_PATH = Path(__file__).parent.parent / 'report_template.html'

def render_report(fusion_result: Optional[Dict[str, Any]]) -> str:
    """
    (主人喵的清洁计划 5.3) [已修改]
    使用 Jinja2 模板引擎渲染报告。
    [Fix FIX-HIGH-004] Robustness against None input.
    """
    # [Task FIX-HIGH-004] Guard clause for empty input
    if not fusion_result:
        logger.warning("Renderer received empty fusion_result.")
        return "<h1>Error: No Data Available</h1><p>The system failed to generate a fusion result.</p>"

    # [主人喵的修复 11.10] 移除过时的 TODO。
    # 此实现现在使用 Jinja2，这是一个真正的模板引擎。
    try:
        with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            template_str = f.read()
    except FileNotFoundError:
        logger.warning(f"Report template not found at {TEMPLATE_PATH}. Using fallback.")
        template_str = "<h1>Report Template Not Found</h1><p>Data: {{ fusion_result }}</p>"

    # (主人喵的清洁计划 5.3) [新] 使用 Jinja2 替换 .replace()
    try:
        template = Template(template_str)
        
        # 我们将整个 fusion_result 字典作为上下文传递。
        # 假设 fusion_result 包含模板所需的所有键 (例如 'causal_graph_data')
        # 我们将 fusion_result 解包，以便模板可以直接访问 {{ causal_graph_data }}
        report_html = template.render(**fusion_result)
        return report_html
        
    except Exception as e:
        logger.error(f"Failed to render Jinja2 template: {e}", exc_info=True)
        # [Task FIX-HIGH-004] Safe Fallback
        return "<h1>Render Error</h1><p>An error occurred while rendering the report template.</p>"
