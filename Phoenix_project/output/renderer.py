# Phoenix_project/output/renderer.py
from typing import Dict, Any
from pathlib import Path
import logging # (主人喵的清洁计划 5.3) [新]
from jinja2 import Template # (主人喵的清洁计划 5.3) [新]

# (主人喵的清洁计划 5.3) [新]
logger = logging.getLogger(__name__)

# 我们知道这个模板存在于根目录中，根据我们的文件分析。
# 也许我们稍后应该将它移动到 `output/`，但现在我们遵循它的位置。
TEMPLATE_PATH = Path(__file__).parent.parent / 'report_template.html'

def render_report(fusion_result: dict) -> str:
    """
    (主人喵的清洁计划 5.3) [已修改]
    使用 Jinja2 模板引擎渲染报告。
    """
    
    # TODO: 实现一个真正的模板引擎 (例如 Jinja2)。
    # 这是一个模拟实现。
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
        # 回退到简单的字符串替换 (以防模板或数据出错)
        report_html = template_str.replace(
            "{{causal_graph_data}}", 
            str(fusion_result.get("causal_graph_data", "Error: Data missing"))
        )
        return report_html
