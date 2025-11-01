# Phoenix_project/output/renderer.py
from typing import Dict, Any
from pathlib import Path

# 我们知道这个模板存在于根目录中，根据我们的文件分析。
# 也许我们稍后应该将它移动到 `output/`，但现在我们遵循它的位置。
TEMPLATE_PATH = Path(__file__).parent.parent / 'report_template.html'

def render_report(fusion_result: dict) -> str:
    """Returns markdown/html report."""
    
    # TODO: 实现一个真正的模板引擎 (例如 Jinja2)。
    # 这是一个模拟实现。
    try:
        with open(TEMPLATE_PATH, 'r') as f:
            template_str = f.read()
    except FileNotFoundError:
        template_str = "<h1>Report Template Not Found</h1><p>{{content}}</p>"

    # 模拟渲染：只替换一个占位符。
    report_html = template_str.replace("{{content}}", str(fusion_result))
    return report_html
