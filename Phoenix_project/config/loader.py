import yaml
import os
from typing import Dict, Any, Optional
# import logging # <-- 已替换
from Phoenix_project.monitor.logging import get_logger # <-- 已添加

# 使用标准 logging，因为 monitor.logging 可能尚未配置
# logger = logging.getLogger("PhoenixProject.ConfigLoader") # <-- 已替换
# 使用统一的 get_logger，它能处理早期初始化问题
logger = get_logger("PhoenixProject.ConfigLoader")


def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    从给定路径加载 YAML 配置文件。
    
    这是为了修复 phoenix_project.py 中
    试图从 .yaml 文件导入函数的严重启动错误。
    [主人喵 Phase 0 修复]: 增加路径解析鲁棒性。
    """
    # 尝试解析绝对路径，解决从不同目录运行脚本时的路径问题
    if not os.path.isabs(config_path):
        # 假设 config_path 是相对于项目根目录 config/ 的 (例如 "system.yaml")
        # 或者相对于此 loader.py 文件的
        base_dir = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join(base_dir, config_path)
        
        # 如果拼接后的路径不存在，回退到原始路径尝试
        if os.path.exists(abs_path):
            config_path = abs_path
            
    if not os.path.exists(config_path):
        logger.critical(f"配置文件未找到: {config_path}")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        logger.info(f"成功从 {config_path} 加载配置")
        return config_data
    except yaml.YAMLError as e:
        logger.critical(f"解析 YAML 配置文件 {config_path} 出错: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.critical(f"读取配置文件 {config_path} 失败: {e}", exc_info=True)
        return None
