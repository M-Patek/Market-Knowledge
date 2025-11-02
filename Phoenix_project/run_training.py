import typer
import os
import sys
from typing_extensions import Annotated

# 将项目根目录添加到 Python 路径
# 这样我们就可以 `from training.walk_forward_trainer`
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.loader import load_config
from monitor.logging import get_logger

# 动态导入训练器，如果路径修复正确，这里应该能工作
try:
    from training.base_trainer import BaseTrainer
    from training.walk_forward_trainer import WalkForwardTrainer
    from training.drl.multi_agent_trainer import MultiAgentTrainer
    # from training.drl.trading_env import TradingEnv
except ImportError as e:
    print(f"Error: 无法导入训练器模块。请确保文件已按 REFACTOR_PLAN.md 迁移。")
    print(f"Details: {e}")
    sys.exit(1)

logger = get_logger('TrainingRunner')
app = typer.Typer(help="凤凰计划 - 离线模型训练器入口")

def load_deps(config_path: str):
    """加载通用的配置和依赖"""
    logger.info(f"从 {config_path} 加载配置...")
    config = load_config(config_path)
    if not config:
        logger.error("配置加载失败。")
        raise typer.Abort()
    return config

@app.command("wf", help="运行滚动优化 (Walk-Forward) 训练器。")
def run_walk_forward(
    config_path: Annotated[str, typer.Option(help="系统配置文件路径。")] = "config/system.yaml"
):
    """
    启动基于历史数据的滚动优化（Walk-Forward）训练。
    这将产出 MetaLearner 或其他基础模型。
    """
    logger.info("--- 启动滚动优化 (WF) 训练 ---")
    try:
        config = load_deps(config_path)
        
        # 1. 初始化 DataManager (训练器需要它来获取历史数据)
        # ( ... 此处需要初始化 DataManager ... )
        
        # 2. 初始化训练器
        # wf_trainer = WalkForwardTrainer(config, data_manager)
        
        # 3. 运行训练
        # await wf_trainer.run_training_loop()
        
        logger.info("滚动优化 (WF) 训练（模拟）完成。")
        logger.info("请将 'run_training.py' 中的依赖项（如 DataManager）初始化补全。")
        
    except Exception as e:
        logger.error(f"WF 训练失败: {e}", exc_info=True)
        raise typer.Abort()

@app.command("drl", help="运行 DRL (深度强化学习) 智能体训练器。")
def run_drl_trainer(
    config_path: Annotated[str, typer.Option(help="系统配置文件路径。")] = "config/system.yaml"
):
    """
    启动 DRL 智能体（例如 AlphaAgent, RiskAgent）的离线训练。
    这将在模拟环境 (TradingEnv) 中运行数千次迭代。
    """
    logger.info("--- 启动 DRL 智能体训练 ---")
    try:
        config = load_deps(config_path)
        
        # 1. 初始化 DRL 环境
        # drl_env = TradingEnv(config)
        
        # 2. 初始化 DRL 训练器
        # drl_trainer = MultiAgentTrainer(config, drl_env)
        
        # 3. 运行训练
        # await drl_trainer.train_agents()
        
        # 4. 保存模型 (例如到 models/ 目录)
        # drl_trainer.save_models("models/drl_agents_v1.zip")
        
        logger.info("DRL 智能体训练（模拟）完成。")
        logger.info("请将 'run_training.py' 中的依赖项（如 TradingEnv）初始化补全。")
        
    except Exception as e:
        logger.error(f"DRL 训练失败: {e}", exc_info=True)
        raise typer.Abort()

if __name__ == "__main__":
    app()
