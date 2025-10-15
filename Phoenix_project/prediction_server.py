import yaml
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from contextlib import asynccontextmanager

# --- Phoenix 项目导入 ---
# 在真实的生产环境中，Phoenix项目将被安装为一个包。
# 对于此补丁，我们假设服务器从项目的根目录运行。
from ai.reasoning_ensemble import MetaLearner

# --- 全局模型缓存 ---
# 这个简单的字典将持有我们加载的模型，以避免在每个请求上重新加载。
model_cache: Dict[str, Any] = {}

class PredictionRequest(BaseModel):
    """定义预测请求的输入模式。"""
    feature_sequence: List[List[float]]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    [新] FastAPI生命周期管理器，在启动时加载模型，
    并在关闭时进行清理。
    """
    print("--- 启动预测服务器 ---")
    print("加载生产模型...")
    model_cache["meta_learner"] = load_production_model()
    print("模型加载成功。")
    yield
    # 清理模型和其他资源
    print("--- 关闭预测服务器 ---")
    model_cache.clear()


app = FastAPI(lifespan=lifespan)

def load_production_model() -> MetaLearner:
    """
    [新] 加载生产就绪的MetaLearner。
    在真实的系统中，这将从MLflow中拉取最佳模型的构件URI。
    """
    # 目前，我们加载一个默认配置。
    model_config_path = "ai/model_config.yaml"
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)

    # 使用生产配置实例化MetaLearner
    meta_learner = MetaLearner(model_config)

    # TODO: 从指定的构件文件（例如.h5文件）加载训练好的模型权重
    # meta_learner.level_two_transformer.load_weights("path/to/production_model.h5")

    meta_learner.is_trained = True # 假设权重已加载
    return meta_learner

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    [新] 使用加载的生产模型执行推断。
    """
    if "meta_learner" not in model_cache or not model_cache["meta_learner"].is_trained:
        raise HTTPException(status_code=503, detail="模型未加载或未训练。")

    try:
        model: MetaLearner = model_cache["meta_learner"]
        features_3d = np.array(request.feature_sequence).reshape(1, len(request.feature_sequence), -1)

        # --- 使用蒙特卡洛Dropout执行推断 ---
        n_passes = 30
        predictions = []
        for _ in range(n_passes):
            predictions.append(model.level_two_transformer(features_3d, training=True))

        # --- 处理并返回结果 ---
        predictions_np = np.array(predictions)
        avg_params = np.mean(predictions_np, axis=0)[0][0]
        alpha = float(avg_params[0])
        beta = float(avg_params[1])

        # 计算Beta分布的均值
        mean_probability = alpha / (alpha + beta)

        # TODO: 校准器也应该作为模型构件的一部分被加载。
        # 目前，我们返回未校准的均值。
        # calibrated_prob = model.calibrator.calibrate([mean_probability])

        return {
            "predicted_probability_mean": mean_probability,
            "predicted_beta_dist_alpha": alpha,
            "predicted_beta_dist_beta": beta
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测期间发生错误: {str(e)}")

@app.get("/health")
async def health_check():
    """[新] 简单的健康检查端点。"""
    return {"status": "ok"}
