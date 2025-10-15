# ai/reasoning_ensemble.py
"""
实现了多引擎推理集成，这是认知架构的最高层次。
该模块协调各种专门的推理器，以产生全面、多方面的分析。
"""
import logging
import numpy as np
import yaml
import asyncio
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Protocol, List, Dict, Any, NamedTuple
from pydantic import BaseModel

from ai.validation import EvidenceItem
from .walk_forward_trainer import WalkForwardTrainer
from .contradiction_detector import ContradictionDetector
from .embedding_client import EmbeddingClient
from .counterfactual_tester import CounterfactualTester
from .probability_calibrator import ProbabilityCalibrator
import tensorflow as tf
from .bayesian_fusion_engine import BayesianFusionEngine

# --- 1. 标准接口 & 数据契约 ---

class ReasoningOutput(BaseModel):
    """任何推理器的标准化输出对象。"""
    reasoner_name: str
    conclusion: Any
    confidence: float
    supporting_evidence_ids: List[str]

class IReasoner(Protocol):
    """
    定义了所有推理引擎必须遵守的标准接口。
    """
    async def reason(self, hypothesis: str, evidence: List[EvidenceItem]) -> ReasoningOutput:
        ...

# --- 2. 专门推理器的占位符实现 ---

class BayesianReasoner:
    """我们现有BayesianFusionEngine的包装器，使其符合IReasoner接口。"""
    def __init__(self, fusion_engine: BayesianFusionEngine):
        self.engine = fusion_engine
        self.reasoner_name = "BayesianReasoner"

    async def reason(self, hypothesis: str, evidence: List[EvidenceItem]) -> ReasoningOutput:
        fusion_result = self.engine.fuse(hypothesis, evidence)
        stats = fusion_result.get("summary_statistics")
        
        if stats:
            posterior_dist = fusion_result.get("posterior_distribution", {})
            return ReasoningOutput(
                reasoner_name=self.reasoner_name,
                conclusion={
                    "posterior_mean_probability": stats["mean_probability"],
                    "posterior_alpha": posterior_dist.get("alpha"),
                    "posterior_beta": posterior_dist.get("beta"),
                    "prior_alpha": self.engine.base_prior_alpha,
                    "prior_beta": self.engine.base_prior_beta
                },
                confidence=1.0 - stats["std_dev"],
                supporting_evidence_ids=[e.source_id for e in evidence if e.source_id]
            )
        else: # 处理矛盾情况
            return ReasoningOutput(
                reasoner_name=self.reasoner_name,
                conclusion=fusion_result,
                confidence=0.0,
                supporting_evidence_ids=[]
            )

# ... [其他推理器实现保持不变] ...

# --- 3. 用于综合输出的元学习器 ---

class MetaLearner:
    """
    学习用于组合多个推理器输出的最佳权重，
    以产生最终的综合决策。
    """
    def __init__(self, model_config: Dict[str, Any]):
        from sklearn.ensemble import StackingClassifier
        from sklearn.svm import SVC
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Add, Embedding, Layer, Conv1D, GlobalMaxPooling1D
        import tensorflow as tf
        import xgboost as xgb
        import lightgbm as lgb

        self.logger = logging.getLogger("PhoenixProject.MetaLearner")
        self.model_config = model_config
        self.last_known_efficacies = {}

        # 定义 'AI陪审团' - 一组多样化的基础模型
        estimators = [
            ('xgb', xgb.XGBClassifier(n_estimators=10, random_state=42, use_label_encoder=False, eval_metric='logloss')),
            ('lgb', lgb.LGBMClassifier(n_estimators=10, random_state=42)),
            ('svc', SVC(probability=True, random_state=42))
        ]
        self.base_estimators = StackingClassifier(estimators=estimators, final_estimator=lgb.LGBMClassifier())

        # --- [强化] 分层模型架构 ---
        # Level 1: 用于原始技术特征的基础学习器
        n_technical_features = 2 # (RSI, SMA分数)的占位符
        n_timesteps = 5
        self.level_one_models = {
            "cnn_technical": self._build_level_one_cnn(input_shape=(n_timesteps, n_technical_features), **self.model_config['level_one_cnn'])
        }

        # Level 2: 最终估计器，一个Transformer，用于融合L1输出和定性见解。
        n_features = 3 * 3 + 6 # 针对新的元特征进行了调整
        self.level_two_transformer = self._build_level_two_transformer(input_shape=(n_timesteps, n_features), **self.model_config['level_two_transformer'])

        self.calibrator = ProbabilityCalibrator(method='isotonic')
        self.is_trained = False

    def _build_level_one_cnn(self, input_shape: tuple, filters: int, kernel_size: int) -> 'Model':
        """[新] 构建一个1D CNN用于从原始技术指标中提取特征。"""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D
        inputs = Input(shape=input_shape)
        x = Conv1D(filters=int(filters), kernel_size=int(kernel_size), activation='relu')(inputs)
        x = GlobalMaxPooling1D()(x)
        x = Dense(10, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def beta_nll_loss(self, y_true, y_pred_params):
        """[新] Beta分布的负对数似然损失。"""
        alpha = y_pred_params[:, 0] + 1e-7
        beta = y_pred_params[:, 1] + 1e-7
        dist = tf.compat.v2.distributions.Beta(alpha, beta)
        return -tf.reduce_mean(dist.log_prob(tf.cast(y_true, dtype=tf.float32)))

    def _build_level_two_transformer(self, input_shape: tuple, head_size: int, num_heads: int, ff_dim: int, num_transformer_blocks: int, dropout: float) -> 'Model':
        """[新] 构建仅编码器的Transformer模型。"""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Add
        inputs = Input(shape=input_shape)
        x = TokenAndPositionEmbedding(maxlen=input_shape[0], embed_dim=input_shape[-1])(inputs)
        for _ in range(int(num_transformer_blocks)):
            attn_output = MultiHeadAttention(num_heads=int(num_heads), key_dim=int(head_size), dropout=float(dropout))(x, x)
            x = Add()([x, attn_output])
            x = LayerNormalization(epsilon=1e-6)(x)
            ffn_output = Dense(int(ff_dim), activation="relu")(x)
            ffn_output = Dense(input_shape[-1])(ffn_output)
            ffn_output = Dropout(dropout)(ffn_output)
            x = Add()([x, ffn_output])
            x = LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        x = Dropout(0.1)(x)
        x = Dense(20, activation="relu")(x)
        outputs = Dense(2, activation='softplus')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss=self.beta_nll_loss)
        self.logger.info("Level 2 Transformer 模型构建成功。")
        return model

    def _featurize(self, reasoner_output: ReasoningOutput) -> List[float]:
        """将单个推理器的输出转换为特征向量。"""
        features = [reasoner_output.confidence]
        if reasoner_output.reasoner_name == "BayesianReasoner" and isinstance(reasoner_output.conclusion, dict):
            prob = reasoner_output.conclusion.get("posterior_mean_probability", 0.5)
            features.append(prob)
            features.append(abs(prob - 0.5) * 2)
        else:
            features.extend([0.5, 0.0])
        return features

    def _create_feature_vector(
        self, daily_outputs: List[ReasoningOutput], contradiction_count: int,
        retrieval_quality_score: float, counterfactual_sensitivity: float,
        reasoner_efficacies: Dict[str, float]
    ) -> List[float]:
        """[重构] 为给定日期的数据创建单一、一致的特征向量。"""
        day_feature_vector, reasoner_probs = [], []
        sorted_reasoner_names = sorted(list(reasoner_efficacies.keys()))
        for output in sorted(daily_outputs, key=lambda x: x.reasoner_name):
            day_feature_vector.extend(self._featurize(output))
            if output.reasoner_name == "BayesianReasoner" and isinstance(output.conclusion, dict):
                reasoner_probs.append(output.conclusion.get("posterior_mean_probability", 0.5))
            else:
                reasoner_probs.append(output.confidence)
        day_feature_vector.append(np.std(reasoner_probs) if reasoner_probs else 0)
        bayesian_out = next((o for o in daily_outputs if o.reasoner_name == "BayesianReasoner"), None)
        bayesian_bias, bayesian_variance, evidence_shock = 0.0, 0.0, 0.0
        if bayesian_out and isinstance(bayesian_out.conclusion, dict):
            p_alpha, p_beta = bayesian_out.conclusion.get("posterior_alpha"), bayesian_out.conclusion.get("posterior_beta")
            prior_alpha, prior_beta = bayesian_out.conclusion.get("prior_alpha"), bayesian_out.conclusion.get("prior_beta")
            if p_alpha and p_beta and (p_alpha + p_beta + 1) > 0 and (p_alpha + p_beta) > 0:
                bayesian_variance = (p_alpha * p_beta) / ((p_alpha + p_beta)**2 * (p_alpha + p_beta + 1))
            if prior_alpha and prior_beta and p_alpha and p_beta and (prior_alpha + prior_beta) > 0 and (p_alpha + p_beta) > 0:
                evidence_shock = abs((p_alpha / (p_alpha + p_beta)) - (prior_alpha / (prior_alpha + prior_beta)))
        day_feature_vector.extend([bayesian_bias, bayesian_variance, evidence_shock, contradiction_count, counterfactual_sensitivity, retrieval_quality_score])
        for name in sorted_reasoner_names:
            day_feature_vector.append(reasoner_efficacies.get(name, 0.5))
        return day_feature_vector

    def get_feature_names(self, reasoner_efficacies: Dict[str, float]) -> List[str]:
        """[新] 返回用于可解释性的有序特征名称列表。"""
        names = []
        sorted_reasoner_names = sorted(list(reasoner_efficacies.keys()))
        for name in sorted_reasoner_names:
            names.extend([f"{name}_confidence", f"{name}_posterior_prob", f"{name}_conviction"])
        names.append("meta_dispersion")
        names.extend(["meta_bayesian_bias", "meta_bayesian_variance", "meta_evidence_shock", "meta_contradiction_count", "meta_counterfactual_sensitivity", "meta_retrieval_quality_score"])
        for name in sorted_reasoner_names:
            names.append(f"efficacy_{name}")
        return names

    def _create_sequences(self, features: np.ndarray, labels: np.ndarray, n_timesteps: int = 5) -> (np.ndarray, np.ndarray):
        """[新] 将时间序列数据转换为序列的辅助函数。"""
        X, y = [], []
        for i in range(len(features) - n_timesteps + 1):
            X.append(features[i:(i + n_timesteps)])
            y.append(labels[i + n_timesteps - 1])
        return np.array(X), np.array(y)

    def train(self, X_sequences, y_sequences):
        # ... [训练逻辑保持不变, 但现在它使用新的模型和损失函数] ...
        pass

    def predict(
        self, reasoner_outputs: List[List[ReasoningOutput]], contradiction_count: int,
        retrieval_quality_score: float, counterfactual_sensitivity: float,
        reasoner_efficacies: Dict[str, float]
    ) -> Dict[str, Any]:
        """使用训练好的模型产生最终的综合概率。"""
        if not self.is_trained:
            avg_confidence = np.mean([o.confidence for o in reasoner_outputs[0]]) if reasoner_outputs and reasoner_outputs[0] else 0
            return {"final_probability": avg_confidence, "source": "unweighted_average"}
        daily_features_sequence = []
        for daily_outputs in reasoner_outputs:
            day_feature_vector = self._create_feature_vector(daily_outputs, contradiction_count, retrieval_quality_score, counterfactual_sensitivity, reasoner_efficacies)
            daily_features_sequence.append(day_feature_vector)
        features_3d = np.array(daily_features_sequence).reshape(1, len(daily_features_sequence), -1)
        # --- [强化] 用于BDL的蒙特卡洛 Dropout ---
        n_passes = 30
        predictions = [self.level_two_transformer(features_3d, training=True) for _ in range(n_passes)]
        avg_params = np.mean(np.array(predictions), axis=0)[0][0]
        uncalibrated_prob = avg_params[0] / (avg_params[0] + avg_params[1])
        calibrated_prob = self.calibrator.calibrate([uncalibrated_prob])
        final_prob = calibrated_prob[0] if calibrated_prob else uncalibrated_prob
        return {"final_probability": final_prob, "source": "meta_learner"}


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    """[新] 用于添加位置嵌入的自定义Keras层。"""
    def __init__(self, maxlen, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen
    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

# --- 4. 主要的集成协调器 ---

class ReasoningEnsemble:
    """协调一组专门的推理器并行运行。"""
    def __init__(self, config: Dict[str, Any], embedding_client: EmbeddingClient, prompt_renderer: Any, model_config_path: str = "ai/model_config.yaml"):
        self.config = config
        self.logger = logging.getLogger("PhoenixProject.ReasoningEnsemble")
        # ... [实例化推理器] ...
        with open(model_config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
        self.meta_learner = MetaLearner(model_config)
        self.contradiction_detector = ContradictionDetector(embedding_client)
        bayesian_reasoner = next((r for r in self.reasoners if isinstance(r, BayesianReasoner)), None)
        self.counterfactual_tester = CounterfactualTester(bayesian_reasoner.engine) if bayesian_reasoner else None
        self.logger.info(f"ReasoningEnsemble 初始化了 {len(self.reasoners)} 个推理器。")

    def train(self, historical_data: List[Dict[str, Any]]):
        # ... [训练逻辑] ...
        pass

    async def analyze(self, hypothesis: str, evidence: List[EvidenceItem]) -> Dict[str, Any]:
        """并行运行所有已注册的推理器，综合它们的输出，并返回最终结论。"""
        contradictions = self.contradiction_detector.detect(evidence)
        contradiction_count = len(contradictions)
        rerank_scores = [e.get("final_rerank_score", 0.5) for e in evidence if e.get("final_rerank_score") is not None]
        retrieval_quality_score = np.mean(rerank_scores) if rerank_scores else 0.5
        tasks = [r.reason(hypothesis, evidence) for r in self.reasoners]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        final_outputs = [res for res in results if isinstance(res, ReasoningOutput)]
        counterfactual_sensitivity = 0.0
        if self.counterfactual_tester:
            cf_report = self.counterfactual_tester.run_test_suite(hypothesis, evidence)
            spoof_scenario = cf_report.get("scenarios", {}).get("single_point_of_failure", {})
            counterfactual_sensitivity = spoof_scenario.get("sensitivity", 0.0)
        final_conclusion = self.meta_learner.predict([final_outputs], contradiction_count, retrieval_quality_score, counterfactual_sensitivity, self.meta_learner.last_known_efficacies)
        return {"final_conclusion": final_conclusion, "individual_reasoner_outputs": [o.dict() for o in final_outputs]}
