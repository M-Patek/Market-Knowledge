import logging
from typing import List, Dict, Any, Optional
from deepforest import CascadeForestClassifier
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np
import abc

from .probability_calibrator import IsotonicCalibrator

# Data Structures
class EvidenceItem(BaseModel):
    source_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    vector_similarity_score: Optional[float] = None
    final_score: Optional[float] = None

class ReasoningOutput(BaseModel):
    reasoner_name: str
    confidence: float = Field(..., ge=0, le=1)
    explanation: str

# Reasoner Interface
class IReasoner(abc.ABC):
    @abc.abstractmethod
    def reason(self, hypothesis: str, evidence: List[EvidenceItem]) -> ReasoningOutput:
        pass

    def train(self, historical_evidence: List[EvidenceItem], historical_outcomes: List[float]):
        # Default implementation does nothing.
        pass

# Level 1 Reasoners (Base Learners)
class SymbolicLogicReasoner(IReasoner):
    """
    A Level 1 reasoner that uses a set of symbolic rules to evaluate a hypothesis.
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("PhoenixProject.SymbolicLogicReasoner")
        self.rules = self._load_rules(config.get("rules_path", ""))

    def _load_rules(self, path: str) -> List[Any]:
        # In a real system, this would load and compile rules from a YAML/JSON file.
        self.logger.info(f"Loading symbolic rules from '{path}'. (Placeholder)")
        return [{"type": "keyword", "keyword": "positive outlook", "score": 0.8}]

    def reason(self, hypothesis: str, evidence: List[EvidenceItem]) -> ReasoningOutput:
        total_score = 0
        activated_rules = 0
        for rule in self.rules:
            for item in evidence:
                if rule["keyword"] in item.content.lower():
                    total_score += rule["score"]
                    activated_rules += 1
        
        confidence = total_score / activated_rules if activated_rules > 0 else 0.5
        return ReasoningOutput(
            reasoner_name="SymbolicLogicReasoner",
            confidence=min(1.0, confidence),
            explanation=f"Activated {activated_rules} symbolic rules."
        )

class BayesianBeliefNetworkReasoner(IReasoner):
    """
    A Level 1 reasoner that uses a probabilistic graphical model.
    This is a placeholder for a more complex implementation.
    """
    def reason(self, hypothesis: str, evidence: List[EvidenceItem]) -> ReasoningOutput:
        # Placeholder logic
        num_positive_sources = sum(1 for item in evidence if item.final_score and item.final_score > 0.6)
        confidence = 0.5 + (0.1 * num_positive_sources)
        return ReasoningOutput(
            reasoner_name="BayesianBeliefNetworkReasoner",
            confidence=min(1.0, confidence),
            explanation=f"Found {num_positive_sources} high-scoring evidence items."
        )

class LLMExplainerReasoner(IReasoner):
    """
    A Level 1 reasoner that uses a large language model to provide a qualitative
    explanation and a soft confidence score.
    """
    def reason(self, hypothesis: str, evidence: List[EvidenceItem]) -> ReasoningOutput:
        # In a real system, this would make an API call to a model like Gemini.
        # The prompt would be carefully engineered to ask for a confidence score and explanation.
        # For now, we simulate this.
        # This is a placeholder for a more sophisticated implementation
        return ReasoningOutput(reasoner_name="LLMExplainerReasoner", confidence=0.6, explanation="Based on the evidence, the hypothesis seems plausible.")

class DeepForestReasoner(IReasoner):
    """
    [NEW] A Level 1 reasoner using a Deep Forest (Cascade Forest) model.
    This provides a non-gradient-based learning path, adding cognitive diversity.
    """
    def __init__(self):
        self.logger = logging.getLogger("PhoenixProject.DeepForestReasoner")
        self.model = CascadeForestClassifier(random_state=42)
        self.is_trained = False

    def train(self, historical_evidence: List[EvidenceItem], historical_outcomes: List[float]):
        """Trains the Deep Forest model."""
        self.logger.info("Training DeepForestReasoner...")
        # Placeholder: Feature extraction from evidence would be needed here.
        # For now, we simulate with random data to demonstrate the structure.
        if not historical_evidence:
             self.logger.warning("No historical evidence to train DeepForestReasoner.")
             return
        X_train = np.random.rand(len(historical_evidence), 10) # 10 dummy features
        y_train = np.array(historical_outcomes).round() # Deep Forest expects class labels
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.logger.info("DeepForestReasoner training complete.")

    def reason(self, hypothesis: str, evidence: List[EvidenceItem]) -> ReasoningOutput:
        if not self.is_trained:
            return ReasoningOutput(reasoner_name="DeepForestReasoner", confidence=0.5, explanation="Model not trained.")
        X_live = np.random.rand(1, 10) # Simulate live features
        proba = self.model.predict_proba(X_live)[0]
        # Assuming class 1 is "positive outcome"
        confidence = proba[1] if len(proba) > 1 else 0.5
        return ReasoningOutput(reasoner_name="DeepForestReasoner", confidence=confidence, explanation=f"Deep Forest model predicts with {confidence:.2%} confidence.")

class ReasoningEnsemble:
    """
    The master orchestrator for the AI cognitive process. It manages multiple
    Level 1 reasoners and a Level 2 MetaLearner to produce a final, robust conclusion.
    """
    def __init__(self, model_config: Dict[str, Any]):
        self.logger = logging.getLogger("PhoenixProject.ReasoningEnsemble")
        self.reasoners: List[IReasoner] = []
        self._initialize_reasoners(model_config)
        
        meta_learner_config = model_config.get('meta_learner', {})
        self.meta_learner = MetaLearner(
            n_timesteps=meta_learner_config.get('n_timesteps', 5),
            n_features=meta_learner_config.get('n_features', 16),
            loss_config=model_config.get('loss_function'),
            adv_config=model_config.get('adversarial_training')
        )
        
        # Other specialized components
        # self.contradiction_detector = ContradictionDetector()
        # self.counterfactual_tester = CounterfactualTester()
        self.logger.info(f"ReasoningEnsemble initialized with {len(self.reasoners)} active reasoners.")

    def _initialize_reasoners(self, model_config: Dict[str, Any]):
        """Instantiates reasoners based on the provided configuration."""
        reasoner_class_map = {
            "SymbolicLogicReasoner": SymbolicLogicReasoner,
            "BayesianBeliefNetworkReasoner": BayesianBeliefNetworkReasoner,
            "LLMExplainerReasoner": LLMExplainerReasoner,
            "DeepForestReasoner": DeepForestReasoner,
        }

        for reasoner_config in model_config.get('reasoners', []):
            if reasoner_config.get('enabled', False):
                class_name = reasoner_config['class'].split('.')[-1]
                if class_name in reasoner_class_map:
                    self.logger.info(f"Initializing reasoner: {class_name}")
                    cls = reasoner_class_map[class_name]
                    # Pass the specific config dict if it exists, otherwise empty dict
                    instance = cls(**reasoner_config.get('config', {}))
                    self.reasoners.append(instance)
                else:
                    self.logger.warning(f"Unknown reasoner class '{class_name}' in config.")
    
    def train_all(self, historical_data: List[Dict[str, Any]]):
        """
        Trains all components of the ensemble, including Level 1 reasoners and the Level 2 MetaLearner.
        """
        self.logger.info("Starting training for the entire Reasoning Ensemble...")
        # Placeholder for extracting features and labels from historical data
        # historical_evidence = ...
        # historical_outcomes = ...
        # meta_learner_features = ...
        # meta_learner_labels = ...

        # 1. Train all Level 1 reasoners
        # for reasoner in self.reasoners:
        #     reasoner.train(historical_evidence, historical_outcomes)
        
        # 2. Train the Level 2 MetaLearner
        # self.meta_learner.train(meta_learner_features, meta_learner_labels)

        self.logger.info("Ensemble training complete.")


# Level 2 MetaLearner
def beta_nll_loss(y_true, y_pred):
    """Negative log-likelihood of the Beta distribution."""
    # Ensure y_true is within (0, 1) to avoid log(0)
    epsilon = tf.keras.backend.epsilon()
    y_true = tf.clip_by_value(y_true, epsilon, 1. - epsilon)
    
    alpha = y_pred[:, 0]
    beta = y_pred[:, 1]
    return -tf.math.lgamma(alpha) - tf.math.lgamma(beta) + tf.math.lgamma(alpha + beta) - (alpha - 1) * tf.math.log(y_true) - (beta - 1) * tf.math.log(1 - y_true)

def focal_loss(y_true, y_pred, gamma):
    """Focal loss for multi-class classification."""
    alpha = y_pred[:, 0]
    beta = y_pred[:, 1]
    p = alpha / (alpha + beta)
    
    # For binary case, y_true is a float, not one-hot
    # We calculate focal loss for both positive and negative cases and add them.
    pt_1 = tf.where(tf.equal(y_true, 1), p, tf.ones_like(p))
    pt_0 = tf.where(tf.equal(y_true, 0), 1 - p, tf.ones_like(p))
    
    # Clip values to prevent log(0)
    epsilon = tf.keras.backend.epsilon()
    pt_1 = tf.clip_by_value(pt_1, epsilon, 1. - epsilon)
    pt_0 = tf.clip_by_value(pt_0, epsilon, 1. - epsilon)
    
    return -tf.reduce_sum((1 - pt_1) ** gamma * tf.math.log(pt_1)) - tf.reduce_sum((1 - pt_0) ** gamma * tf.math.log(pt_0))

class MetaLearner:
    """
    A Level 2 learner that uses a Transformer-based model to synthesize the outputs
    of Level 1 reasoners into a final, calibrated probability.
    """
    def __init__(self, n_timesteps: int = 5, n_features: int = 10, loss_config: Dict[str, Any] = None, adv_config: Dict[str, Any] = None):
        self.logger = logging.getLogger("PhoenixProject.MetaLearner")
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.loss_config = loss_config if loss_config else {'beta_nll_weight': 0.5, 'focal_loss_gamma': 2.0}
        self.adv_config = adv_config if adv_config else {'enabled': True, 'epsilon': 0.02}
        self.level_two_transformer = self._build_model()
        self.logger.info(f"MetaLearner initialized with loss_config={self.loss_config} and adv_config={self.adv_config}")
        self.is_trained = False
        self.calibrator = IsotonicCalibrator()
        self.last_known_efficacies = {}

    def _build_model(self):
        """Builds the Transformer model."""
        # ... [Transformer model architecture remains the same] ...
        # This is a simplified placeholder
        input_layer = tf.keras.layers.Input(shape=(self.n_timesteps, self.n_features))
        # A real transformer would be more complex
        lstm_layer = tf.keras.layers.LSTM(64)(input_layer)
        dense_layer = tf.keras.layers.Dense(32, activation='relu')(lstm_layer)
        # Output two positive values for the parameters of the Beta distribution
        output_params = tf.keras.layers.Dense(2, activation='softplus')(dense_layer)

        model = tf.keras.Model(inputs=input_layer, outputs=output_params)

        # --- [NEW] Create the compound loss function ---
        def compound_loss(y_true, y_pred):
            beta_loss = beta_nll_loss(y_true, y_pred)
            f_loss = focal_loss(y_true, y_pred, gamma=self.loss_config['focal_loss_gamma'])
            
            return self.loss_config['beta_nll_weight'] * beta_loss + (1.0 - self.loss_config['beta_nll_weight']) * f_loss

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=self.optimizer, loss=compound_loss)
        return model

    def _create_feature_vector(
        self, daily_outputs: List[ReasoningOutput], contradiction_count: int,
        retrieval_quality_score: float, counterfactual_sensitivity: float,
        reasoner_efficacies: Dict[str, float],
        market_state_confidence: float
    ) -> List[float]:
        """[重构] 为给定日期的数据创建单一、一致的特征向量。"""
        day_feature_vector, reasoner_probs = [], []
        # --- [强化] 特征工程 ---
        sorted_reasoner_names = sorted([r.reasoner_name for r in daily_outputs])
        output_map = {r.reasoner_name: r.confidence for r in daily_outputs}
        for name in sorted_reasoner_names:
            reasoner_probs.append(output_map.get(name, 0.5))
        
        day_feature_vector.extend(reasoner_probs)
        # 统计特征
        day_feature_vector.extend([np.mean(reasoner_probs), np.std(reasoner_probs), np.max(reasoner_probs) - np.min(reasoner_probs)])
        # 贝叶斯融合器的先验/后验差异作为特征
        bayesian_bias, bayesian_variance, evidence_shock = 0, 0, 0
        # ... [此处应有从贝叶斯融合器获取这些值的逻辑] ...
        # prior_alpha, prior_beta = 1, 1
        # p_alpha, p_beta = ...
        # if (p_alpha + p_beta) > 0 and (prior_alpha + prior_beta) > 0:
        # evidence_shock = abs((p_alpha / (p_alpha + p_beta)) - (prior_alpha / (prior_alpha + prior_beta)))
        day_feature_vector.extend([bayesian_bias, bayesian_variance, evidence_shock, contradiction_count, counterfactual_sensitivity, retrieval_quality_score])
        for name in sorted_reasoner_names:
            day_feature_vector.append(reasoner_efficacies.get(name, 0.5))
        # [NEW] Add the market state confidence as a new meta-feature
        day_feature_vector.append(market_state_confidence)
        return day_feature_vector

    def get_feature_names(self, reasoner_efficacies: Dict[str, float]) -> List[str]:
        """[新增] 返回与特征向量顺序一致的特征名称列表。"""
        names = []
        sorted_reasoner_names = sorted(reasoner_efficacies.keys())
        for name in sorted_reasoner_names:
            names.append(f"prob_{name}")
        names.extend(["meta_mean_prob", "meta_std_prob", "meta_range_prob"])
        names.extend(["meta_bayesian_bias", "meta_bayesian_variance", "meta_evidence_shock", "meta_contradiction_count", "meta_counterfactual_sensitivity", "meta_retrieval_quality_score"])
        for name in sorted_reasoner_names:
            names.append(f"efficacy_{name}")
        names.append("meta_market_state_confidence")
        return names

    def _create_sequences(self, features: np.ndarray, labels: np.ndarray, n_timesteps: int = 5) -> (np.ndarray, np.ndarray):
        """[新增] 将扁平的时间序列数据转换为适用于LSTM/Transformer的序列数据。"""
        X_seq, y_seq = [], []
        for i in range(len(features) - n_timesteps + 1):
            X_seq.append(features[i:i + n_timesteps])
            y_seq.append(labels[i + n_timesteps - 1])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        return X_seq, y_seq

    def train(self, features: np.ndarray, labels: np.ndarray, epochs: int = 10, batch_size: int = 32):
        """
        [NEW] Custom training loop to incorporate adversarial training.
        """
        self.logger.info("Starting MetaLearner training with custom adversarial loop...")
        train_dataset = tf.data.Dataset.from_tensor_slices((features, labels)).shuffle(len(features)).batch(batch_size)

        for epoch in range(epochs):
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    # 1. Standard forward pass to calculate the original loss
                    predictions = self.level_two_transformer(x_batch_train, training=True)
                    loss = self.level_two_transformer.loss(y_batch_train, predictions)

                    # 2. --- FGSM Adversarial Training Step ---
                    if self.adv_config.get('enabled', False):
                        # Get gradients of the loss w.r.t the input
                        input_gradient = tape.gradient(loss, x_batch_train)
                        signed_grad = tf.sign(input_gradient)
                        adversarial_perturbation = self.adv_config['epsilon'] * signed_grad
                        x_adversarial = x_batch_train + adversarial_perturbation
                        
                        # Run a second forward pass with the adversarial examples
                        adv_predictions = self.level_two_transformer(x_adversarial, training=True)
                        adversarial_loss = self.level_two_transformer.loss(y_batch_train, adv_predictions)
                        loss = (loss + adversarial_loss) / 2.0
                
                # 3. Apply gradients from the combined loss
                grads = tape.gradient(loss, self.level_two_transformer.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.level_two_transformer.trainable_variables))
            self.logger.info(f"Epoch {epoch+1}/{epochs} complete.")
        self.is_trained = True
        self.logger.info("MetaLearner training complete.")

    def predict(
        self, reasoner_outputs: List[List[ReasoningOutput]], contradiction_count: int,
        retrieval_quality_score: float, counterfactual_sensitivity: float,
        reasoner_efficacies: Dict[str, float],
        market_state_confidence: float
    ) -> Dict[str, Any]:
        """使用训练好的模型产生最终的综合概率。"""
        if not self.is_trained:
            avg_confidence = np.mean([o.confidence for o in reasoner_outputs[0]]) if reasoner_outputs and reasoner_outputs[0] else 0
            return {"final_probability": avg_confidence, "source": "unweighted_average"}
        daily_features_sequence = []
        for daily_outputs in reasoner_outputs:
            day_feature_vector = self._create_feature_vector(daily_outputs, contradiction_count, retrieval_quality_score, counterfactual_sensitivity, reasoner_efficacies, market_state_confidence)
            daily_features_sequence.append(day_feature_vector)
        features_3d = np.array(daily_features_sequence).reshape(1, len(daily_features_sequence), -1)
        # --- [强化] 用于BDL的蒙特卡洛 Dropout ---
        n_passes = 30
        predictions = [self.level_two_transformer(features_3d, training=True) for _ in range(n_passes)]
        avg_params = np.mean(np.array(predictions), axis=0)[0]
        alpha, beta = avg_params[0], avg_params[1]

        # --- [NEW] Calculate posterior variance as the uncertainty metric ---
        # Variance of a Beta distribution: (a*b) / ((a+b)^2 * (a+b+1))
        if alpha > 0 and beta > 0:
            posterior_variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        else:
            posterior_variance = 0.0 # Default to no uncertainty if params are invalid

        uncalibrated_prob = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
        calibrated_prob = self.calibrator.calibrate([uncalibrated_prob])
        final_prob = calibrated_prob[0] if calibrated_prob else uncalibrated_prob
        return {"final_probability": final_prob, "posterior_variance": posterior_variance, "source": "meta_learner"}


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    # ... [Implementation remains the same] ...
    pass

class TransformerBlock(tf.keras.layers.Layer):
    # ... [Implementation remains the same] ...
    pass


# Main Ensemble Client Facade
class EnsembleAIClient:
    def __init__(self, config_path: str):
        # ... [Implementation remains the same] ...
        pass

    async def analyze(self, hypothesis: str, evidence: List[EvidenceItem], market_state_confidence: float = 0.0) -> Dict[str, Any]:
        """并行运行所有已注册的推理器，综合它们的输出，并返回最终结论。"""
        # This is a simplified passthrough, in a real system it would have more logic
        # contradictions = self.contradiction_detector.detect(evidence)
        # contradiction_count = len(contradictions)
        # cf_report = self.counterfactual_tester.run_test_suite(hypothesis, evidence)
        # spoof_scenario = cf_report.get("scenarios", {}).get("single_point_of_failure", {})
        # counterfactual_sensitivity = spoof_scenario.get("sensitivity", 0.0)
        # final_conclusion = self.meta_learner.predict([final_outputs], contradiction_count, retrieval_quality_score, 
        #                                              counterfactual_sensitivity, self.meta_learner.last_known_efficacies, market_state_confidence)
        # return {"final_conclusion": final_conclusion, "individual_reasoner_outputs": [o.dict() for o in final_outputs]}
        pass
