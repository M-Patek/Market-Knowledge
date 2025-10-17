import tensorflow as tf
import numpy as np
import logging
import shap
import pandas as pd
from typing import Dict, Any, List

# --- Transformer核心构建块 ---

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    """处理特征嵌入和层级化位置编码。"""
    def __init__(self, max_len, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.feature_emb = tf.keras.layers.Dense(embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        # 用于日历特征的层级化嵌入
        self.day_of_week_emb = tf.keras.layers.Embedding(input_dim=7, output_dim=embed_dim)
        self.month_of_year_emb = tf.keras.layers.Embedding(input_dim=12, output_dim=embed_dim)

    def call(self, inputs):
        x, calendar_features = inputs
        x_emb = self.feature_emb(x)
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        pos_emb = self.pos_emb(positions)
        day_emb = self.day_of_week_emb(calendar_features[:, :, 0])
        month_emb = self.month_of_year_emb(calendar_features[:, :, 1])
        return x_emb + pos_emb + day_emb + month_emb

class AdaNorm(tf.keras.layers.Layer):
    """自适应层归一化，根据外部宏观状态动态调整参数。"""
    def __init__(self, epsilon=1e-6):
        super(AdaNorm, self).__init__()
        self.gamma_beta_generator = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(2)
        ])
        self.epsilon = epsilon

    def call(self, inputs):
        x, context = inputs
        mean, variance = tf.nn.moments(x, axes=[-1], keepdims=True)
        gamma, beta = tf.split(self.gamma_beta_generator(context), 2, axis=-1)
        return (x - mean) * tf.math.rsqrt(variance + self.epsilon) * gamma + beta

class TransformerBlock(tf.keras.layers.Layer):
    """核心Transformer块，包含因果多头注意力和自适应归一化。"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.adanorm1 = AdaNorm(epsilon=1e-6)
        self.adanorm2 = AdaNorm(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs_with_context, training=False):
        inputs, context = inputs_with_context
        attn_output = self.att(inputs, inputs, use_causal_mask=True) # 强制因果性
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.adanorm1([inputs + attn_output, context])
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.adanorm2([out1 + ffn_output, context])

class FeatureGatingNetwork(tf.keras.layers.Layer):
    """特征门控网络，动态加权输入特征。"""
    def __init__(self, num_features):
        super(FeatureGatingNetwork, self).__init__()
        self.gate_generator = tf.keras.layers.Dense(num_features, activation='sigmoid')

    def call(self, inputs):
        gate = self.gate_generator(inputs)
        return inputs * gate

class ReasonerAttention(tf.keras.layers.Layer):
    """内生推理器加权，在每个时间步动态加权输入特征（推理器）。"""
    def __init__(self, num_reasoners, **kwargs):
        super(ReasonerAttention, self).__init__(**kwargs)
        self.num_reasoners = num_reasoners
        self.attention_network = tf.keras.Sequential([
            tf.keras.layers.Dense(num_reasoners, activation='tanh'),
            tf.keras.layers.Dense(num_reasoners, activation='softmax')
        ])
    def call(self, sequence_input):
        attention_weights = self.attention_network(sequence_input)
        return sequence_input * attention_weights

class MetaLearner:
    """
    第二层MetaLearner，一个为金融序列特征深度定制的、
    超鲁棒的Transformer决策引擎。
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("PhoenixProject.MetaLearner")
        self.config = config
        model_params = self.config.get('meta_learner', {}).get('model_params', {})
        self.n_timesteps = model_params.get('n_timesteps', 10)
        self.n_features = model_params.get('n_features', 5)
        self.loss_config = self.config.get('meta_learner', {}).get('loss_params', {})
        self.adv_config = self.config.get('meta_learner', {}).get('adversarial_training', {})
        self.explainer = None
        self.level_two_transformer = self._build_model()

        # [V2.0+] Cost Feedback Loop
        self.cost_feedback_enabled = self.config.get('cost_feedback_enabled', False)
        self.cost_feature_ema_span = self.config.get('cost_feature_ema_span', 20)
        self.execution_cost_ema = 0.0

    def _build_model(self):
        """构建最终的、深度优化的Transformer模型。"""
        sequence_input = tf.keras.layers.Input(shape=(self.n_timesteps, self.n_features), name="sequence_input")
        context_input = tf.keras.layers.Input(shape=(1,), name="context_input")
        calendar_input = tf.keras.layers.Input(shape=(self.n_timesteps, 2), dtype=tf.int32, name="calendar_input")
        graph_input = tf.keras.layers.Input(shape=(32,), name="graph_input")

        # 1. 特征嵌入与层级化位置编码
        embedding_layer = TokenAndPositionEmbedding(max_len=self.n_timesteps, vocab_size=1, embed_dim=self.n_features)
        x = embedding_layer([sequence_input, calendar_input])

        # 1.2 内生推理器加权
        reasoner_attention_layer = ReasonerAttention(num_reasoners=self.n_features)
        x = reasoner_attention_layer(x)

        # 1.5. 特征门控网络
        feature_gating_layer = FeatureGatingNetwork(num_features=self.n_features)
        x = feature_gating_layer(x)
        
        # 2. 核心Transformer块
        transformer_block = TransformerBlock(embed_dim=self.n_features, num_heads=4, ff_dim=32)
        x = transformer_block([x, context_input])

        # 3. [NEW] Cross-Attention Dynamic Fusion
        # Project graph embedding to the same dimension as the sequence features
        projected_graph_emb = tf.keras.layers.Dense(self.n_features)(graph_input)
        # Reshape graph input to be a sequence of length 1 for the attention mechanism
        graph_input_reshaped = tf.keras.layers.Reshape((1, self.n_features))(projected_graph_emb)
        
        # The sequence representation (Query) attends to the graph embedding (Key/Value).
        cross_attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=self.n_features)
        fused_representation = cross_attention_layer(query=x, value=graph_input_reshaped, key=graph_input_reshaped)
        fused_representation = tf.keras.layers.GlobalAveragePooling1D()(fused_representation) # Pool the resulting sequence
        
        # 分支1: 主要BDL头
        bdl_head = tf.keras.layers.Dropout(0.1)(fused_representation)
        bdl_head = tf.keras.layers.Dense(20, activation="relu")(bdl_head)
        bdl_head = tf.keras.layers.Dropout(0.1)(bdl_head)
        output_params = tf.keras.layers.Dense(2, activation='softplus', name='bdl_output')(bdl_head)

        # 分支2: 辅助市场状态头
        regime_head = tf.keras.layers.Dense(10, activation='relu')(fused_representation)
        regime_output = tf.keras.layers.Dense(3, activation='softmax', name='regime_output')(regime_head)

        model = tf.keras.Model(inputs=[sequence_input, context_input, calendar_input, graph_input], outputs=[output_params, regime_output])

        def compound_loss(y_true, y_pred):
            alpha, beta = y_pred[:, 0], y_pred[:, 1]
            beta_dist = tf.compat.v2.distributions.Beta(alpha, beta)
            y_true_clipped = tf.clip_by_value(y_true, 1e-6, 1.0 - 1e-6)
            beta_loss = -beta_dist.log_prob(y_true_clipped)
            f_loss = tf.keras.losses.binary_focal_crossentropy(y_true, y_pred, from_logits=False)
            return self.loss_config.get('beta_nll_weight', 0.5) * beta_loss + (1.0 - self.loss_config.get('beta_nll_weight', 0.5)) * f_loss

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=self.optimizer,
            loss={'bdl_output': compound_loss, 'regime_output': tf.keras.losses.SparseCategoricalCrossentropy()},
            loss_weights={'bdl_output': 1.0, 'regime_output': 0.3}
        )
        return model
    
    def update_execution_costs(self, new_costs: List[float]):
        """Updates the smoothed EMA of execution costs."""
        if not self.cost_feedback_enabled or not new_costs:
            return

        # Use pandas to simplify EMA calculation on the series of costs
        costs_series = pd.Series(new_costs)
        new_ema = costs_series.ewm(span=self.cost_feature_ema_span, adjust=False).mean().iloc[-1]
        
        # Smoothly update the existing EMA
        self.execution_cost_ema = new_ema


    def train(self, features: Dict[str, np.ndarray], labels: np.ndarray, market_uncertainties: np.ndarray, epochs: int = 10, batch_size: int = 32):
        """包含自适应对抗性训练的自定义训练循环。"""
        self.logger.info("使用自定义对抗循环开始MetaLearner训练...")
        train_dataset = tf.data.Dataset.from_tensor_slices((features, labels, market_uncertainties)).shuffle(len(labels)).batch(batch_size)

        for epoch in range(epochs):
            for step, (x_batch_train, y_batch_train, uncertainty_batch) in enumerate(train_dataset):
                with tf.GradientTape(persistent=True) as tape:
                    predictions = self.level_two_transformer(x_batch_train, training=True)
                    y_true_dict = {'bdl_output': y_batch_train, 'regime_output': y_batch_train} # Assuming same label for simplicity
                    loss = self.level_two_transformer.compiled_loss(y_true_dict, predictions)

                    if self.adv_config.get('enabled', False):
                        # Gradient calculation requires model inputs, not the whole x_batch_train dictionary
                        trainable_vars = self.level_two_transformer.trainable_variables
                        # We need to get gradients with respect to inputs, so we watch them
                        tape.watch(list(x_batch_train.values()))
                        # Recalculate loss to ensure tape is watching
                        predictions_for_grad = self.level_two_transformer(x_batch_train, training=True)
                        loss_for_grad = self.level_two_transformer.compiled_loss(y_true_dict, predictions_for_grad)
                        
                        input_gradients = tape.gradient(loss_for_grad, x_batch_train)
                        
                        # Apply perturbation only to the sequence input
                        signed_grad = tf.sign(input_gradients['sequence_input'])
                        base_epsilon = self.adv_config.get('epsilon', 0.02)
                        adaptive_epsilon = base_epsilon * (1.0 - tf.reshape(uncertainty_batch, [-1, 1, 1]))
                        adversarial_perturbation = adaptive_epsilon * signed_grad
                        
                        x_adversarial = x_batch_train.copy()
                        x_adversarial['sequence_input'] = x_batch_train['sequence_input'] + adversarial_perturbation
                        
                        adv_predictions = self.level_two_transformer(x_adversarial, training=True)
                        adversarial_loss = self.level_two_transformer.compiled_loss(y_true_dict, adv_predictions)
                        loss = (loss + adversarial_loss) / 2.0
                
                gradients = tape.gradient(loss, self.level_two_transformer.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.level_two_transformer.trainable_variables))
                del tape
            
            # [NEW] After each epoch, create/update the SHAP explainer
            # We take a background sample formatted correctly as a dictionary for the explainer
            background_features, _, _ = next(iter(train_dataset))
            self.explainer = shap.DeepExplainer(self.level_two_transformer, background_features)

            self.logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.numpy():.4f}")

    def predict(self, features: Dict[str, np.ndarray], num_mc_samples: int = 50) -> Dict[str, Any]:
        """使用蒙特卡洛Dropout进行预测以量化不确定性。"""
        # [V2.0+] Inject cost feature
        if self.cost_feedback_enabled:
            # This assumes features['context_input'] is a placeholder that can be updated.
            # A more robust implementation would have a dedicated feature engineering pipeline.
            self.logger.info(f"Injecting execution cost feature into prediction: {self.execution_cost_ema:.2f} bps EMA")
            # The context input might need to be expanded to include this new feature.
            # For simplicity, we'll just log it here. The model architecture would need to be adapted.
            
        mc_predictions = tf.stack([self.level_two_transformer(features, training=True) for _ in range(num_mc_samples)], axis=0)
        # Correctly average over the Monte Carlo samples (axis=0) for both model outputs
        mean_outputs = tf.reduce_mean(mc_predictions, axis=0)
        mean_params = mean_outputs[0] # The bdl_output is the first element
        alpha, beta = mean_params[:, 0], mean_params[:, 1]
        final_prob = alpha / (alpha + beta)
        posterior_variance = tf.math.reduce_variance(mc_predictions[:, 0, :, :], axis=0)[:, 0]

        shap_values = None
        if self.explainer:
            # We are interested in the shap values for the first output (bdl_output)
            shap_values = self.explainer.shap_values(features)[0]

        return {"final_probability": final_prob.numpy(), "posterior_variance": posterior_variance.numpy(), "shap_values": shap_values, "source": "meta_learner"}
