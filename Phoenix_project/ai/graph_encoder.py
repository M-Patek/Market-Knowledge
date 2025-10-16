# Phoenix_project/ai/graph_encoder.py
import tensorflow as tf
import tensorflow_gnn as tfgnn
from typing import List
from .relation_extractor import KnowledgeGraph

class GNNEncoder(tf.keras.Model):
    """
    使用GNN将KnowledgeGraph编码为单个密集嵌入向量。
    """
    def __init__(self, embedding_dim: int = 32, num_node_features: int = 16):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_node_features = num_node_features
        # 用于节点类型的简单嵌入层
        self.node_type_embedding = tf.keras.layers.Embedding(input_dim=10, output_dim=self.num_node_features) # 假设最多10种实体类型

        # GNN层
        self.graph_conv = tfgnn.keras.layers.GCNConv(self.embedding_dim, activation='relu')
        # 将节点状态池化为单个图嵌入的层
        self.pool = tfgnn.keras.layers.Pool(tfgnn.CONTEXT, "mean")

    def call(self, graph_tensor: tfgnn.GraphTensor) -> tf.Tensor:
        """GNN编码器的前向传播。"""
        # 嵌入节点类型以创建初始特征
        node_features = self.node_type_embedding(graph_tensor.node_sets["entities"]["type"])
        graph_tensor = graph_tensor.replace_features(node_sets={"entities": {"features": node_features}})

        # 应用图卷积
        graph = self.graph_conv(graph_tensor, edge_set_name="relations")
        # 池化节点状态以获得图级别的嵌入
        embedding = self.pool(graph)
        return embedding

    def encode_graph(self, kg: KnowledgeGraph) -> tf.Tensor:
        """
        将Pydantic的KnowledgeGraph转换为GraphTensor并进行编码。
        """
        if not kg.entities:
            return tf.zeros((1, self.embedding_dim))

        # 将实体类型简单映射为整数
        type_map = {t: i for i, t in enumerate(set(e.type for e in kg.entities))}

        graph_tensor = tfgnn.GraphTensor.from_pieces(
            node_sets={
                "entities": tfgnn.NodeSet.from_fields(
                    sizes=tf.constant([len(kg.entities)]),
                    features={
                        "type": tf.constant([[type_map.get(e.type, 0) for e in kg.entities]], dtype=tf.int32),
                    })
            },
            edge_sets={
                "relations": tfgnn.EdgeSet.from_fields(
                    sizes=tf.constant([len(kg.relations)]),
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("entities", tf.constant([r.source_id for r in kg.relations], dtype=tf.int32)),
                        target=("entities", tf.constant([r.target_id for r in kg.relations], dtype=tf.int32)),
                    ))
            })
        return self(graph_tensor)
