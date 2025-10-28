# ai/contradiction_detector.py
"""
实现了在证据集中进行对抗性验证和矛盾检测的服务。
"""
import logging
import asyncio
import numpy as np
import os
from typing import List, Dict, Any, Tuple
import google.generativeai as genai

from api.gemini_pool_manager import GeminiPoolManager # 导入我们的池
from .embedding_client import EmbeddingClient
from ai.validation import EvidenceItem

class ContradictionDetector:
    """
    使用语义相似性和分数对立来识别矛盾的证据项。
    (作为 Task 1.4 的 LLM 仲裁器)
    """
    def __init__(self,
                 pool_manager: GeminiPoolManager, # 添加池
                 embedding_client: EmbeddingClient,
                 similarity_threshold: float = 0.85,
                 positive_threshold: float = 0.7,
                 negative_threshold: float = 0.3):
        """
        初始化检测器。

        Args:
            pool_manager: 用于 LLM 仲裁的统一API池。
            embedding_client: 用于为证据发现生成嵌入的客户端。
            similarity_threshold: 余弦相似度分数，高于此值则认为发现“相似”。
            positive_threshold: 证据分数，高于此值则认为发现“正面”。
            negative_threshold: 证据分数，低于此值则认为发现“负面”。
        """
        self.logger = logging.getLogger("PhoenixProject.ContradictionDetector")
        self.embedding_client = embedding_client
        self.pool_manager = pool_manager # 存储池
        self.similarity_threshold = similarity_threshold
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.model_name = "gemini-1.5-pro-latest" # 存储模型名称，按任务要求升级
        self.logger.info(f"ContradictionDetector configured with model '{self.model_name}' and GeminiPoolManager.")

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """计算两个向量之间的余弦相似度。"""
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    async def detect(self, evidence_list: List[EvidenceItem]) -> List[Tuple[EvidenceItem, EvidenceItem]]:
        """
        在给定列表中检测成对的矛盾证据。

        Returns:
            一个元组列表，每个元组包含一对矛盾的 EvidenceItems。
        """
        if len(evidence_list) < 2:
            return []

        # 1. 为所有有内容的证据发现生成嵌入
        docs_to_embed = [{"content": ev.finding, "_original_item": ev} for ev in evidence_list if ev.finding]
        # 注意: 假设 embedding_client 已被重构为异步的
        embedded_docs = await self.embedding_client.create_text_embeddings(docs_to_embed)

        # 过滤掉任何嵌入失败的项
        valid_evidence = [doc for doc in embedded_docs if 'vector' in doc]
        if len(valid_evidence) < 2:
            return []

        potential_contradictions = []
        
        # 2. 执行向量化余弦相似度计算
        embeddings = np.array([doc['vector'] for doc in valid_evidence])
        # 将每个向量归一化为单位长度
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        # 计算余弦相似度矩阵
        similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)

        # 3. 查找高于阈值的对，并检查分数是否对立
        # 我们使用 np.where 来找到满足条件的索引
        indices_row, indices_col = np.where(similarity_matrix > self.similarity_threshold)

        for i, j in zip(indices_row, indices_col):
            if i >= j: continue # 只考虑 i < j 的对，避免自比较和重复

            item_i = valid_evidence[i]['_original_item']
            item_j = valid_evidence[j]['_original_item']

            is_i_positive = item_i.score >= self.positive_threshold
            is_j_positive = item_j.score >= self.positive_threshold
            is_i_negative = item_i.score <= self.negative_threshold
            is_j_negative = item_j.score <= self.negative_threshold

            if (is_i_positive and is_j_negative) or (is_i_negative and is_j_positive):
                self.logger.info(f"Potential contradiction found (Sim: {similarity_matrix[i, j]:.2f}). Sending to LLM Arbitrator.")
                potential_contradictions.append((item_i, item_j))

        # 3. [NEW] 使用 LLM 仲裁潜在的矛盾
        if not potential_contradictions:
            return []

        tasks = [self._arbitrate_pair(item_i, item_j) for item_i, item_j in potential_contradictions]
        arbitration_results = await asyncio.gather(*tasks)

        # 筛选出 LLM 确认为矛盾的对
        confirmed_contradictions = [pair for pair, is_contradictory in zip(potential_contradictions, arbitration_results) if is_contradictory]
        if confirmed_contradictions:
            self.logger.warning(f"LLM Arbitrator confirmed {len(confirmed_contradictions)} contradictions.")

        return confirmed_contradictions

    async def _arbitrate_pair(self, item1: EvidenceItem, item2: EvidenceItem) -> bool:
        """使用 GeminiPoolManager 确定两个项是否真的矛盾。"""
        prompt = f"""
Analyze the following two pieces of evidence. Do they represent a direct logical contradiction?
Your response MUST be a single word: Contradiction, Neutral, or Entailment.

Evidence A: "{item1.finding}" (Score: {item1.score})
Evidence B: "{item2.finding}" (Score: {item2.score})
"""
        try:
            contents = [{"parts": [{"text": prompt}]}]
            response = await self.pool_manager.generate_content(
                model_name=self.model_name,
                contents=contents,
                generation_config={"temperature": 0.1} # 低温用于仲裁
            )
            # 假设池管理器返回一个类似的对象
            decision = response.text.strip()
            return decision == "Contradiction"
        except Exception as e:
            self.logger.error(f"LLM arbitration call failed: {e}")
            return False
