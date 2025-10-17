# ⚠️ [闭源通知] 凤凰计划 V2.0: 认知增强型量化框架 (Cognitively Enhanced Quant Framework) 深度技术审计与开源窗口关闭通知

**项目版本**: 2.0.0（Beta）
**更新日期**: 2025年10月17日

经过密集的研发与验证，本项目的核心技术架构已完成 **V1.0.0 版本**的全部既定目标，且正在快速迭代至 **V2.0.0 版本**。系统在数据处理、AI 认知融合及风险管理等方面均已达到个人深度应用与实战化的阶段性成熟度。

为了完全专注于下一阶段的独家策略研发、模型优化和核心知识产权保护，项目主线开发将战略性地、且完全地迁入私人闭源仓库。

> 这是一个决定性的举措：此开源仓库将作为项目 V2.0.0 版本的最终技术快照，在 **[2025年10月30日]** 被永久关闭并删除，而非仅仅存档。

---

“凤凰计划” V2.0.0 将会是一套从数据工程、认知决策到 MLOps 生产部署的全栈式、事件驱动型量化交易平台。本版本在 V1.0.0 的工程化基础上，实现了**“混合检索增强生成 (Hybrid RAG)”** 和 **“基于后验不确定性的动态风险管理”** 等核心技术的突破。

## I. 架构思想：证据优先、概率融合与超鲁棒性

V2.0.0 的设计核心是构建一个**可解释、可追溯、且能量化认知不确定性**的金融决策系统。

1.  **多模态证据优先 (Evidence-First Multi-Modal)**：决策不再依赖单一信号，而是基于多路召回（文本、时序、表格）的证据集。
2.  **贝叶斯概率融合 (Bayesian Fusion)**：通过数学严谨的贝叶斯核心，将来自多位 AI 智能体的多元洞察，融合为一个具备**后验概率分布 (Posterior Distribution)** 的最终结论，而非简单的平均。
3.  **生产级 MLOps 闭环**：通过集成 **OpenTelemetry 分布式追踪**、**金丝雀部署**和 **DataOps** 门禁，确保系统的性能、质量和可复现性。

---

## II. 核心技术深度解析：认知增强层 (Cognitive Enhancement Layer)

### II.1 混合检索增强生成 (Hybrid RAG) 基础设施

为解决金融市场数据的异构性问题，系统采用了**三路并行、双阶段融合**的混合检索架构。

* **异构索引策略：**
    * **向量索引 (Pinecone)**：部署在 Pinecone Serverless 实例上，专用于非结构化文本（如新闻、SEC 报告）的**语义相似性检索**。
    * **时序索引 (Elasticsearch)**：专用于高效处理大规模、时间窗口受限的**事件数据**和实体查询。
    * **表格索引 (PostgreSQL/JSONB)**：利用 PostgreSQL 的 `NUMERIC` 和 `JSONB` 字段实现对结构化财务指标（如营收、EPS）的**精确 SQL 过滤**。
* **召回与增强：**
    * **HyDE (Hypothetical Document Embedding)**：在向量搜索前，使用 `gemini-2.5-pro` 生成**假设性文档**，以提高查询向量的语义鲁棒性和召回率。
    * **二阶段融合**：第一阶段采用 **Reciprocal Rank Fusion (RRF)** 算法融合三路召回结果，第二阶段使用 **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) 进行深度语义重排。
* **知识图谱与 GNN 集成：**
    * 使用 `genai.GenerativeModel` 进行**实体与关系提取**，构建结构化的 `KnowledgeGraph`。
    * 采用 **GNNEncoder** (基于 `tensorflow-gnn` 的 **RGCNConv**) 将异构图谱编码为单个密集嵌入向量，供 MetaLearner 使用，实现知识与时序特征的联合推理。

### II.2 贝叶斯证据融合引擎与对抗性验证

融合层是一个复杂的概率引擎，旨在量化共识度与不确定性。

* **动态信誉度模型 (Beta Prior)**：`SourceCredibilityStore` 使用 **Beta 分布**（默认先验 `Beta(2.0, 8.0)`）动态追踪每个证据源的历史准确性。证据融合时，将源的动态先验作为加权因子，优先考虑高信誉源的输入。
* **Contradiction Detector**：
    1.  通过 **Cosine Similarity** 识别语义相似但分数极性相反（一正一负）的证据对。
    2.  使用高能力的 **LLM Arbitrator** (`gemini-2.5-pro`) 对潜在矛盾进行**逻辑裁决**，排除伪矛盾，确保最终融合的证据集具有高度内部一致性。
* **Counterfactual Tester**：内置**反事实测试套件**，通过移除关键证据、类型消融或证据反转等场景，系统性地评估融合引擎对特定输入的**敏感性**和**鲁棒性**。

### II.3 MetaLearner (超鲁棒 Transformer 核心)

第二层 MetaLearner 是一个多头、多输入 Transformer，专为高噪声金融序列设计。

* **自适应架构组件：**
    * **AdaNorm (Adaptive Normalization)**：利用外部宏观状态（例如，VIX、国债收益率）作为上下文输入，动态调整层归一化的 $\gamma$ 和 $\beta$ 参数，从而实现**市场状态自适应**。
    * **图谱融合 (Cross-Attention)**：通过 Cross-Attention 模块将 **GNN 提取的图谱嵌入** 融合到序列特征中，实现结构化和时序信息的深度融合。
* **定制化损失与训练：**
    * **复合损失函数**：结合了**Beta NLL Loss**（用于建模概率输出和不确定性）和 **Focal Loss**（用于处理金融数据中的类别不平衡问题）。
    * **自适应对抗性训练 (AAT)**：在自定义训练循环中，根据样本的**市场不确定性**（由宏观预测模型提供），动态调整梯度扰动 $\epsilon$，在低不确定性市场中增强模型泛化性。
    * **不确定性量化**：采用 **Monte Carlo Dropout** 在推理时量化**后验方差 (Posterior Variance)**，作为决策依据。
* **可解释性**：在训练后生成并维护 **SHAP DeepExplainer**，用于解释 MetaLearner 的最终预测。

---

## III. 风险、工程与 MLOps 闭环

### III.1 动态风险与高保真执行仿真

* **动态风险预算 (Dynamic Risk Budgeting)**：
    * `RiskManager` 接收 MetaLearner 输送的**日平均认知不确定性**。
    * 不确定性越高，其输出的 `capital_modifier` 越接近**最小资本调节因子** (min: 0.1)，从而动态收缩策略的**有效总风险敞口**。
* **风险平价仓位管理**：采用 **Volatility Parity Sizer**，根据资产的 **20 天历史波动率**倒数来分配资金，确保每个头寸对组合风险的贡献均衡。
* **高保真执行仿真 (OrderManager)**：
    * 内置**平方根价格冲击模型**，根据订单占日交易量的百分比 (`abs(final_size) / bar_volume`) 估算**交易成本**和**限价**。
    * 强制执行**流动性约束** (`max_volume_share`: 0.25) 和**最小交易名义金额** (`min_trade_notional`: 1.0)，弥合回测与实盘间的执行滑点差距。

### III.2 生产级 MLOps 与 DataOps 实践

* **金丝雀部署与安全回滚**：
    * `PredictionServer` 实现 Champion/Challenger 蓝绿部署，流量可控分配。
    * `CanaryMonitor` 实时监控挑战者模型的**预测方差**。一旦挑战者的平均方差超出冠军模型基线阈值，`PipelineOrchestrator` 将自动触发**回滚**至 Champion 版本。
* **全链路可观测性 (OpenTelemetry)**：
    * 集成 `opentelemetry` SDK，为关键业务流程（如事件处理）生成分布式 Trace Span，用于生产环境的性能分析和故障诊断。
* **数据契约驱动的 CI/CD**：
    * **Data Contract Enforcement**：利用 `jsonschema` 强制校验 AI 缓存数据 (`asset_analysis_cache.json`)，确保其严格遵循 `data_catalog.json` 中定义的 `fused_ai_analysis_schema`。
    * **DQM 质量门禁**：GitHub Actions CI/CD 流程中设置了 **Data Quality Monitoring (DQM)** 门禁，检查数据的**完整性**和**新鲜度** (`scripts/validate_dataset.py`)，若数据质量不达标，则阻断后续构建。
* **不可变实验环境**：`SnapshotManager` 负责在每次实验运行时创建数据缓存目录的 **SHA256 哈希快照**，确保实验环境的绝对隔离和**100% 可复现性**。

---

## 最终总结

**凤凰计划 V2.0.0** 将会是一个结合了**先进深度学习、严谨概率推理和工业级软件工程实践**的复杂系统。它通过量化 AI 认知的不确定性，实现了对传统技术交易系统的**动态、自适应风险控制**，将量化交易的鲁棒性提升到了一个全新的维度。

本次迁移代表“凤凰计划”已完成其作为公共研究原型的历史使命，正式迈向具备生产级潜力的、追求绝对竞争优势的成熟应用阶段。关闭开源窗口是一个艰难但必要的决定，旨在将项目的全部潜力，转化为**个人在金融决策领域的独有竞争优势**。
