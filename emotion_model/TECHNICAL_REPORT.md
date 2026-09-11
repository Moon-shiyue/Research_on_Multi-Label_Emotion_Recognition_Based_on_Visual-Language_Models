# 基于视觉语言模型的多标记情感识别 — 基础架构技术说明

> **文档定位**：本文档说明模型的**基础架构**（CLIP 双塔编码 + 层次化注意力融合 + 标签关联 + 多标记分类头）。
> 该架构同时作为消融实验的对照组件使用；项目完整方案的三大核心模块（冲突感知融合 / 情感环形分类头 /
> VL-Adapter）详见 [`CORE_MODULES.md`](CORE_MODULES.md)。

## 1. 总体架构概述

基础架构基于 CLIP (ViT-B/32) 视觉语言模型构建**层次化跨模态融合多标记情感识别架构**。核心设计理念是：**先局部聚焦情感关键区域，再全局对齐多模态特征，最后利用标签关系知识进行推理校准**。

```
输入层         Image + Emotion Labels (Text)
  │
  ▼
编码层         CLIP 双塔 (ViT-B/32 Visual + Text Transformer)
  │
  ▼
融合层  ⭐     层次化注意力融合 (Hierarchical Attention Fusion)
  │            Step 1: 局部高权重特征融合
  │            Step 2: 全局对齐特征校准
  ▼
标签层  ⭐     标签关联建模 (Label Association)
  │            Label Attention + Lightweight GCN
  ▼
输出层         多标记分类头 (Multi-Label Sigmoid)
  │
  ▼
输出           [P(joy), P(sadness), ..., P(excitement)] × 12
```

---

## 2. 核心创新点一：层次化注意力跨模态融合

### 2.1 设计动机

现有的视觉-语言多模态融合方法（如简单的拼接、逐元素相乘或单层交叉注意力）存在以下不足：

1. **粗粒度融合**：将视觉全局特征与文本特征直接融合，忽略了图像中不同区域对应不同情感的事实
2. **单向注意力**：仅关注视觉→文本或文本→视觉的单向信息流，缺乏双向交互
3. **缺乏层次结构**：没有渐进式的从局部到全局的特征抽象过程

### 2.2 创新方案：两步层次化融合

本模型提出「**局部高权重特征融合 → 全局对齐特征校准**」的两步层次化融合策略。

#### Step 1: 局部高权重特征融合 (Local High-Weight Fusion)

**视觉情绪注意力 (Visual Emotion Attention)**

```
Text Features (Query) ──→ Attend to ──→ Visual Patch Features (Key/Value)
       ↑                                        ↑
  "joy" "fear" ...                     49 image patches from ViT
                                           ↓
                              每个情感标签找到图像中
                              最相关的视觉区域
```

- 以**文本情感特征为 Query**，对 ViT 输出的 49 个图像 patch 做交叉注意力
- 模型自动学会：「joy」关注面部微笑区域，「awe」关注广阔天空，「fear」关注暗部或威胁物
- 输出：每个情感标签加权后的局部视觉特征 (B, L, D)

**文本情绪注意力 (Text Emotion Attention)**

```
Visual Global Feature (Query) ──→ Attend to ──→ Text Emotion Features (Key/Value)
       ↑                                              ↑
  图像整体语义                            12 个情感标签的文本表示
       ↓                                              ↓
                              筛选出与当前图像内容
                              最匹配的情感语义
```

- 以**视觉全局特征为 Query**，对多个情感标签的文本特征做交叉注意力
- 模型学会：一张日落照片与「peace」「awe」「contentment」更相关，而不是「fear」「disgust」
- 输出：视觉加权的文本情感特征 (B, L, D)

**局部融合**

将两个方向的注意力输出拼接后通过 MLP 融合，形成同时包含「视觉聚焦」和「语义筛选」信息的局部融合特征。

#### Step 2: 全局对齐特征校准 (Global Alignment Calibration)

```
Local Fused Features (B, L, D)
        │
        ▼
┌──────────────────────┐
│  Multi-Head Self-Attn │  ← 标签间全局依赖建模
│  (Layer 1)            │     例：joy+surprise → excitement
├──────────────────────┤
│  Feed-Forward Network │
├──────────────────────┤
│  Multi-Head Self-Attn │  ← 层级化特征精炼
│  (Layer 2)            │
├──────────────────────┤
│  Global Context Pool  │  ← 多标签全局上下文聚合
│  + Residual Calib.    │
└──────────────────────┘
        │
        ▼
Calibrated Features (B, L, D)
```

- **多层自注意力**：建模多个情感标签之间的全局依赖关系（如 joy 和 sadness 互斥，joy 和 excitement 共现）
- **全局上下文校准**：将所有标签的特征聚合为全局上下文向量，再注入回每个标签特征，消除模态偏移
- **残差连接**：每一步都保留原始信息流，防止深层网络中的特征漂移

### 2.3 创新性总结

| 对比维度 | 传统方法 | 本模型方法 |
|---------|---------|----------|
| 融合粒度 | 全局特征拼接 | patch 级 + 标签级细粒度 |
| 注意力方向 | 单向 | 双向（视觉⇄文本） |
| 融合策略 | 单步融合 | 两步层次化（局部→全局） |
| 校准机制 | 无 | 全局上下文残差校准 |

---

## 3. 核心创新点二：先验知识注入的标签关联建模

### 3.1 设计动机

多标记情感识别与普通多标记分类的关键区别在于：**情感标签之间存在丰富的结构关系**。

- **共现关系**：joy 与 excitement 常同时出现，love 与 peace 高度共现
- **互斥关系**：joy 与 sadness 几乎不会同时出现，peace 与 anger 互斥
- **层级关系**：amusement 是 joy 的子类，awe 与 surprise 语义相近

忽略这些关系会导致不合理的预测（如同时高置信度预测 joy 和 sadness）。

### 3.2 创新方案：双重标签关联建模

本模型采用「数据驱动 + 先验知识」的双重标签关系建模策略。

#### 组件一：标签注意力机制 (Label Attention)

```
可学习标签嵌入 (12 × 128)
        │
        ▼
┌─────────────────────┐
│   Self-Attention     │  ← 从数据中自动学习标签间关系
│   (4 heads)          │
└─────────────────────┘
        │
        ▼
  标签关系矩阵 (12 × 12)    学习到的关系权重
        │
        ▼
┌─────────────────────┐
│   门控融合            │  ← 自适应决定融入多少标签关系信息
│   Gate = σ(W·[feat, label_ctx]) │
└─────────────────────┘
```

- 可学习的标签嵌入经过自注意力，自动发现训练数据中的标签共现模式
- 门控机制保护原始特征，仅在标签关系有用时才融入

#### 组件二：轻量图卷积标签关联网络 (Lightweight Label GCN)

```
先验知识（心理学理论）
        │
        ▼
┌─────────────────────┐
│  标签关系图构建       │
│  ┌───────────────┐  │
│  │ joy──(+)──excitement │  共现（正边）
│  │ joy──(-)──sadness    │  互斥（负边）
│  │ anger──(+)──disgust  │
│  └───────────────┘  │
└─────────────────────┘
        │ 有符号邻接矩阵
        ▼
┌─────────────────────┐
│  Graph Convolution   │
│  (2 layers, signed)  │
│                      │
│  正边：聚合共现标签特征 │
│  负边：抑制互斥标签特征 │
└─────────────────────┘
        │
        ▼
  先验增强的标签表示
```

**关键创新：有符号图卷积**

标准 GCN 仅支持非负邻接矩阵，本模型设计的 GCN 支持正负边：

```
h_i^(l+1) = σ( Σ_{j∈N+(i)} A_ij^+ · W^+ · h_j^(l)    ← 共现标签信息聚合
              + Σ_{j∈N-(i)} A_ij^- · W^- · h_j^(l) )   ← 互斥标签信息抑制
```

- **正边**（共现关系）：鼓励同时激活的标签特征互相增强
- **负边**（互斥关系）：利用负权重抑制不应同时出现的标签

#### 组件三：自适应融合

```
α = σ(w_fusion)  ∈ [0, 1]

enhanced = α · (Label Attention 输出) + (1-α) · (GCN 输出)
```

- 可学习的融合权重 α，在训练中自适应平衡两种标签关系信息来源
- 初期 α≈0.5，两种信息同等重要
- 训练后根据数据特点自动调整

### 3.3 先验知识来源

本模型的标签关系图基于以下心理学理论构建：

| 关系类型 | 来源 | 示例 |
|---------|------|------|
| 基本情感理论 | Ekman (1992) 六种基本情感分类 | anger, disgust, fear, joy, sadness, surprise |
| 情感维度理论 | Russell (1980) Circumplex Model | Valence-Arousal 空间的临近关系 |
| 数据集统计 | ArtPhoto, Emotion6 共现统计 | 从真实标注中提取的共现频率 |

---

## 4. 核心创新点三：非对称损失与标签特定分类

### 4.1 非对称损失函数 (Asymmetric Loss)

多标记情感识别的关键挑战是**极端标签不平衡**：

- 大多数图像仅表达 1-3 种情感（正样本稀疏）
- 负样本（不表达的情感）远多于正样本
- 标准 BCE Loss 会被大量易分负样本主导

本模型实现的 Asymmetric Loss：

```
ASL = -y · (1-p)^γ⁺ · log(p) - (1-y) · p̃^γ⁻ · log(1-p̃)

其中 γ⁺ = 1 (正样本聚焦)，γ⁻ = 4 (强负样本抑制)
     p̃ = max(p, 1-ε) (概率裁剪，防止过自信)
```

- 对正样本使用较小的 γ（保留学习信号）
- 对负样本使用较大的 γ（大幅抑制简单负样本的贡献）
- 概率裁剪防止模型对负类过自信

### 4.2 双模式分类器

支持两种分类器切换，适应不同的部署场景：

| 分类器 | 参数量 | 优势 | 适用场景 |
|-------|--------|------|---------|
| Shared Attention | 较少 | 参数效率高，标签嵌入可迁移 | 标签数多、数据有限 |
| Label-Specific | 较多 | 每个标签独立决策边界 | 标签差异大、数据充分 |

---

## 5. 模型技术指标

### 5.1 参数规模

| 组件 | 参数量（约） | 可训练 |
|------|------------|--------|
| CLIP ViT-B/32 (视觉) | 87.8M | 可选冻结 |
| CLIP Text (文本) | 37.8M | 可选冻结 |
| 层次化融合模块 | ~5.2M | ✓ |
| 标签关联模块 | ~1.1M | ✓ |
| 分类输出头 | ~0.5M | ✓ |
| **总计** | **~132M** | ~7M (冻结编码器) |

### 5.2 计算复杂度

- 单张图像推理时间：~50ms (V100 GPU)
- 训练显存占用：~8GB (batch_size=32, 冻结编码器)
- GCN 额外开销：<1% 总计算量（标签数固定为 12）

---

## 6. 模块依赖关系

```
config.py  ← 全局配置（情感标签、超参数）
   │
   ├── base_encoder.py  ← CLIP 双塔封装
   │      └── VisualEncoder, TextEncoder, CLIPEncoder
   │
   ├── fusion_module.py  ← 层次化注意力融合 ⭐
   │      ├── VisualEmotionAttention
   │      ├── TextEmotionAttention
   │      ├── GlobalAlignmentCalibration
   │      └── HierarchicalAttentionFusion
   │
   ├── label_association.py  ← 标签关联建模 ⭐
   │      ├── build_emotion_label_graph()
   │      ├── GraphConvolution (signed)
   │      ├── LightweightLabelGCN
   │      ├── LabelAttentionModule
   │      └── LabelAssociationModule
   │
   ├── classification_head.py  ← 多标记输出
   │      ├── LabelSpecificClassifier
   │      ├── SharedAttentionClassifier
   │      └── MultiLabelClassificationHead
   │
   ├── full_model.py  ← 完整模型组装
   │      └── MultiLabelEmotionModel
   │
   ├── dataset.py  ← 数据加载
   │      └── MultiLabelEmotionDataset
   │
   └── utils.py  ← 工具函数
          ├── compute_metrics()
          ├── EarlyStopping
          └── LRSchedulerWrapper
```

---

## 7. 使用示例

```python
from emotion_model import MultiLabelEmotionModel
from emotion_model.config import ModelConfig, UNIFIED_EMOTIONS

# 创建模型
config = ModelConfig(
    freeze_visual=True,   # 冻结 CLIP 视觉编码器
    freeze_text=True,     # 冻结 CLIP 文本编码器
    num_emotions=12,
)
model = MultiLabelEmotionModel(config)

# 训练
outputs = model(
    pixel_values=images,           # (B, 3, 224, 224)
    emotion_labels=UNIFIED_EMOTIONS,
)
loss = model.compute_loss(outputs["logits"], targets, loss_type="asymmetric")
loss.backward()

# 推理
results = model.predict(images)
# results["probabilities"]: 每个标签的置信度 [0,1]
# results["predictions"]: 阈值化后的 0/1 预测

# 可解释性分析
attention = model.get_attention_maps(images)
# attention["attention_maps"]: 每个情感标签对图像各区域的关注度
```

---

## 8. 参考文献

1. Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML, 2021. (CLIP)
2. Ekman, P. "An argument for basic emotions." Cognition & Emotion, 1992.
3. Russell, J.A. "A circumplex model of affect." JPSP, 1980.
4. Kipf, T.N. & Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks." ICLR, 2017.
5. Vaswani, A., et al. "Attention Is All You Need." NeurIPS, 2017.
6. Ridnik, T., et al. "Asymmetric Loss for Multi-Label Classification." ICCV, 2021.

---

*文档生成时间: 2026-08*
*项目: 基于视觉语言模型的多标记情感识别研究*
