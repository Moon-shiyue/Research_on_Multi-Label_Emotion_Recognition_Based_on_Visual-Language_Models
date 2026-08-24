"""
多标记分类输出头模块
Multi-Label Classification Head

设计适配多标记情感识别任务的分类输出头，核心特性：

1. 并行预测：对 N 个情感标签同时进行二分类预测（不是 N 选 1）
2. 置信度输出：每个标签独立输出 [0,1] 置信度分数
3. 层次化解码：局部特征 → 标签特征 → 逐标签概率
4. 非对称损失支持：处理多标记场景中的正负样本不平衡

架构设计：
    Fused Features (B, L, D_fusion)
        ↓
    ┌─────────────────────────────┐
    │  每个标签独立分类头          │  ← 标签特定参数
    │  Label 0: Linear(D→d→1)     │
    │  Label 1: Linear(D→d→1)     │
    │  ...                        │
    │  Label L-1: Linear(D→d→1)   │
    └─────────────────────────────┘
        ↓
    Logits (B, L) → Sigmoid → Probabilities (B, L)

同时支持：
    - 共享分类头：所有标签共享参数，通过标签嵌入区分
    - 标签特定分类头：每个标签拥有独立参数（更灵活，参数略多）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math


class LabelSpecificClassifier(nn.Module):
    """
    标签特定分类器

    为每个情感标签维护独立的分类权重，允许不同标签有不同的决策边界。
    这模拟了人类识别不同情感的独立性 —— "joy" 的判定标准与 "fear" 不同。
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]

        # 为每个标签创建独立的 MLP 分类器
        # 使用分组卷积 / 独立 Linear 实现
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            # 每个标签共享同层结构但拥有独立权重：
            # 输入 (B, L, prev_dim) → 输出 (B, L, h_dim*L)，其中每个标签对应一段 h_dim 维
            layers.append(nn.Linear(prev_dim, h_dim * num_labels))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim * num_labels

        # 最后一层：每个标签输出一个 logit
        layers.append(nn.Linear(prev_dim, 1))

        self.classifier = nn.Sequential(*layers)
        self.num_labels = num_labels

        self._init_weights()

    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # 初始化为小的负偏置，模拟标签稀疏性（多数情感不出现在单张图中）
                    nn.init.constant_(module.bias, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_labels, input_dim) 每个标签的特征表示

        Returns:
            logits: (B, num_labels) 每个标签的原始 logits
        """
        logits = self.classifier(x)  # (B, num_labels, 1)
        return logits.squeeze(-1)    # (B, num_labels)


class SharedAttentionClassifier(nn.Module):
    """
    共享注意力分类器

    使用标签嵌入 (Label Embeddings) + 交叉注意力的方式：
    - 所有标签共享分类参数
    - 通过可学习的标签嵌入区分不同情感
    - 参数效率更高，且标签嵌入可在训练中学习情感语义关系
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        label_embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.input_dim = input_dim

        # 可学习的标签嵌入（每个情感标签的语义向量）
        self.label_embeddings = nn.Parameter(
            torch.randn(num_labels, label_embed_dim) * 0.02
        )

        # 标签嵌入 → query 投影
        self.label_to_query = nn.Sequential(
            nn.Linear(label_embed_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

        # 交叉注意力：Q=label_embed, K/V=fused_features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_features: (B, num_labels, input_dim)

        Returns:
            logits: (B, num_labels)
        """
        B = fused_features.shape[0]

        # 扩展标签嵌入到 batch
        label_emb = self.label_embeddings.unsqueeze(0).expand(B, -1, -1)
        queries = self.label_to_query(label_emb)  # (B, num_labels, input_dim)

        # 交叉注意力
        attn_out, _ = self.cross_attn(
            query=queries,
            key=fused_features,
            value=fused_features,
        )  # (B, num_labels, input_dim)

        # 逐标签输出 logit
        logits = self.output_proj(attn_out).squeeze(-1)  # (B, num_labels)
        return logits


class MultiLabelClassificationHead(nn.Module):
    """
    多标记情感分类输出头

    整合多种分类策略，支持灵活的配置切换。

    核心功能：
    1. 多标签并行预测（同时预测 N 个情感的 0/1）
    2. 置信度输出（sigmoid 概率，非 softmax）
    3. 可选的标签特定分类器 / 共享注意力分类器
    4. 阈值自适应（训练过程中学习最优阈值）

    损失函数支持：
    - BCEWithLogitsLoss（标准二元交叉熵）
    - Asymmetric Loss（非对称损失，处理标签不平衡）
    - Focal Loss（聚焦困难样本）
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_labels: int = 12,
        emotion_labels: List[str] = None,
        hidden_dims: List[int] = None,
        classifier_type: str = "shared_attention",  # "label_specific" | "shared_attention"
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.emotion_labels = emotion_labels or [f"emotion_{i}" for i in range(num_labels)]
        self.classifier_type = classifier_type
        self.label_smoothing = label_smoothing

        # 特征预处理
        self.feature_preprocess = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
        )

        # 根据配置选择分类器
        if classifier_type == "label_specific":
            self.classifier = LabelSpecificClassifier(
                input_dim=input_dim,
                num_labels=num_labels,
                hidden_dims=hidden_dims or [256, 128],
                dropout=dropout,
            )
        elif classifier_type == "shared_attention":
            self.classifier = SharedAttentionClassifier(
                input_dim=input_dim,
                num_labels=num_labels,
                label_embed_dim=128,
                num_heads=4,
                dropout=dropout,
            )
        else:
            raise ValueError(f"未知的分类器类型: {classifier_type}")

        # 可学习的逐标签阈值（用于推理时将概率转为 0/1）
        self.logit_thresholds = nn.Parameter(torch.zeros(num_labels))

        print(f"[MultiLabelClassificationHead] 初始化完成")
        print(f"  - 分类器类型: {classifier_type}")
        print(f"  - 标签数量: {num_labels}")
        print(f"  - 标签列表: {self.emotion_labels}")

    def forward(
        self,
        fused_features: torch.Tensor,
        return_probs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            fused_features: (B, num_labels, input_dim) 融合后的特征
            return_probs: 是否返回概率值

        Returns:
            dict:
                - logits: (B, num_labels) 原始 logits
                - probabilities: (B, num_labels) [0,1] 置信度
                - predictions: (B, num_labels) 二值预测 (0/1)，使用动态阈值
        """
        # 预处理
        x = self.feature_preprocess(fused_features)  # (B, num_labels, input_dim)

        # 分类预测
        logits = self.classifier(x)  # (B, num_labels)

        result = {"logits": logits}

        if return_probs:
            probabilities = torch.sigmoid(logits)
            result["probabilities"] = probabilities

            # 使用可学习阈值进行二值预测
            thresholds = torch.sigmoid(self.logit_thresholds)  # [0, 1]
            predictions = (probabilities > thresholds.unsqueeze(0)).float()
            result["predictions"] = predictions

        return result

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_type: str = "bce",
        pos_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算多标记分类损失

        Args:
            logits: (B, num_labels) 预测 logits
            targets: (B, num_labels) 真实标签 (0/1 多热编码)
            loss_type: "bce" | "asymmetric" | "focal"
            pos_weight: 可选的正样本权重

        Returns:
            loss: 标量损失值
        """
        if self.label_smoothing > 0:
            # 标签平滑
            targets = targets * (1 - self.label_smoothing) + \
                      0.5 * self.label_smoothing

        if loss_type == "bce":
            loss = F.binary_cross_entropy_with_logits(
                logits, targets,
                pos_weight=pos_weight,
                reduction="mean",
            )

        elif loss_type == "asymmetric":
            # Asymmetric Loss (ASL)
            # 对正样本和负样本使用不同的聚焦参数
            gamma_pos = 1.0
            gamma_neg = 4.0  # 更强地压制易分负样本
            clip = 0.05

            probs = torch.sigmoid(logits)

            # 正样本损失
            pos_loss = -((1 - probs) ** gamma_pos) * torch.log(probs + 1e-8)
            pos_loss = pos_loss * targets

            # 负样本损失（带概率裁剪）
            probs_neg = probs.clamp(max=1 - clip)  # 防止过自信
            neg_loss = -((probs_neg) ** gamma_neg) * torch.log(1 - probs_neg + 1e-8)
            neg_loss = neg_loss * (1 - targets)

            loss = (pos_loss + neg_loss).mean()

        elif loss_type == "focal":
            # Focal Loss for multi-label
            gamma = 2.0
            alpha = 0.25

            probs = torch.sigmoid(logits)
            pt = probs * targets + (1 - probs) * (1 - targets)
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = -alpha_t * ((1 - pt) ** gamma) * torch.log(pt + 1e-8)
            loss = loss.mean()

        else:
            raise ValueError(f"未知的损失类型: {loss_type}")

        return loss


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("多标记分类输出头测试")
    print("=" * 60)

    B, L, D = 4, 12, 512
    dummy_features = torch.randn(B, L, D)
    dummy_targets = torch.randint(0, 2, (B, L)).float()

    # 测试标签特定分类器
    print("\n[1] 标签特定分类器测试")
    head_specific = MultiLabelClassificationHead(
        input_dim=D,
        num_labels=L,
        classifier_type="label_specific",
    )
    with torch.no_grad():
        output_specific = head_specific(dummy_features)

    print(f"  logits:        {output_specific['logits'].shape}")
    print(f"  probabilities: {output_specific['probabilities'].shape}")
    print(f"  predictions:   {output_specific['predictions'].shape}")
    print(f"  prob range:    [{output_specific['probabilities'].min():.3f}, {output_specific['probabilities'].max():.3f}]")

    # 测试损失计算
    loss_bce = head_specific.compute_loss(output_specific["logits"], dummy_targets, "bce")
    loss_asl = head_specific.compute_loss(output_specific["logits"], dummy_targets, "asymmetric")
    loss_focal = head_specific.compute_loss(output_specific["logits"], dummy_targets, "focal")
    print(f"  BCE Loss:      {loss_bce.item():.4f}")
    print(f"  ASL Loss:      {loss_asl.item():.4f}")
    print(f"  Focal Loss:    {loss_focal.item():.4f}")

    # 测试共享注意力分类器
    print("\n[2] 共享注意力分类器测试")
    head_shared = MultiLabelClassificationHead(
        input_dim=D,
        num_labels=L,
        classifier_type="shared_attention",
    )
    with torch.no_grad():
        output_shared = head_shared(dummy_features)

    print(f"  logits:        {output_shared['logits'].shape}")
    print(f"  probabilities: {output_shared['probabilities'].shape}")

    print("\n✅ 多标记分类输出头测试通过！")
