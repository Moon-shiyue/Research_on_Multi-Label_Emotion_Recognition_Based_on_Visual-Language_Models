"""
多标记情感识别损失函数

支持的损失函数:
    1. BCE Loss             — 二元交叉熵（多标签学习基础损失）
    2. Focal Loss           — 聚焦难样本，缓解标签不平衡
    3. Asymmetric Loss (ASL) — 非对称损失，分别控制正负样本
    4. Label Smoothing BCE   — 标签平滑
    5. Distribution-aware Loss — LDL 分布感知损失（KL散度）
    6. Combined Loss         — 组合多个损失

参考文献:
    - ASL: "Asymmetric Loss For Multi-Label Classification", ICCV 2021
    - Focal Loss: "Focal Loss for Dense Object Detection", ICCV 2017
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class AsymmetricLoss(nn.Module):
    """
    非对称损失 (Asymmetric Loss for Multi-Label Classification).

    核心思想: 对正/负样本使用不同的聚焦参数，重点处理标签不平衡。

    Args:
        gamma_neg: 负样本聚焦参数 (默认 4.0，更大的值更关注难负样本)
        gamma_pos: 正样本聚焦参数 (默认 1.0)
        clip: 概率裁剪值，防止 log(0)
        eps: 数值稳定
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) 原始 logits（未经过 sigmoid）
            targets: (B, C) 二值标签 (0 or 1)

        Returns:
            scalar loss
        """
        # 转概率并裁剪
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, self.clip, 1.0 - self.clip)

        # 正样本损失
        pos_loss = targets * torch.log(probs + self.eps)
        pos_weight = (1 - probs) ** self.gamma_pos
        pos_loss = -pos_weight * pos_loss

        # 负样本损失
        neg_loss = (1 - targets) * torch.log(1 - probs + self.eps)
        neg_weight = probs ** self.gamma_neg
        neg_loss = -neg_weight * neg_loss

        # 均值
        loss = pos_loss + neg_loss
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss 用于多标签分类。

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

        # 正样本
        pos_loss = -self.alpha * (1 - probs) ** self.gamma * targets * torch.log(probs)
        # 负样本
        neg_loss = -(1 - self.alpha) * probs ** self.gamma * (1 - targets) * torch.log(1 - probs)

        loss = pos_loss + neg_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingBCE(nn.Module):
    """
    标签平滑 BCE。

    将硬标签 [0, 1] 平滑为 [ε/(C-1), 1-ε]。
    缓解过拟合和标签噪声问题。
    """

    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C = targets.shape[1]
        # 平滑标签
        smooth_targets = targets * (1 - self.epsilon) + self.epsilon / C
        return F.binary_cross_entropy_with_logits(logits, smooth_targets)


class DistributionAwareLoss(nn.Module):
    """
    分布感知损失: 结合 KL 散度和 BCE，用于 LDL 标签。

    适用于 Emotion6 等提供情感强度分布的数据集。
    """

    def __init__(self, kl_weight: float = 0.3, temperature: float = 2.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        distributions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C)
            targets: (B, C) 二值 multi-hot
            distributions: (B, C) 连续分布标签 (LDL)，可为 None
        """
        # BCE 基础损失
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        if distributions is not None:
            # KL 散度: 让预测的概率分布逼近真实分布
            pred_probs = torch.sigmoid(logits / self.temperature)
            pred_probs = pred_probs / (pred_probs.sum(dim=1, keepdim=True) + 1e-8)
            target_dist = distributions / (distributions.sum(dim=1, keepdim=True) + 1e-8)

            kl_loss = F.kl_div(
                torch.log(pred_probs + 1e-8),
                target_dist,
                reduction='batchmean',
            )
            return bce_loss + self.kl_weight * kl_loss

        return bce_loss


class CombinedLoss(nn.Module):
    """
    组合损失: BCE + Focal + (可选) 标签平滑

    适用于大多数多标记情感识别场景。
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean")
        self.label_smoothing = label_smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            C = targets.shape[1]
            smooth_targets = targets * (1 - self.label_smoothing) + self.label_smoothing / C
            bce_loss = self.bce(logits, smooth_targets)
        else:
            bce_loss = self.bce(logits, targets)

        focal_loss = self.focal(logits, targets)

        return self.bce_weight * bce_loss + self.focal_weight * focal_loss


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    损失函数工厂。

    Args:
        loss_type: bce / asymmetric / focal / focal_bce / label_smooth / combined / distribution
        **kwargs: 传递给具体损失函数的参数

    Returns:
        损失函数模块
    """
    loss_type = loss_type.lower()

    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()

    elif loss_type == "asymmetric" or loss_type == "asl":
        return AsymmetricLoss(
            gamma_neg=kwargs.get("asymmetric_gamma_neg", 4.0),
            gamma_pos=kwargs.get("asymmetric_gamma_pos", 1.0),
        )

    elif loss_type == "focal":
        return FocalLoss(
            alpha=kwargs.get("focal_alpha", 0.25),
            gamma=kwargs.get("focal_gamma", 2.0),
        )

    elif loss_type == "focal_bce" or loss_type == "combined":
        return CombinedLoss(
            bce_weight=kwargs.get("bce_weight", 0.5),
            focal_weight=kwargs.get("focal_weight", 0.5),
            focal_alpha=kwargs.get("focal_alpha", 0.25),
            focal_gamma=kwargs.get("focal_gamma", 2.0),
            label_smoothing=kwargs.get("label_smoothing", 0.0),
        )

    elif loss_type == "label_smooth" or loss_type == "label_smoothing":
        return LabelSmoothingBCE(epsilon=kwargs.get("label_smoothing", 0.1))

    elif loss_type == "distribution":
        return DistributionAwareLoss(
            kl_weight=kwargs.get("kl_weight", 0.3),
            temperature=kwargs.get("temperature", 2.0),
        )

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. "
                         f"Available: bce, asymmetric, focal, focal_bce, label_smooth, distribution")
