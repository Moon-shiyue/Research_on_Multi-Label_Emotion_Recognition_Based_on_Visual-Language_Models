"""
情感环形表示多标记分类头
Emotion Circle Representation Multi-Label Classification Head

理论依据:
  - Mikels, J. A., et al. "Emotional category data on images from the
    International Affective Picture System." Behavior Research Methods, 2005.
  - Yang, J., et al. "A Circular-Structured Representation for Visual Emotion
    Distribution Learning." CVPR 2021.

核心思想:
  情感并非相互独立的离散标签，而是在心理学环形模型（Mikel's Wheel）上
  按「极性-类型-强度」三个属性连续分布。本模块将离散的情感标签映射到
  三维连续情感空间，使模型能够刻画情感的共存性、相似性与强弱差异。

情感环形三个属性（论文 Eq.1）:
  e_i = (p_i, θ_i, r_i)
    - p_i: 情感极性 (polarity)  — 0 = 积极(positive), 1 = 消极(negative)
    - θ_i: 情感类型 (type)      — 极角 θ ∈ [0, 2π)，不同情感按角度相邻
    - r_i: 情感强度 (intensity) — 极径 r ∈ [0, 1]，1 为最强

8 类基本情感的角度分配（论文 Eq.3: θ_j = (2j-1)/8 · π）:
  沿环形逆时针排列，积极情感位于 [0, π/2) ∪ [3π/2, 2π)，
  消极情感位于 [π/2, 3π/2)，相邻情感在心理学语义上最相近。

三维直角坐标转换（申请书公式14）:
  x_k = r · a_k · cos(θ_k)
  y_k = r · a_k · sin(θ_k)
  z_k = v_k
  其中 a_k 为强度、θ_k 为角度、v_k 为极性值（z 轴）。

三分支输出（申请书公式15-18）:
  p_k = σ(W_pk · Z + b_pk)          极性判别分支
  v_k = tanh(W_vk · Z + b_vk)       连续极性值
  θ_k = 2π · σ(W_θk · Z + b_θk)     角度类型分支
  a_k = σ(W_ak · Z + b_ak)          强度回归分支

渐进式环形损失 Progressive Circular Loss（论文 Eq.7-10）:
  L_p  = (1/N) Σ (p_i - p̂_i)²                        极性约束（粗）
  L_t  = (1/N) Σ (θ_i - θ̂_i)²                        类型约束（中）
  L_PC = (1/N) Σ r_i · ((p_i - p̂_i)² + (θ_i - θ̂_i)²)  强度加权约束（细）
  L    = (1-μ)·L_KL + μ·L_PC                          与分布损失联合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


# ============================================================
# 情感环形常量定义
# ============================================================

# Mikels Wheel 8 类基本情感（按环形角度逆时针排列）
# 角度 = (2j-1)/8 · π, j = 1..8
CIRCULAR_EMOTIONS = [
    "amusement",    # j=1 → 22.5°
    "excitement",   # j=2 → 67.5°
    "anger",        # j=3 → 112.5°
    "disgust",      # j=4 → 157.5°
    "fear",         # j=5 → 202.5°
    "sadness",      # j=6 → 247.5°
    "awe",          # j=7 → 292.5°
    "contentment",  # j=8 → 337.5°
]

# 每类情感的环形角度（弧度）
EMOTION_ANGLES = {
    name: (2 * (i + 1) - 1) / 8 * math.pi
    for i, name in enumerate(CIRCULAR_EMOTIONS)
}

# 情感极性: 0 = 积极, 1 = 消极
# 积极: θ ∈ [0, π/2) ∪ [3π/2, 2π)   → amusement, excitement, awe, contentment
# 消极: θ ∈ [π/2, 3π/2)             → anger, disgust, fear, sadness
POLARITY_POSITIVE = 0
POLARITY_NEGATIVE = 1
POLARITY_NAMES = ["positive", "neutral", "negative"]  # 三分支判别（含中性）


def polarity_of_angle(theta: torch.Tensor) -> torch.Tensor:
    """
    根据极角判断极性（论文 Eq.2）

    Args:
        theta: (...,) 极角，取值 [0, 2π)

    Returns:
        polarity: (...,) 0 = 积极, 1 = 消极
    """
    theta = theta % (2 * math.pi)
    positive = ((theta >= 0) & (theta < math.pi / 2)) | \
               ((theta >= 3 * math.pi / 2) & (theta < 2 * math.pi))
    return torch.where(positive, torch.zeros_like(theta), torch.ones_like(theta))


def build_emotion_angle_vector(num_labels: int = 8) -> torch.Tensor:
    """构建 8 类基本情感的角度向量（弧度）"""
    return torch.tensor(
        [EMOTION_ANGLES[name] for name in CIRCULAR_EMOTIONS[:num_labels]],
        dtype=torch.float32,
    )


def build_emotion_polarity_vector(num_labels: int = 8) -> torch.Tensor:
    """构建 8 类基本情感的极性向量（0 = 积极, 1 = 消极）"""
    angles = build_emotion_angle_vector(num_labels)
    return polarity_of_angle(angles)


# ============================================================
# 情感环形映射器
# ============================================================

class EmotionCircleMapper(nn.Module):
    """
    情感环形映射器

    实现论文 Algorithm 1：将情感分布映射为情感环形向量。

    流程:
        1. 用各情感的描述度（分布值）加权基本情感向量
        2. 在环形上做向量加法，得到复合情感向量
        3. 由直角坐标反解出极坐标 (p, θ, r)

    该映射同时支持正反两个方向：
        - distribution_to_vector: 标签分布 → 环形向量（用于构造监督信号）
        - vector_to_distribution: 环形向量 → 标签分布（用于评估/解释）
    """

    def __init__(self, num_labels: int = 8, radius: float = 1.0):
        super().__init__()
        self.num_labels = num_labels
        self.radius = radius

        # 基本情感的单位向量（角度 + 极性）
        angles = build_emotion_angle_vector(num_labels)
        polarities = build_emotion_polarity_vector(num_labels)

        self.register_buffer("base_angles", angles)          # (C,)
        self.register_buffer("base_polarities", polarities)  # (C,)
        # 基本情感的单位向量 (C, 2)：xy 平面
        self.register_buffer(
            "base_vectors",
            torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1),
        )

    def distribution_to_vector(self, distribution: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        情感分布 → 复合情感环形向量（论文 Algorithm 1）

        Args:
            distribution: (B, C) 情感分布（可为概率分布或多热标签）

        Returns:
            dict:
                - polarity: (B,) 极性（0/1）
                - angle: (B,) 复合情感极角 θ ∈ [0, 2π)
                - intensity: (B,) 复合情感强度 r ∈ [0, 1)
                - cartesian: (B, 3) 三维直角坐标 (x, y, z)
        """
        B = distribution.shape[0]
        # 1) 用描述度加权基本情感向量，并在环形上求和
        #    x_i = Σ_j d_j · cos(θ_j), y_i = Σ_j d_j · sin(θ_j)
        weighted = distribution.unsqueeze(-1) * self.base_vectors.unsqueeze(0)  # (B, C, 2)
        sum_xy = weighted.sum(dim=1)  # (B, 2)

        x, y = sum_xy[:, 0], sum_xy[:, 1]

        # 2) 直角坐标 → 极坐标
        intensity_raw = torch.sqrt(x ** 2 + y ** 2 + 1e-8)  # (B,)
        angle = torch.atan2(y, x) % (2 * math.pi)           # (B,)

        # 归一化强度到 [0, 1]（除以最大可能半径 = 分布总和）
        total = distribution.sum(dim=-1).clamp(min=1e-8)
        intensity = (intensity_raw / total.clamp(min=1e-8)).clamp(0, 1)

        # 3) 极性由角度决定
        polarity = polarity_of_angle(angle)

        # 4) z 轴：极性符号（积极为正、消极为负），幅度用强度
        z = (1.0 - 2.0 * polarity) * intensity

        return {
            "polarity": polarity,                                        # (B,)
            "angle": angle,                                              # (B,)
            "intensity": intensity,                                      # (B,)
            "cartesian": torch.stack([x, y, z], dim=-1),                 # (B, 3)
        }

    def vector_to_distribution(
        self,
        angle: torch.Tensor,
        intensity: torch.Tensor,
    ) -> torch.Tensor:
        """
        环形向量 → 情感分布（反向映射，用于解释与评估）

        以极角到各基本情感角度的环形距离构造 softmax 权重，
        再以强度缩放，得到各情感的描述度。

        Args:
            angle: (B,) 极角
            intensity: (B,) 强度

        Returns:
            distribution: (B, C) 情感描述度分布
        """
        # 环形角度距离（考虑 2π 周期性）
        diff = angle.unsqueeze(-1) - self.base_angles.unsqueeze(0)  # (B, C)
        circ_dist = torch.abs(torch.atan2(torch.sin(diff), torch.cos(diff)))  # (B, C)

        # 距离越小权重越大（温度系数控制尖锐程度）
        weights = F.softmax(-circ_dist * 4.0, dim=-1)  # (B, C)
        return weights * intensity.unsqueeze(-1)


# ============================================================
# 情感环形多标记分类头
# ============================================================

class CircularMultiLabelHead(nn.Module):
    """
    基于情感环形表示的多标记分类头

    在 CLIP 统一视觉-语言语义空间基础上，构建融合「情感极性、角度类型、
    强度」三个维度约束的多标记分类头，通过三个并行分支实现：

        分支1 — 极性判别 (polarity):
            输出三种极性（积极/中性/消极）的概率 + 连续极性值 v_k
            损失: 交叉熵（粗粒度约束）

        分支2 — 角度类型 (type):
            输出各离散情感标签的存在概率（sigmoid 激活，天然适配多标记）
            + 该情感在环形上的角度 θ_k
            损失: BCE（多标记）+ 环形角度回归

        分支3 — 强度回归 (intensity):
            输出每个激活情感的强度值 a_k ∈ (0, 1)
            损失: MSE（细粒度约束）

    最终通过三维直角坐标转换（申请书公式14）得到情感环形向量:
        x_k = r · a_k · cos(θ_k)
        y_k = r · a_k · sin(θ_k)
        z_k = v_k
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_labels: int = 8,
        emotion_labels: List[str] = None,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        radius: float = 1.0,
        use_label_correlation: bool = True,
        label_correlation_matrix: Optional[torch.Tensor] = None,
        correlation_scale: float = 0.3,
    ):
        """
        Args:
            input_dim: 输入特征维度（融合模块输出维度）
            num_labels: 情感标签数（默认 8 类 Mikels 基本情感）
            emotion_labels: 情感标签名称列表
            hidden_dim: 分支隐层维度
            dropout: dropout 比例
            radius: 环形半径常数 r
            use_label_correlation: 是否引入标签共现关联矩阵
            label_correlation_matrix: (C, C) 标签关联矩阵（先验共现/互斥）
            correlation_scale: 标签关联项的缩放系数
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.emotion_labels = emotion_labels or CIRCULAR_EMOTIONS[:num_labels]
        self.radius = radius
        self.use_label_correlation = use_label_correlation
        self.correlation_scale = correlation_scale

        # 共享特征提取（三个分支共享底层表示）
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- 分支1: 极性判别 ----
        # 每个标签位置独立输出其极性三分类 logits（积极/中性/消极）
        self.polarity_branch = nn.Linear(hidden_dim, 3)
        # 连续极性值 v_k ∈ (-1, 1)
        self.polarity_value_branch = nn.Linear(hidden_dim, 1)

        # ---- 分支2: 角度类型 ----
        # 各情感标签存在 logits（sigmoid 后为多标记置信度）
        self.type_branch = nn.Linear(hidden_dim, 1)
        # 每个标签的角度偏移量（在基本情感角度基础上的偏移，用于刻画复合情感）
        self.angle_offset_branch = nn.Linear(hidden_dim, 1)

        # ---- 分支3: 强度回归 ----
        self.intensity_branch = nn.Linear(hidden_dim, 1)

        # 环形常量
        mapper = EmotionCircleMapper(num_labels=num_labels, radius=radius)
        self.register_buffer("base_angles", mapper.base_angles.clone())
        self.register_buffer("base_polarities", mapper.base_polarities.clone())

        # 标签关联矩阵（先验知识：共现为正、互斥为负）
        if use_label_correlation:
            if label_correlation_matrix is not None:
                corr = label_correlation_matrix.float()
            else:
                corr = torch.eye(num_labels)  # 无先验时退化为自连接
            self.register_buffer("label_correlation", corr)

        self.mapper = mapper
        self._init_weights()

    def _init_weights(self):
        """分支输出层初始化为小值，避免训练初期分布过于极端"""
        for module in [self.polarity_branch, self.type_branch,
                       self.intensity_branch, self.angle_offset_branch,
                       self.polarity_value_branch]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # 强度分支初始偏置设为正值，避免初始强度全为 0
        nn.init.constant_(self.intensity_branch.bias, 0.5)

    def forward(
        self,
        features: torch.Tensor,
        return_circle: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            features: (B, num_labels, input_dim) 融合后的多模态特征
            return_circle: 是否返回环形向量表示

        Returns:
            dict:
                - type_logits:     (B, L) 各情感存在 logits
                - probabilities:   (B, L) [0,1] 情感存在概率（多标记置信度）
                - polarity_logits: (B, L, 3) 极性三分支 logits
                - polarity_probs:  (B, L, 3) 极性概率分布
                - polarity_value:  (B, L) 连续极性值 v ∈ (-1,1)
                - angle:           (B, L) 环形角度 θ ∈ [0, 2π)
                - intensity:       (B, L) 情感强度 a ∈ (0,1)
                - circle_vector:   (B, L, 3) 三维直角坐标 (x, y, z)（可选）
        """
        B, L, _ = features.shape

        # 共享表示
        h = self.shared(features)  # (B, L, hidden_dim)

        # ---- 分支1: 极性判别 ----
        polarity_logits = self.polarity_branch(h)                       # (B, L, 3)
        polarity_probs = F.softmax(polarity_logits, dim=-1)             # (B, L, 3)
        polarity_value = torch.tanh(
            self.polarity_value_branch(h)
        ).squeeze(-1)                                                    # (B, L)

        # ---- 分支2: 角度类型 ----
        type_logits = self.type_branch(h).squeeze(-1)                   # (B, L)
        probabilities = torch.sigmoid(type_logits)                      # (B, L) 多标记置信度

        # 角度 = 先验基本角度 + 预测偏移（偏移限制在 ±π/8，保证不跨越相邻情感）
        angle_offset = torch.tanh(
            self.angle_offset_branch(h)
        ).squeeze(-1) * (math.pi / 8)                                   # (B, L)
        base = self.base_angles.view(1, L).expand(B, L)                 # (B, L)
        angle = (base + angle_offset) % (2 * math.pi)                   # (B, L)

        # ---- 分支3: 强度回归 ----
        intensity = torch.sigmoid(
            self.intensity_branch(h)
        ).squeeze(-1)                                                    # (B, L)

        # ---- 标签共现关联增强（申请书公式21）----
        # y_i = σ(W·F + b + M · y_{-i})
        if self.use_label_correlation:
            # 用当前预测概率作为 y_{-i}，通过关联矩阵 M 注入其他标签的信息
            # correlated[b, i] = Σ_j M[i, j] · (y[b, j] - mean_j y[b, j])
            prob_centered = probabilities - probabilities.mean(dim=-1, keepdim=True)
            correlated = torch.einsum("ij,bj->bi", self.label_correlation, prob_centered)
            type_logits = type_logits + self.correlation_scale * correlated
            probabilities = torch.sigmoid(type_logits)

        result = {
            "logits": type_logits,              # 与通用多标记分类头接口保持一致
            "type_logits": type_logits,
            "probabilities": probabilities,
            "polarity_logits": polarity_logits,
            "polarity_probs": polarity_probs,
            "polarity_value": polarity_value,
            "angle": angle,
            "intensity": intensity,
        }

        if return_circle:
            # 三维直角坐标转换（申请书公式14）
            radius = self.radius
            x = radius * intensity * torch.cos(angle)   # (B, L)
            y = radius * intensity * torch.sin(angle)   # (B, L)
            z = polarity_value                           # (B, L)
            result["circle_vector"] = torch.stack([x, y, z], dim=-1)  # (B, L, 3)

        return result

    def predict(
        self,
        features: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        推理接口：输出多标记预测与情感环形表示

        Returns:
            dict:
                - probabilities: (B, L) 情感置信度
                - predictions:   (B, L) 0/1 预测
                - circle_vector: (B, L, 3) 环形三维表示
                - polarity:      (B, L) 每个标签的极性（0=积极, 2=消极）
        """
        out = self.forward(features, return_circle=True)
        probs = out["probabilities"]

        return {
            "probabilities": probs,
            "predictions": (probs > threshold).float(),
            "circle_vector": out["circle_vector"],
            "polarity": out["polarity_probs"].argmax(dim=-1),  # (B, L) 0/1/2
            "intensity": out["intensity"],
            "angle": out["angle"],
        }


# ============================================================
# 渐进式环形损失
# ============================================================

class ProgressiveCircularLoss(nn.Module):
    """
    渐进式环形损失 (Progressive Circular Loss, PC Loss)

    论文 CVPR 2021 Eq.7-10，从粗到细分三步施加约束：

        第一步（粗）— 极性约束:
            L_p = (1/N) Σ (p_i - p̂_i)²
            先保证情感整体倾向（积极/消极）的正确性

        第二步（中）— 类型约束:
            L_t = (1/N) Σ (θ_i - θ̂_i)²
            利用环形角度距离度量情感类型的相似性

        第三步（细）— 强度加权约束:
            L_PC = (1/N) Σ r_i · ((p_i - p̂_i)² + (θ_i - θ̂_i)²)
            以情感强度 r_i 作为置信度，对强情感施加更严格的约束

    另外提供与 KL 散度损失的联合形式（论文 Eq.10）:
        L = (1-μ)·L_KL + μ·L_PC
    """

    def __init__(
        self,
        mu: float = 0.5,
        angle_mode: str = "circular",
        use_intensity_weight: bool = True,
    ):
        """
        Args:
            mu: PC 损失在与 KL 联合时的权重
            angle_mode: 角度误差计算方式
                - "circular": 环形距离 min(|Δθ|, 2π-|Δθ|)（推荐，尊重环形结构）
                - "raw": 直接平方差（论文原始 Eq.8 形式）
            use_intensity_weight: 是否使用强度加权（论文 Eq.9）
        """
        super().__init__()
        self.mu = mu
        self.angle_mode = angle_mode
        self.use_intensity_weight = use_intensity_weight

    @staticmethod
    def circular_angle_error(pred_angle: torch.Tensor, target_angle: torch.Tensor) -> torch.Tensor:
        """环形角度误差（考虑 2π 周期性）"""
        diff = pred_angle - target_angle
        return torch.abs(torch.atan2(torch.sin(diff), torch.cos(diff)))

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        计算渐进式环形损失

        Args:
            pred: 分类头输出，需含
                - polarity_probs: (B, L, 3) 或 polarity_value: (B, L)
                - angle: (B, L)
                - intensity: (B, L)
            target: 监督信号，需含
                - polarity: (B, L) 极性标签（0/1）或 (B, L, 3) 极性分布
                - angle: (B, L) 目标角度（可由情感分布经 EmotionCircleMapper 得到）
                - intensity: (B, L) 目标强度（通常为情感分布值）

        Returns:
            dict: {polar_loss, type_loss, pc_loss, total}
        """
        # ---- 第一步: 极性约束 ----
        if "polarity_probs" in pred:
            # 用极性概率的期望值作为连续极性预测（0=积极, 1=消极）
            polarity_soft = pred["polarity_probs"]  # (B, L, 3)
            # 0: positive, 1: neutral, 2: negative → 极性值 0 / 0.5 / 1
            polarity_levels = torch.tensor([0.0, 0.5, 1.0], device=polarity_soft.device)
            pred_polarity = (polarity_soft * polarity_levels).sum(dim=-1)  # (B, L)
        else:
            pred_polarity = (pred["polarity_value"] + 1) / 2  # tanh(-1,1) → (0,1)

        target_polarity = target["polarity"].float()  # (B, L)
        polar_loss = F.mse_loss(pred_polarity, target_polarity)

        # ---- 第二步: 类型（角度）约束 ----
        pred_angle = pred["angle"]           # (B, L)
        target_angle = target["angle"]       # (B, L)
        if self.angle_mode == "circular":
            angle_err = self.circular_angle_error(pred_angle, target_angle)
            type_loss = (angle_err ** 2).mean()
        else:
            type_loss = F.mse_loss(pred_angle, target_angle)

        # ---- 第三步: 强度加权的细粒度约束 ----
        if self.use_intensity_weight and "intensity" in target:
            weight = target["intensity"].float()  # (B, L) 作为置信度
            if self.angle_mode == "circular":
                per_sample = (pred_polarity - target_polarity) ** 2 + \
                             self.circular_angle_error(pred_angle, target_angle) ** 2
            else:
                per_sample = (pred_polarity - target_polarity) ** 2 + \
                             (pred_angle - target_angle) ** 2
            pc_loss = (weight * per_sample).mean()
        else:
            pc_loss = polar_loss + type_loss

        return {
            "polar_loss": polar_loss,
            "type_loss": type_loss,
            "pc_loss": pc_loss,
            "total": pc_loss,
        }

    def combine_with_distribution_loss(
        self,
        pc_loss: torch.Tensor,
        distribution_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        与分布损失（KL 散度）联合（论文 Eq.10）

        L = (1-μ)·L_KL + μ·L_PC

        Args:
            pc_loss: 渐进式环形损失
            distribution_loss: KL 散度等分布损失

        Returns:
            联合损失
        """
        return (1 - self.mu) * distribution_loss + self.mu * pc_loss


def build_label_correlation_matrix(
    emotion_labels: List[str],
    cooccurrence: Dict[Tuple[str, str], float] = None,
    mutual_exclusion: Dict[Tuple[str, str], float] = None,
) -> torch.Tensor:
    """
    构建标签共现关联矩阵 M（申请书公式21 中的 M）

    正值表示共现（互相增强），负值表示互斥（互相抑制）。

    Args:
        emotion_labels: 情感标签列表
        cooccurrence: 共现关系 {(e1, e2): strength}
        mutual_exclusion: 互斥关系 {(e1, e2): strength}

    Returns:
        (C, C) 关联矩阵
    """
    C = len(emotion_labels)
    M = torch.zeros(C, C)
    idx = {label: i for i, label in enumerate(emotion_labels)}

    if cooccurrence:
        for (e1, e2), strength in cooccurrence.items():
            if e1 in idx and e2 in idx:
                M[idx[e1], idx[e2]] = strength
                M[idx[e2], idx[e1]] = strength

    if mutual_exclusion:
        for (e1, e2), strength in mutual_exclusion.items():
            if e1 in idx and e2 in idx:
                M[idx[e1], idx[e2]] = -strength
                M[idx[e2], idx[e1]] = -strength

    return M


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("情感环形表示多标记分类头 — 模块测试")
    print("=" * 60)

    B, L, D = 4, 8, 512
    features = torch.randn(B, L, D)

    # 1. 情感环形映射
    print("\n[1] 情感环形映射测试")
    mapper = EmotionCircleMapper(num_labels=8)
    dist = torch.tensor([
        [0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2],  # 偏积极
        [0.0, 0.0, 0.5, 0.3, 0.2, 0.0, 0.0, 0.0],  # 偏消极
        [0.125] * 8,                                 # 均匀
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 单一情感
    ])
    vec = mapper.distribution_to_vector(dist)
    print(f"  极性: {vec['polarity'].tolist()}")
    print(f"  角度(度): {[round(a * 180 / 3.14159265, 1) for a in vec['angle'].tolist()]}")
    print(f"  强度: {[round(r, 3) for r in vec['intensity'].tolist()]}")
    print(f"  三维坐标形状: {vec['cartesian'].shape}")

    rev = mapper.vector_to_distribution(vec["angle"], vec["intensity"])
    print(f"  反向映射形状: {rev.shape}")

    # 2. 分类头前向
    print("\n[2] 三分支分类头前向传播")
    head = CircularMultiLabelHead(
        input_dim=D, num_labels=L, emotion_labels=CIRCULAR_EMOTIONS,
    )
    out = head(features)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")

    # 3. 渐进式环形损失
    print("\n[3] 渐进式环形损失")
    target_vec = mapper.distribution_to_vector(dist[:B] if B <= 4 else dist)
    loss_fn = ProgressiveCircularLoss(mu=0.5)
    losses = loss_fn(out, {
        "polarity": target_vec["polarity"].unsqueeze(-1).expand(B, L),
        "angle": target_vec["angle"].unsqueeze(-1).expand(B, L),
        "intensity": target_vec["intensity"].unsqueeze(-1).expand(B, L),
    })
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # 4. 推理接口
    print("\n[4] 推理接口")
    pred = head.predict(features)
    print(f"  probabilities: {tuple(pred['probabilities'].shape)}")
    print(f"  predictions:   {tuple(pred['predictions'].shape)}")
    print(f"  circle_vector: {tuple(pred['circle_vector'].shape)}")

    print("\n✅ 情感环形表示多标记分类头测试通过！")
