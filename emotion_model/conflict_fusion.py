"""
注意力引导的冲突感知跨模态融合模块
Attention-Guided Conflict-Aware Cross-Modal Fusion

理论依据:
  - Yao, T., et al. "Cross-Modal Semantic Interference Suppression (CMSIS)
    for image-text matching." Engineering Applications of Artificial
    Intelligence, 2024.
  - 申请书「研究内容1」：基于注意力引导的跨模态融合模块构建

解决的痛点:
  现有模型对「图文情感极性对立」「语义无关」等多元语义冲突识别能力不足，
  以及对图文情感特征语义识别错位、对齐失效。例如：
    - 反讽：文本"我喜欢这宽敞的腿部空间" + 图像（经济舱狭窄实景）
    - 图文错位：图像悲伤但文本欢快

三层结构与核心机制:
  ┌────────────────────────────────────────────────────────────┐
  │ ① 文本情绪注意力子模块（双路径冲突注意力）                    │
  │    - 路径A：字面情感特征（原始文本语义）                       │
  │    - 路径B：真实意图特征（情感词/转折词/否定词引导）            │
  │    - 冲突注意力层：计算双路径余弦差异度                        │
  │    - 动态片段划分：差异度 < 阈值 → 情感一致；否则 → 情感冲突    │
  │    - 模态内对比学习 + 三元组损失，推开"字面相似但情感相反"的文本 │
  ├────────────────────────────────────────────────────────────┤
  │ ② 视觉情绪注意力子模块（文本引导 + 冲突区域捕捉）              │
  │    - 以文本情感特征为引导，计算 patch 级余弦相似度              │
  │    - 筛选核心情感区域（人脸表情、场景氛围、情感化物体）          │
  │    - 动态划分匹配/非匹配区域，对情感对立区域赋予负相似度         │
  │    - 图像-图像对比学习抑制视觉模态内语义干扰                    │
  ├────────────────────────────────────────────────────────────┤
  │ ③ 跨模态冲突感知注意力对齐机制                                │
  │    - 情感一致图文对 → 强化双向注意力权重（特征互补）            │
  │    - 情感冲突图文对 → 弱化一致性对齐，保留独立冲突特征           │
  │    - 联合建模图文情感极性关系与语义关联强度                     │
  └────────────────────────────────────────────────────────────┘

损失函数（CMSIS Eq.15-20）:
  L_v  = Σ[τ_v - S(I,I+) + S(I,Î+)]₊                    图像模态内三元组
  L_t  = Σ[τ_t - S(T,T+) + S(T,T̂+)]₊                    文本模态内三元组
  L_vt = Σ[γ - S(I,T) + S(I,T̂)]₊ + [γ - S(I,T) + S(Î,T)]₊  双向跨模态三元组
  L    = L_vt + L_v + L_t
  其中 Î+ / T̂+ 为 hardest negative（batch 内最难负样本），
  τ_v, τ_t, γ 为 margin 超参数（论文默认 0.3, 0.3, 0.2）。

融合公式（申请书公式4-5, 13）:
  α_v = Softmax(Q_v·K_vᵀ/√d_k)·V_v
  α_t = Softmax(Q_t·K_tᵀ/√d_k)·V_t
  F_fusion = α_v · F_v + α_t · F_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


# ============================================================
# 可学习的动态相关性边界（CMSIS 核心设计）
# ============================================================

class DynamicBoundary(nn.Module):
    """
    动态学习的相关性边界 p

    CMSIS 论文中用于区分「匹配片段」与「不匹配（冲突）片段」：
      - 相似度 > p → 正片段（情感一致）
      - 相似度 < p → 负片段（情感冲突）

    与固定阈值的区别：边界通过正态分布参数化并参与训练，
    能够自适应不同数据集的相似度分布尺度。
    """

    def __init__(self, init_boundary: float = 0.0, learnable: bool = True):
        super().__init__()
        self.boundary = nn.Parameter(
            torch.tensor(float(init_boundary)), requires_grad=learnable
        )

    def forward(self) -> torch.Tensor:
        return self.boundary


# ============================================================
# ① 文本情绪注意力子模块（双路径冲突注意力）
# ============================================================

class TextConflictAttention(nn.Module):
    """
    文本情绪注意力子模块 — 双路径冲突注意力

    设计（对应申请书「文本端冲突提纯」）:
      路径A（字面路径）: 原始文本语义 → 提取字面情感
      路径B（意图路径）: 情感词性标注引导 → 提取真实意图
        若能提供 token 级特征与情感词掩码（emotion_word_mask），
        则直接聚焦情感词/转折词/否定词；否则用可学习的意图提取器模拟。

      冲突注意力层:
        1. 计算两条路径特征的余弦差异度
        diff = 1 - cos(path_A, path_B)  ∈ [0, 2]
        2. 动态片段划分（借助可学习边界 p）:
           - diff ≤ p  → 情感一致片段：融合双路径特征
           - diff > p  → 情感冲突片段：保留独立特征并输出冲突信号
        3. 双向上下文自注意力捕捉长距离依赖

    输出:
      - consistency_features: (B, L, D) 融合后的文本情绪特征
      - conflict_scores:      (B, L) 每个标签位置的冲突强度
      - segment_mask:         (B, L) 1 = 情感冲突片段, 0 = 情感一致片段
    """

    def __init__(
        self,
        text_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        boundary_init: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 字面路径投影
        self.literal_proj = nn.Linear(text_dim, hidden_dim)
        # 意图路径投影（情感词性引导）
        self.intent_proj = nn.Linear(text_dim, hidden_dim)

        # 情感词性引导注意力（当无显式掩码时，学习关注情感触发词）
        self.emotion_word_attn = nn.Sequential(
            nn.Linear(text_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 双向上下文自注意力（捕捉长距离依赖）
        self.context_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_ctx = nn.LayerNorm(hidden_dim)

        # 冲突感知门控：情感一致时融合，冲突时保留独立特征
        self.conflict_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

        # 动态相关性边界
        self.boundary = DynamicBoundary(init_boundary=boundary_init)

    def forward(
        self,
        text_features: torch.Tensor,               # (B, L, text_dim)
        text_token_features: Optional[torch.Tensor] = None,  # (B, M, text_dim)
        emotion_word_mask: Optional[torch.Tensor] = None,    # (B, M)
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_features: 标签级文本特征（每个情感标签的语义表示）
            text_token_features: 可选的 token 级文本特征序列
            emotion_word_mask: 可选的情感词掩码（1 = 情感触发词）

        Returns:
            dict:
                - text_features:   (B, L, D) 冲突感知的文本情绪特征
                - conflict_scores: (B, L) 冲突强度
                - segment_mask:    (B, L) 冲突片段标记
                - literal_features:(B, L, D) 字面路径特征
                - intent_features: (B, L, D) 意图路径特征
        """
        B, L, _ = text_features.shape

        # 字面路径特征
        literal = self.literal_proj(text_features)      # (B, L, D)

        # 意图路径特征
        if text_token_features is not None:
            # 有 token 级特征：按情感词掩码加权聚合出"真实意图"
            token_h = self.intent_proj(text_token_features)  # (B, M, D)
            if emotion_word_mask is not None:
                # 掩码加权（归一化，避免全 0）
                mask = emotion_word_mask.unsqueeze(-1).float()      # (B, M, 1)
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
                intent_global = (token_h * mask).sum(dim=1, keepdim=True) / denom  # (B, 1, D)
            else:
                # 无掩码：用可学习注意力自动定位情感触发词
                attn_logits = self.emotion_word_attn(text_token_features)  # (B, M, 1)
                attn_w = F.softmax(attn_logits, dim=1)                     # (B, M, 1)
                intent_global = (token_h * attn_w).sum(dim=1, keepdim=True)  # (B, 1, D)
            intent = literal + intent_global.expand(-1, L, -1)
        else:
            # 无 token 特征：用自注意力提取"上下文意图"并与字面特征对比
            ctx, _ = self.context_attn(literal, literal, literal)
            intent = literal + self.dropout(ctx)

        # ---- 冲突注意力层：双路径余弦差异度 ----
        cos_sim = F.cosine_similarity(literal, intent, dim=-1)  # (B, L) ∈ [-1, 1]
        diff = (1.0 - cos_sim) / 2.0                            # (B, L) ∈ [0, 1]

        # ---- 动态片段划分（软化，保证边界可学习）----
        # soft_mask 参与前向计算（可导），hard_mask 用于统计与解释
        boundary = self.boundary()
        soft_mask = torch.sigmoid((diff - boundary) * 10.0)     # (B, L) ∈ (0,1)
        hard_mask = (diff > boundary).float()
        segment_mask = soft_mask

        # ---- 冲突感知融合 ----
        # 情感一致 → 门控融合双路径；情感冲突 → 保留意图路径为主（真实表意）
        gate = self.conflict_gate(torch.cat([literal, intent], dim=-1))  # (B, L, D)
        fused_consistent = gate * intent + (1 - gate) * literal
        # 冲突片段的权重：更偏向意图路径（真实情感），并保留字面差异信号
        conflict_weight = segment_mask.unsqueeze(-1)                     # (B, L, 1)
        text_out = fused_consistent * (1 - conflict_weight) + \
                   (0.7 * intent + 0.3 * literal) * conflict_weight

        # 双向上下文自注意力 + FFN
        ctx_out, _ = self.context_attn(text_out, text_out, text_out)
        text_out = self.norm1(text_out + self.dropout(ctx_out))
        text_out = self.norm2(text_out + self.ffn(text_out))

        return {
            "text_features": text_out,
            "conflict_scores": diff,
            "segment_mask": soft_mask,
            "segment_mask_hard": hard_mask,
            "literal_features": literal,
            "intent_features": intent,
            "boundary": boundary,
        }


# ============================================================
# ② 视觉情绪注意力子模块（文本引导 + 冲突区域捕捉）
# ============================================================

class VisualConflictAttention(nn.Module):
    """
    视觉情绪注意力子模块 — 文本引导的细粒度区域提取与冲突捕捉

    设计（对应申请书「视觉端冲突捕捉」）:
      1. 以文本输出的情感特征为引导，计算每个视觉 patch 与情感特征的余弦相似度
      2. 相似度作为注意力权重对 patch 加权，筛选核心情感区域
         （人脸表情、场景氛围、情感化物体等），剔除无关背景干扰
      3. 动态划分匹配 / 非匹配区域（借助可学习边界）:
         - 相似度 > p → 匹配区域（情感证据一致）
         - 相似度 < p → 非匹配区域（情感对立证据），赋予负相似度

    输出:
      - emotion_visual_features: (B, L, D) 情感加权的视觉特征
      - attention_maps:          (B, H, L, P) 注意力热图
      - region_scores:           (B, L, P) patch 与情感的相似度
      - negative_mask:           (B, L, P) 1 = 情感对立区域
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        boundary_init: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Q/K/V 投影（申请书公式4: α_v = Softmax(Q_v·K_vᵀ/√d_k)·V_v）
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

        # 动态边界（用于划分匹配/非匹配区域）
        self.boundary = DynamicBoundary(init_boundary=boundary_init)

    def forward(
        self,
        visual_patches: torch.Tensor,   # (B, P, visual_dim)
        text_features: torch.Tensor,    # (B, L, text_dim)
    ) -> Dict[str, torch.Tensor]:
        B, P, _ = visual_patches.shape
        L = text_features.shape[1]

        V = self.visual_proj(visual_patches)   # (B, P, D)
        T = self.text_proj(text_features)      # (B, L, D)

        # ---- 引导相似度：每个情感标签对每个 patch 的余弦相似度 ----
        V_norm = F.normalize(V, dim=-1)        # (B, P, D)
        T_norm = F.normalize(T, dim=-1)        # (B, L, D)
        region_scores = torch.einsum("bpd,bld->blp", V_norm, T_norm)  # (B, L, P)

        # ---- 动态划分匹配 / 非匹配区域 ----
        # soft mask 参与前向（可导，边界可学习），hard mask 用于统计
        boundary = self.boundary()
        soft_negative = torch.sigmoid((boundary - region_scores) * 10.0)  # (B, L, P)
        negative_mask = (region_scores < boundary).float()               # 硬标记

        # ---- 多头注意力（文本为 Q，视觉为 K/V）----
        Q = self.q_proj(T)   # (B, L, D)
        K = self.k_proj(V)   # (B, P, D)
        Vv = self.v_proj(V)  # (B, P, D)

        h = self.num_heads
        d = self.hidden_dim // h
        Qh = Q.view(B, L, h, d).transpose(1, 2)    # (B, H, L, d)
        Kh = K.view(B, P, h, d).transpose(1, 2)    # (B, H, P, d)
        Vh = Vv.view(B, P, h, d).transpose(1, 2)   # (B, H, P, d)

        attn_scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(d)  # (B, H, L, P)

        # 对情感对立区域施加负偏置（弱化其贡献，抑制语义干扰）
        neg_bias = soft_negative.unsqueeze(1) * (-2.0)  # (B, 1, L, P)
        attn_scores = attn_scores + neg_bias

        attn_weights = F.softmax(attn_scores, dim=-1)   # (B, H, L, P)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, Vh)       # (B, H, L, d)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.hidden_dim)
        attn_out = self.out_proj(attn_out)

        output = self.norm1(T + attn_out)
        output = self.norm2(output + self.ffn(output))

        return {
            "emotion_visual_features": output,
            "attention_maps": attn_weights,
            "region_scores": region_scores,
            "negative_mask": soft_negative,
            "negative_mask_hard": negative_mask,
            "boundary": boundary,
        }


# ============================================================
# ③ 跨模态冲突感知注意力对齐机制
# ============================================================

class ConflictAwareAlignment(nn.Module):
    """
    跨模态冲突感知注意力对齐机制

    设计（对应申请书「细粒度图文对齐」）:
      1. 计算图文对的情感一致性（视觉全局特征 vs 文本情感特征）
      2. 根据冲突程度动态调整对齐权重:
         - 情感一致 → 强化双向注意力权重，实现特征互补
         - 情感冲突 → 弱化一致性对齐约束，保留两个模态的独立冲突特征，
                      避免错误对齐（这是反讽场景的关键）
      3. 输出对齐后的融合特征

    参照申请书公式:
      α_v = Softmax(Q_v·K_vᵀ/√d_k)·V_v
      α_t = Softmax(Q_t·K_tᵀ/√d_k)·V_t
      F_fusion = α_v · F_v + α_t · F_t
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 模态投影
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # 视觉侧注意力（申请书公式4）
        self.q_v = nn.Linear(hidden_dim, hidden_dim)
        self.k_v = nn.Linear(hidden_dim, hidden_dim)
        self.v_v = nn.Linear(hidden_dim, hidden_dim)

        # 文本侧注意力（申请书公式5）
        self.q_t = nn.Linear(hidden_dim, hidden_dim)
        self.k_t = nn.Linear(hidden_dim, hidden_dim)
        self.v_t = nn.Linear(hidden_dim, hidden_dim)

        # 冲突感知的对齐门控
        # base_weight 由冲突程度显式单调决定（保证「冲突 → 弱化对齐」的语义）
        # modulation 由特征决定，提供可学习的自适应调制
        self.align_bias = nn.Parameter(torch.tensor(0.5))    # 对齐强度偏置
        self.align_temp = nn.Parameter(torch.tensor(4.0))    # 冲突敏感度（温度）
        self.align_modulation = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 融合权重（可学习，平衡视觉/文本贡献）
        self.fusion_alpha = nn.Parameter(torch.tensor(0.0))

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """标准多头注意力（返回加权后的 value）"""
        B, Nq, D = query.shape
        Nk = key.shape[1]
        h = self.num_heads
        d = D // h

        Qh = query.view(B, Nq, h, d).transpose(1, 2)
        Kh = key.view(B, Nk, h, d).transpose(1, 2)
        Vh = value.view(B, Nk, h, d).transpose(1, 2)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(d)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, Vh)
        return out.transpose(1, 2).contiguous().view(B, Nq, D)

    def forward(
        self,
        visual_features: torch.Tensor,   # (B, L, visual_dim) 或 (B, P, visual_dim)
        text_features: torch.Tensor,     # (B, L, text_dim)
        consistency: Optional[torch.Tensor] = None,  # (B, L) 图文一致性分数
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict:
                - aligned_features: (B, L, D)
                - alignment_weights:(B, L) 对齐强度（冲突时降低）
                - cross_conflict:   (B, L) 跨模态冲突程度
        """
        V = self.visual_proj(visual_features)   # (B, Nv, D)
        T = self.text_proj(text_features)       # (B, L, D)
        L = T.shape[1]

        # 若视觉侧是 patch 序列，先聚合到标签维度
        if V.shape[1] != L:
            # 用文本特征作 query 池化视觉特征
            V_pooled = self._attention(T, V, V)  # (B, L, D)
        else:
            V_pooled = V

        # ---- 申请书公式4/5：视觉侧与文本侧注意力 ----
        # 视觉侧：Q=text, K/V=visual → 找到与每个情感相关的视觉证据
        alpha_v = self._attention(self.q_v(T), self.k_v(V_pooled), self.v_v(V_pooled))

        # 文本侧：Q=visual, K/V=text → 用视觉引导筛选情感语义
        alpha_t = self._attention(self.q_t(V_pooled), self.k_t(T), self.v_t(T))

        # ---- 跨模态一致性（冲突程度）----
        if consistency is None:
            # 用余弦相似度衡量图文情感一致性
            consistency = F.cosine_similarity(V_pooled, T, dim=-1)  # (B, L) ∈ [-1,1]

        cross_conflict = (1.0 - consistency) / 2.0  # (B, L) ∈ [0,1]，越大越冲突

        # ---- 冲突感知对齐门控 ----
        # 基础对齐权重：结构上保证「冲突越大 → 对齐权重越低」（单调递减）
        #   w_base = σ(τ · (b - c))，c 为跨模态冲突程度，τ > 0
        base_weight = torch.sigmoid(
            self.align_temp * (self.align_bias - cross_conflict)
        )  # (B, L) ∈ (0,1)

        # 特征相关的可学习调制（0.5 ~ 1.5 倍），提供场景自适应能力
        modulation = 0.5 + self.align_modulation(
            torch.cat([alpha_v, alpha_t], dim=-1)
        ).squeeze(-1)  # (B, L) ∈ (0.5, 1.5)

        align_weight = (base_weight * modulation).clamp(0.0, 1.0)  # (B, L)
        align_weight_3d = align_weight.unsqueeze(-1)               # (B, L, 1)

        # 情感一致 → 权重高（强化融合）；冲突 → 权重低（保留独立冲突特征）
        # 申请书公式13: F_fusion = α_v · F_v + α_t · F_t （用门控动态调节）
        w = torch.sigmoid(self.fusion_alpha)
        fused = align_weight_3d * (w * alpha_v + (1 - w) * alpha_t) + \
                (1 - align_weight_3d) * (0.5 * alpha_v + 0.5 * alpha_t)

        aligned = self.norm(fused)

        return {
            "aligned_features": aligned,
            "alignment_weights": align_weight,              # (B, L)
            "base_alignment_weights": base_weight,          # (B, L) 结构决定的权重
            "cross_conflict": cross_conflict,               # (B, L)
            "alpha_v": alpha_v,
            "alpha_t": alpha_t,
        }


# ============================================================
# 模态内 / 跨模态对比学习与三元组排序损失
# ============================================================

class ConflictContrastiveLoss(nn.Module):
    """
    冲突感知的对比学习损失（CMSIS Eq.15-20）

    三项损失:
      L_t  = Σ[τ_t - S(T,T+) + S(T,T̂+)]₊         文本模态内三元组
      L_v  = Σ[τ_v - S(I,I+) + S(I,Î+)]₊         图像模态内三元组
      L_vt = Σ[γ - S(I,T) + S(I,T̂)]₊
             + [γ - S(I,T) + S(Î,T)]₊            双向跨模态三元组

    正样本构造（CMSIS 做法）: 对特征施加随机 dropout 增强得到 (X, X+)
    负样本: batch 内其他样本，取 hardest negative（相似度最高者）
    """

    def __init__(
        self,
        margin_visual: float = 0.3,
        margin_text: float = 0.3,
        margin_cross: float = 0.2,
        use_hardest_negative: bool = True,
    ):
        super().__init__()
        self.margin_visual = margin_visual
        self.margin_text = margin_text
        self.margin_cross = margin_cross
        self.use_hardest_negative = use_hardest_negative

    @staticmethod
    def _pairwise_similarity(x: torch.Tensor) -> torch.Tensor:
        """批量两两余弦相似度矩阵 (B, B)"""
        x = F.normalize(x, dim=-1)
        return torch.matmul(x, x.t())

    def _intra_modal_triplet(
        self,
        features: torch.Tensor,     # (B, D) 模态特征
        augmented: torch.Tensor,    # (B, D) dropout 增强版本
        margin: float,
    ) -> torch.Tensor:
        """模态内三元组排序损失（CMSIS Eq.15/17）"""
        B = features.shape[0]
        if B < 2:
            return features.new_zeros(())

        sim_pos = F.cosine_similarity(features, augmented, dim=-1)  # (B,) S(X, X+)
        sim_matrix = self._pairwise_similarity(features)            # (B, B)

        # hardest negative：排除自身，取最相似的负样本
        eye = torch.eye(B, dtype=torch.bool, device=features.device)
        if self.use_hardest_negative:
            sim_neg = sim_matrix.masked_fill(eye, -float("inf")).max(dim=-1).values
        else:
            sim_neg = sim_matrix.masked_fill(eye, 0.0).mean(dim=-1)

        loss = F.relu(margin - sim_pos + sim_neg)
        return loss.mean()

    def _cross_modal_triplet(
        self,
        visual: torch.Tensor,   # (B, D)
        text: torch.Tensor,     # (B, D)
        margin: float,
    ) -> torch.Tensor:
        """双向跨模态三元组排序损失（CMSIS Eq.19）"""
        B = visual.shape[0]
        if B < 2:
            return visual.new_zeros(())

        sim_matrix = torch.matmul(
            F.normalize(visual, dim=-1), F.normalize(text, dim=-1).t()
        )  # (B, B)，对角为匹配对
        sim_pos = sim_matrix.diag()  # (B,) S(I, T)

        eye = torch.eye(B, dtype=torch.bool, device=visual.device)
        # 图像→文本方向：S(I, T̂) 取该图像最难的非匹配文本
        sim_neg_vt = sim_matrix.masked_fill(eye, -float("inf")).max(dim=-1).values
        # 文本→图像方向：S(Î, T) 取该文本最难的非匹配图像
        sim_neg_tv = sim_matrix.masked_fill(eye, -float("inf")).max(dim=0).values

        loss_vt = F.relu(margin - sim_pos + sim_neg_vt)
        loss_tv = F.relu(margin - sim_pos + sim_neg_tv)
        return (loss_vt + loss_tv).mean()

    def forward(
        self,
        text_features: torch.Tensor,       # (B, L, D)
        visual_features: torch.Tensor,     # (B, L, D)
        text_augmented: Optional[torch.Tensor] = None,
        visual_augmented: Optional[torch.Tensor] = None,
        return_details: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        计算冲突对比损失

        Args:
            text_features: 文本情绪特征 (B, L, D)
            visual_features: 情感加权视觉特征 (B, L, D)
            text_augmented / visual_augmented: dropout 增强副本（正样本）；
                为空时使用 dropout 在线生成

        Returns:
            dict: {loss_text, loss_visual, loss_cross, total}
        """
        # 池化为样本级表示（对标签维度取平均）
        t_sample = text_features.mean(dim=1)   # (B, D)
        v_sample = visual_features.mean(dim=1)  # (B, D)

        # 正样本：若未提供，用 dropout 增强模拟（CMSIS 随机 dropout 做区域增强）
        if text_augmented is None:
            text_augmented = F.dropout(t_sample, p=0.1, training=self.training)
        else:
            text_augmented = text_augmented.mean(dim=1)
        if visual_augmented is None:
            visual_augmented = F.dropout(v_sample, p=0.1, training=self.training)
        else:
            visual_augmented = visual_augmented.mean(dim=1)

        loss_text = self._intra_modal_triplet(t_sample, text_augmented, self.margin_text)
        loss_visual = self._intra_modal_triplet(v_sample, visual_augmented, self.margin_visual)
        loss_cross = self._cross_modal_triplet(v_sample, t_sample, self.margin_cross)

        total = loss_text + loss_visual + loss_cross

        return {
            "loss_text": loss_text,
            "loss_visual": loss_visual,
            "loss_cross": loss_cross,
            "total": total,
        }


# ============================================================
# 整合模块：注意力引导的冲突感知跨模态融合
# ============================================================

class ConflictAwareFusionModule(nn.Module):
    """
    注意力引导的冲突感知跨模态融合模块（完整实现）

    整合三个子模块，形成「文本端冲突提纯 → 视觉端冲突捕捉 → 跨模态对齐」的
    完整流程，输出冲突感知的多模态融合特征。

    与层次化注意力融合 (HierarchicalAttentionFusion) 的区别:
      - 层次化注意力融合：局部→全局渐进融合，无显式冲突建模
      - 本模块：显式建模图文情感冲突（反讽/极性对立场景），
                引入模态内对比学习与三元组排序损失

    输入输出接口与层次化注意力融合保持一致，便于在模型中切换与消融对比。
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_output_dim: int = 512,
        text_boundary_init: float = 0.3,
        visual_boundary_init: float = 0.0,
        use_contrastive_loss: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_output_dim = fusion_output_dim
        self.use_contrastive_loss = use_contrastive_loss

        # ① 文本情绪注意力（双路径冲突注意力）
        self.text_attention = TextConflictAttention(
            text_dim=text_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, dropout=dropout,
            boundary_init=text_boundary_init,
        )

        # ② 视觉情绪注意力（文本引导 + 冲突捕捉）
        self.visual_attention = VisualConflictAttention(
            visual_dim=visual_dim, text_dim=hidden_dim,
            hidden_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, boundary_init=visual_boundary_init,
        )

        # ③ 跨模态冲突感知对齐
        self.cross_alignment = ConflictAwareAlignment(
            visual_dim=hidden_dim, text_dim=hidden_dim,
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
        )

        # 局部融合（视觉 + 文本 → 统一表示）
        self.local_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, fusion_output_dim),
            nn.LayerNorm(fusion_output_dim),
        )

        # 对比学习损失
        if use_contrastive_loss:
            self.contrastive_loss = ConflictContrastiveLoss()

    def forward(
        self,
        visual_patches: torch.Tensor,                     # (B, P, visual_dim)
        visual_global: torch.Tensor = None,               # (B, visual_dim)（可选）
        text_features: torch.Tensor = None,               # (B, L, text_dim)
        text_token_features: Optional[torch.Tensor] = None,  # (B, M, text_dim)
        emotion_word_mask: Optional[torch.Tensor] = None,    # (B, M)
    ) -> Dict[str, torch.Tensor]:
        """
        冲突感知融合前向传播

        Args:
            visual_patches: ViT patch 特征 (B, P, 768)
            visual_global: 视觉全局特征 (B, 512)，兼容通用接口
            text_features: 文本情感特征 (B, L, 512)
            text_token_features: 可选 token 级文本特征
            emotion_word_mask: 可选情感词掩码

        Returns:
            dict:
                - fused_features:    (B, L, fusion_dim) 冲突感知融合特征
                - attention_maps:    (B, H, L, P) 视觉注意力图
                - conflict_scores:   (B, L) 文本冲突强度
                - cross_conflict:    (B, L) 跨模态冲突程度
                - alignment_weights: (B, L) 对齐强度
                - text_features:     (B, L, D) 文本情绪特征（供对比损失使用）
                - visual_features:   (B, L, D) 视觉情绪特征（供对比损失使用）
        """
        if text_features is None:
            raise ValueError("ConflictAwareFusionModule 需要 text_features 输入")

        # ---- ① 文本端冲突提纯 ----
        text_out = self.text_attention(
            text_features,
            text_token_features=text_token_features,
            emotion_word_mask=emotion_word_mask,
        )
        text_feat = text_out["text_features"]           # (B, L, D)

        # ---- ② 视觉端冲突捕捉（以文本情感特征为引导）----
        visual_out = self.visual_attention(
            visual_patches=visual_patches,
            text_features=text_feat,
        )
        visual_feat = visual_out["emotion_visual_features"]  # (B, L, D)

        # ---- ③ 跨模态冲突感知对齐 ----
        align_out = self.cross_alignment(
            visual_features=visual_feat,
            text_features=text_feat,
        )
        aligned = align_out["aligned_features"]         # (B, L, D)

        # ---- 局部融合 ----
        concat = torch.cat([visual_feat, text_feat], dim=-1)   # (B, L, 2D)
        local_fused = self.local_fusion(concat)                # (B, L, D)

        # 对齐特征与局部融合特征残差相加（保留冲突信息）
        combined = local_fused + aligned
        fused_features = self.output_proj(combined)            # (B, L, fusion_dim)

        result = {
            "fused_features": fused_features,
            "attention_maps": visual_out["attention_maps"],
            "conflict_scores": text_out["conflict_scores"],
            "segment_mask": text_out["segment_mask"],
            "cross_conflict": align_out["cross_conflict"],
            "alignment_weights": align_out["alignment_weights"],
            "region_scores": visual_out["region_scores"],
            "negative_mask": visual_out["negative_mask"],
            "text_features": text_feat,
            "visual_features": visual_feat,
            "literal_features": text_out["literal_features"],
            "intent_features": text_out["intent_features"],
        }

        # 对比损失（训练时使用）
        if self.use_contrastive_loss and self.training:
            contrastive = self.contrastive_loss(
                text_features=text_feat,
                visual_features=visual_feat,
            )
            result["contrastive_loss"] = contrastive["total"]
            result["contrastive_details"] = contrastive

        return result

    def compute_conflict_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算冲突相关损失（供训练脚本调用）

        返回:
            dict: {loss_text, loss_visual, loss_cross, total}
        """
        if not self.use_contrastive_loss:
            zero = outputs["fused_features"].new_zeros(())
            return {"loss_text": zero, "loss_visual": zero, "loss_cross": zero, "total": zero}

        return self.contrastive_loss(
            text_features=outputs["text_features"],
            visual_features=outputs["visual_features"],
        )


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("注意力引导的冲突感知跨模态融合模块 — 测试")
    print("=" * 60)

    B, P, L = 4, 49, 8
    visual_dim, text_dim, hidden_dim = 768, 512, 512

    visual_patches = torch.randn(B, P, visual_dim)
    visual_global = torch.randn(B, text_dim)
    text_features = torch.randn(B, L, text_dim)
    text_tokens = torch.randn(B, 32, text_dim)
    word_mask = (torch.rand(B, 32) > 0.7).float()

    # 1. 文本端双路径冲突注意力
    print("\n[1] 文本情绪注意力（双路径冲突注意力）")
    tca = TextConflictAttention(text_dim=text_dim, hidden_dim=hidden_dim)
    tca.train()  # 训练模式以启用手册中的 dropout 增强
    tca_out = tca(text_features, text_token_features=text_tokens, emotion_word_mask=word_mask)
    print(f"  文本特征:     {tuple(tca_out['text_features'].shape)}")
    print(f"  冲突分数:     {tuple(tca_out['conflict_scores'].shape)}, "
          f"范围 [{tca_out['conflict_scores'].min():.3f}, {tca_out['conflict_scores'].max():.3f}]")
    print(f"  动态边界 p:   {tca_out['boundary'].item():.4f}")
    print(f"  冲突片段比例: {tca_out['segment_mask_hard'].mean().item():.3f}")

    # 2. 视觉端冲突捕捉
    print("\n[2] 视觉情绪注意力（冲突区域捕捉）")
    vca = VisualConflictAttention(visual_dim=visual_dim, text_dim=hidden_dim)
    vca.train()
    vca_out = vca(visual_patches, tca_out["text_features"])
    print(f"  视觉特征:   {tuple(vca_out['emotion_visual_features'].shape)}")
    print(f"  注意力图:   {tuple(vca_out['attention_maps'].shape)}")
    print(f"  动态边界 p: {vca_out['boundary'].item():.4f}")
    print(f"  对立区域比例: {vca_out['negative_mask_hard'].mean().item():.3f}")

    # 3. 跨模态冲突感知对齐
    print("\n[3] 跨模态冲突感知对齐")
    caa = ConflictAwareAlignment(visual_dim=hidden_dim, text_dim=hidden_dim)
    caa_out = caa(vca_out["emotion_visual_features"], tca_out["text_features"])
    print(f"  对齐特征:   {tuple(caa_out['aligned_features'].shape)}")
    print(f"  对齐权重:   {tuple(caa_out['alignment_weights'].shape)}, "
          f"均值 {caa_out['alignment_weights'].mean():.3f}")
    print(f"  跨模态冲突: 均值 {caa_out['cross_conflict'].mean():.3f}")

    # 4. 对比学习损失
    print("\n[4] 冲突对比学习损失")
    loss_fn = ConflictContrastiveLoss()
    loss_fn.train()  # 启用 dropout 正样本增强
    loss_out = loss_fn(
        text_features=tca_out["text_features"],
        visual_features=vca_out["emotion_visual_features"],
    )
    for k, v in loss_out.items():
        print(f"  {k}: {v.item():.4f}")

    # 5. 完整融合模块
    print("\n[5] 完整冲突感知融合模块")
    fusion = ConflictAwareFusionModule(
        visual_dim=visual_dim, text_dim=text_dim, hidden_dim=hidden_dim,
    )
    fusion.train()
    out = fusion(visual_patches, visual_global, text_features,
                 text_token_features=text_tokens, emotion_word_mask=word_mask)
    print(f"  融合特征:   {tuple(out['fused_features'].shape)}")
    print(f"  注意力图:   {tuple(out['attention_maps'].shape)}")
    print(f"  对比损失:   {out['contrastive_loss'].item():.4f}")

    # 6. 梯度流验证
    print("\n[6] 梯度反向传播验证")
    loss = out["fused_features"].sum() + out["contrastive_loss"]
    loss.backward()
    grad_params = sum(1 for _, p in fusion.named_parameters()
                      if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for _ in fusion.parameters())
    print(f"  有梯度的参数: {grad_params}/{total_params}")

    print("\n✅ 冲突感知跨模态融合模块测试通过！")
