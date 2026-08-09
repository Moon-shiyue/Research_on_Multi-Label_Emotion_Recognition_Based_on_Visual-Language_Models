"""
跨模态层次化注意力融合模块
Hierarchical Cross-Modal Attention Fusion Module

实现「局部高权重特征融合 → 全局对齐特征校准」两步融合策略：

第一步 — 局部高权重特征融合 (Local High-Weight Fusion):
  ├── 视觉情绪注意力 (Visual Emotion Attention):
  │     文本特征引导的视觉 patch 注意力，定位图像中情绪相关的关键区域
  │     输出: 情绪加权的局部视觉特征
  ├── 文本情绪注意力 (Text Emotion Attention):
  │     视觉特征引导的文本注意力，筛选与图像内容匹配的情感语义
  │     输出: 视觉相关的文本情感特征
  └── 局部融合:
        将视觉和文本的高权重特征进行交叉融合

第二步 — 全局对齐特征校准 (Global Alignment Calibration):
  ├── 跨模态双向注意力 (Bi-directional Cross-Attention):
  │     视觉→文本、文本→视觉的交互建模
  ├── 全局自注意力校准 (Global Self-Attention):
  │     在融合空间中进一步对齐多模态特征
  └── 残差连接 + LayerNorm → 稳定训练

核心创新点：
  1. 层次化设计：先局部聚焦再全局对齐，模拟人类「先看重点→再整体理解」的认知过程
  2. 双向注意力引导：视觉和文本互相作为查询，实现真正的跨模态交互
  3. 残差校准机制：每一步融合都保留原始信息流，防止特征漂移
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple


# ============================================================
# 基础组件
# ============================================================

class MultiHeadCrossAttention(nn.Module):
    """
    多头交叉注意力模块

    支持两种模式:
    - Q 来自模态 A，K/V 来自模态 B（标准交叉注意力）
    - 可选的门控机制，控制跨模态信息流强度
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_gating: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_gating = use_gating

        # Q, K, V 投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # 门控机制
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid(),
            )

        self.dropout = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,       # (B, N_q, dim)
        key_value: torch.Tensor,   # (B, N_kv, dim)
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N_q, _ = query.shape
        N_kv = key_value.shape[1]

        # 线性投影
        Q = self.q_proj(query)      # (B, N_q, dim)
        K = self.k_proj(key_value)  # (B, N_kv, dim)
        V = self.v_proj(key_value)  # (B, N_kv, dim)

        # 重塑为多头格式
        Q = Q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)    # (B, H, N_q, d)
        K = K.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, N_kv, d)
        V = V.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, N_kv, d)

        # 注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, N_q, N_kv)

        if key_padding_mask is not None:
            # key_padding_mask: (B, N_kv) → (B, 1, 1, N_kv)
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, N_q, N_kv)
        attn_weights = self.dropout_attn(attn_weights)

        # 加权聚合
        attn_output = torch.matmul(attn_weights, V)  # (B, H, N_q, d)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_q, self.dim)

        # 输出投影
        output = self.out_proj(attn_output)
        output = self.dropout(output)

        # 门控机制：自适应控制跨模态信息融合强度
        if self.use_gating:
            gate_input = torch.cat([query, output], dim=-1)  # (B, N_q, 2*dim)
            gate_value = self.gate(gate_input)               # (B, N_q, dim)
            output = gate_value * output + (1 - gate_value) * query

        if return_attention:
            return output, attn_weights
        return output, None


class FeedForwardNetwork(nn.Module):
    """Transformer 标准前馈网络"""

    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 层次化注意力融合核心模块
# ============================================================

class VisualEmotionAttention(nn.Module):
    """
    视觉情绪注意力子模块 (Visual Emotion Attention)

    功能：
      以文本情感特征为查询 (Query)，对视觉 patch 特征进行注意力加权，
      自动定位图像中与特定情感最相关的区域。

    设计理念：
      不同情感往往关联图像中不同区域：
      - "joy" 通常关联面部微笑区域
      - "fear" 通常关联暗部或威胁性物体
      - "awe" 通常关联广阔景观的上半部分
      通过文本引导的视觉注意力，模型学会聚焦最相关的区域。

    输入:
        - visual_patches: (B, num_patches, dim_vit)  视觉 patch 特征
        - text_features: (B, num_labels, dim_text)   文本情感特征

    输出:
        - emotion_visual_features: (B, num_labels, dim_out)  情感加权的视觉特征
        - attention_maps: (B, num_labels, num_patches)       注意力热图（可解释性）
    """

    def __init__(
        self,
        visual_dim: int = 768,       # ViT-B/32 patch 特征维度
        text_dim: int = 512,         # 文本特征维度
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 将视觉 patch 特征和文本特征投影到统一空间
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # 文本→视觉的交叉注意力
        # Q=text(情感), K/V=visual(图像区域) → 找到每个情感对应的图像区域
        self.cross_attn = MultiHeadCrossAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_gating=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim * 4, dropout)

    def forward(
        self,
        visual_patches: torch.Tensor,  # (B, num_patches, visual_dim)
        text_features: torch.Tensor,   # (B, num_labels, text_dim)
    ) -> Dict[str, torch.Tensor]:
        B = visual_patches.shape[0]
        num_patches = visual_patches.shape[1]
        num_labels = text_features.shape[1]

        # 投影到统一空间
        V = self.visual_proj(visual_patches)   # (B, num_patches, hidden_dim)
        T = self.text_proj(text_features)       # (B, num_labels, hidden_dim)

        # 文本引导的视觉注意力：Q=text, K/V=visual
        # 对于每个情感标签，在图像中找到最相关的区域
        attended_visual, attn_weights = self.cross_attn(
            query=T,          # (B, num_labels, hidden_dim)
            key_value=V,      # (B, num_patches, hidden_dim)
        )  # → (B, num_labels, hidden_dim)

        # 残差连接 + FFN
        output = self.norm1(T + attended_visual)
        output = self.norm2(output + self.ffn(output))  # (B, num_labels, hidden_dim)

        return {
            "emotion_visual_features": output,
            "attention_maps": attn_weights,  # (B, H, num_labels, num_patches)
        }


class TextEmotionAttention(nn.Module):
    """
    文本情绪注意力子模块 (Text Emotion Attention)

    功能：
      以视觉全局特征为查询，对多个情感标签的文本特征进行注意力加权，
      筛选出与当前图像内容最匹配的情感语义。

    设计理念：
      对于给定的图像，不同情感标签的相关性不同：
      - 一张日落图片与 "peace", "awe", "contentment" 更相关
      - 一张车祸图片与 "fear", "sadness" 更相关
      通过视觉引导的文本注意力，模型学会筛选最相关的情绪描述。

    输入:
        - text_features: (B, num_labels, dim_text)    文本情感特征
        - visual_global: (B, dim_visual)              视觉全局特征

    输出:
        - weighted_text_features: (B, num_labels, dim_out)  视觉加权的文本特征
    """

    def __init__(
        self,
        text_dim: int = 512,
        visual_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)

        # 视觉→文本的交叉注意力
        # Q=visual(图像), K/V=text(情感标签) → 找到与图像最匹配的情感
        self.cross_attn = MultiHeadCrossAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_gating=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim * 4, dropout)

    def forward(
        self,
        text_features: torch.Tensor,   # (B, num_labels, text_dim)
        visual_global: torch.Tensor,   # (B, visual_dim)
    ) -> Dict[str, torch.Tensor]:
        B = text_features.shape[0]
        num_labels = text_features.shape[1]

        T = self.text_proj(text_features)  # (B, num_labels, hidden_dim)

        # 将视觉全局特征扩展为序列形式
        V = self.visual_proj(visual_global).unsqueeze(1)  # (B, 1, hidden_dim)

        # 视觉引导的文本注意力：Q=visual, K/V=text
        attended_text, attn_weights = self.cross_attn(
            query=V,          # (B, 1, hidden_dim)
            key_value=T,      # (B, num_labels, hidden_dim)
        )  # → (B, 1, hidden_dim)

        # 扩展回标签维度
        attended_text = attended_text.expand(-1, num_labels, -1)  # (B, num_labels, hidden_dim)

        output = self.norm1(T + attended_text)
        output = self.norm2(output + self.ffn(output))  # (B, num_labels, hidden_dim)

        return {
            "weighted_text_features": output,
            "text_attention_weights": attn_weights,  # (B, H, 1, num_labels)
        }


class GlobalAlignmentCalibration(nn.Module):
    """
    全局对齐特征校准模块

    功能：
      在局部融合之后，通过双向跨模态注意力实现全局特征对齐，
      使用自注意力建模标签间的全局依赖关系。

    设计理念：
      局部融合确定了「哪些区域和哪些情感相关」，
      全局校准进一步解决「这些情感之间如何协调共存」的问题。
      例如：joy + surprise → 可能是 "excitement"
            sadness + fear → 可能是 "despair"

    输入:
        - fused_features: (B, num_labels, hidden_dim)  局部融合后的特征

    输出:
        - calibrated_features: (B, num_labels, hidden_dim)  全局校准后的特征
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_layers = num_layers

        # 多层 Transformer 编码器（自注意力），建模标签间全局依赖
        self.self_attn_layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                "norm1": nn.LayerNorm(hidden_dim),
                "ffn": FeedForwardNetwork(hidden_dim, hidden_dim * 4, dropout),
                "norm2": nn.LayerNorm(hidden_dim),
            })
            for _ in range(num_layers)
        ])

        # 全局上下文压缩：将 num_labels 个特征聚合成一个全局表示
        self.global_context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        fused_features: torch.Tensor,  # (B, num_labels, hidden_dim)
    ) -> torch.Tensor:
        x = fused_features

        # 多层自注意力 → 建模标签间全局依赖
        for layer in self.self_attn_layers:
            residual = x
            attn_out, _ = layer["self_attn"](x, x, x)
            x = layer["norm1"](residual + attn_out)
            x = layer["norm2"](x + layer["ffn"](x))

        # 全局上下文聚合
        # 对所有标签特征取平均，然后通过 MLP 生成全局校准向量
        global_context = x.mean(dim=1)  # (B, hidden_dim)
        global_context = self.global_context_proj(global_context)  # (B, hidden_dim)

        # 全局校准：将全局上下文融入每个标签特征
        calibrated = self.norm_out(x + global_context.unsqueeze(1))

        return calibrated


class HierarchicalAttentionFusion(nn.Module):
    """
    层次化注意力融合模块（完整实现）

    两步融合策略：

    Step 1 — 局部高权重特征融合:
        并行执行视觉情绪注意力和文本情绪注意力，分别从两个模态中
        提取情绪相关的高权重特征，然后进行局部交叉融合。

    Step 2 — 全局对齐特征校准:
        对局部融合后的特征进行全局自注意力建模，捕捉标签间关系，
        并通过全局上下文进行特征校准，消除模态间偏移。

    输入:
        - visual_patches:  (B, num_patches, visual_dim)  — ViT patch 特征
        - visual_global:   (B, visual_dim)               — 视觉全局特征
        - text_features:   (B, num_labels, text_dim)      — 文本情感特征

    输出:
        - fused_features:  (B, num_labels, fusion_dim)   — 融合后的多模态特征
        - attention_maps:  (B, H, num_labels, num_patches) — 视觉注意力图（可用于可视化）
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 2,
        fusion_output_dim: int = 512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ---- Step 1: 局部高权重特征融合 ----

        # 1a. 视觉情绪注意力：文本→视觉，定位情绪相关图像区域
        self.visual_emotion_attn = VisualEmotionAttention(
            visual_dim=visual_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 1b. 文本情绪注意力：视觉→文本，筛选图像相关情感语义
        self.text_emotion_attn = TextEmotionAttention(
            text_dim=text_dim,
            visual_dim=visual_dim if visual_dim == text_dim else hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 1c. 局部融合层：将视觉和文本的高权重特征进行融合
        self.local_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ---- Step 2: 全局对齐特征校准 ----
        self.global_calibration = GlobalAlignmentCalibration(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers,
        )

        # ---- 输出投影 ----
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, fusion_output_dim),
            nn.LayerNorm(fusion_output_dim),
        )

        self.fusion_output_dim = fusion_output_dim

    def forward(
        self,
        visual_patches: torch.Tensor,    # (B, num_patches, visual_dim)
        visual_global: torch.Tensor,     # (B, visual_dim)
        text_features: torch.Tensor,     # (B, num_labels, text_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        层次化注意力融合前向传播

        Args:
            visual_patches: ViT patch 特征 (B, P, 768)
            visual_global: 视觉全局特征 (B, 512)
            text_features: 文本情感特征 (B, L, 512)

        Returns:
            dict:
                - fused_features: (B, L, fusion_dim) 融合特征
                - attention_maps: (B, H, L, P) 视觉注意力图
                - local_fused: (B, L, hidden_dim) 局部融合特征（中间表示）
                - calibrated: (B, L, hidden_dim) 全局校准特征（中间表示）
        """
        # ============================================
        # Step 1: 局部高权重特征融合
        # ============================================

        # 1a. 视觉情绪注意力
        visual_emotion_out = self.visual_emotion_attn(
            visual_patches=visual_patches,
            text_features=text_features,
        )
        emotion_visual = visual_emotion_out["emotion_visual_features"]  # (B, L, hidden_dim)
        attention_maps = visual_emotion_out["attention_maps"]

        # 1b. 文本情绪注意力
        text_emotion_out = self.text_emotion_attn(
            text_features=text_features,
            visual_global=visual_global,
        )
        weighted_text = text_emotion_out["weighted_text_features"]  # (B, L, hidden_dim)

        # 1c. 局部融合
        concat_features = torch.cat([emotion_visual, weighted_text], dim=-1)  # (B, L, 2*hidden_dim)
        local_fused = self.local_fusion(concat_features)  # (B, L, hidden_dim)

        # ============================================
        # Step 2: 全局对齐特征校准
        # ============================================
        calibrated = self.global_calibration(local_fused)  # (B, L, hidden_dim)

        # ============================================
        # 输出投影
        # ============================================
        fused_features = self.output_proj(calibrated)  # (B, L, fusion_dim)

        return {
            "fused_features": fused_features,
            "attention_maps": attention_maps,
            "local_fused": local_fused,
            "calibrated": calibrated,
        }


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("层次化注意力融合模块测试")
    print("=" * 60)

    # 测试参数
    B = 2           # batch size
    P = 49          # ViT patches (7×7)
    L = 12          # emotion labels
    visual_dim = 768
    text_dim = 512
    hidden_dim = 512

    # 模拟输入
    visual_patches = torch.randn(B, P, visual_dim)
    visual_global = torch.randn(B, text_dim)
    text_features = torch.randn(B, L, text_dim)

    # 初始化模块
    fusion = HierarchicalAttentionFusion(
        visual_dim=visual_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        num_heads=8,
        dropout=0.1,
        num_layers=2,
        fusion_output_dim=512,
    )

    # 前向传播
    with torch.no_grad():
        output = fusion(visual_patches, visual_global, text_features)

    print(f"\n输入:")
    print(f"  visual_patches:  {visual_patches.shape}")
    print(f"  visual_global:   {visual_global.shape}")
    print(f"  text_features:   {text_features.shape}")

    print(f"\n输出:")
    print(f"  fused_features:  {output['fused_features'].shape}")
    print(f"  attention_maps:  {output['attention_maps'].shape}")
    print(f"  local_fused:     {output['local_fused'].shape}")
    print(f"  calibrated:      {output['calibrated'].shape}")

    # 验证维度
    assert output["fused_features"].shape == (B, L, 512), "融合特征维度错误！"
    assert output["attention_maps"].shape[2:] == (L, P), "注意力图形状错误！"
    assert output["local_fused"].shape == (B, L, hidden_dim), "局部融合维度错误！"
    assert output["calibrated"].shape == (B, L, hidden_dim), "校准特征维度错误！"

    # 统计参数量
    total_params = sum(p.numel() for p in fusion.parameters())
    trainable_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    print(f"\n参数量: {total_params:,} (可训练: {trainable_params:,})")
    print("✅ 层次化注意力融合模块测试通过！")
