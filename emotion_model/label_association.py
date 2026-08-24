"""
标签关联建模模块
Label Association Modeling Module

建模情感标签之间的结构关系（共现、互斥、层级），将先验知识融入模型推理。

两种互补的建模方式：

1. 标签注意力机制 (Label Attention)
   - 将情感标签嵌入作为可学习的语义向量
   - 通过自注意力自动发现标签间的关系
   - 输出: 标签关系感知的特征调制向量

2. 轻量图卷积标签关联网络 (Lightweight GCN)
   - 基于情感先验知识构建标签关系图
   - 共现关系 → 正边 (鼓励同时激活)
   - 互斥关系 → 负边 (抑制同时激活)
   - 输出: 图结构增强的标签表示

核心创新点：
  - 将心理学先验（Ekman 情感理论）编码为图结构
  - 可学习的标签关系 + 先验知识注入的双重机制
  - 轻量级设计：GCN 仅 2 层，参数少且不影响推理速度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math


# ============================================================
# 标签关系图构建
# ============================================================

def build_emotion_label_graph(
    emotion_labels: List[str],
    cooccurrence: Dict[Tuple[str, str], float] = None,
    mutual_exclusion: Dict[Tuple[str, str], float] = None,
) -> torch.Tensor:
    """
    根据情感先验知识构建标签关系邻接矩阵

    Args:
        emotion_labels: 情感标签列表
        cooccurrence: 共现关系字典 {(e1, e2): strength}
        mutual_exclusion: 互斥关系字典 {(e1, e2): strength}

    Returns:
        adj_matrix: (num_labels, num_labels) 邻接矩阵
            - 正值为共现关系
            - 负值为互斥关系
            - 对角线为 1.0（自连接）
    """
    num_labels = len(emotion_labels)
    adj_matrix = torch.eye(num_labels)  # 自连接

    label_to_idx = {label: i for i, label in enumerate(emotion_labels)}

    # 添加共现关系（正边）
    if cooccurrence:
        for (e1, e2), strength in cooccurrence.items():
            if e1 in label_to_idx and e2 in label_to_idx:
                i, j = label_to_idx[e1], label_to_idx[e2]
                adj_matrix[i, j] = strength
                adj_matrix[j, i] = strength  # 对称

    # 添加互斥关系（负边）
    if mutual_exclusion:
        for (e1, e2), strength in mutual_exclusion.items():
            if e1 in label_to_idx and e2 in label_to_idx:
                i, j = label_to_idx[e1], label_to_idx[e2]
                adj_matrix[i, j] = -strength
                adj_matrix[j, i] = -strength  # 对称

    return adj_matrix


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    对称归一化邻接矩阵: D^{-1/2} A D^{-1/2}

    这是 GCN 的标准归一化方法，确保图卷积的数值稳定性。
    对于包含负值的邻接矩阵，使用绝对值计算度矩阵。
    """
    # 使用绝对值计算度矩阵（处理负边）
    deg = torch.sum(torch.abs(adj), dim=1)
    deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
    deg_inv_sqrt = torch.diag(deg_inv_sqrt)
    adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt
    return adj_norm


# ============================================================
# 标签注意力机制
# ============================================================

class LabelAttentionModule(nn.Module):
    """
    标签注意力机制

    通过可学习的标签嵌入 + 自注意力，自动发现和建模情感标签之间的语义关系。
    与 GCN 的互补之处在于：
    - GCN 依赖先验知识（手工定义的图结构）
    - LabelAttention 从数据中学习关系（数据驱动）

    两者结合实现了「先验引导 + 数据驱动」的标签关系建模。
    """

    def __init__(
        self,
        num_labels: int,
        feature_dim: int = 512,
        label_embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.feature_dim = feature_dim

        # 可学习的标签语义嵌入
        self.label_embeddings = nn.Parameter(
            torch.randn(num_labels, label_embed_dim) * 0.02
        )

        # 标签嵌入 → 特征空间的投影
        self.label_proj = nn.Sequential(
            nn.Linear(label_embed_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        # 自注意力：建模标签间关系
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 关系感知的特征调制
        self.relation_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid(),
        )

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        fused_features: torch.Tensor,  # (B, num_labels, feature_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused_features: 融合后的特征

        Returns:
            dict:
                - modulated_features: 标签关系调制后的特征
                - label_relations: 学习到的标签关系矩阵
        """
        B = fused_features.shape[0]

        # 获取标签语义嵌入
        label_emb = self.label_proj(self.label_embeddings)  # (num_labels, feature_dim)

        # 自注意力建模标签关系
        label_seq = label_emb.unsqueeze(0).expand(B, -1, -1)  # (B, num_labels, feature_dim)
        attn_out, attn_weights = self.self_attn(label_seq, label_seq, label_seq)
        label_context = self.norm1(label_seq + self.dropout(attn_out))  # (B, num_labels, D)

        # 关系感知门控：决定多少标签关系信息融入原始特征
        gate_input = torch.cat([fused_features, label_context], dim=-1)  # (B, num_labels, 2D)
        gate = self.relation_gate(gate_input)  # (B, num_labels, D)

        # 门控融合
        modulated = self.norm2(
            gate * label_context + (1 - gate) * fused_features
        )

        return {
            "modulated_features": modulated,
            "label_relations": attn_weights,  # 标签间注意力权重
        }


# ============================================================
# 轻量图卷积标签关联网络
# ============================================================

class GraphConvolution(nn.Module):
    """
    单层图卷积

    支持有符号邻接矩阵（正边=共现，负边=互斥），
    分别进行信息传递后融合。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # 正边权重（共现关系的信息传递）
        self.weight_pos = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # 负边权重（互斥关系的抑制信号）
        self.weight_neg = nn.Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_pos)
        nn.init.xavier_uniform_(self.weight_neg)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (num_labels, in_features) 节点特征
            adj: (num_labels, num_labels) 有符号邻接矩阵

        Returns:
            output: (num_labels, out_features)
        """
        # 分离正边和负边
        adj_pos = torch.clamp(adj, min=0)   # 仅保留正边（共现）
        adj_neg = torch.clamp(adj, max=0)   # 仅保留负边（互斥）

        # 正边信息传递（共现标签特征聚合）
        support_pos = torch.mm(x, self.weight_pos)
        output_pos = torch.mm(adj_pos, support_pos)

        # 负边信息传递（互斥标签特征抑制）
        support_neg = torch.mm(x, self.weight_neg)
        output_neg = torch.mm(adj_neg, support_neg)  # 负值 adj * 特征 → 差分信号

        # 融合
        output = output_pos + output_neg

        if self.bias is not None:
            output = output + self.bias

        return output


class LightweightLabelGCN(nn.Module):
    """
    轻量图卷积标签关联网络

    2 层 GCN，在标签关系图上进行消息传递，
    将情感标签的先验关系（共现/互斥）编码到标签表示中。

    设计要点：
    - 仅 2 层，避免过平滑 (over-smoothing)
    - 有符号邻接矩阵：区分共现（正向）和互斥（负向）关系
    - 残差连接：保留原始标签特征，防止先验知识过度主导
    """

    def __init__(
        self,
        num_labels: int,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        emotion_labels: List[str] = None,
        cooccurrence: Dict = None,
        mutual_exclusion: Dict = None,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.output_dim = output_dim

        # 构建标签关系图（基于先验知识）
        if emotion_labels is not None:
            adj = build_emotion_label_graph(
                emotion_labels, cooccurrence, mutual_exclusion
            )
        else:
            # 无先验知识时初始化为单位矩阵（仅自连接）
            adj = torch.eye(num_labels)

        self.register_buffer("adj_matrix", normalize_adjacency(adj))

        # GCN 层
        self.gcn_layers = nn.ModuleList()
        in_dim = input_dim
        for i in range(num_layers - 1):
            self.gcn_layers.append(GraphConvolution(in_dim, hidden_dim))
            in_dim = hidden_dim

        # 最后一层输出
        self.gcn_layers.append(GraphConvolution(hidden_dim, output_dim))

        self.dropout = nn.Dropout(dropout)
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])

        # 残差投影（将输入映射到输出维度）
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = nn.Identity()

        # 中间层残差投影（处理 hidden_dim 与输入维度不一致的情况）
        self.intermediate_residual_projs = nn.ModuleList()
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            if in_dim != out_dim:
                self.intermediate_residual_projs.append(nn.Linear(in_dim, out_dim))
            else:
                self.intermediate_residual_projs.append(nn.Identity())
            in_dim = out_dim

    def forward(
        self,
        fused_features: torch.Tensor,  # (B, num_labels, feature_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused_features: 融合后的特征 (B, num_labels, D)

        Returns:
            dict:
                - gcn_features: GCN 增强的标签特征 (B, num_labels, output_dim)
        """
        B = fused_features.shape[0]

        # 将所有样本的标签特征聚合为图节点特征
        # 对 batch 维度取平均 → 每个标签一个代表向量
        node_features = fused_features.mean(dim=0)  # (num_labels, input_dim)

        # GCN 消息传递
        x = node_features
        for i, (gcn, norm) in enumerate(zip(self.gcn_layers, self.norms)):
            residual = x
            x = gcn(x, self.adj_matrix)
            x = norm(x)
            x = self.dropout(F.relu(x))

            # 残差连接（每层维度不一致时用投影对齐）
            residual_proj = self.intermediate_residual_projs[i]
            x = x + residual_proj(residual)

        # 扩展到整个 batch
        gcn_features = x.unsqueeze(0).expand(B, -1, -1)  # (B, num_labels, output_dim)

        return {
            "gcn_features": gcn_features,
        }


# ============================================================
# 标签关联建模整体模块
# ============================================================

class LabelAssociationModule(nn.Module):
    """
    标签关联建模模块（整合 Label Attention + GCN）

    整合两种互补的标签关系建模方式：
    1. Label Attention: 数据驱动的标签关系学习
    2. Lightweight GCN: 先验知识引导的标签关系建模

    最终通过自适应融合将两者结合，输出标签关系增强的特征。
    """

    def __init__(
        self,
        num_labels: int = 12,
        feature_dim: int = 512,
        label_embed_dim: int = 128,
        gcn_hidden_dim: int = 256,
        num_attention_heads: int = 4,
        gcn_num_layers: int = 2,
        dropout: float = 0.1,
        emotion_labels: List[str] = None,
        cooccurrence: Dict = None,
        mutual_exclusion: Dict = None,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.feature_dim = feature_dim

        # 标签注意力模块
        self.label_attention = LabelAttentionModule(
            num_labels=num_labels,
            feature_dim=feature_dim,
            label_embed_dim=label_embed_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # 轻量 GCN 模块
        self.label_gcn = LightweightLabelGCN(
            num_labels=num_labels,
            input_dim=feature_dim,
            hidden_dim=gcn_hidden_dim,
            output_dim=feature_dim,
            num_layers=gcn_num_layers,
            dropout=dropout,
            emotion_labels=emotion_labels,
            cooccurrence=cooccurrence,
            mutual_exclusion=mutual_exclusion,
        )

        # 自适应融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        # 最终输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )

        print(f"[LabelAssociationModule] 初始化完成")
        print(f"  - 标签数: {num_labels}")
        print(f"  - 特征维度: {feature_dim}")
        print(f"  - GCN 层数: {gcn_num_layers}")
        if emotion_labels:
            print(f"  - 情感标签: {emotion_labels}")

    def forward(
        self,
        fused_features: torch.Tensor,  # (B, num_labels, feature_dim)
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            fused_features: 融合后的多模态特征 (B, num_labels, D)
            return_details: 是否返回中间结果

        Returns:
            dict:
                - enhanced_features: 标签关系增强的特征 (B, num_labels, D)
                - label_relations: (B, H, num_labels, num_labels) 标签关系矩阵
                - gcn_features: GCN 特征（仅 return_details=True）
        """
        # 1. 标签注意力 → 数据驱动的标签关系建模
        attn_out = self.label_attention(fused_features)
        modulated = attn_out["modulated_features"]  # (B, num_labels, D)

        # 2. GCN → 先验知识引导的标签关系建模
        gcn_out = self.label_gcn(fused_features)
        gcn_features = gcn_out["gcn_features"]  # (B, num_labels, D)

        # 3. 自适应融合
        alpha = torch.sigmoid(self.fusion_weight)
        enhanced = alpha * modulated + (1 - alpha) * gcn_features

        # 4. 输出投影
        enhanced = self.output_proj(enhanced)

        result = {
            "enhanced_features": enhanced,
            "label_relations": attn_out["label_relations"],
        }

        if return_details:
            result["gcn_features"] = gcn_features
            result["modulated_features"] = modulated

        return result


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("标签关联建模模块测试")
    print("=" * 60)

    from config import UNIFIED_EMOTIONS, EMOTION_COOCCURRENCE, EMOTION_MUTUAL_EXCLUSION

    B, L, D = 4, 12, 512
    dummy_features = torch.randn(B, L, D)

    # 测试标签关系图构建
    print("\n[1] 标签关系图构建测试")
    adj = build_emotion_label_graph(
        UNIFIED_EMOTIONS,
        EMOTION_COOCCURRENCE,
        EMOTION_MUTUAL_EXCLUSION,
    )
    print(f"  邻接矩阵形状: {adj.shape}")
    print(f"  正边数量: {(adj > 0).sum().item() - L}")  # 减去自连接
    print(f"  负边数量: {(adj < 0).sum().item()}")
    print(f"  例子 - joy↔sadness 权重: {adj[0, 1]:.2f}")  # 互斥

    # 测试邻接矩阵归一化
    adj_norm = normalize_adjacency(adj)
    print(f"  归一化后值域: [{adj_norm.min():.3f}, {adj_norm.max():.3f}]")

    # 测试完整标签关联模块
    print("\n[2] LabelAssociationModule 测试")
    label_module = LabelAssociationModule(
        num_labels=L,
        feature_dim=D,
        emotion_labels=UNIFIED_EMOTIONS,
        cooccurrence=EMOTION_COOCCURRENCE,
        mutual_exclusion=EMOTION_MUTUAL_EXCLUSION,
    )

    with torch.no_grad():
        output = label_module(dummy_features, return_details=True)

    print(f"  enhanced_features: {output['enhanced_features'].shape}")
    print(f"  label_relations:   {output['label_relations'].shape}")
    if "gcn_features" in output:
        print(f"  gcn_features:      {output['gcn_features'].shape}")

    # 验证输出
    assert output["enhanced_features"].shape == (B, L, D), "输出维度错误！"
    assert output["label_relations"].shape[-2:] == (L, L), "关系矩阵形状错误！"

    # 参数量
    total = sum(p.numel() for p in label_module.parameters())
    trainable = sum(p.numel() for p in label_module.parameters() if p.requires_grad)
    print(f"\n  参数量: {total:,} (可训练: {trainable:,})")
    print("✅ 标签关联建模模块测试通过！")
