"""
全局配置文件
定义模型超参数、情感标签体系、训练参数等

标签体系包含两级：
  1. 12 类统一情感标签（UNIFIED_EMOTIONS）— 基础版，兼容多数据集
  2. 8 类 Mikels 基础情感（MIKELS_BASIC_EMOTIONS）— 申请书要求，
     对应 CVPR 2021 情感环形表示（Emotion Circle）
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

# ============================================================
# 情感标签体系
# 基于 Ekman 六种基本情感 + ArtPhoto/GAPED 扩展情感
# ============================================================

# Ekman 六种基本情感
EKMAN_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# ArtPhoto 数据集扩展情感
ARTPHOTO_EMOTIONS = ["amusement", "anger", "awe", "contentment", "disgust",
                     "excitement", "fear", "sadness"]

# Emotion6 情感标签
EMOTION6_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# GAPED 情感标签
GAPED_EMOTIONS = ["anger", "disgust", "fear", "sadness"]

# 统一情感标签集（合并所有数据集的情感类别，用于多标记识别）
UNIFIED_EMOTIONS = [
    "joy", "sadness", "anger", "fear", "surprise", "disgust",
    "love", "peace", "amusement", "awe", "contentment", "excitement"
]

# ============================================================
# Mikels Wheel 8 类基础情感体系（申请书要求）
# 对应 CVPR 2021 情感环形表示（Emotion Circle），角度 = (2j-1)/8·π
# ============================================================

# 8 类基础情感（沿环形逆时针排列）
MIKELS_BASIC_EMOTIONS = [
    "amusement",    # j=1 → 22.5°  积极
    "excitement",   # j=2 → 67.5°  积极
    "anger",        # j=3 → 112.5° 消极
    "disgust",      # j=4 → 157.5° 消极
    "fear",         # j=5 → 202.5° 消极
    "sadness",      # j=6 → 247.5° 消极
    "awe",          # j=7 → 292.5° 积极
    "contentment",  # j=8 → 337.5° 积极
]

# 基础情感的极性分组（Mikels et al. 2005）
MIKELS_POSITIVE = ["amusement", "excitement", "awe", "contentment"]
MIKELS_NEGATIVE = ["anger", "disgust", "fear", "sadness"]

# 复合情感映射规则（环形相邻基础情感的组合，构成两级标签体系）
COMPOUND_EMOTION_RULES = {
    ("amusement", "excitement"): "joy",           # 有趣+兴奋 → 喜悦
    ("amusement", "contentment"): "satisfaction", # 有趣+满足 → 满意
    ("awe", "contentment"): "serenity",           # 敬畏+满足 → 宁静
    ("sadness", "awe"): "melancholy",             # 悲伤+敬畏 → 忧郁
    ("fear", "sadness"): "despair",               # 恐惧+悲伤 → 绝望
    ("disgust", "fear"): "aversion",              # 厌恶+恐惧 → 反感
    ("anger", "disgust"): "contempt",             # 愤怒+厌恶 → 轻蔑
    ("excitement", "anger"): "agitation",         # 兴奋+愤怒 → 激越（跨极性高唤醒）
}

# 可派生出的复合情感列表
COMPOUND_EMOTIONS = sorted(set(COMPOUND_EMOTION_RULES.values()))

# 8 类基础情感的共现关系（环形相邻情感共现概率更高）
MIKELS_COOCCURRENCE = {
    ("amusement", "excitement"): 0.7,
    ("amusement", "contentment"): 0.6,
    ("awe", "contentment"): 0.5,
    ("sadness", "awe"): 0.3,
    ("fear", "sadness"): 0.6,
    ("disgust", "fear"): 0.5,
    ("anger", "disgust"): 0.6,
    ("excitement", "anger"): 0.2,
    ("fear", "anger"): 0.4,
    ("sadness", "disgust"): 0.4,
}

# 8 类基础情感的互斥关系（跨极性情感互斥）
MIKELS_MUTUAL_EXCLUSION = {
    ("amusement", "anger"): 0.8,
    ("amusement", "disgust"): 0.8,
    ("amusement", "fear"): 0.7,
    ("amusement", "sadness"): 0.7,
    ("excitement", "sadness"): 0.6,
    ("contentment", "anger"): 0.8,
    ("contentment", "fear"): 0.7,
    ("awe", "anger"): 0.5,
    ("awe", "disgust"): 0.5,
}


def map_to_compound_emotion(positive_emotions: List[str]) -> List[str]:
    """
    基础情感组合 → 复合情感映射（两级标签体系的第二级）

    Args:
        positive_emotions: 被激活的基础情感列表

    Returns:
        派生出的复合情感列表

    示例:
        >>> map_to_compound_emotion(["amusement", "excitement"])
        ['joy']
    """
    positive_set = set(positive_emotions)
    compounds = []
    for (e1, e2), compound in COMPOUND_EMOTION_RULES.items():
        if e1 in positive_set and e2 in positive_set and compound not in compounds:
            compounds.append(compound)
    return compounds


def map_legacy_to_basic(legacy_label: str) -> List[str]:
    """
    旧 12 类标签 → 8 类基础情感映射（用于数据集标签体系转换）

    映射依据（语义/极性/唤醒度最接近）:
        joy      → amusement, excitement
        surprise → excitement, awe
        love     → contentment, amusement
        peace    → contentment, awe
        其余标签本身即为 Mikels 基础情感
    """
    mapping = {
        "joy": ["amusement", "excitement"],
        "surprise": ["excitement", "awe"],
        "love": ["contentment", "amusement"],
        "peace": ["contentment", "awe"],
    }
    if legacy_label in MIKELS_BASIC_EMOTIONS:
        return [legacy_label]
    return mapping.get(legacy_label, [])


def multihot_12_to_8(multihot: torch.Tensor) -> torch.Tensor:
    """
    12 类多热标签 → 8 类基础情感多热标签

    Args:
        multihot: (N, 12) 旧标签体系多热编码

    Returns:
        (N, 8) Mikels 基础情感多热编码
    """
    N = multihot.shape[0]
    out = torch.zeros(N, len(MIKELS_BASIC_EMOTIONS))
    idx = {name: i for i, name in enumerate(MIKELS_BASIC_EMOTIONS)}

    for j, legacy in enumerate(UNIFIED_EMOTIONS):
        if multihot[:, j].sum() == 0:
            continue
        for basic in map_legacy_to_basic(legacy):
            if basic in idx:
                # 取最大值，避免一条标签映射到多个基础情感时重复累加
                out[:, idx[basic]] = torch.maximum(out[:, idx[basic]], multihot[:, j])
    return out

# 情感标签共现关系（先验知识）
# 值域: [0, 1]，表示两个情感同时出现的先验概率/关联强度
EMOTION_COOCCURRENCE = {
    ("joy", "amusement"): 0.7,      ("joy", "excitement"): 0.8,
    ("joy", "love"): 0.6,           ("joy", "contentment"): 0.5,
    ("joy", "peace"): 0.4,          ("joy", "awe"): 0.3,
    ("sadness", "fear"): 0.5,       ("sadness", "disgust"): 0.4,
    ("anger", "disgust"): 0.5,      ("anger", "fear"): 0.3,
    ("fear", "surprise"): 0.6,      ("sadness", "anger"): 0.3,
    ("disgust", "fear"): 0.3,       ("love", "peace"): 0.5,
    ("amusement", "excitement"): 0.6, ("contentment", "peace"): 0.7,
    ("awe", "surprise"): 0.5,
}

# 情感标签互斥关系
# 某些情感几乎不会同时出现
EMOTION_MUTUAL_EXCLUSION = {
    ("joy", "sadness"): 0.9,
    ("joy", "anger"): 0.8,
    ("joy", "fear"): 0.7,
    ("joy", "disgust"): 0.8,
    ("peace", "anger"): 0.9,
    ("peace", "fear"): 0.8,
    ("love", "anger"): 0.8,
    ("contentment", "anger"): 0.8,
}


@dataclass
class ModelConfig:
    """模型架构配置"""

    # ---- CLIP 编码器 ----
    clip_model_name: str = "openai/clip-vit-base-patch32"  # ViT-B/32
    visual_feature_dim: int = 512      # CLIP ViT-B/32 视觉特征维度
    text_feature_dim: int = 512        # CLIP ViT-B/32 文本特征维度
    projection_dim: int = 512          # 统一投影维度

    # ---- 编码器控制 ----
    freeze_visual: bool = False        # 是否冻结视觉编码器
    freeze_text: bool = False          # 是否冻结文本编码器
    use_grad_checkpoint: bool = True   # 梯度检查点（节省显存）

    # ---- 层次化注意力融合 ----
    fusion_hidden_dim: int = 512       # 融合模块隐层维度
    fusion_num_heads: int = 8          # 多头注意力头数
    fusion_dropout: float = 0.1        # 融合模块 dropout
    fusion_num_layers: int = 2         # 层次化融合层数

    # ---- 多标记分类头 ----
    num_emotions: int = 12             # 情感类别数 (= len(UNIFIED_EMOTIONS))
    emotion_labels: List[str] = field(default_factory=lambda: UNIFIED_EMOTIONS)
    classifier_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])

    # ---- 标签关联建模 ----
    use_label_association: bool = True  # 是否启用标签关联模块
    label_embed_dim: int = 128         # 标签嵌入维度
    gcn_num_layers: int = 2            # 图卷积层数
    gcn_dropout: float = 0.1           # GCN dropout
    label_attention_heads: int = 4     # 标签注意力头数

    # ---- 损失函数 ----
    use_asymmetric_loss: bool = True   # 使用非对称损失处理标签不平衡
    pos_weight: Optional[float] = None # 正样本权重
    label_smoothing: float = 0.0       # 标签平滑

    # ========================================================
    # 申请书「研究内容」三大核心创新模块开关
    # 全部为 False 时 = 基础版（消融实验的对照组）
    # ========================================================

    # ---- 模块① 注意力引导的冲突感知跨模态融合 ----
    use_conflict_fusion: bool = False      # 替换基础版层次化融合
    conflict_text_boundary_init: float = 0.3   # 文本冲突动态边界初值
    conflict_visual_boundary_init: float = 0.0 # 视觉冲突动态边界初值
    use_conflict_contrastive: bool = True      # 是否启用模态内对比学习损失

    # ---- 模块② 情感环形表示多标记分类头 ----
    use_circular_head: bool = False        # 替换基础版分类头
    circular_radius: float = 1.0           # 环形半径 r
    circular_mu: float = 0.5               # PC 损失与 KL 损失的权重（论文 Eq.10）
    circular_angle_mode: str = "circular"  # 角度误差模式: circular | raw
    use_compound_emotion: bool = True      # 是否输出复合情感（两级标签体系）

    # ---- 模块③ VL-Adapter 跨场景泛化 ----
    use_vl_adapter: bool = False           # 启用参数解耦型适配器
    adapter_bottleneck_ratio: int = 4      # 瓶颈压缩比（4 → 约 4.6% 可训练参数）
    adapter_num_domains: int = 4           # 场景（域）数量
    adapter_domain_rank: int = 4           # 域特定低秩分支秩
    adapter_domain: int = 0                # 当前场景编号

    # ---- 标签体系 ----
    use_mikels_basic: bool = False         # 使用 8 类 Mikels 基础情感（环形表示要求）


def create_innovation_config(**overrides) -> "ModelConfig":
    """
    创建启用三大核心创新模块的完整配置（申请书「研究内容」完整方案）

    对应论文实验矩阵中的「完整方案」（A-2），用于与基础版对照。

    启用内容:
      - 模块① 冲突感知跨模态融合（含模态内对比学习与三元组排序损失）
      - 模块② 情感环形表示分类头（8 类 Mikels 基础情感 + 渐进式环形损失）
      - 模块③ VL-Adapter 跨场景泛化（参数解耦适配器）
      - 冻结 CLIP 编码器（配合 VL-Adapter 的参数高效微调策略）

    使用方式:
        config = create_innovation_config(freeze_visual=True, freeze_text=True)
        model = MultiLabelEmotionModel(config)
    """
    defaults = dict(
        # 三大创新模块
        use_conflict_fusion=True,
        use_circular_head=True,
        use_vl_adapter=True,
        # 标签体系切换为 8 类 Mikels 基础情感（环形表示的前提）
        use_mikels_basic=True,
        num_emotions=len(MIKELS_BASIC_EMOTIONS),
        emotion_labels=MIKELS_BASIC_EMOTIONS,
        # 模块③ 要求冻结主干
        freeze_visual=True,
        freeze_text=True,
        # 环形表示替代基础版标签关联（环形本身已建模标签关系）
        use_label_association=False,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def create_ablation_configs() -> dict:
    """
    生成消融实验配置集合（对应申请书的消融实验设计）

    Returns:
        dict: {实验名: ModelConfig}

    实验组:
        - baseline:  基础版（层次化融合 + 普通多标记头）
        - full:      完整方案（三大模块全开）
        - w/o_conflict_fusion:  移除模块①
        - w/o_circular_head:    移除模块②（换回普通多标记头）
        - w/o_vl_adapter:       移除模块③
        - w/o_contrastive:      仅移除模块①中的对比学习损失
    """
    return {
        "baseline": ModelConfig(
            use_conflict_fusion=False,
            use_circular_head=False,
            use_vl_adapter=False,
            use_mikels_basic=False,
        ),
        "full": create_innovation_config(),
        "w/o_conflict_fusion": create_innovation_config(use_conflict_fusion=False),
        "w/o_circular_head": create_innovation_config(
            use_circular_head=False, use_mikels_basic=False,
            num_emotions=len(UNIFIED_EMOTIONS), emotion_labels=UNIFIED_EMOTIONS,
        ),
        "w/o_vl_adapter": create_innovation_config(use_vl_adapter=False),
        "w/o_contrastive": create_innovation_config(use_conflict_contrastive=False),
    }


@dataclass
class TrainingConfig:
    """训练配置"""

    # ---- 优化器 ----
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # ---- 训练循环 ----
    num_epochs: int = 50
    batch_size: int = 32
    gradient_accumulation_steps: int = 2

    # ---- 学习率调度 ----
    lr_scheduler: str = "cosine"       # cosine / linear / constant
    lr_warmup_ratio: float = 0.05

    # ---- 早停 ----
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_f1_macro"

    # ---- 数据 ----
    image_size: Tuple[int, int] = (224, 224)
    num_workers: int = 4
    prefetch_factor: int = 2

    # ---- 日志 ----
    log_interval: int = 50
    eval_interval: int = 500
    save_total_limit: int = 3

    # ---- 混合精度 ----
    use_amp: bool = True               # 自动混合精度


# 默认配置实例
default_model_config = ModelConfig()
default_training_config = TrainingConfig()
