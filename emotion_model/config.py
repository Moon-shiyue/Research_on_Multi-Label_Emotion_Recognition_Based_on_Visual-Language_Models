"""
全局配置文件
定义模型超参数、情感标签体系、训练参数等

标签体系（全项目统一为单一体系）:
  采用 Mikels Wheel 8 类基础情感（MIKELS_BASIC_EMOTIONS），
  对应情感环形表示（Emotion Circle）。

  历史扩展标签集（LEGACY_EXTENDED_EMOTIONS，12 类）仅为兼容早期版本的数据
  格式而保留，**不作为模型输出空间**；如数据集使用该格式，需先用
  multihot_12_to_8() 转换到 8 类基础情感。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

# ============================================================
# 各数据集原始标签定义（用于数据转换与追溯）
# ============================================================

# Ekman 六种基本情感 (Ekman, 1992)
EKMAN_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# ArtPhoto / FI 数据集标签（Mikels 8 类）
ARTPHOTO_EMOTIONS = ["amusement", "anger", "awe", "contentment", "disgust",
                     "excitement", "fear", "sadness"]

# Emotion6 情感标签
EMOTION6_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# GAPED 情感标签
GAPED_EMOTIONS = ["anger", "disgust", "fear", "sadness"]

# 历史扩展标签集（12 类）—— 仅用于兼容早期数据格式，不作为模型标签空间
#
# 构成：Ekman 6 类 + Mikels 独有的 4 类正性情感 + love/peace
# 注意：love 与 peace 不属于 Ekman、Mikels 等经典情感模型，在 ArtPhoto /
# Emotion6 / GAPED 中也没有对应标注，不建议继续使用。
LEGACY_EXTENDED_EMOTIONS = [
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


# ============================================================
# 12 类 → 8 类标签映射（数据预处理用）
# ============================================================

# 映射依据:
#   1. Mikels et al. (2005) —— Mikels 8 类中的 4 类正性情感
#      (amusement, awe, contentment, excitement) 是对 Ekman (1992) 单一
#      "joy" 的细分，故 joy 的映射有明确理论出处。
#   2. Russell (1980) 环形模型 —— 依据 valence-arousal 坐标就近映射。
#
# 映射类型:
#   · 一对一（8 个）: sadness / anger / fear / disgust / amusement /
#                     awe / contentment / excitement（本身即 Mikels 基础情感）
#   · 一对多（2 个）: joy / surprise（Ekman 情感在 Mikels 体系中无单点对应）
#   · 无依据（2 个）: love / peace —— 不属于 Ekman、Mikels 等经典情感模型，
#                     映射仅为兼容旧数据格式，**不建议使用**
LEGACY_TO_BASIC_MAPPING = {
    # Ekman 的 joy → Mikels 的正性情感（Mikels et al. 2005）
    "joy": ["amusement", "excitement"],
    # Ekman 的 surprise 在 Mikels 8 类中无对应，按唤醒度就近映射
    "surprise": ["excitement", "awe"],
    # 以下两项无理论依据，仅作旧格式兼容
    "love": ["contentment", "amusement"],
    "peace": ["contentment", "awe"],
}


def map_legacy_to_basic(legacy_label: str) -> List[str]:
    """
    旧 12 类标签 → 8 类基础情感映射

    Args:
        legacy_label: 旧标签体系中的情感标签

    Returns:
        对应的基础情感列表（可能一对多）

    注意:
        love / peace 不属于经典情感模型，映射缺乏理论与数据依据，
        建议在数据侧直接剔除这两个标签，而非依赖映射。
    """
    if legacy_label in MIKELS_BASIC_EMOTIONS:
        return [legacy_label]
    return LEGACY_TO_BASIC_MAPPING.get(legacy_label, [])


def multihot_12_to_8(multihot: torch.Tensor) -> torch.Tensor:
    """
    12 类多热标签 → 8 类基础情感多热标签（数据预处理工具）

    Args:
        multihot: (N, 12) 旧标签体系多热编码

    Returns:
        (N, 8) Mikels 基础情感多热编码
    """
    N = multihot.shape[0]
    out = torch.zeros(N, len(MIKELS_BASIC_EMOTIONS))
    idx = {name: i for i, name in enumerate(MIKELS_BASIC_EMOTIONS)}

    for j, legacy in enumerate(LEGACY_EXTENDED_EMOTIONS):
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
    num_emotions: int = len(MIKELS_BASIC_EMOTIONS)   # 情感类别数（8 类 Mikels 基础情感）
    emotion_labels: List[str] = field(default_factory=lambda: list(MIKELS_BASIC_EMOTIONS))
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
    # 三大核心模块开关（对应「研究内容」的三项工作）
    # 全部为 False 时 = 消融基线（逐项移除核心模块，用于对照实验）
    # ========================================================

    # ---- 模块① 注意力引导的冲突感知跨模态融合 ----
    use_conflict_fusion: bool = False      # 关闭时改用层次化注意力融合（消融对照）
    conflict_text_boundary_init: float = 0.3   # 文本冲突动态边界初值
    conflict_visual_boundary_init: float = 0.0 # 视觉冲突动态边界初值
    use_conflict_contrastive: bool = True      # 是否启用模态内对比学习损失

    # ---- 模块② 情感环形表示多标记分类头 ----
    use_circular_head: bool = False        # 关闭时改用通用多标记分类头（消融对照）
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
    # 全项目统一采用 Mikels 8 类基础情感（由 num_emotions / emotion_labels 决定），
    # 不再支持切换标签空间；不同标签格式的数据集请先用 multihot_12_to_8() 转换。


def create_full_config(**overrides) -> "ModelConfig":
    """
    创建项目完整模型配置（三大核心模块全部启用）

    即「研究内容」中确定的技术方案：冲突感知融合 + 情感环形分类头 + VL-Adapter。

    启用内容:
      - 模块① 冲突感知跨模态融合（含模态内对比学习与三元组排序损失）
      - 模块② 情感环形表示分类头（8 类 Mikels 基础情感 + 渐进式环形损失）
      - 模块③ VL-Adapter 跨场景泛化（参数解耦适配器）
      - 冻结 CLIP 编码器（配合 VL-Adapter 的参数高效微调策略）

    使用方式:
        config = create_full_config()
        model = MultiLabelEmotionModel(config)
    """
    defaults = dict(
        # 三大核心模块
        use_conflict_fusion=True,
        use_circular_head=True,
        use_vl_adapter=True,
        # 标签体系：8 类 Mikels 基础情感
        num_emotions=len(MIKELS_BASIC_EMOTIONS),
        emotion_labels=list(MIKELS_BASIC_EMOTIONS),
        # 模块③ 要求冻结主干
        freeze_visual=True,
        freeze_text=True,
        # 环形表示自身已建模标签关系，不再叠加标签关联模块
        use_label_association=False,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def create_ablation_configs() -> dict:
    """
    生成消融实验配置集合

    以完整模型为基准，逐项移除核心模块，用于验证各模块贡献。

    .. important::
       所有配置使用**相同的 8 类 Mikels 标签空间**，仅改变模块开关，
       以保证消融对比的公平性。

    Returns:
        dict: {实验名: ModelConfig}

    实验组:
        - full:                 完整模型（三大模块全开）
        - w/o_conflict_fusion:  移除模块①（改用层次化注意力融合）
        - w/o_circular_head:    移除模块②（改用通用多标记分类头）
        - w/o_vl_adapter:       移除模块③
        - w/o_contrastive:      仅移除模块①中的对比学习损失
        - baseline:             消融基线（三项核心模块全部移除）
    """
    return {
        "full": create_full_config(),
        "w/o_conflict_fusion": create_full_config(use_conflict_fusion=False),
        # 仅替换分类头，标签空间保持 8 类不变
        "w/o_circular_head": create_full_config(use_circular_head=False),
        "w/o_vl_adapter": create_full_config(use_vl_adapter=False),
        "w/o_contrastive": create_full_config(use_conflict_contrastive=False),
        # 消融基线：关闭全部核心模块，标签空间仍为 8 类
        "baseline": ModelConfig(
            use_conflict_fusion=False,
            use_circular_head=False,
            use_vl_adapter=False,
        ),
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
