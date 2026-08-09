"""
全局配置文件
定义模型超参数、情感标签体系、训练参数等
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
