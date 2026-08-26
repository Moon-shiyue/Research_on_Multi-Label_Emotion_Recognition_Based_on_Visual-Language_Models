"""
全局配置管理 — 所有模块的统一配置入口
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class DataConfig:
    """数据路径与处理配置"""
    # 原始数据集路径（下载后放置的位置）
    data_root: str = "C:/Users/32934/Desktop/数据集"
    emotion6_root: str = ""
    flickr30k_root: str = ""
    gaped_root: str = ""
    artphoto_root: str = ""
    fi_root: str = ""

    # 处理后数据保存路径
    processed_root: str = "C:/Users/32934/Desktop/EMR_project/data/processed"

    # 图像统一尺寸
    image_size: int = 224

    # 数据集划分比例
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # 随机种子
    seed: int = 42

    # 每个epoch是否做shuffle
    shuffle_train: bool = True

    def __post_init__(self):
        if not self.emotion6_root:
            self.emotion6_root = os.path.join(self.data_root, "Emotion6")
        if not self.flickr30k_root:
            self.flickr30k_root = os.path.join(self.data_root, "flickr 30k")
        if not self.gaped_root:
            self.gaped_root = os.path.join(self.data_root, "GAPED")
        if not self.artphoto_root:
            self.artphoto_root = os.path.join(self.data_root, "Artphoto")
        if not self.fi_root:
            self.fi_root = os.path.join(self.data_root, "FI")


@dataclass
class ModelConfig:
    """模型配置"""
    # CLIP 变体: "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"
    clip_model_name: str = "openai/clip-vit-base-patch32"

    # 特征维度（由 CLIP 模型决定）
    image_embed_dim: int = 512   # ViT-B/32
    text_embed_dim: int = 512    # ViT-B/32
    projection_dim: int = 256    # 融合后投影维度

    # 融合方式: "concat" | "cross_attention" | "gated"
    fusion_method: str = "concat"

    # 预测头
    num_emotions: int = 8
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])

    # Dropout
    dropout: float = 0.3

    # 是否冻结 CLIP 骨干
    freeze_image_encoder: bool = False
    freeze_text_encoder: bool = False

    # 用于生成文本描述的模板
    text_prompt_template: str = "a photo expressing {emotion}"


@dataclass
class TrainingConfig:
    """训练配置"""
    # 批次大小（CPU 训练建议减小）
    batch_size: int = 16
    eval_batch_size: int = 32

    # 训练轮数
    num_epochs: int = 50

    # 学习率
    learning_rate: float = 2e-5
    backbone_lr_ratio: float = 0.1      # 骨干网络学习率倍率

    # 优化器
    optimizer: str = "adamw"           # adamw / sgd
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # 学习率调度
    lr_scheduler: str = "cosine"       # cosine / linear / step
    warmup_steps: int = 500
    warmup_ratio: float = 0.1

    # 损失函数
    loss_type: str = "bce"             # bce / asymmetric / focal / focal_bce
    asymmetric_gamma_neg: float = 4.0   # ASL 负样本 gamma
    asymmetric_gamma_pos: float = 1.0   # ASL 正样本 gamma
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # 早停
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_f1_micro"

    # 混合精度（CPU 不可用，仅 GPU 生效）
    use_amp: bool = False

    # 日志
    log_interval: int = 50              # 每 N 步打印一次
    eval_interval: int = 1              # 每 N 个 epoch 验证一次

    # 输出
    output_dir: str = "C:/Users/32934/Desktop/EMR_project/outputs"


@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # 设备
    device: str = "cpu"                # 自动检测，可手动覆盖

    # 实验名称
    experiment_name: str = "baseline_clip_concat"

    def __post_init__(self):
        import torch
        if torch.cuda.is_available():
            self.device = "cuda"
            self.training.use_amp = True

    def save(self, path: str):
        """保存配置到 YAML"""
        import yaml
        from dataclasses import asdict
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, allow_unicode=True, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 加载配置"""
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f)
        cfg = cls()
        cfg.data = DataConfig(**d.get("data", {}))
        cfg.model = ModelConfig(**d.get("model", {}))
        cfg.training = TrainingConfig(**d.get("training", {}))
        cfg.device = d.get("device", cfg.device)
        cfg.experiment_name = d.get("experiment_name", cfg.experiment_name)
        return cfg


# 单例
_default_config = Config()


def get_config() -> Config:
    """获取全局默认配置"""
    return _default_config


def set_config(cfg: Config):
    """设置全局配置"""
    global _default_config
    _default_config = cfg
