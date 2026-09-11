"""
基于视觉语言模型的多标记情感识别研究
Multi-Label Emotion Recognition Based on Visual-Language Models

核心模型架构包

模块组织（对应「研究内容」的三大核心模块）:
  1. 基础框架: base_encoder (CLIP 双塔编码器)
  2. 模块① 冲突感知融合: conflict_fusion (注意力引导的跨模态融合 + 冲突抑制)
  3. 模块② 环形表示分类头: circular_head (双头设计 + 渐进式环形损失)
       - 头1 CircularMultiLabelHead   → 有哪些情感（多标记）
       - 头2 CompoundEmotionHead      → 整体落在环上哪里（复合情感向量）
       - CircularConsistencyLoss      → 双头一致性约束
  4. 模块③ VL-Adapter: vl_adapter (跨场景情感特征泛化)
  5. 消融对照组件: fusion_module (层次化注意力融合)
  6. 标签关联建模: label_association (标签注意力 + 轻量 GCN，消融对照组件)
"""

__version__ = "0.3.0"
__author__ = "Shen X"

from .base_encoder import CLIPEncoder, VisualEncoder, TextEncoder
from .fusion_module import HierarchicalAttentionFusion
from .classification_head import MultiLabelClassificationHead
from .label_association import LabelAssociationModule
from .full_model import MultiLabelEmotionModel

# 三大核心模块
from .conflict_fusion import (
    ConflictAwareFusionModule,
    TextConflictAttention,
    VisualConflictAttention,
    ConflictAwareAlignment,
    ConflictContrastiveLoss,
)
from .circular_head import (
    CircularMultiLabelHead,
    CompoundEmotionHead,
    CircularConsistencyLoss,
    EmotionCircleMapper,
    ProgressiveCircularLoss,
    CIRCULAR_EMOTIONS,
    EMOTION_ANGLES,
)
