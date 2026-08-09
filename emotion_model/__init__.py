"""
基于视觉语言模型的多标记情感识别研究
Multi-Label Emotion Recognition Based on Visual-Language Models

核心模型架构包
"""

__version__ = "0.1.0"
__author__ = "Shen X"

from .base_encoder import CLIPEncoder, VisualEncoder, TextEncoder
from .fusion_module import HierarchicalAttentionFusion
from .classification_head import MultiLabelClassificationHead
from .label_association import LabelAssociationModule
from .full_model import MultiLabelEmotionModel
