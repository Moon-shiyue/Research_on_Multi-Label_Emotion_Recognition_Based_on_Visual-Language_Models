"""
Qwen2.5-VL LoRA 微调 — 多标记情感识别
Multi-Label Emotion Recognition via Vision-Language Model Fine-tuning

与 emotion_model/ (CLIP方案) 形成对比实验：
  - CLIP方案: 双塔编码 + 手写跨模态融合 + GCN标签关联
  - Qwen-VL方案: 原生多模态模型 + LoRA微调 + Prompt策略

优势：
  1. 模型天然理解情感语义，无需手写标签关联
  2. LoRA 仅训练 ~2% 参数，单卡即可微调
  3. 架构简洁，调参门槛低
"""

__version__ = "0.1.0"
