"""
Qwen2.5-VL LoRA 微调 — 技术路线探索记录
Vision-Language Model Fine-tuning (Exploratory Implementation)

本模块记录项目早期对「原生多模态大模型微调」路线的探索实现：
  - 原生多模态模型 + LoRA 轻量微调 + Prompt 策略

未纳入项目最终技术方案（最终方案为 CLIP + 三大核心模块，见 emotion_model/）。
保留本模块用于记录技术选型的调研与论证过程。
详见同目录 README.md。
"""

__version__ = "0.1.0"
