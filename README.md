# 基于视觉语言模型的多标记情感识别研究

Research on Multi-Label Emotion Recognition Based on Visual-Language Models

## 项目简介

基于视觉语言模型（VLM）的多标记情感识别研究，提供两套可对比的实验方案：

| 方案 | 目录 | 技术路线 | 特点 |
|------|------|---------|------|
| **方案A** | `emotion_model/` | CLIP (ViT-B/32) 双塔编码 + 层次化注意力融合 + 标签关联建模 | 模块化、可解释、可消融 |
| **方案B** | `qwenvl_emotion/` | Qwen2.5-VL-3B + LoRA 轻量微调 | 原生多模态理解、Prompt 策略 |

两方案共享统一的情感标签体系（12 类）与评估指标，用于论文对比实验。

## 统一情感标签体系

基于 Ekman 六种基本情感 + ArtPhoto/Emotion6/GAPED 扩展情感：

```
joy, sadness, anger, fear, surprise, disgust,
love, peace, amusement, awe, contentment, excitement
```

## 方案A：CLIP 双塔融合（emotion_model/）

架构：`CLIP双塔 → 层次化注意力融合 → 标签关联(GCN+Label Attention) → 多标记分类头`

核心创新：
- **层次化跨模态融合**：局部高权重特征融合（视觉情绪注意力 + 文本情绪注意力）→ 全局对齐特征校准
- **先验知识注入的标签关联建模**：有符号 GCN（共现正边/互斥负边）+ 数据驱动标签注意力
- **非对称损失**：处理多标记场景的极端标签不平衡

### 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 模块功能验证（无需数据集）
cd emotion_model
python verify.py                 # 全部模块验证
python verify.py --quick         # 快速验证（跳过部分）
python verify.py --output report.txt   # 输出验证报告

# 3. 训练（需要数据）
python -m emotion_model.train --data_root ./data/ArtPhoto ./data/Emotion6 --epochs 50
# 常用选项：
#   --loss_type asymmetric|bce|focal    损失函数（默认 asymmetric）
#   --freeze_visual --freeze_text       冻结 CLIP 编码器
#   --no_label_association              禁用标签关联（消融）
#   --eval_only                         仅评估

# 4. 单模块测试
python base_encoder.py          # 编码器测试
python fusion_module.py         # 融合模块测试
python classification_head.py   # 分类头测试
python label_association.py     # 标签关联测试
python dataset.py               # 数据加载测试
python full_model.py            # 完整模型测试
```

## 方案B：Qwen2.5-VL LoRA 微调（qwenvl_emotion/）

- Qwen2.5-VL-3B 基座 + LoRA（仅训练约 2% 参数）
- 三种 Prompt 策略：Direct / CoT / Contrastive
- 生成式训练（Next Token Prediction），输出结构化 JSON

### 快速开始

```bash
# 验证配置（无需 GPU）
cd qwenvl_emotion
python verify.py

# GPU 服务器训练
python -m qwenvl_emotion.train --data_root ./data --epochs 5 --prompt_strategy cot
# 显存不足时: --load_in_4bit
```

## 数据集

| 数据集 | 说明 | 地址 |
|--------|------|------|
| ArtPhoto | 806 张艺术照片，8 类情感 | https://www.imageemotion.org/ |
| Emotion6 | 1980 张，6 类基本情感 | http://chenlab.ece.cornell.edu/downloads.html |
| GAPED | 730 张，负面情感为主 | https://www.unige.ch/cisa/research/materials-and-online-research/research-material/ |

数据格式要求：每个数据集目录下 `images/`（图像）+ `labels.csv` 或 `labels.json`（多热标签）。

## 目录结构

```
项目根目录/
├── emotion_model/          ← 方案A：CLIP 双塔融合
│   ├── base_encoder.py     CLIP ViT + Text 编码器封装
│   ├── fusion_module.py    ⭐ 层次化注意力融合（核心创新）
│   ├── label_association.py ⭐ GCN + Label Attention 标签关联（核心创新）
│   ├── classification_head.py  多标记分类头（BCE/ASL/Focal）
│   ├── full_model.py       完整模型
│   ├── config.py           配置（标签体系/超参数/先验关系）
│   ├── dataset.py          数据加载
│   ├── train.py            训练脚本
│   ├── utils.py            评估指标/工具函数
│   ├── verify.py           模块验证脚本
│   └── TECHNICAL_REPORT.md 核心创新点技术说明文档
├── qwenvl_emotion/         ← 方案B：Qwen2.5-VL LoRA
│   ├── model.py            ⭐ Qwen-VL + LoRA 模型
│   ├── config.py           配置 + Prompt 模板
│   ├── train.py            训练脚本
│   ├── dataset.py          数据加载
│   ├── utils.py            工具函数
│   ├── verify.py           验证脚本
│   └── README.md           双方案对比说明
└── requirements.txt        依赖清单
```

## 相关文档

- `emotion_model/TECHNICAL_REPORT.md` — 核心创新点技术说明（层次化融合/标签关联/非对称损失）
- `qwenvl_emotion/README.md` — 双方案对比与实验设计建议
- `数据集下载地址` — 各数据集官方地址
