# Qwen2.5-VL LoRA 微调方案 vs CLIP 双塔融合方案

## 架构对比一览

```
┌─────────────────────────────────────────────────────────────────┐
│                    方案A：CLIP 双塔融合                           │
│                    (emotion_model/)                              │
│                                                                  │
│   Image ──→ ViT-B/32 ──→ 512d ──┐                               │
│                                   ├──→ 手写融合 ──→ GCN ──→ 分类头 │
│   Text ──→ CLIP Text ──→ 512d ──┘    (599行代码)  (529行)        │
│                                                                  │
│   特点：模块化、可解释、需要手写架构                               │
│   参数：~132M（冻结编码器后 ~7M 可训）                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    方案B：Qwen-VL LoRA 微调                       │
│                    (qwenvl_emotion/)                              │
│                                                                  │
│   Image ──┐                                                      │
│            ├──→ Qwen2.5-VL-3B (LoRA) ──→ JSON {"joy":0.9,...}   │
│   Prompt ─┘    (模型内部自动编码+融合+推理)                        │
│                                                                  │
│   特点：简洁、情感语义强、仅需设计 Prompt                          │
│   参数：~3B 总量，LoRA ~60M 可训（约2%）                          │
└─────────────────────────────────────────────────────────────────┘
```

## 关键差异

| 维度 | CLIP 方案 | Qwen-VL 方案 |
|------|----------|-------------|
| **模型类型** | 双塔编码器（需手写融合） | 原生多模态大模型 |
| **编码器** | ViT-B/32 (87M) + Text (38M) | Qwen2.5-VL 内部 ViT + LLM |
| **融合方式** | 手写层次化注意力（599行） | 模型内部 Transformer 自动完成 |
| **标签关联** | 手写 GCN + Label Attention（529行） | 模型内部注意力自动捕获 |
| **可训练参数** | ~7M（冻结编码器） | ~60M（LoRA，约2%） |
| **训练方式** | 分类式（BCE Loss） | 生成式（Next Token Prediction） |
| **推理速度** | ~50ms/图 (V100) | ~200ms/图 (V100) |
| **显存需求** | ~8GB (batch=32) | ~16GB (batch=4) |
| **情感理解** | 依赖 CLIP 预训练对齐 | 原生多模态指令微调 |
| **中文支持** | 较弱（英文为主） | 强（中英双语） |
| **调参难度** | 高（超参数多） | 低（主要是 LoRA rank + lr） |

## 实验设计建议

推荐在你的论文中做四组实验，构成完整的对比研究：

### 实验矩阵

| 实验组 | 方案 | 变体 | 目的 |
|--------|------|------|------|
| **A-1** | CLIP | 基础（ViT-B/32 + 简单拼接） | Baseline |
| **A-2** | CLIP | + 层次化融合 + GCN（完整 CLIP 方案） | 验证手写模块有效性 |
| **A-3** | CLIP | + ViT-L/14（更大编码器） | 验证 scale 效果 |
| **B-1** | Qwen-VL | Direct Prompt | VLM 最简方案 |
| **B-2** | Qwen-VL | CoT Prompt | 验证思维链效果 |
| **B-3** | Qwen-VL | 对比 Qwen2.5-VL-7B | 验证模型规模影响 |

### 消融实验（CLIP 方案内）

| 消融项 | 移除模块 | 预期影响 |
|--------|---------|---------|
| 无融合 | 移除层次化融合，直接拼接 | f1 ↓ 3-5% |
| 无 GCN | 移除标签关联模块 | 互斥标签误判增多 |
| 无校准 | 移除全局对齐校准 | 跨模态偏移增大 |

### 消融实验（Qwen-VL 方案内）

| 消融项 | 变体 | 预期影响 |
|--------|------|---------|
| Prompt 策略 | Direct vs CoT vs Contrastive | CoT 对复杂情感更准 |
| LoRA rank | r=4/8/16/32 | r=16 性价比最高 |
| 模型规模 | 2B vs 3B vs 7B | 7B 优势在细微情感 |

## 预期论文贡献

通过两套方案的对比，你的论文可以提出以下核心主张：

1. **VLM 时代的多标记情感识别**不再需要手写跨模态融合——原生多模态模型内部已天然完成
2. **Prompt 策略比模型架构更重要**——好的 Prompt（CoT）带来的提升可能超过复杂的融合模块
3. **标签先验知识已经编码在模型中**——LLM 的情感世界知识可以替代手工构建的 GCN 标签图
4. **但 CLIP 方案仍有价值**——在小数据集、低资源场景下，轻量方案可能更稳定

## 目录结构

```
项目根目录/
├── emotion_model/          ← 方案A：CLIP 双塔融合
│   ├── base_encoder.py     CLIP ViT + Text 编码器
│   ├── fusion_module.py    ⭐ 层次化注意力融合
│   ├── label_association.py ⭐ GCN + Label Attention
│   ├── classification_head.py  多标记分类头
│   ├── full_model.py       完整模型
│   ├── config.py           配置
│   ├── dataset.py          数据加载
│   ├── utils.py            工具函数
│   ├── verify.py           验证脚本
│   └── TECHNICAL_REPORT.md 技术文档
│
├── qwenvl_emotion/         ← 方案B：Qwen-VL LoRA
│   ├── model.py            ⭐ Qwen-VL + LoRA 模型
│   ├── config.py           配置 + Prompt 模板
│   ├── train.py            训练脚本
│   ├── dataset.py          数据加载
│   ├── utils.py            工具函数
│   ├── verify.py           验证脚本
│   └── README.md           ← 本文件
│
└── 数据集下载地址           共享数据集
```

## 快速上手

```bash
# 方案A（CLIP）:
cd emotion_model
python base_encoder.py          # 测试编码器
python fusion_module.py         # 测试融合模块
python full_model.py            # 测试完整模型
python verify.py                # 全模块验证

# 方案B（Qwen-VL）:
cd qwenvl_emotion
python verify.py                # 配置验证（无需GPU）
# GPU 服务器上:
python -m qwenvl_emotion.train --data_root ./data --epochs 5
```
