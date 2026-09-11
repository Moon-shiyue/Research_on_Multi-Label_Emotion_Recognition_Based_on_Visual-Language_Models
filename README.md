# 基于视觉语言模型的多标记情感识别研究

Research on Multi-Label Emotion Recognition Based on Visual-Language Models

## 项目简介

基于视觉语言模型（VLM）的多标记情感识别研究。项目采用 **CLIP + 三大核心模块**的技术方案，
围绕图文语义冲突、标签关系建模与跨场景泛化三个痛点展开。

| 目录 | 定位 | 内容 |
|------|------|------|
| `emotion_model/` | **项目技术方案** | CLIP (ViT-B/32) 双塔编码 + 冲突感知融合 + 情感环形分类头 + VL-Adapter |
| `qwenvl_emotion/` | 早期技术路线探索记录 | Qwen2.5-VL-3B + LoRA 微调路线的调研与代码（未纳入最终方案，详见其 README） |

## 三大核心模块

「研究内容」中确定的三大核心模块（详见
[`emotion_model/CORE_MODULES.md`](emotion_model/CORE_MODULES.md)）：


| 模块 | 解决的问题 | 核心机制 | 代码 |
|------|-----------|---------|------|
| **① 冲突感知跨模态融合** | 图文语义冲突（反讽）、模态内语义干扰 | 双路径冲突注意力 + 模态内对比学习 + 双向三元组排序损失 + 冲突感知对齐 | `conflict_fusion.py` |
| **② 情感环形表示分类头** | 复合情感共存、强度差异建模 | Mikels Wheel 环形表示（极性-类型-强度三维）+ 三分支输出 + 渐进式环形损失 | `circular_head.py` |
| **③ VL-Adapter 跨场景泛化** | 跨场景特征漂移、灾难性遗忘 | 参数解耦适配器（共享参数 + 域特定参数），可训练参数仅 4.57% | `vl_adapter.py` |

**验证状态**：`verify_modules.py` — **102/102 项测试全部通过**

## 情感标签体系（两级）

**扩展标签集（12 类）** — 兼容 ArtPhoto/Emotion6/GAPED 等数据集：
```
joy, sadness, anger, fear, surprise, disgust,
love, peace, amusement, awe, contentment, excitement
```

**项目标签体系（8 类 Mikels 基础情感 + 复合情感）** — 对应情感环形表示：
```
amusement(22.5°), excitement(67.5°), anger(112.5°), disgust(157.5°),
fear(202.5°), sadness(247.5°), awe(292.5°), contentment(337.5°)

复合情感示例: amusement+excitement → joy,  anger+disgust → contempt
```

## 项目方案：CLIP 双塔 + 三大核心模块（emotion_model/）

**完整模型**（项目技术方案）：

```
CLIP 双塔编码 → 模块①冲突感知融合 → 模块②情感环形分类头
                     ↑
              模块③ VL-Adapter（挂载双塔各层，跨场景泛化）
```

**消融对照组件**（关闭核心模块时启用的基础实现）：

```
CLIP 双塔编码 → 层次化注意力融合 → 标签关联(GCN + Label Attention) → 通用多标记分类头
```

前者用于实现「研究内容」的技术方案，后者在消融实验中作为对照，
通过 `create_ablation_configs()` 可生成 6 组配置一键切换。

### 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 模块功能验证（无需数据集）
cd emotion_model
python verify.py                        # 基础组件验证（69 项）
python verify_modules.py                # 三大核心模块验证（102 项）
python verify_modules.py --module circular       # 单模块验证
python verify_modules.py --output report.txt     # 输出验证报告

# 3. 训练（需要数据）
# 消融基线（12 类扩展标签，核心模块全部关闭）
python -m emotion_model.train --data_root ./data/ArtPhoto ./data/Emotion6 --epochs 50

# 完整模型（8 类基础情感 + 三大核心模块，冻结主干配合 VL-Adapter）
python -m emotion_model.train --data_root ./data --output ./output_full --epochs 50 --full

# 消融实验（自动套用对应配置）
python -m emotion_model.train --data_root ./data --ablation w/o_circular_head --epochs 50

# 常用选项：
#   --loss_type asymmetric|bce|focal    损失函数（默认 asymmetric）
#   --freeze_visual --freeze_text       冻结 CLIP 编码器
#   --no_label_association              禁用标签关联（消融）
#   --full                              使用完整模型配置（三大核心模块）
#   --ablation <name>                   消融配置：baseline/full/w_o_conflict_fusion/...
#   --eval_only                         仅评估

# 4. 单模块测试
python base_encoder.py          # 编码器测试
python fusion_module.py         # 融合模块测试
python conflict_fusion.py       # ⭐ 模块① 冲突感知融合测试
python circular_head.py         # ⭐ 模块② 情感环形分类头测试
python vl_adapter.py            # ⭐ 模块③ VL-Adapter 测试
python classification_head.py   # 分类头测试
python label_association.py     # 标签关联测试
python dataset.py               # 数据加载测试
python full_model.py            # 完整模型测试
```

## 早期技术路线探索（qwenvl_emotion/）

项目启动初期对「原生多模态大模型微调」路线做过调研与实现，**未纳入最终技术方案**，
代码保留以供结题报告说明技术选型依据。

- Qwen2.5-VL-3B 基座 + LoRA（约 2% 可训练参数）
- 三种 Prompt 策略：Direct / CoT / Contrastive
- 生成式训练（Next Token Prediction），输出结构化 JSON

**未采用的原因**（详见 [`qwenvl_emotion/README.md`](qwenvl_emotion/README.md)）：
与「研究内容」以 CLIP 为基座的三大模块不兼容；多标记输出需解析文本而非直接 sigmoid 置信度；
显存需求 16GB（vs 8GB）且需额外下载 6-7GB 权重；可解释性分析难以直接获得注意力分布。

```bash
# 配置验证（无需 GPU）
cd qwenvl_emotion
python verify.py
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
├── emotion_model/          ← 项目技术方案：CLIP + 三大核心模块
│   ├── base_encoder.py     CLIP ViT + Text 编码器封装
│   ├── conflict_fusion.py  ⭐⭐ 模块① 冲突感知跨模态融合
│   ├── circular_head.py    ⭐⭐ 模块② 情感环形表示分类头
│   ├── vl_adapter.py       ⭐⭐ 模块③ VL-Adapter 跨场景泛化
│   ├── fusion_module.py    层次化注意力融合（消融对照组件）
│   ├── label_association.py GCN + Label Attention 标签关联
│   ├── classification_head.py  通用多标记分类头（BCE/ASL/Focal，消融对照组件）
│   ├── full_model.py       完整模型（支持完整配置/消融配置切换）
│   ├── config.py           配置（两级标签体系/超参数/先验关系/消融配置）
│   ├── dataset.py          数据加载
│   ├── train.py            训练脚本（支持 --full / --ablation）
│   ├── utils.py            评估指标/工具函数
│   ├── verify.py           基础组件验证（69 项）
│   ├── verify_modules.py   三大核心模块验证（102 项）
│   ├── TECHNICAL_REPORT.md 基础架构技术说明
│   ├── CORE_MODULES.md     ⭐ 三大核心模块技术实现说明
│   ├── verification_report.txt          基础组件验证报告
│   └── module_verification_report.txt     核心模块验证报告
├── qwenvl_emotion/         ← 早期技术路线探索记录（未纳入方案）
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

- `emotion_model/CORE_MODULES.md` — **三大核心模块技术实现**（「研究内容」对应）
- `emotion_model/TECHNICAL_REPORT.md` — 基础架构技术说明（层次化融合/标签关联/非对称损失）
- `emotion_model/module_verification_report.txt` — 核心模块验证报告（102/102 通过）
- `qwenvl_emotion/README.md` — 双方案对比与实验设计建议
- `数据集下载地址` — 各数据集官方地址
