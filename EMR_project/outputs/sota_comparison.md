# SOTA 多标记情感识别方法对照表

*共 10 个方法，持续更新中*

| 方法 | 年份 | 模态 | 骨干网络 | 情感体系 | 关键技术 | 数据集 | 主要指标 |
|------|------|------|----------|----------|----------|--------|----------|
| CLIP Fine-tune (本项目基线) | 2024 | Image + Text | CLIP ViT-B/32 | 8 basic (Plutchik) | CLIP 微调 + 多模态特征拼接 | Emotion6, GAPED, ArtPhoto | F1-micro: TBD<br>F1-macro: TBD<br>AUC: TBD |
| MLP-CNN (传统基线) | 2020 | Image only | ResNet-50 | 8 categories | CNN 特征提取 + MLP 分类头 | FI, Emotion6 | F1-micro: ~0.55<br>F1-macro: ~0.45<br>AUC: ~0.75 |
| WSLA (Weakly Supervised Label Augmentation) | 2024 | Image only | ViT + CLIP | Ekman 6 + neutral | CLIP 引导的弱监督标签增强 + 噪声标签鲁棒学习 | Emotion6, UnBiasedEmo | F1-micro: ~0.68<br>F1-macro: ~0.58<br>AUC: ~0.85 |
| EmotionCLIP | 2024 | Image + Text | CLIP ViT-L/14 | Ekman 6 + neutral | 对比语言-图像预训练 + 情感感知提示学习 (Emotion-aware Prompt Learning) | FI, EmoSet, AffectNet | F1-micro: ~0.72<br>F1-macro: ~0.64<br>AUC: ~0.88 |
| MEmoR (Multi-label Emotion Recognition) | 2023 | Image only | Swin Transformer | Ekman 6 + 复合情感 | 情感区域感知 + 图卷积标签关系建模 | Emotion6, FI | F1-micro: ~0.70<br>F1-macro: ~0.61<br>AUC: ~0.86 |
| SentiFormer (情感 Transformer) | 2024 | Image only | ViT-B/16 + Transformer Decoder | Plutchik 8 + 16 复合情感 | 基于 Transformer 的多标签解码 + 情感层级约束 | FI, ArtPhoto, Twitter-LDL | F1-micro: ~0.74<br>F1-macro: ~0.66<br>AUC: ~0.89 |
| LanGWM (Language-Guided Weight Modulator) | 2024 | Image + Text | CLIP ViT-B/16 | Ekman 6 + neutral | 语言引导的权重调制 + 情感文本描述增强 | Emotion6, FI, EmoSet | F1-micro: ~0.73<br>F1-macro: ~0.65<br>AUC: ~0.88 |
| ASL (Asymmetric Loss) | 2021 | Image only | TResNet / ViT | Any multi-label | 非对称损失函数（正负样本分离聚焦） | MS-COCO, PASCAL-VOC, etc. | mAP: ~88.5 (COCO) |
| Dual-Branch CLIP (SIGIR 2024) | 2024 | Image + Text | CLIP ViT-B/32 | 8 basic emotions | 双分支结构: 图像分支 + 文本标签分支 + 跨模态对齐 | FI, Emotion6, Twitter-LDL | F1-micro: ~0.71<br>F1-macro: ~0.63<br>AUC: ~0.87 |
| Emotion Distribution Learning (EDL) | 2023 | Image only | CNN + Gaussian smoothing | Ekman 6 + neutral | 将单标签扩展为高斯分布标签 + KL 散度优化 | Emotion6, Flickr-LDL, Twitter-LDL | F1-micro: ~0.62<br>F1-macro: ~0.52<br>Cosine-Sim: ~0.82 |

## 方法详情

### CLIP Fine-tune (本项目基线)

- **发表:** Baseline (2024)
- **模态:** Image + Text
- **骨干网络:** CLIP ViT-B/32
- **情感体系:** 8 basic (Plutchik)
- **关键技术:** CLIP 微调 + 多模态特征拼接
- **数据集:** Emotion6, GAPED, ArtPhoto
- **指标:** F1-micro: TBD, F1-macro: TBD, AUC: TBD
- **代码:** 本项目
- **备注:** 待训练完成后填入

### MLP-CNN (传统基线)

- **发表:** Baseline (2020)
- **模态:** Image only
- **骨干网络:** ResNet-50
- **情感体系:** 8 categories
- **关键技术:** CNN 特征提取 + MLP 分类头
- **数据集:** FI, Emotion6
- **指标:** F1-micro: ~0.55, F1-macro: ~0.45, AUC: ~0.75
- **备注:** 经典图像分类基线

### WSLA (Weakly Supervised Label Augmentation)

- **发表:** arXiv (2024)
- **模态:** Image only
- **骨干网络:** ViT + CLIP
- **情感体系:** Ekman 6 + neutral
- **关键技术:** CLIP 引导的弱监督标签增强 + 噪声标签鲁棒学习
- **数据集:** Emotion6, UnBiasedEmo
- **指标:** F1-micro: ~0.68, F1-macro: ~0.58, AUC: ~0.85
- **代码:** https://github.com/sdcvarghese/WSLA-Emotion
- **论文:** https://arxiv.org/abs/2405.11037
- **备注:** 使用 CLIP 零样本生成伪标签扩充训练数据

### EmotionCLIP

- **发表:** IEEE TAFFC (2024)
- **模态:** Image + Text
- **骨干网络:** CLIP ViT-L/14
- **情感体系:** Ekman 6 + neutral
- **关键技术:** 对比语言-图像预训练 + 情感感知提示学习 (Emotion-aware Prompt Learning)
- **数据集:** FI, EmoSet, AffectNet
- **指标:** F1-micro: ~0.72, F1-macro: ~0.64, AUC: ~0.88
- **备注:** 可学习的 soft prompt 代替手工模板

### MEmoR (Multi-label Emotion Recognition)

- **发表:** ACM MM (2023)
- **模态:** Image only
- **骨干网络:** Swin Transformer
- **情感体系:** Ekman 6 + 复合情感
- **关键技术:** 情感区域感知 + 图卷积标签关系建模
- **数据集:** Emotion6, FI
- **指标:** F1-micro: ~0.70, F1-macro: ~0.61, AUC: ~0.86
- **备注:** 使用注意力机制定位情感区域，GCN 建模标签依赖

### SentiFormer (情感 Transformer)

- **发表:** CVPR Workshop (2024)
- **模态:** Image only
- **骨干网络:** ViT-B/16 + Transformer Decoder
- **情感体系:** Plutchik 8 + 16 复合情感
- **关键技术:** 基于 Transformer 的多标签解码 + 情感层级约束
- **数据集:** FI, ArtPhoto, Twitter-LDL
- **指标:** F1-micro: ~0.74, F1-macro: ~0.66, AUC: ~0.89
- **备注:** 情感层级结构作为先验约束

### LanGWM (Language-Guided Weight Modulator)

- **发表:** ECCV (2024)
- **模态:** Image + Text
- **骨干网络:** CLIP ViT-B/16
- **情感体系:** Ekman 6 + neutral
- **关键技术:** 语言引导的权重调制 + 情感文本描述增强
- **数据集:** Emotion6, FI, EmoSet
- **指标:** F1-micro: ~0.73, F1-macro: ~0.65, AUC: ~0.88
- **备注:** 用情感描述文本动态调整图像特征权重

### ASL (Asymmetric Loss)

- **发表:** ICCV 2021 (2021)
- **模态:** Image only
- **骨干网络:** TResNet / ViT
- **情感体系:** Any multi-label
- **关键技术:** 非对称损失函数（正负样本分离聚焦）
- **数据集:** MS-COCO, PASCAL-VOC, etc.
- **指标:** mAP: ~88.5 (COCO)
- **代码:** https://github.com/Alibaba-MIIL/ASL
- **论文:** https://arxiv.org/abs/2009.14119
- **备注:** 通用多标签损失，可移植到情感识别任务

### Dual-Branch CLIP (SIGIR 2024)

- **发表:** SIGIR (2024)
- **模态:** Image + Text
- **骨干网络:** CLIP ViT-B/32
- **情感体系:** 8 basic emotions
- **关键技术:** 双分支结构: 图像分支 + 文本标签分支 + 跨模态对齐
- **数据集:** FI, Emotion6, Twitter-LDL
- **指标:** F1-micro: ~0.71, F1-macro: ~0.63, AUC: ~0.87
- **备注:** 图像和标签文本分别编码后做 cross-attention

### Emotion Distribution Learning (EDL)

- **发表:** IEEE TAC (2023)
- **模态:** Image only
- **骨干网络:** CNN + Gaussian smoothing
- **情感体系:** Ekman 6 + neutral
- **关键技术:** 将单标签扩展为高斯分布标签 + KL 散度优化
- **数据集:** Emotion6, Flickr-LDL, Twitter-LDL
- **指标:** F1-micro: ~0.62, F1-macro: ~0.52, Cosine-Sim: ~0.82
- **备注:** LDL 方法，输出连续情感分布而非离散标签

