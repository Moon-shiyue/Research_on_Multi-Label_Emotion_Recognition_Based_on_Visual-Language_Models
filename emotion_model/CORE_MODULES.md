# 三大核心模块 — 技术实现说明

> 本文档说明「研究内容」中三大核心模块的**具体实现**，与
> `TECHNICAL_REPORT.md`（基础架构说明）配套阅读。
>
> 验证结果：**102/102 项测试全部通过**（详见 `module_verification_report.txt`）

---

## 模块① 基于注意力引导的冲突感知跨模态融合

**文件**：`conflict_fusion.py`
**理论来源**：Yao et al. *Cross-Modal Semantic Interference Suppression (CMSIS)*,
Engineering Applications of Artificial Intelligence, 2024

### 解决的痛点

| 场景 | 例子 | 传统方法的失效原因 |
|------|------|------------------|
| **图文语义冲突（反讽）** | 文本"我喜欢这宽敞的腿部空间" + 经济舱狭窄实景 | InfoNCE 强制拉近图文特征，无法识别反讽 |
| **模态内语义干扰** | 字面相似但情感相反的两条文本 | 同模态内特征重叠，污染跨模态对齐 |
| **语义错位对齐** | 图像悲伤但文本欢快 | 平等聚合所有特征，发生"空间崩塌" |

### 三层结构实现

```
┌─────────────────────────────────────────────────────────────┐
│ ① TextConflictAttention — 文本情绪注意力（双路径冲突注意力）  │
│                                                             │
│   路径A（字面）: 原始文本特征 ──┐                            │
│                                ├──→ 余弦差异度 diff ∈ [0,1]   │
│   路径B（意图）: 情感词引导 ────┘                            │
│                                                             │
│   动态片段划分（可学习边界 p，CMSIS 核心设计）:                │
│     diff ≤ p → 情感一致片段 → 门控融合双路径                   │
│     diff > p → 情感冲突片段 → 保留独立特征 + 输出冲突信号       │
│                                                             │
│   实现: conflict_fusion.py:TextConflictAttention             │
│   软化 mask: σ((diff - p) × 10) 保证边界可学习                │
├─────────────────────────────────────────────────────────────┤
│ ② VisualConflictAttention — 视觉情绪注意力（冲突区域捕捉）     │
│                                                             │
│   patch 级余弦相似度 S = cos(V_patch, T_emotion)              │
│     S > p → 核心情感区域（人脸表情/场景氛围/情感化物体）        │
│     S < p → 情感对立区域 → 注意力施加负偏置(-2.0)抑制干扰       │
│                                                             │
│   多头注意力: α_v = Softmax(Q_v·K_vᵀ/√d_k)·V_v               │
├─────────────────────────────────────────────────────────────┤
│ ③ ConflictAwareAlignment — 跨模态冲突感知注意力对齐           │
│                                                             │
│   冲突程度 c = (1 - cos(V, T)) / 2                            │
│   对齐权重 w = σ(τ·(b - c)) × modulation                      │
│     ↑ 结构单调递减：冲突越大 → 权重越低（无需训练即成立）        │
│                                                             │
│   一致图文对 → w 高 → 强化双向注意力（特征互补）               │
│   冲突图文对 → w 低 → 保留独立冲突特征，避免错误对齐            │
│                                                             │
│   融合公式: F = w·(α_v·F_v + α_t·F_t) + ...                     │
└─────────────────────────────────────────────────────────────┘
```

### 损失函数（CMSIS Eq.15-20）

```
L_text  = Σ[τ_t - S(T,T⁺) + S(T,T̂⁺)]₊        文本模态内三元组
L_visual= Σ[τ_v - S(I,I⁺) + S(I,Î⁺)]₊        图像模态内三元组
L_cross = Σ[γ - S(I,T) + S(I,T̂)]₊
        + [γ - S(I,T) + S(Î,T)]₊              双向跨模态三元组
L_conflict = L_text + L_visual + L_cross
```

- **正样本**：随机 dropout 增强（CMSIS 做法）
- **负样本**：batch 内 hardest negative（相似度最高者，`argmax_{x≠I⁺} S(I,x)`）
- **超参数**：τ_t = τ_v = 0.3，γ = 0.2（论文默认值）

### 关键验证结果

| 验证项 | 结果 |
|--------|------|
| 冲突时对齐权重下降（核心语义） | ✅ 一致 0.66 > 冲突 0.23 |
| 对齐权重对冲突单调递减（结构保证） | ✅ 冲突 0→1 时权重严格递减 |
| 三元组损失语义正确 | ✅ 相同正样本 0.0000 ≤ 随机正样本 0.4320 |
| 动态边界可学习 | ✅ 边界梯度 -2.6e-03 |

---

## 模块② 基于情感环形表示的多标记分类头

**文件**：`circular_head.py`
**理论来源**：Yang et al. *A Circular-Structured Representation for Visual Emotion
Distribution Learning*, CVPR 2021；Mikels et al. 2005（Mikels Wheel）

### 情感环形三属性

```
e_i = (p_i, θ_i, r_i)
  p_i — 情感极性 (polarity):  0 = 积极, 1 = 消极
  θ_i — 情感类型 (type):      极角 θ ∈ [0, 2π)
  r_i — 情感强度 (intensity): 极径 r ∈ [0, 1]
```

### 8 类基础情感的角度分配（θ_j = (2j-1)/8·π）

| 情感 | 角度 | 极性 | 环形位置 |
|------|------|------|---------|
| amusement | 22.5° | 积极 | 环形起点 |
| excitement | 67.5° | 积极 | ↓ |
| anger | 112.5° | 消极 | ↓ |
| disgust | 157.5° | 消极 | ↓ |
| fear | 202.5° | 消极 | ↓ |
| sadness | 247.5° | 消极 | ↓ |
| awe | 292.5° | 积极 | ↓ |
| contentment | 337.5° | 积极 | 环形终点 |

**极性半圆划分**：积极 θ ∈ [0°, 90°) ∪ [270°, 360°)，消极 θ ∈ [90°, 270°)
**语义相邻性**：相邻情感语义最接近（如 amusement↔excitement 都是高唤醒积极情感）

### 三分支结构

```
共享特征 Z = LayerNorm → Linear → GELU → Dropout (512 → 256)

分支1 极性判别:  p_k = σ(W_pk·Z + b_pk)              三分类 logits (积极/中性/消极)
                v_k = tanh(W_vk·Z + b_vk)            连续极性值 ∈ (-1,1)
分支2 角度类型:  logits_k = W_θk·Z + b_θk            sigmoid → 多标记置信度
                θ_k = 基本角度 + tanh(·)·(π/8)        偏移受限，不跨越相邻情感
分支3 强度回归:  a_k = σ(W_ak·Z + b_ak)              强度 ∈ (0,1)
```

### 三维直角坐标转换

```
x_k = r · a_k · cos(θ_k)
y_k = r · a_k · sin(θ_k)
z_k = v_k
```

### 标签共现关联增强

```
y_i = σ(W · F_fusion + b + M · y_{-i})
M — 标签关联矩阵（共现为正边、互斥为负边，共 20 正边 / 18 负边）
```

### 渐进式环形损失

```
第一步（粗）L_p  = (1/N)Σ(p_i - p̂_i)²                        极性 MSE
第二步（中）L_t  = (1/N)Σ(θ_i - θ̂_i)²                        环形角度 MSE
第三步（细）L_PC = (1/N)Σ r_i·((p_i-p̂_i)² + (θ_i-θ̂_i)²)      强度加权
联合损失   L    = (1-μ)·L_KL + μ·L_PC                        μ = 0.5
```

**环形角度误差**采用 `|atan2(sin Δθ, cos Δθ)|`，正确处理 2π 周期性
（验证：跨 0 点近邻误差 0.01 rad，而欧氏误差 6.27 rad）。

### 复合情感派生（基础情感 → 复合情感）

在 8 类基础情感之上，可按环形相邻关系派生出复合情感（8 条规则），例如：
- amusement + excitement → **joy**（喜悦）
- anger + disgust → **contempt**（轻蔑）
- fear + sadness → **despair**（绝望）

### 关键验证结果

| 验证项 | 结果 |
|--------|------|
| 单一情感角度还原准确 | ✅ 最大误差 < 1e-4 弧度 |
| 复合情感角度落在组成情感之间 | ✅ amusement+excitement → 48.7°（介于 22.5°~67.5°） |
| 环形角度误差处理 2π 周期性 | ✅ 环形 0.0100 ≪ 欧氏 6.2732 |
| 角度偏移受限于 ±π/8 | ✅ 最大偏移 0.299 ≤ 0.3927 |
| 标签关联项影响输出 | ✅ 最大差异 0.107 |

---

## 模块③ 基于 VL-Adapter 的跨场景情感特征泛化优化

**文件**：`vl_adapter.py`
**理论来源**：Sung et al. *VL-Adapter: Parameter-Efficient Transfer Learning for
Vision-and-Language Tasks*, CVPR 2022

### 分治型参数解耦架构

```
┌──────────────────────────────────────────────────────────────┐
│ 冻结 CLIP 预训练主干（保留通用跨模态语义能力）                  │
│                                                              │
│ CLIP 视觉塔 (12 层)              CLIP 文本塔 (12 层)           │
│   Layer 1 ──→ [Adapter]            Layer 1 ──→ [Adapter]      │
│   Layer 2 ──→ [Adapter]            Layer 2 ──→ [Adapter]      │
│      ...                              ...                    │
│   Layer 12 ─→ [Adapter]            Layer 12 ─→ [Adapter]      │
│                                                              │
│ 适配器结构（降维–激活–升维–残差）:                             │
│   Adapter(x) = x + W_up·GELU(W_down·x)                       │
│   W_down: d → k (k = d/4),  W_up: k → d                      │
│                        ↓                                      │
│ 参数解耦:                                                     │
│   共享参数 (shared)        — 所有场景共用，建模域不变知识        │
│   域特定参数 (domain)      — 低秩分支 B_d·A_d，每场景独立        │
│                                                              │
│ 前向: output = x + h_shared + α·(B_d·A_d·x)                   │
└──────────────────────────────────────────────────────────────┘
```

### 插入位置

```
H_mid = LN(H_in + MSA(H_in) + Adapter_MSA(H_in + MSA(H_in)))
H_out = LN(H_mid + MLP(H_mid) + Adapter_MLP(H_mid + MLP(H_mid)))
```
通过 **forward hook** 挂载，无需修改 transformers 源码。

### 参数效率

| 配置 | 可训练参数 | 占主干比例 |
|------|-----------|-----------|
| bottleneck_ratio = 4（默认） | 6,017,280 | **4.57%** ✅ |
| bottleneck_ratio = 8 | ~3.4M | 2.73% |
| bottleneck_ratio = 16 | ~1.8M | 1.46% |

- 共享参数：5,131,008
- 域特定参数：491,520（4 场景 × 24 层）
- 视觉投影层：394,240

### 跨场景工作流

```python
# 1. 源域训练：训练共享参数 + 源域特定参数
manager.set_domain(0)                    # social_media
manager.unfreeze_shared()

# 2. 适配新域：冻结共享参数，仅训练新域的低秩偏置
manager.add_domain(1)                    # 新增场景
manager.set_domain(4)                    # 新场景
manager.freeze_shared()                  # 保护域不变知识

# 3. 推理：按场景 ID 选择对应参数
manager.set_domain(2)                    # education
```

### 域分布对齐损失（抑制特征漂移）

```
L_align = MMD²(源域, 目标域) + 0.1·‖Cov_s - Cov_t‖²_F
```
相比 DANN 对抗训练更稳定，无需额外判别器。

### 关键验证结果

| 验证项 | 结果 |
|--------|------|
| 可训练参数占比符合 3%~5% | ✅ 4.57% |
| 梯度仅流入当前场景（场景隔离） | ✅ 有梯度场景 = [1] |
| 近恒等初始化（不破坏预训练特征） | ✅ 相对偏移 < 5% |
| 域扩展不破坏已有参数 | ✅ 4 → 6 场景，原参数不变 |
| 分布偏移越大对齐损失越大 | ✅ 偏移 1.02 > 同分布 0.40 |

---

## 集成结果

### 完整模型配置对比

| 配置 | 标签数 | 模块① | 模块② | 模块③ | 可训练参数 |
|------|--------|-------|-------|-------|-----------|
| **消融基线** | 12 | ✗ | ✗ | ✗ | 22.1M |
| **完整模型** | 8 | ✓ | ✓ | ✓ | 18.9M（11.15%） |

### 消融实验配置矩阵

`create_ablation_configs()` 自动生成 6 组配置：

| 实验组 | 说明 |
|--------|------|
| `baseline` | 消融基线（核心模块全部关闭） |
| `full` | 完整方案（三大模块全开） |
| `w/o_conflict_fusion` | 移除模块① |
| `w/o_circular_head` | 移除模块② |
| `w/o_vl_adapter` | 移除模块③ |
| `w/o_contrastive` | 仅移除模块①中的对比学习损失 |

### 使用方式

```python
from emotion_model.config import create_full_config, create_ablation_configs
from emotion_model.full_model import MultiLabelEmotionModel

# 完整模型（三大核心模块）
config = create_full_config()
model = MultiLabelEmotionModel(config)

# 训练时计算多目标联合损失
outputs = model(images)
loss = model.compute_loss(
    outputs["logits"], targets,
    loss_type="asymmetric",
    circular_outputs=outputs,              # 模块② 环形损失
    contrastive_loss=outputs.get("contrastive_loss"),  # 模块① 对比损失
)
```

---

## 验证报告

完整验证报告：`module_verification_report.txt`

```
总测试数: 102
通过: 102 ✅
失败: 0 ❌
```

| 验证模块 | 测试数 | 通过 |
|---------|-------|------|
| 模块① 冲突感知融合 | 29 | 29 ✅ |
| 模块② 环形表示分类头 | 33 | 33 ✅ |
| 模块③ VL-Adapter | 23 | 23 ✅ |
| 集成验证 | 17 | 17 ✅ |

运行方式：
```bash
python emotion_model/verify_modules.py                      # 全部验证
python emotion_model/verify_modules.py --module circular     # 单模块验证
python emotion_model/verify_modules.py --output report.txt   # 输出报告
```

---

*文档更新时间: 2026-09*
*对应「研究内容」技术方案*
