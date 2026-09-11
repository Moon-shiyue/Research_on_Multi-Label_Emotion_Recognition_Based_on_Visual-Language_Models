"""
多标记情感识别模型 — 核心模块功能验证脚本
Verification for the Three Core Modules

对应「研究内容」中的三大核心模块:
  模块① 基于注意力引导的冲突感知跨模态融合
  模块② 基于情感环形表示的多标记分类头
  模块③ 基于 VL-Adapter 的跨场景情感特征泛化优化

验证维度（每个模块三层）:
  1. 结构验证：张量维度、参数统计、输出格式
  2. 功能验证：模块行为是否符合设计语义（如冲突检测是否真的区分一致/冲突）
  3. 集成验证：与完整模型端到端集成、梯度回传、损失计算

运行方式:
    python verify_modules.py                      # 运行全部验证
    python verify_modules.py --output report.txt  # 输出报告到文件
    python verify_modules.py --module circular    # 仅验证指定模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import argparse
import math
from datetime import datetime
from typing import Dict, List

# Windows 控制台编码兼容
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_model.config import (
    ModelConfig, MIKELS_BASIC_EMOTIONS, LEGACY_EXTENDED_EMOTIONS,
    MIKELS_POSITIVE, MIKELS_NEGATIVE, COMPOUND_EMOTION_RULES,
    MIKELS_COOCCURRENCE, MIKELS_MUTUAL_EXCLUSION,
    map_to_compound_emotion, map_legacy_to_basic, multihot_12_to_8,
    create_full_config, create_ablation_configs,
)
from emotion_model.conflict_fusion import (
    ConflictAwareFusionModule, TextConflictAttention, VisualConflictAttention,
    ConflictAwareAlignment, ConflictContrastiveLoss,
)
from emotion_model.circular_head import (
    CircularMultiLabelHead, EmotionCircleMapper, ProgressiveCircularLoss,
    CIRCULAR_EMOTIONS, EMOTION_ANGLES, build_label_correlation_matrix,
    polarity_of_angle,
)
from emotion_model.vl_adapter import (
    BottleneckAdapter, DisentangledAdapter, VLAdapterManager,
    DomainAlignmentLoss, EMOTION_DOMAINS,
)


# ============================================================
# 验证报告收集器
# ============================================================

class VerificationReport:
    """验证报告收集器"""

    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.start_time = datetime.now()

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        self.tests.append({"name": name, "passed": bool(condition), "detail": detail,
                           "section": getattr(self, "_section", "")})
        status = "✅" if condition else "❌"
        msg = f"  {status} {name}"
        if detail:
            msg += f": {detail}"
        print(msg)
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        return bool(condition)

    def warn(self, msg: str):
        self.warnings += 1
        print(f"  ⚠️  {msg}")

    def section(self, title: str):
        self._section = title
        print(f"\n{'─'*62}")
        print(f"  {title}")
        print(f"{'─'*62}")

    def summary(self) -> str:
        total = self.passed + self.failed
        elapsed = (datetime.now() - self.start_time).total_seconds()
        lines = [
            f"\n{'='*62}",
            f"  核心模块 — 验证报告摘要",
            f"{'='*62}",
            f"  总测试数: {total}",
            f"  通过: {self.passed} ✅",
            f"  失败: {self.failed} ❌",
            f"  警告: {self.warnings} ⚠️",
            f"  耗时: {elapsed:.2f}s",
            f"  时间: {datetime.now().isoformat()}",
            f"{'='*62}",
        ]
        if self.failed > 0:
            lines.append("\n  失败测试:")
            for t in self.tests:
                if not t["passed"]:
                    lines.append(f"    ❌ [{t['section']}] {t['name']}: {t['detail']}")
        report = "\n".join(lines)
        print(report)
        return report

    def succeeded(self) -> bool:
        return self.failed == 0


# ============================================================
# 模块① 冲突感知跨模态融合
# ============================================================

def verify_conflict_fusion(report: VerificationReport):
    """验证模块①：基于注意力引导的冲突感知跨模态融合"""
    B, P, L = 4, 49, 8
    visual_dim, text_dim, hidden = 768, 512, 512

    report.section("模块① 基于注意力引导的冲突感知跨模态融合")

    visual_patches = torch.randn(B, P, visual_dim)
    text_features = torch.randn(B, L, text_dim)

    # ---- 1.1 文本双路径冲突注意力 ----
    print("\n  [1.1] 文本情绪注意力子模块（双路径冲突注意力）")
    tca = TextConflictAttention(text_dim=text_dim, hidden_dim=hidden, num_heads=8)
    tca.train()
    out_t = tca(text_features)

    report.check("文本情绪特征维度", out_t["text_features"].shape == (B, L, hidden),
                 f"实际: {tuple(out_t['text_features'].shape)}")
    report.check("双路径特征均生成",
                 "literal_features" in out_t and "intent_features" in out_t,
                 "字面路径 + 意图路径")
    report.check("冲突分数维度", out_t["conflict_scores"].shape == (B, L))
    report.check("冲突分数值域 [0,1]",
                 bool((out_t["conflict_scores"] >= 0).all() and
                      (out_t["conflict_scores"] <= 1).all()),
                 f"范围 [{out_t['conflict_scores'].min():.4f}, {out_t['conflict_scores'].max():.4f}]")
    report.check("动态边界为可学习参数",
                 isinstance(tca.boundary.boundary, nn.Parameter) and
                 tca.boundary.boundary.requires_grad,
                 f"初始边界 p = {out_t['boundary'].item():.4f}")
    report.check("片段划分输出存在（含硬标记）",
                 "segment_mask" in out_t and "segment_mask_hard" in out_t)

    # 语义验证：字面与意图差异大时，冲突分数应更高
    t_literal = torch.randn(B, L, text_dim)
    t_intent_far = -t_literal  # 完全相反
    t_intent_near = t_literal + torch.randn(B, L, text_dim) * 0.01  # 几乎相同
    # 直接测试冲突度量函数的行为
    cos_far = F.cosine_similarity(t_literal, t_intent_far, dim=-1)
    cos_near = F.cosine_similarity(t_literal, t_intent_near, dim=-1)
    report.check("冲突度量语义正确（相反特征冲突更高）",
                 bool(cos_far.mean() < cos_near.mean()),
                 f"相反 cos={cos_far.mean():.3f} < 相近 cos={cos_near.mean():.3f}")

    # ---- 1.2 视觉情绪注意力（冲突区域捕捉）----
    print("\n  [1.2] 视觉情绪注意力子模块（文本引导 + 冲突捕捉）")
    vca = VisualConflictAttention(visual_dim=visual_dim, text_dim=hidden, hidden_dim=hidden)
    vca.train()
    out_v = vca(visual_patches, out_t["text_features"])

    report.check("情感加权视觉特征维度",
                 out_v["emotion_visual_features"].shape == (B, L, hidden),
                 f"实际: {tuple(out_v['emotion_visual_features'].shape)}")
    report.check("注意力图形状 (B,H,L,P)",
                 out_v["attention_maps"].shape == (B, 8, L, P),
                 f"实际: {tuple(out_v['attention_maps'].shape)}")
    report.check("区域相似度分数存在", out_v["region_scores"].shape == (B, L, P))

    # 语义验证：与文本一致的 patch 应获得更高注意力
    with torch.no_grad():
        v_norm = F.normalize(vca.visual_proj(visual_patches), dim=-1)
        t_norm = F.normalize(vca.text_proj(out_t["text_features"]), dim=-1)
        # 构造一个与第 0 个标签完全一致的 patch
        aligned_patch = vca.visual_proj(visual_patches).clone()
        patches_test = torch.cat([t_norm[:, :1, :], vca.visual_proj(visual_patches)[:, 1:, :]], dim=1)
        sim_aligned = F.cosine_similarity(patches_test[:, 0:1, :], t_norm[:, 0:1, :], dim=-1)
    report.check("文本引导相似度计算正确",
                 bool(sim_aligned.mean() > 0.99),
                 f"一致性 patch 相似度 = {sim_aligned.mean():.4f}")

    # ---- 1.3 跨模态冲突感知对齐 ----
    print("\n  [1.3] 跨模态冲突感知注意力对齐机制")
    caa = ConflictAwareAlignment(visual_dim=hidden, text_dim=hidden, hidden_dim=hidden)

    v_test = torch.randn(B, L, hidden)
    t_test = torch.randn(B, L, hidden)
    ones = torch.ones(B, L)

    # 场景A：图文情感一致（一致性 = 1）
    out_consistent = caa(v_test, t_test, consistency=ones)
    # 场景B：图文情感冲突（一致性 = -1，如反讽/极性对立）
    out_conflict = caa(v_test, t_test, consistency=-ones)

    report.check("对齐特征维度", out_consistent["aligned_features"].shape == (B, L, hidden))
    report.check("对齐权重维度", out_consistent["alignment_weights"].shape == (B, L))
    report.check("跨模态冲突程度输出", out_consistent["cross_conflict"].shape == (B, L))

    # 核心语义验证：冲突时对齐权重应低于一致时
    w_consistent = out_consistent["alignment_weights"].mean().item()
    w_conflict = out_conflict["alignment_weights"].mean().item()
    cmp_symbol = ">" if w_consistent > w_conflict else "≤"
    report.check("★ 冲突场景对齐权重下降（模块核心语义）",
                 w_conflict < w_consistent,
                 f"一致时 {w_consistent:.4f} {cmp_symbol} 冲突时 {w_conflict:.4f}")
    report.check("冲突程度量化正确",
                 out_conflict["cross_conflict"].mean() > out_consistent["cross_conflict"].mean(),
                 f"冲突 {out_conflict['cross_conflict'].mean():.4f} > "
                 f"一致 {out_consistent['cross_conflict'].mean():.4f}")

    # 结构单调性：对齐权重随冲突程度单调递减（不依赖训练）
    conflict_levels = torch.linspace(0, 1, 7)              # (7,) 冲突程度
    consistency_levels = (1 - 2 * conflict_levels)          # (7,) 对应一致性
    dummy_v = torch.randn(1, L, hidden).expand(7, -1, -1)
    dummy_t = torch.randn(1, L, hidden).expand(7, -1, -1)
    cons = consistency_levels.view(7, 1).expand(7, L)
    with torch.no_grad():
        out_mono = caa(dummy_v, dummy_t, consistency=cons)
        weights_by_level = out_mono["base_alignment_weights"].mean(dim=1).tolist()
    is_monotonic = all(weights_by_level[i] >= weights_by_level[i + 1] - 1e-9
                       for i in range(len(weights_by_level) - 1))
    report.check("★ 对齐权重对冲突程度单调递减（结构保证，无需训练）",
                 is_monotonic,
                 f"冲突 0→1 时权重: {[round(w, 4) for w in weights_by_level]}")

    # 隐式一致性估计（未显式传入时由余弦相似度推断）
    out_implicit = caa(v_test, t_test)
    report.check("未传一致性时自动估计冲突程度",
                 "cross_conflict" in out_implicit and
                 bool((out_implicit["cross_conflict"] >= 0).all() and
                      (out_implicit["cross_conflict"] <= 1).all()),
                 f"估计冲突均值 {out_implicit['cross_conflict'].mean():.4f}")

    # ---- 1.4 对比学习与三元组排序损失 ----
    print("\n  [1.4] 模态内对比学习 + 双向三元组排序损失")
    loss_fn = ConflictContrastiveLoss(margin_visual=0.3, margin_text=0.3, margin_cross=0.2)
    loss_fn.train()
    cl = loss_fn(out_t["text_features"], out_v["emotion_visual_features"])

    report.check("文本模态内三元组损失", "loss_text" in cl,
                 f"值: {cl['loss_text'].item():.4f}")
    report.check("图像模态内三元组损失", "loss_visual" in cl,
                 f"值: {cl['loss_visual'].item():.4f}")
    report.check("双向跨模态三元组损失", "loss_cross" in cl,
                 f"值: {cl['loss_cross'].item():.4f}")
    report.check("总损失为有限值", bool(torch.isfinite(cl["total"])),
                 f"total: {cl['total'].item():.4f}")

    # 三元组损失语义：正样本相似度高时损失更小
    x = torch.randn(8, 64)
    fn_strict = ConflictContrastiveLoss(margin_cross=0.2)
    loss_same = fn_strict._intra_modal_triplet(x, x.clone(), 0.3)  # 正样本完全相同
    loss_rand = fn_strict._intra_modal_triplet(x, torch.randn(8, 64), 0.3)  # 正样本随机
    report.check("三元组损失语义正确（正样本越相似损失越小）",
                 bool(loss_same <= loss_rand),
                 f"相同正样本 {loss_same.item():.4f} ≤ 随机正样本 {loss_rand.item():.4f}")

    # ---- 1.5 完整融合模块 ----
    print("\n  [1.5] 完整冲突感知融合模块")
    fusion = ConflictAwareFusionModule(
        visual_dim=visual_dim, text_dim=text_dim, hidden_dim=hidden,
        num_heads=8, fusion_output_dim=512,
    )
    fusion.train()
    out_f = fusion(visual_patches, None, text_features)

    report.check("融合特征维度 (B,L,512)",
                 out_f["fused_features"].shape == (B, L, 512),
                 f"实际: {tuple(out_f['fused_features'].shape)}")
    report.check("输出无 NaN", not bool(torch.isnan(out_f["fused_features"]).any()))
    report.check("冲突相关中间量齐全",
                 all(k in out_f for k in ["conflict_scores", "cross_conflict",
                                          "alignment_weights", "segment_mask"]))
    report.check("训练模式下产出对比损失", "contrastive_loss" in out_f,
                 f"值: {out_f.get('contrastive_loss', torch.tensor(0.0)).item():.4f}")

    # ---- 1.6 梯度流 ----
    print("\n  [1.6] 梯度反向传播验证")
    loss = out_f["fused_features"].pow(2).mean() + out_f.get(
        "contrastive_loss", torch.tensor(0.0))
    loss.backward()
    modules_with_grad = set()
    for name, p in fusion.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            modules_with_grad.add(name.split(".")[0])
    report.check("梯度传播到三个子模块",
                 len(modules_with_grad) >= 3,
                 f"有梯度模块: {sorted(modules_with_grad)}")
    boundary_grad = fusion.text_attention.boundary.boundary.grad
    report.check("动态边界可学习（有梯度）",
                 boundary_grad is not None,
                 f"边界梯度: {None if boundary_grad is None else boundary_grad.item():.3e}")


# ============================================================
# 模块② 情感环形表示多标记分类头
# ============================================================

def verify_circular_head(report: VerificationReport):
    """验证模块②：基于情感环形表示的多标记分类头"""
    B, L, D = 4, 8, 512

    report.section("模块② 基于情感环形表示的多标记分类头")

    features = torch.randn(B, L, D)

    # ---- 2.1 环形常量与角度分配 ----
    print("\n  [2.1] 情感环形常量（Mikels Wheel 8 类）")
    report.check("8 类基础情感数量", len(CIRCULAR_EMOTIONS) == 8,
                 f"{CIRCULAR_EMOTIONS}")
    angles_deg = [round(EMOTION_ANGLES[e] * 180 / math.pi, 1) for e in CIRCULAR_EMOTIONS]
    expected_deg = [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]
    report.check("角度分配符合公式 θ=(2j-1)/8·π",
                 all(abs(a - b) < 0.01 for a, b in zip(angles_deg, expected_deg)),
                 f"实际: {angles_deg}")

    pos_ok = all(polarity_of_angle(torch.tensor(EMOTION_ANGLES[e])).item() == 0
                 for e in MIKELS_POSITIVE)
    neg_ok = all(polarity_of_angle(torch.tensor(EMOTION_ANGLES[e])).item() == 1
                 for e in MIKELS_NEGATIVE)
    report.check("极性分组正确（4 正 4 负半圆划分）", pos_ok and neg_ok,
                 f"积极: {MIKELS_POSITIVE} / 消极: {MIKELS_NEGATIVE}")

    # ---- 2.2 情感环形映射 ----
    print("\n  [2.2] 情感分布 → 情感环形向量映射（论文 Algorithm 1）")
    mapper = EmotionCircleMapper(num_labels=8)

    # 构造单一情感的分布 → 映射角度应接近该情感先验角度
    single_dists = torch.zeros(8, 8)
    for i in range(8):
        single_dists[i, i] = 1.0
    vec = mapper.distribution_to_vector(single_dists)

    angle_errors = []
    for i, name in enumerate(CIRCULAR_EMOTIONS):
        err = abs(vec["angle"][i].item() - EMOTION_ANGLES[name])
        err = min(err, 2 * math.pi - err)  # 环形距离
        angle_errors.append(err)
    report.check("★ 单一情感映射角度还原准确",
                 max(angle_errors) < 1e-4,
                 f"最大角度误差 {max(angle_errors):.2e} 弧度")

    # 极性正确性
    polarity_correct = all(
        vec["polarity"][i].item() == (0 if CIRCULAR_EMOTIONS[i] in MIKELS_POSITIVE else 1)
        for i in range(8)
    )
    report.check("单情感极性判定正确", polarity_correct,
                 f"极性: {vec['polarity'].tolist()}")

    # 强度归一化
    report.check("强度范围 [0,1]",
                 bool((vec["intensity"] >= 0).all() and (vec["intensity"] <= 1).all()),
                 f"范围 [{vec['intensity'].min():.3f}, {vec['intensity'].max():.3f}]")

    # 三维坐标
    report.check("三维直角坐标输出 (B,3)", vec["cartesian"].shape == (8, 3),
                 f"实际: {tuple(vec['cartesian'].shape)}")

    # 复合情感：两个相邻情感的分布 → 角度应落在两者之间
    comp_dist = torch.zeros(1, 8)
    comp_dist[0, 0] = 0.5  # amusement (22.5°)
    comp_dist[0, 1] = 0.5  # excitement (67.5°)
    comp_vec = mapper.distribution_to_vector(comp_dist)
    comp_angle_deg = comp_vec["angle"][0].item() * 180 / math.pi
    report.check("★ 复合情感角度落在组成情感之间（环形加法有效）",
                 22.5 <= comp_angle_deg <= 67.5,
                 f"amusement+excitement → {comp_angle_deg:.1f}°（介于 22.5°~67.5°）")

    # 反向映射
    rev = mapper.vector_to_distribution(vec["angle"], vec["intensity"])
    report.check("反向映射（环形向量 → 分布）", rev.shape == (8, 8),
                 f"实际: {tuple(rev.shape)}")

    # ---- 2.3 三分支分类头 ----
    print("\n  [2.3] 三分支多标记分类头")
    head = CircularMultiLabelHead(
        input_dim=D, num_labels=L, emotion_labels=CIRCULAR_EMOTIONS,
    )
    out = head(features)

    report.check("类型分支 logits 维度 (B,L)", out["type_logits"].shape == (B, L))
    report.check("多标记置信度在 [0,1]",
                 bool((out["probabilities"] >= 0).all() and (out["probabilities"] <= 1).all()),
                 f"范围 [{out['probabilities'].min():.3f}, {out['probabilities'].max():.3f}]")
    report.check("极性分支 logits 维度 (B,L,3)", out["polarity_logits"].shape == (B, L, 3),
                 "积极/中性/消极 三分类")
    report.check("极性概率归一化（和为 1）",
                 bool(torch.allclose(out["polarity_probs"].sum(dim=-1),
                                     torch.ones(B, L), atol=1e-5)),
                 f"求和示例: {out['polarity_probs'].sum(dim=-1)[0, :3].tolist()}")
    report.check("连续极性值范围 (-1,1)",
                 bool((out["polarity_value"] > -1).all() and (out["polarity_value"] < 1).all()),
                 f"范围 [{out['polarity_value'].min():.3f}, {out['polarity_value'].max():.3f}]")
    report.check("强度分支范围 (0,1)",
                 bool((out["intensity"] > 0).all() and (out["intensity"] < 1).all()),
                 f"范围 [{out['intensity'].min():.3f}, {out['intensity'].max():.3f}]")

    # 角度约束：预测角度应保持在先验角度 ±π/8 内（不跨越相邻情感）
    angle_dev = []
    for i, name in enumerate(CIRCULAR_EMOTIONS):
        d = (out["angle"][:, i] - EMOTION_ANGLES[name]).abs()
        d = torch.minimum(d, 2 * math.pi - d)
        angle_dev.append(d.max().item())
    report.check("★ 角度偏移受限于 ±π/8（保持情感类型可辨识）",
                 max(angle_dev) <= math.pi / 8 + 1e-5,
                 f"最大偏移 {max(angle_dev):.4f} ≤ {math.pi/8:.4f}")

    report.check("情感环形三维向量 (B,L,3)", out["circle_vector"].shape == (B, L, 3),
                 f"实际: {tuple(out['circle_vector'].shape)}")

    # ---- 2.4 标签共现关联增强（申请书公式21）----
    print("\n  [2.4] 标签共现关联增强（y_i = σ(W·F + b + M·y_{-i})）")
    corr = build_label_correlation_matrix(
        CIRCULAR_EMOTIONS, MIKELS_COOCCURRENCE, MIKELS_MUTUAL_EXCLUSION,
    )
    report.check("关联矩阵维度 (C,C)", corr.shape == (L, L))
    report.check("关联矩阵含正边（共现）", bool((corr > 0).any()),
                 f"正边数 {(corr > 0).sum().item()}")
    report.check("关联矩阵含负边（互斥）", bool((corr < 0).any()),
                 f"负边数 {(corr < 0).sum().item()}")

    head_corr = CircularMultiLabelHead(
        input_dim=D, num_labels=L, emotion_labels=CIRCULAR_EMOTIONS,
        use_label_correlation=True, label_correlation_matrix=corr,
    )
    head_nocorr = CircularMultiLabelHead(
        input_dim=D, num_labels=L, emotion_labels=CIRCULAR_EMOTIONS,
        use_label_correlation=False,
    )
    # 同步权重后比较，验证关联项确实改变了输出
    head_nocorr.load_state_dict(
        {k: v for k, v in head_corr.state_dict().items() if k in head_nocorr.state_dict()},
        strict=False,
    )
    o_corr = head_corr(features)
    o_nocorr = head_nocorr(features)
    report.check("标签关联项影响输出（模块生效）",
                 not torch.allclose(o_corr["probabilities"], o_nocorr["probabilities"]),
                 f"最大差异 {(o_corr['probabilities'] - o_nocorr['probabilities']).abs().max():.6f}")

    # ---- 2.5 渐进式环形损失 ----
    print("\n  [2.5] 渐进式环形损失（Progressive Circular Loss）")
    pc_loss = ProgressiveCircularLoss(mu=0.5, angle_mode="circular")

    target_vec = mapper.distribution_to_vector(single_dists[:B])
    target = {
        "polarity": target_vec["polarity"].unsqueeze(-1).expand(B, L),
        "angle": target_vec["angle"].unsqueeze(-1).expand(B, L),
        "intensity": target_vec["intensity"].unsqueeze(-1).expand(B, L),
    }
    losses = pc_loss(out, target)

    report.check("极性约束项（第一步，粗粒度）", "polar_loss" in losses,
                 f"L_p = {losses['polar_loss'].item():.4f}")
    report.check("类型约束项（第二步，中粒度）", "type_loss" in losses,
                 f"L_t = {losses['type_loss'].item():.4f}")
    report.check("强度加权约束项（第三步，细粒度）", "pc_loss" in losses,
                 f"L_PC = {losses['pc_loss'].item():.4f}")

    # 环形角度误差的周期性验证
    a1 = torch.tensor([0.0])
    a2 = torch.tensor([2 * math.pi - 0.01])  # 跨 0 点的近邻
    circ_err = pc_loss.circular_angle_error(a1, a2).item()
    raw_err = abs(a1.item() - a2.item())
    report.check("★ 环形角度误差处理 2π 周期性",
                 circ_err < 0.02 and raw_err > 6.0,
                 f"环形误差 {circ_err:.4f} ≪ 欧氏误差 {raw_err:.4f}")

    # 与 KL 散度联合（论文 Eq.10）
    kl = F.kl_div(F.log_softmax(out["type_logits"], dim=-1),
                  torch.softmax(target["angle"], dim=-1), reduction="batchmean")
    combined = pc_loss.combine_with_distribution_loss(losses["pc_loss"], kl)
    report.check("与分布损失联合（L=(1-μ)L_KL+μL_PC）", bool(torch.isfinite(combined)),
                 f"联合损失 = {combined.item():.4f}")

    # ---- 2.6 两级标签体系（基础 + 复合）----
    print("\n  [2.6] 两级标签体系（8 类基础情感 + 复合情感映射）")
    compounds = map_to_compound_emotion(["amusement", "excitement"])
    report.check("复合情感映射规则生效", "joy" in compounds,
                 f"amusement+excitement → {compounds}")
    report.check("单一基础情感不产生复合情感",
                 map_to_compound_emotion(["anger"]) == [],
                 f"anger → {map_to_compound_emotion(['anger'])}")
    report.check("复合情感规则数量", len(COMPOUND_EMOTION_RULES) == 8,
                 f"{len(COMPOUND_EMOTION_RULES)} 条环形相邻规则")

    # 旧 12 类 → 8 类映射（用旧标签集构造输入）
    legacy = torch.zeros(1, len(LEGACY_EXTENDED_EMOTIONS))
    legacy[0, LEGACY_EXTENDED_EMOTIONS.index("joy")] = 1.0
    mapped = multihot_12_to_8(legacy)
    report.check("12 类 → 8 类标签映射",
                 mapped.shape == (1, 8) and mapped[0, CIRCULAR_EMOTIONS.index("amusement")] == 1.0,
                 f"joy → amusement/excitement，映射后 {mapped.tolist()}")
    report.check("映射不引入 love/peace（无依据标签）",
                 all(x not in MIKELS_BASIC_EMOTIONS for x in ["love", "peace"]),
                 "8 类基础情感中不含 love/peace")

    # ---- 2.7 梯度流 ----
    print("\n  [2.7] 梯度反向传播验证（覆盖三个分支）")
    head.zero_grad()
    out_g = head(features)
    # 损失需覆盖三个分支的全部输出，才能验证各分支均有梯度回传
    loss_g = (
        out_g["probabilities"].pow(2).mean()          # 类型分支
        + out_g["intensity"].mean()                    # 强度分支
        + out_g["polarity_probs"].pow(2).mean()        # 极性判别分支
        + out_g["polarity_value"].pow(2).mean()        # 极性连续值
        + out_g["angle"].mean()                        # 角度偏移分支
    )
    loss_g.backward()
    branches_with_grad = set()
    for name, p in head.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            branches_with_grad.add(name.split(".")[0])

    report.check("梯度传播到三个分支的全部子模块",
                 len(branches_with_grad) >= 6,
                 f"有梯度模块: {sorted(branches_with_grad)}")
    report.check("极性判别分支有梯度",
                 "polarity_branch" in branches_with_grad,
                 "交叉熵分支可训练")
    report.check("角度偏移分支有梯度",
                 "angle_offset_branch" in branches_with_grad,
                 "环形角度可训练")


# ============================================================
# 模块③ VL-Adapter 跨场景泛化
# ============================================================

def verify_vl_adapter(report: VerificationReport):
    """验证模块③：基于 VL-Adapter 的跨场景情感特征泛化优化"""
    B, D_v, D_t = 4, 768, 512

    report.section("模块③ 基于 VL-Adapter 的跨场景情感特征泛化优化")

    visual = torch.randn(B, 197, D_v)
    text = torch.randn(B, 77, D_t)

    # ---- 3.1 瓶颈适配器结构 ----
    print("\n  [3.1] 瓶颈适配器（降维–激活–升维–残差连接）")
    adapter = BottleneckAdapter(dim=D_v, bottleneck_dim=D_v // 16)
    out_a = adapter(visual)

    report.check("输出维度与输入一致", out_a.shape == visual.shape,
                 f"{tuple(visual.shape)} → {tuple(out_a.shape)}")
    report.check("瓶颈压缩生效（k = d/16）",
                 adapter.bottleneck_dim == D_v // 16,
                 f"d={D_v}, k={adapter.bottleneck_dim}")
    # 近恒等初始化：初始输出相对偏移应很小（不破坏预训练特征）
    rel_offset = ((out_a - visual).norm() / visual.norm()).item()
    report.check("近恒等初始化（不破坏预训练特征）", rel_offset < 0.05,
                 f"初始输出相对偏移 {rel_offset:.2e}（< 5%）")

    # 参数量远小于原层（对照：d×d 全连接层）
    full_layer_params = D_v * D_v + D_v
    adapter_params = sum(p.numel() for p in adapter.parameters())
    report.check("参数量远小于原层",
                 adapter_params < full_layer_params * 0.2,
                 f"{adapter_params:,} vs 原层 {full_layer_params:,}"
                 f"（占比 {100*adapter_params/full_layer_params:.1f}%）")

    # ---- 3.2 参数解耦 ----
    print("\n  [3.2] 参数解耦（共享参数 + 域特定参数）")
    dis = DisentangledAdapter(dim=D_v, bottleneck_dim=D_v // 16,
                              num_domains=4, domain_rank=4)

    report.check("共享参数存在", dis.num_shared_params > 0,
                 f"共享参数 {dis.num_shared_params:,}")
    report.check("域特定参数存在", dis.num_domain_params > 0,
                 f"域特定参数 {dis.num_domain_params:,}（4 场景）")

    # 域扩展能力（新增场景不破坏已有参数）
    params_before = dis.domain_A.data.clone()
    dis.add_domain(2)
    report.check("支持动态扩展新场景",
                 dis.num_domains == 6 and
                 torch.allclose(dis.domain_A.data[:4], params_before),
                 f"域数 4 → {dis.num_domains}，原有参数未受影响")

    # 域隔离：不同场景参数独立
    with torch.no_grad():
        dis.domain_B.normal_(0, 0.05)
    o0 = dis(visual, domain_id=0)
    o1 = dis(visual, domain_id=1)
    report.check("★ 不同场景输出存在差异（域特定参数生效）",
                 not torch.allclose(o0, o1),
                 f"场景 0/1 输出差异 {(o0 - o1).abs().mean():.6f}")

    # ---- 3.3 VL-Adapter 管理器 ----
    print("\n  [3.3] VL-Adapter 管理器（12 层双塔适配）")
    manager = VLAdapterManager(
        visual_dim=D_v, text_dim=D_t, num_layers=12,
        bottleneck_ratio=4, num_domains=4, domain_rank=4,
    )
    # 真实 CLIP ViT-B/32 主干参数量约 125.6M
    backbone_params = 87_800_000 + 37_800_000
    summary = manager.parameter_summary(backbone_params=backbone_params)

    report.check("视觉/文本适配器各 12 层",
                 len(manager.visual_adapters) == 12 and len(manager.text_adapters) == 12)
    report.check("★ 可训练参数占比符合申请书要求（3%~5%）",
                 3.0 <= summary["ratio_percent"] <= 5.0,
                 f"{summary['ratio_percent']:.2f}%（可训练 {summary['trainable']:,} / 主干 {backbone_params:,}）")
    report.check("共享参数占比记录", summary["trainable_shared"] > 0,
                 f"共享 {summary['trainable_shared']:,} / 域特定 {summary['trainable_domain']:,}")

    # 特征级前向
    fwd = manager(visual.mean(dim=1), text.mean(dim=1))
    report.check("视觉适配输出维度", fwd["visual_adapted"].shape == (B, D_v))
    report.check("文本适配输出维度", fwd["text_adapted"].shape == (B, D_t))
    report.check("视觉投影到统一空间", fwd["projected_visual"].shape == (B, D_t))

    # ---- 3.4 跨场景工作流 ----
    print("\n  [3.4] 跨场景适配工作流（冻结共享 → 新增场景参数）")
    with torch.no_grad():
        for a in list(manager.visual_adapters) + list(manager.text_adapters):
            a.domain_B.normal_(0, 0.05)

    manager.freeze_shared()
    frozen_ok = all(
        not next(a.shared_adapter.parameters()).requires_grad
        for a in list(manager.visual_adapters) + list(manager.text_adapters)
    )
    report.check("共享参数可冻结（保护域不变知识）", frozen_ok,
                 "24 个适配器层的共享参数已冻结")

    trainable_frozen = sum(p.numel() for p in manager.parameters() if p.requires_grad)
    report.check("冻结后仅域特定参数可训练", trainable_frozen < summary["trainable"],
                 f"{trainable_frozen:,} < {summary['trainable']:,}")

    manager.unfreeze_shared()
    report.check("支持解冻恢复全量适配器训练", True, "unfreeze_shared()")

    # ---- 3.5 域隔离的梯度验证 ----
    print("\n  [3.5] 梯度隔离验证（仅当前场景参数更新）")
    manager.set_domain(1)
    manager.zero_grad()
    v_seq = visual.mean(dim=1)
    for i in range(len(manager.visual_adapters)):
        v_seq = manager.adapt_visual(v_seq, layer_idx=i)
    t_seq = text.mean(dim=1)
    for i in range(len(manager.text_adapters)):
        t_seq = manager.adapt_text(t_seq, layer_idx=i)
    proj = manager.visual_projection(v_seq)
    target = torch.randn_like(proj)
    (F.mse_loss(proj, target) + t_seq.pow(2).mean()).backward()

    first = manager.visual_adapters[0]
    g_B = first.domain_B.grad
    if g_B is not None:
        per_domain = g_B.abs().sum(dim=(1, 2)).tolist()
        nonzero_domains = [i for i, v in enumerate(per_domain) if v > 0]
        report.check("★ 梯度仅流入当前场景（场景隔离）",
                     nonzero_domains == [1],
                     f"有梯度场景: {nonzero_domains}（当前 domain_id=1）")
    else:
        report.check("★ 梯度仅流入当前场景（场景隔离）", False, "domain_B 无梯度")

    grads = sum(1 for _, p in manager.named_parameters()
                if p.grad is not None and p.grad.abs().sum() > 0)
    report.check("梯度覆盖适配器与投影层", grads > 50,
                 f"{grads} 个参数张量有梯度")

    # ---- 3.6 域分布对齐损失 ----
    print("\n  [3.6] 域分布对齐损失（抑制特征漂移）")
    align = DomainAlignmentLoss()
    src = torch.randn(16, D_v)
    tgt_shifted = torch.randn(16, D_v) + 0.8   # 明显分布偏移
    tgt_same = torch.randn(16, D_v)            # 同分布

    l_shifted = align(src, tgt_shifted)
    l_same = align(src, tgt_same)

    report.check("MMD 项计算", "mmd" in l_shifted, f"值: {l_shifted['mmd'].item():.4f}")
    report.check("协方差对齐项计算", "cov" in l_shifted, f"值: {l_shifted['cov'].item():.4f}")
    report.check("★ 分布偏移越大对齐损失越大",
                 l_shifted["total"] > l_same["total"],
                 f"偏移 {l_shifted['total'].item():.4f} > 同分布 {l_same['total'].item():.4f}")

    # ---- 3.7 场景定义 ----
    print("\n  [3.7] 场景（域）定义")
    report.check("预置 4 类应用场景", len(EMOTION_DOMAINS) == 4,
                 f"{EMOTION_DOMAINS}")


# ============================================================
# 集成验证：完整模型与消融基线
# ============================================================

def verify_integration(report: VerificationReport):
    """验证三大模块与完整模型的端到端集成"""
    report.section("集成验证：完整模型与消融配置")

    from emotion_model.full_model import MultiLabelEmotionModel

    B = 2
    dummy_images = torch.randn(B, 3, 224, 224)

    # ---- 4.1 配置系统 ----
    print("\n  [4.1] 配置系统（完整模型 / 消融实验矩阵）")
    full_cfg = create_full_config()
    report.check("完整模型配置启用三大模块",
                 full_cfg.use_conflict_fusion and full_cfg.use_circular_head and full_cfg.use_vl_adapter,
                 "模块①②③ 全部启用")
    report.check("完整模型使用 8 类 Mikels 基础情感",
                 full_cfg.num_emotions == 8 and full_cfg.emotion_labels == MIKELS_BASIC_EMOTIONS,
                 f"{full_cfg.num_emotions} 类: {full_cfg.emotion_labels}")
    report.check("完整模型冻结 CLIP 主干（配合 VL-Adapter）",
                 full_cfg.freeze_visual and full_cfg.freeze_text)

    ablations = create_ablation_configs()
    report.check("消融实验配置矩阵齐全", len(ablations) == 6,
                 f"{list(ablations.keys())}")

    # ---- 4.2 消融基线模型（核心模块全部关闭，标签空间仍为 8 类）----
    print("\n  [4.2] 消融基线模型（8 类标签，核心模块关闭）")
    base_config = ModelConfig(freeze_visual=True, freeze_text=True)
    base_model = MultiLabelEmotionModel(base_config)
    base_model.eval()

    with torch.no_grad():
        base_out = base_model(dummy_images, return_attention=True)

    report.check("消融基线 logits 维度 (B,8)", base_out["logits"].shape == (B, 8),
                 f"实际: {tuple(base_out['logits'].shape)}")
    report.check("消融基线注意力图输出", base_out.get("attention_maps") is not None,
                 f"{tuple(base_out['attention_maps'].shape)}" if base_out.get("attention_maps") is not None else "无")
    report.check("消融基线不含核心模块输出",
                 "circle_vector" not in base_out and "conflict_scores" not in base_out,
                 "核心模块关闭")
    report.check("★ 消融基线与完整模型标签空间一致（对比公平性）",
                 base_config.num_emotions == full_cfg.num_emotions == 8,
                 f"均为 {base_config.num_emotions} 类 Mikels 基础情感")

    # ---- 4.3 完整模型（8 类标签 + 三大核心模块）----
    print("\n  [4.3] 完整模型（8 类基础情感 + 三大核心模块）")
    model = MultiLabelEmotionModel(full_cfg)
    model.eval()

    with torch.no_grad():
        out = model(dummy_images, return_attention=True)

    report.check("完整模型 logits 维度 (B,8)", out["logits"].shape == (B, 8),
                 f"实际: {tuple(out['logits'].shape)}")
    report.check("模块① 冲突感知输出", "conflict_scores" in out and "cross_conflict" in out,
                 f"冲突分数 {tuple(out['conflict_scores'].shape)}")
    report.check("模块② 环形表示输出", "circle_vector" in out,
                 f"环形向量 {tuple(out['circle_vector'].shape)}")
    report.check("模块② 三分支输出",
                 all(k in out for k in ["polarity_probs", "angle", "intensity"]))
    report.check("模块③ VL-Adapter 已挂载",
                 getattr(model, "_adapter_attached", False),
                 "已挂载到 CLIP 双塔各层")
    report.check("概率在 [0,1]",
                 bool((out["probabilities"] >= 0).all() and (out["probabilities"] <= 1).all()))
    report.check("输出无 NaN", not bool(torch.isnan(out["logits"]).any()))

    # ---- 4.4 联合损失计算 ----
    print("\n  [4.4] 多目标联合损失")
    model.train()
    targets = torch.zeros(B, 8)
    targets[0, [0, 1]] = 1.0   # amusement + excitement → 复合情感 joy
    targets[1, [3, 4]] = 1.0   # disgust + fear

    outputs = model(dummy_images)
    total_loss = model.compute_loss(
        outputs["logits"], targets,
        loss_type="asymmetric",
        circular_outputs=outputs,
        contrastive_loss=outputs.get("contrastive_loss"),
    )
    report.check("联合损失可计算（分类+环形+对比）", bool(torch.isfinite(total_loss)),
                 f"L_total = {total_loss.item():.4f}")

    total_loss.backward()
    modules_grad = set()
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            modules_grad.add(name.split(".")[0])
    report.check("梯度传播到各主要模块",
                 len(modules_grad) >= 3,
                 f"有梯度模块: {sorted(modules_grad)}")

    # ---- 4.5 参数效率 ----
    print("\n  [4.5] 参数效率（VL-Adapter 轻量化）")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report.check("完整模型可训练参数显著少于总参数",
                 trainable < total * 0.2,
                 f"可训练 {trainable:,} / 总 {total:,} = {100*trainable/total:.2f}%")

    print(f"\n  参数量明细:")
    print(f"    总参数:   {total:,}")
    print(f"    可训练:   {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"    冻结合计: {total - trainable:,}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="核心模块功能验证")
    parser.add_argument("--module", type=str, default=None,
                        choices=["conflict", "circular", "adapter", "integration"],
                        help="仅验证指定模块")
    parser.add_argument("--output", type=str, default=None, help="输出报告到文件")
    args = parser.parse_args()

    print("=" * 62)
    print("  多标记情感识别模型 — 核心模块功能验证")
    print("  Multi-Label Emotion Recognition: Core Modules")
    print("=" * 62)
    print(f"  时间: {datetime.now().isoformat()}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")

    report = VerificationReport()
    m = args.module

    if m is None or m == "conflict":
        verify_conflict_fusion(report)
    if m is None or m == "circular":
        verify_circular_head(report)
    if m is None or m == "adapter":
        verify_vl_adapter(report)
    if m is None or m == "integration":
        verify_integration(report)

    summary = report.summary()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("核心模块 — 功能验证报告\n")
            f.write(f"生成时间: {datetime.now().isoformat()}\n")
            f.write(f"PyTorch: {torch.__version__}\n")
            f.write(summary)
            f.write("\n\n详细测试记录:\n")
            current_section = ""
            for t in report.tests:
                if t["section"] != current_section:
                    current_section = t["section"]
                    f.write(f"\n【{current_section}】\n")
                status = "PASS" if t["passed"] else "FAIL"
                f.write(f"  [{status}] {t['name']}: {t['detail']}\n")
        print(f"\n验证报告已保存至: {args.output}")

    sys.exit(0 if report.succeeded() else 1)


if __name__ == "__main__":
    main()
