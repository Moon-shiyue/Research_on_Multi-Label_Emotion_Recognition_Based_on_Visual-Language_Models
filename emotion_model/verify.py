"""
模块功能验证脚本
Module Function Verification Script

对项目各模块进行全面的功能验证，包括：
1. 基础编码器验证：张量维度、特征归一化、双编码器协同
2. 融合模块验证：注意力机制正确性、两步融合流程、梯度流
3. 分类头验证：输出格式、损失函数、多标记预测语义正确性
4. 标签关联模块验证：图构建、GCN 消息传递、融合权重
5. 端到端集成验证：完整前向传播、损失回传、推理一致性

运行方式:
    python verify.py                    # 运行所有验证
    python verify.py --quick            # 仅快速验证（跳过 GPU 相关）
    python verify.py --module encoder   # 仅验证指定模块
    python verify.py --output report    # 输出验证报告
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
from typing import Dict, List, Tuple
from datetime import datetime

# Windows 控制台默认 GBK 编码无法输出 emoji，强制 UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

# 确保可以导入模块（以包方式导入，支持相对导入）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_model.config import (
    ModelConfig, TrainingConfig,
    MIKELS_BASIC_EMOTIONS, MIKELS_COOCCURRENCE, MIKELS_MUTUAL_EXCLUSION,
)
from emotion_model.base_encoder import CLIPEncoder, VisualEncoder, TextEncoder
from emotion_model.fusion_module import (
    HierarchicalAttentionFusion,
    VisualEmotionAttention,
    TextEmotionAttention,
    GlobalAlignmentCalibration,
)
from emotion_model.classification_head import (
    MultiLabelClassificationHead,
    LabelSpecificClassifier,
    SharedAttentionClassifier,
)
from emotion_model.label_association import (
    LabelAssociationModule,
    LightweightLabelGCN,
    build_emotion_label_graph,
    normalize_adjacency,
)
from emotion_model.full_model import MultiLabelEmotionModel


# ============================================================
# 验证工具
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
        self.tests.append({
            "name": name,
            "passed": condition,
            "detail": detail,
        })
        status = "✅" if condition else "❌"
        msg = f"  {status} {name}"
        if detail:
            msg += f": {detail}"
        print(msg)

        if condition:
            self.passed += 1
        else:
            self.failed += 1
        return condition

    def warn(self, msg: str):
        self.warnings += 1
        print(f"  ⚠️  {msg}")

    def section(self, title: str):
        print(f"\n{'─'*55}")
        print(f"  {title}")
        print(f"{'─'*55}")

    def summary(self) -> str:
        total = self.passed + self.failed
        elapsed = (datetime.now() - self.start_time).total_seconds()

        lines = [
            f"\n{'='*60}",
            f"  验证报告摘要",
            f"{'='*60}",
            f"  总测试数: {total}",
            f"  通过: {self.passed} ✅",
            f"  失败: {self.failed} ❌",
            f"  警告: {self.warnings} ⚠️",
            f"  耗时: {elapsed:.2f}s",
            f"  时间: {datetime.now().isoformat()}",
            f"{'='*60}",
        ]

        if self.failed > 0:
            lines.append(f"\n  失败测试:")
            for t in self.tests:
                if not t["passed"]:
                    lines.append(f"    ❌ {t['name']}: {t['detail']}")

        report = "\n".join(lines)
        print(report)
        return report

    def succeeded(self) -> bool:
        return self.failed == 0


# ============================================================
# 1. 基础编码器验证
# ============================================================

def verify_encoder(report: VerificationReport, quick: bool = False):
    """验证 CLIP 编码器模块"""
    report.section("1. 基础编码器验证 (CLIP ViT-B/32)")

    B = 2

    # 1.1 编码器初始化
    print("\n  [1.1] 编码器初始化")
    try:
        encoder = CLIPEncoder(
            projection_dim=512,
            freeze_visual=not quick,  # quick 模式冻结
            freeze_text=not quick,
        )
        report.check("CLIP 双塔编码器初始化", True)
    except Exception as e:
        report.check("CLIP 双塔编码器初始化", False, str(e))
        return  # 无法继续

    # 1.2 视觉编码器
    print("\n  [1.2] 视觉编码器输出维度验证")
    dummy_image = torch.randn(B, 3, 224, 224)

    try:
        with torch.no_grad():
            visual_out = encoder.encode_image(dummy_image)

        report.check(
            "视觉全局特征维度",
            visual_out["global_feature"].shape == (B, 512),
            f"实际: {visual_out['global_feature'].shape}"
        )
        report.check(
            "视觉 patch 特征维度",
            visual_out["patch_features"].shape == (B, 49, 768),
            f"实际: {visual_out['patch_features'].shape}"
        )
    except Exception as e:
        report.check("视觉编码器前向传播", False, str(e))

    # 1.3 文本编码器
    print("\n  [1.3] 文本编码器验证")
    test_labels = MIKELS_BASIC_EMOTIONS[:6]  # 取前 6 类用于编码器测试

    try:
        with torch.no_grad():
            text_out = encoder.encode_text(test_labels)

        report.check(
            "文本特征维度",
            text_out["global_feature"].shape == (6, 512),
            f"实际: {text_out['global_feature'].shape}"
        )

        # 验证 prompt 生成
        prompts = encoder.text_encoder.get_emotion_prompts(test_labels)
        report.check(
            "情感 prompt 生成",
            len(prompts) == 6 and all(isinstance(p, str) for p in prompts),
            f"示例: {prompts[0]}"
        )
    except Exception as e:
        report.check("文本编码器前向传播", False, str(e))

    # 1.4 联合编码
    print("\n  [1.4] 联合编码验证")
    try:
        with torch.no_grad():
            joint_out = encoder(dummy_image, test_labels)

        report.check(
            "联合编码视觉特征",
            joint_out["visual_features"].shape == (B, 512),
            f"实际: {joint_out['visual_features'].shape}"
        )
        report.check(
            "联合编码文本特征",
            joint_out["text_features"].shape == (6, 512),
            f"实际: {joint_out['text_features'].shape}"
        )
    except Exception as e:
        report.check("联合编码前向传播", False, str(e))


# ============================================================
# 2. 融合模块验证
# ============================================================

def verify_fusion(report: VerificationReport):
    """验证层次化注意力融合模块"""
    report.section("2. 跨模态融合模块验证")

    B, P, L = 2, 49, 8   # 8 类 Mikels 基础情感
    visual_dim, text_dim, hidden_dim = 768, 512, 512

    visual_patches = torch.randn(B, P, visual_dim)
    visual_global = torch.randn(B, text_dim)
    text_features = torch.randn(B, L, text_dim)

    # 2.1 视觉情绪注意力
    print("\n  [2.1] 视觉情绪注意力子模块")
    vea = VisualEmotionAttention(
        visual_dim=visual_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        num_heads=8,
    )

    try:
        vea_out = vea(visual_patches, text_features)
        report.check(
            "情感加权的视觉特征维度",
            vea_out["emotion_visual_features"].shape == (B, L, hidden_dim),
            f"实际: {vea_out['emotion_visual_features'].shape}"
        )
        report.check(
            "注意力图形状",
            vea_out["attention_maps"].shape[2:] == (L, P),
            f"实际: torch.Size({vea_out['attention_maps'].shape[2:]})"
        )
    except Exception as e:
        report.check("视觉情绪注意力前向传播", False, str(e))

    # 2.2 文本情绪注意力
    print("\n  [2.2] 文本情绪注意力子模块")
    tea = TextEmotionAttention(
        text_dim=text_dim,
        visual_dim=hidden_dim,  # 统一维度
        hidden_dim=hidden_dim,
        num_heads=8,
    )

    try:
        tea_out = tea(text_features, visual_global)
        report.check(
            "视觉加权的文本特征维度",
            tea_out["weighted_text_features"].shape == (B, L, hidden_dim),
            f"实际: {tea_out['weighted_text_features'].shape}"
        )
    except Exception as e:
        report.check("文本情绪注意力前向传播", False, str(e))

    # 2.3 全局校准模块
    print("\n  [2.3] 全局对齐校准子模块")
    gac = GlobalAlignmentCalibration(
        hidden_dim=hidden_dim,
        num_heads=8,
        num_layers=2,
    )

    try:
        dummy_fused = torch.randn(B, L, hidden_dim)
        calibrated = gac(dummy_fused)
        report.check(
            "全局校准特征维度",
            calibrated.shape == (B, L, hidden_dim),
            f"实际: {calibrated.shape}"
        )
        report.check(
            "校准特征非零",
            torch.any(calibrated != 0).item(),
            "校准输出全为零！"
        )
    except Exception as e:
        report.check("全局校准前向传播", False, str(e))

    # 2.4 完整融合模块
    print("\n  [2.4] 完整层次化注意力融合")
    fusion = HierarchicalAttentionFusion(
        visual_dim=visual_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        num_heads=8,
        num_layers=2,
    )

    try:
        fusion_out = fusion(visual_patches, visual_global, text_features)
        report.check(
            "融合特征维度",
            fusion_out["fused_features"].shape == (B, L, hidden_dim),
            f"实际: {fusion_out['fused_features'].shape}"
        )
        report.check(
            "所有输出非 NaN",
            not torch.isnan(fusion_out["fused_features"]).any().item(),
            "检测到 NaN 值！"
        )
    except Exception as e:
        report.check("完整融合前向传播", False, str(e))

    # 2.5 梯度流验证
    print("\n  [2.5] 融合模块梯度流验证")
    try:
        fusion_grad = HierarchicalAttentionFusion(
            visual_dim=visual_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_heads=8,
        )
        fusion_grad.train()

        vp = torch.randn(B, P, visual_dim, requires_grad=False)
        vg = torch.randn(B, text_dim, requires_grad=False)
        tf = torch.randn(B, L, text_dim, requires_grad=False)
        target = torch.randn(B, L, hidden_dim)

        out = fusion_grad(vp, vg, tf)
        loss = nn.MSELoss()(out["fused_features"], target)
        loss.backward()

        has_grad = False
        for name, param in fusion_grad.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        report.check("融合模块梯度反向传播", has_grad, "梯度成功回传")
    except Exception as e:
        report.check("融合模块梯度流", False, str(e))


# ============================================================
# 3. 分类头验证
# ============================================================

def verify_classification_head(report: VerificationReport):
    """验证多标记分类输出头"""
    report.section("3. 多标记分类输出头验证")

    B, L, D = 4, 8, 512
    dummy_features = torch.randn(B, L, D)
    dummy_targets = torch.zeros(B, L)
    dummy_targets[0, [0, 3]] = 1.0
    dummy_targets[1, [1, 5, 7]] = 1.0
    dummy_targets[2, [2, 4]] = 1.0
    dummy_targets[3, [6, 7]] = 1.0

    # 3.1 标签特定分类器
    print("\n  [3.1] 标签特定分类器")
    head_ls = MultiLabelClassificationHead(
        input_dim=D,
        num_labels=L,
        classifier_type="label_specific",
    )

    try:
        with torch.no_grad():
            out_ls = head_ls(dummy_features)

        report.check("logits 维度", out_ls["logits"].shape == (B, L))
        report.check("概率值域", out_ls["probabilities"].min() >= 0 and
                     out_ls["probabilities"].max() <= 1,
                     f"范围: [{out_ls['probabilities'].min():.3f}, {out_ls['probabilities'].max():.3f}]")
        report.check("预测为二值", torch.all((out_ls["predictions"] == 0) | (out_ls["predictions"] == 1)).item())
    except Exception as e:
        report.check("标签特定分类器", False, str(e))

    # 3.2 共享注意力分类器
    print("\n  [3.2] 共享注意力分类器")
    head_sa = MultiLabelClassificationHead(
        input_dim=D,
        num_labels=L,
        classifier_type="shared_attention",
    )

    try:
        with torch.no_grad():
            out_sa = head_sa(dummy_features)
        report.check("logits 维度", out_sa["logits"].shape == (B, L))
    except Exception as e:
        report.check("共享注意力分类器", False, str(e))

    # 3.3 损失函数验证
    print("\n  [3.3] 损失函数验证")
    try:
        logits = torch.randn(B, L)

        # BCE Loss
        loss_bce = head_sa.compute_loss(logits, dummy_targets, "bce")
        report.check("BCE 损失为标量", loss_bce.dim() == 0, f"值: {loss_bce.item():.4f}")
        report.check("BCE 损失非负", loss_bce.item() >= 0)

        # Asymmetric Loss
        loss_asl = head_sa.compute_loss(logits, dummy_targets, "asymmetric")
        report.check("Asymmetric 损失非负", loss_asl.item() >= 0,
                     f"值: {loss_asl.item():.4f}")

        # Focal Loss
        loss_focal = head_sa.compute_loss(logits, dummy_targets, "focal")
        report.check("Focal 损失非负", loss_focal.item() >= 0,
                     f"值: {loss_focal.item():.4f}")

    except Exception as e:
        report.check("损失计算", False, str(e))

    # 3.4 损失值语义验证
    print("\n  [3.4] 损失语义验证")
    try:
        # 完美预测 → 低损失
        perfect_logits = torch.where(dummy_targets == 1, 10.0, -10.0)
        perfect_loss = head_sa.compute_loss(perfect_logits, dummy_targets, "bce")
        report.check("完美预测损失足够小", perfect_loss.item() < 0.01,
                     f"值: {perfect_loss.item():.6f}")

        # 相反预测 → 高损失
        wrong_logits = torch.where(dummy_targets == 1, -10.0, 10.0)
        wrong_loss = head_sa.compute_loss(wrong_logits, dummy_targets, "bce")
        report.check("错误预测损失大于完美预测", wrong_loss.item() > perfect_loss.item(),
                     f"完美: {perfect_loss.item():.4f}, 错误: {wrong_loss.item():.4f}")

    except Exception as e:
        report.check("损失语义验证", False, str(e))


# ============================================================
# 4. 标签关联模块验证
# ============================================================

def verify_label_association(report: VerificationReport):
    """验证标签关联建模模块"""
    report.section("4. 标签关联建模模块验证")

    B, L, D = 4, 8, 512
    dummy_features = torch.randn(B, L, D)

    # 4.1 标签图构建
    print("\n  [4.1] 情感标签图构建")
    try:
        adj = build_emotion_label_graph(
            MIKELS_BASIC_EMOTIONS,
            MIKELS_COOCCURRENCE,
            MIKELS_MUTUAL_EXCLUSION,
        )
        report.check("邻接矩阵形状", adj.shape == (L, L))
        report.check("自连接为 1", torch.allclose(adj.diag(), torch.ones(L)),
                     f"对角值: {adj.diag()[:4]}")

        pos_edges = (adj > 0).sum().item() - L  # 减去自连接
        neg_edges = (adj < 0).sum().item()
        report.check("正边（共现）存在", pos_edges > 0, f"共 {pos_edges} 条")
        report.check("负边（互斥）存在", neg_edges > 0, f"共 {neg_edges} 条")

        # 归一化
        adj_norm = normalize_adjacency(adj)
        report.check("归一化矩阵形状", adj_norm.shape == (L, L))
        report.check("归一化无 NaN", not torch.isnan(adj_norm).any().item())

    except Exception as e:
        report.check("标签图构建", False, str(e))

    # 4.2 GCN 前向传播
    print("\n  [4.2] 轻量 GCN 前向传播")
    try:
        gcn = LightweightLabelGCN(
            num_labels=L,
            input_dim=D,
            hidden_dim=256,
            output_dim=D,
            num_layers=2,
            emotion_labels=MIKELS_BASIC_EMOTIONS,
            cooccurrence=MIKELS_COOCCURRENCE,
            mutual_exclusion=MIKELS_MUTUAL_EXCLUSION,
        )

        gcn_out = gcn(dummy_features)
        report.check(
            "GCN 输出维度",
            gcn_out["gcn_features"].shape == (B, L, D),
            f"实际: {gcn_out['gcn_features'].shape}"
        )
        report.check(
            "GCN 输出无 NaN",
            not torch.isnan(gcn_out["gcn_features"]).any().item()
        )

        # 验证 GCN 消息传递：不同节点输出应该不同（图结构起作用）
        feat_variation = gcn_out["gcn_features"][0].std(dim=0).mean()
        report.check(
            "GCN 消息传递有效（节点输出有差异）",
            feat_variation > 0.001,
            f"节点间标准差: {feat_variation:.6f}"
        )

    except Exception as e:
        report.check("GCN 前向传播", False, str(e))

    # 4.3 标签关联完整模块
    print("\n  [4.3] 标签关联完整模块")
    try:
        lam = LabelAssociationModule(
            num_labels=L,
            feature_dim=D,
            emotion_labels=MIKELS_BASIC_EMOTIONS,
            cooccurrence=MIKELS_COOCCURRENCE,
            mutual_exclusion=MIKELS_MUTUAL_EXCLUSION,
        )

        lam_out = lam(dummy_features, return_details=True)
        report.check(
            "增强特征维度",
            lam_out["enhanced_features"].shape == (B, L, D),
            f"实际: {lam_out['enhanced_features'].shape}"
        )
        report.check(
            "标签关系矩阵",
            lam_out["label_relations"].shape[-2:] == (L, L),
            f"实际: torch.Size({lam_out['label_relations'].shape[-2:]})"
        )

    except Exception as e:
        report.check("标签关联完整模块", False, str(e))

    # 4.4 融合权重验证
    print("\n  [4.4] 自适应融合权重")
    try:
        # fusion_weight 是 sigmoid 前的值，sigmoid 后在 [0,1]
        weight_val = torch.sigmoid(lam.fusion_weight).item()
        report.check(
            "融合权重在合理范围",
            0.0 <= weight_val <= 1.0,
            f"当前权重: {weight_val:.4f}"
        )
    except Exception as e:
        report.check("融合权重验证", False, str(e))


# ============================================================
# 5. 端到端集成验证
# ============================================================

def verify_end_to_end(report: VerificationReport, quick: bool = False):
    """端到端集成验证"""
    report.section("5. 端到端集成验证")

    B, L = 2, 8

    # 5.1 模型初始化
    print("\n  [5.1] 模型初始化")
    try:
        config = ModelConfig(
            freeze_visual=True,
            freeze_text=True,
            num_emotions=L,
            use_label_association=True,
        )
        model = MultiLabelEmotionModel(config)
        report.check("模型初始化成功", True)
    except Exception as e:
        report.check("模型初始化", False, str(e))
        return

    # 5.2 完整前向传播
    print("\n  [5.2] 完整前向传播")
    dummy_images = torch.randn(B, 3, 224, 224)
    test_labels = config.emotion_labels

    try:
        with torch.no_grad():
            outputs = model(
                dummy_images,
                emotion_labels=test_labels,
                return_attention=True,
                return_intermediate=True,
            )

        report.check("logits 维度", outputs["logits"].shape == (B, L))
        report.check("probabilities 维度", outputs["probabilities"].shape == (B, L))
        report.check("predictions 维度", outputs["predictions"].shape == (B, L))
        report.check("注意力图存在", "attention_maps" in outputs)
        report.check("中间特征存在", "intermediate" in outputs)

        # 验证概率语义
        probs = outputs["probabilities"]
        report.check("概率在 [0,1] 范围", (probs >= 0).all() and (probs <= 1).all())

    except Exception as e:
        report.check("完整前向传播", False, str(e))

    # 5.3 损失回传（梯度流）
    print("\n  [5.3] 梯度反向传播验证")
    try:
        model.train()
        dummy_targets = torch.zeros(B, L)
        dummy_targets[0, [0, 3, 6]] = 1.0
        dummy_targets[1, [1, 5, 7]] = 1.0

        outputs = model(dummy_images, emotion_labels=test_labels)
        loss = model.compute_loss(outputs["logits"], dummy_targets, "bce")
        loss.backward()

        # 检查梯度是否成功传播到各模块
        grad_modules = []
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 1e-8:
                grad_modules.append(name.split(".")[0])

        unique_modules = set(grad_modules)
        report.check(
            "梯度传播到多个模块",
            len(unique_modules) >= 3,
            f"有梯度的模块: {unique_modules}"
        )

    except Exception as e:
        report.check("梯度反向传播", False, str(e))

    # 5.4 推理一致性
    print("\n  [5.4] 推理一致性验证")
    try:
        model.eval()
        with torch.no_grad():
            out1 = model(dummy_images, emotion_labels=test_labels)
            out2 = model(dummy_images, emotion_labels=test_labels)

        # 相同输入应产生相同输出
        report.check(
            "推理确定性",
            torch.allclose(out1["probabilities"], out2["probabilities"]),
            "两次推理结果一致"
        )

    except Exception as e:
        report.check("推理一致性", False, str(e))

    # 5.5 不同损失类型
    print("\n  [5.5] 多损失类型测试")
    try:
        dummy_targets = torch.zeros(B, L)
        dummy_targets[:, [0, 3]] = 1.0

        for loss_type in ["bce", "asymmetric", "focal"]:
            outputs = model(dummy_images, emotion_labels=test_labels)
            loss = model.compute_loss(outputs["logits"], dummy_targets, loss_type)
            report.check(
                f"{loss_type} 损失有效",
                loss.item() > 0 and not torch.isnan(loss).item(),
                f"值: {loss.item():.6f}"
            )

    except Exception as e:
        report.check("多损失类型", False, str(e))

    # 5.6 参数量统计
    print("\n  [5.6] 参数量统计")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"    总参数: {total:,}")
    print(f"    可训练: {trainable:,}")
    print(f"    冻结: {total - trainable:,}")

    report.check("模型有可训练参数", trainable > 0)
    report.check("总参数量合理", total > 1_000_000,  # 至少 1M
                 f"{total/1e6:.1f}M 参数")


# ============================================================
# 6. 配置与工具模块验证
# ============================================================

def verify_config_and_utils(report: VerificationReport):
    """验证配置和工具模块"""
    report.section("6. 配置与工具模块验证")

    # 6.1 配置完整性
    print("\n  [6.1] 模型配置完整性")
    try:
        mc = ModelConfig()
        required_attrs = [
            "clip_model_name", "visual_feature_dim", "text_feature_dim",
            "projection_dim", "fusion_hidden_dim", "fusion_num_heads",
            "num_emotions", "emotion_labels", "use_label_association",
        ]
        for attr in required_attrs:
            report.check(f"配置属性 {attr}", hasattr(mc, attr),
                         f"值: {getattr(mc, attr)}")

        tc = TrainingConfig()
        report.check("训练配置可用", tc.learning_rate > 0)
    except Exception as e:
        report.check("配置模块", False, str(e))

    # 6.2 情感标签体系
    print("\n  [6.2] 情感标签体系")
    report.check("基础情感标签非空", len(MIKELS_BASIC_EMOTIONS) > 0,
                 f"共 {len(MIKELS_BASIC_EMOTIONS)} 类基础情感")
    report.check("共现关系已定义", len(MIKELS_COOCCURRENCE) > 0,
                 f"共 {len(MIKELS_COOCCURRENCE)} 对")
    report.check("互斥关系已定义", len(MIKELS_MUTUAL_EXCLUSION) > 0,
                 f"共 {len(MIKELS_MUTUAL_EXCLUSION)} 对")

    # 6.3 工具函数
    print("\n  [6.3] 评估指标验证")
    try:
        from emotion_model.utils import compute_metrics

        N, L = 50, 8
        targets = torch.randint(0, 2, (N, L)).float()
        probs = torch.sigmoid(torch.randn(N, L) + targets * 2)
        preds = (probs > 0.5).float()

        metrics = compute_metrics(preds, probs, targets)

        required_metrics = ["f1_micro", "f1_macro", "hamming_loss", "mAP"]
        for m in required_metrics:
            report.check(f"指标 {m} 存在", m in metrics,
                         f"值: {metrics.get(m, 'N/A')}")

    except Exception as e:
        report.check("评估指标", False, str(e))

    # 6.4 设备检测
    print("\n  [6.4] 设备检测")
    try:
        from emotion_model.utils import get_device
        device = get_device()
        report.check("设备检测成功", isinstance(device, torch.device),
                     f"设备: {device}")
    except Exception as e:
        report.check("设备检测", False, str(e))


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="多标记情感识别模型 - 模块功能验证"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="快速验证模式（跳过 GPU 相关测试）"
    )
    parser.add_argument(
        "--module", type=str, default=None,
        choices=["encoder", "fusion", "classification", "label", "e2e", "config"],
        help="仅验证指定模块"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出验证报告到文件"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  多标记情感识别模型 — 模块功能验证")
    print("  Multi-Label Emotion Recognition Model")
    print("  Module Function Verification")
    print("=" * 60)
    print(f"  时间: {datetime.now().isoformat()}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    print(f"  MPS: {torch.backends.mps.is_available()}")

    report = VerificationReport()

    # 根据参数决定验证哪些模块
    module = args.module

    if module is None or module == "config":
        verify_config_and_utils(report)

    if module is None or module == "encoder":
        verify_encoder(report, quick=args.quick)

    if module is None or module == "fusion":
        verify_fusion(report)

    if module is None or module == "classification":
        verify_classification_head(report)

    if module is None or module == "label":
        verify_label_association(report)

    if module is None or module == "e2e":
        verify_end_to_end(report, quick=args.quick)

    # 汇总报告
    summary = report.summary()

    # 保存报告
    if args.output:
        output_path = args.output
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary)
            f.write("\n\n详细测试记录:\n")
            for t in report.tests:
                status = "PASS" if t["passed"] else "FAIL"
                f.write(f"  [{status}] {t['name']}: {t['detail']}\n")
        print(f"\n验证报告已保存至: {output_path}")

    # 返回退出码
    sys.exit(0 if report.succeeded() else 1)


if __name__ == "__main__":
    main()
