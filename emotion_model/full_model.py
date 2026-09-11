"""
完整的多标记情感识别模型
Multi-Label Emotion Recognition Model

基于视觉语言模型 (CLIP ViT-B/32) 的端到端多标记情感识别架构。

架构总览:

    ┌─────────────────────────────────────────────────────┐
    │                    输入层                            │
    │   Image (B,3,224,224)    Emotion Labels (List[str])  │
    └──────────┬────────────────────┬─────────────────────┘
               │                    │
    ┌──────────▼────────────────────▼─────────────────────┐
    │                CLIP 双塔编码器                        │
    │  ┌──────────────┐    ┌──────────────────┐           │
    │  │ Visual Enc.  │    │  Text Encoder    │           │
    │  │ (ViT-B/32)   │    │  (CLIP Text)     │           │
    │  └──────┬───────┘    └────────┬─────────┘           │
    │         │ visual_features     │ text_features        │
    │         │ (B,512) + patches   │ (B,N_labels,512)     │
    └─────────┼─────────────────────┼─────────────────────┘
              │                     │
    ┌─────────▼─────────────────────▼─────────────────────┐
    │         层次化注意力融合模块                           │
    │  ┌──────────────────────────────────────────┐       │
    │  │ Step 1: 局部高权重特征融合                 │       │
    │  │  ├── Visual Emotion Attention            │       │
    │  │  └── Text Emotion Attention              │       │
    │  │         ↓ 局部融合                         │       │
    │  │ Step 2: 全局对齐特征校准                   │       │
    │  │  ├── Bi-directional Cross-Attention      │       │
    │  │  └── Global Self-Attention Calibration   │       │
    │  └──────────────────────────────────────────┘       │
    │  Output: fused_features (B, N_labels, 512)          │
    └─────────────────┬───────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────┐
    │              标签关联建模模块                         │
    │  ┌────────────────┐  ┌──────────────────┐           │
    │  │ Label          │  │ Lightweight GCN  │           │
    │  │ Attention      │  │ (Label Graph)    │           │
    │  │ (数据驱动)      │  │ (先验知识引导)    │           │
    │  └───────┬────────┘  └────────┬─────────┘           │
    │          └──────────┬─────────┘                     │
    │                     ▼ 自适应融合                      │
    │          enhanced_features (B, N_labels, 512)       │
    └─────────────────┬───────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────┐
    │              多标记分类输出头                         │
    │          每个标签独立 Sigmoid → (B, N_labels)         │
    │  Output: {logits, probabilities, predictions}       │
    └─────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
import warnings

# 兼容两种运行方式：
#   1. 作为包导入:  python -m emotion_model.train / from emotion_model import ...
#   2. 直接运行:    python emotion_model/full_model.py
try:
    from .base_encoder import CLIPEncoder
    from .fusion_module import HierarchicalAttentionFusion
    from .classification_head import MultiLabelClassificationHead
    from .label_association import LabelAssociationModule
    from .config import EMOTION_COOCCURRENCE, EMOTION_MUTUAL_EXCLUSION
    # 申请书三大核心创新模块
    from .conflict_fusion import ConflictAwareFusionModule
    from .circular_head import (
        CircularMultiLabelHead, EmotionCircleMapper, ProgressiveCircularLoss,
        build_label_correlation_matrix,
    )
    from .vl_adapter import VLAdapterManager
except ImportError:  # 直接运行脚本时相对导入不可用
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from base_encoder import CLIPEncoder
    from fusion_module import HierarchicalAttentionFusion
    from classification_head import MultiLabelClassificationHead
    from label_association import LabelAssociationModule
    from config import EMOTION_COOCCURRENCE, EMOTION_MUTUAL_EXCLUSION
    from conflict_fusion import ConflictAwareFusionModule
    from circular_head import (
        CircularMultiLabelHead, EmotionCircleMapper, ProgressiveCircularLoss,
        build_label_correlation_matrix,
    )
    from vl_adapter import VLAdapterManager


class MultiLabelEmotionModel(nn.Module):
    """
    基于视觉语言模型的多标记情感识别完整模型

    端到端的训练与推理接口。

    使用方式:
        # 初始化
        model = MultiLabelEmotionModel(config)

        # 训练
        outputs = model(
            pixel_values=images,
            emotion_labels=["joy", "sadness", ..., "excitement"],
        )
        loss = model.compute_loss(outputs["logits"], targets)

        # 推理
        with torch.no_grad():
            results = model.predict(images)
            # results["probabilities"]: 每个标签的置信度
            # results["predictions"]: 阈值化后的二值预测
    """

    def __init__(
        self,
        model_config=None,
        training_config=None,
    ):
        super().__init__()

        # 延迟导入配置（避免循环依赖）
        if model_config is None:
            from .config import default_model_config
            model_config = default_model_config

        self.config = model_config

        # ---- 1. CLIP 双塔编码器 ----
        self.encoder = CLIPEncoder(
            model_name=model_config.clip_model_name,
            projection_dim=model_config.projection_dim,
            freeze_visual=model_config.freeze_visual,
            freeze_text=model_config.freeze_text,
            use_grad_checkpoint=model_config.use_grad_checkpoint,
        )

        visual_dim = self.encoder.visual_encoder.hidden_size  # 768 (ViT-B/32)
        text_dim = model_config.text_feature_dim  # 512
        fusion_dim = model_config.fusion_hidden_dim  # 512

        # ---- 1.5 模块③：VL-Adapter 跨场景泛化（可选）----
        self.use_vl_adapter = getattr(model_config, "use_vl_adapter", False)
        if self.use_vl_adapter:
            self.vl_adapter = VLAdapterManager(
                visual_dim=visual_dim,
                text_dim=text_dim,
                num_layers=12,  # ViT-B/32 与 CLIP Text 均为 12 层
                bottleneck_ratio=getattr(model_config, "adapter_bottleneck_ratio", 4),
                num_domains=getattr(model_config, "adapter_num_domains", 4),
                domain_rank=getattr(model_config, "adapter_domain_rank", 4),
                dropout=model_config.fusion_dropout,
            )
            # 挂载到 CLIP 双塔各层（通过 forward hook）
            try:
                self.vl_adapter.attach(
                    vision_model=self.encoder.visual_encoder.vision_model,
                    text_model=self.encoder.text_encoder.text_model,
                )
                # 冻结主干：仅适配器、层归一化与视觉投影层可训练
                self.vl_adapter.freeze_backbone(
                    self.encoder.visual_encoder.vision_model,
                    self.encoder.text_encoder.text_model,
                )
                self.vl_adapter.set_domain(getattr(model_config, "adapter_domain", 0))
                self._adapter_attached = True
            except Exception as e:
                warnings.warn(f"VL-Adapter 挂载失败，将仅使用特征级适配: {e}")
                self._adapter_attached = False
        else:
            self.vl_adapter = None
            self._adapter_attached = False

        # ---- 2. 融合模块：模块① 冲突感知融合 或 基础版层次化融合 ----
        self.use_conflict_fusion = getattr(model_config, "use_conflict_fusion", False)
        if self.use_conflict_fusion:
            self.fusion_module = ConflictAwareFusionModule(
                visual_dim=visual_dim,
                text_dim=text_dim,
                hidden_dim=fusion_dim,
                num_heads=model_config.fusion_num_heads,
                dropout=model_config.fusion_dropout,
                fusion_output_dim=fusion_dim,
                text_boundary_init=getattr(model_config, "conflict_text_boundary_init", 0.3),
                visual_boundary_init=getattr(model_config, "conflict_visual_boundary_init", 0.0),
                use_contrastive_loss=getattr(model_config, "use_conflict_contrastive", True),
            )
        else:
            self.fusion_module = HierarchicalAttentionFusion(
                visual_dim=visual_dim,
                text_dim=text_dim,
                hidden_dim=fusion_dim,
                num_heads=model_config.fusion_num_heads,
                dropout=model_config.fusion_dropout,
                num_layers=model_config.fusion_num_layers,
                fusion_output_dim=fusion_dim,
            )

        # ---- 3. 标签关联建模（环形分类头自带标签关联，避免重复）----
        self.use_circular_head = getattr(model_config, "use_circular_head", False)
        use_la = model_config.use_label_association and not self.use_circular_head
        if use_la:
            # 根据标签体系选择先验关系表
            if getattr(model_config, "use_mikels_basic", False):
                from .config import MIKELS_COOCCURRENCE, MIKELS_MUTUAL_EXCLUSION
                cooc, mutex = MIKELS_COOCCURRENCE, MIKELS_MUTUAL_EXCLUSION
            else:
                cooc, mutex = EMOTION_COOCCURRENCE, EMOTION_MUTUAL_EXCLUSION
            self.label_association = LabelAssociationModule(
                num_labels=model_config.num_emotions,
                feature_dim=fusion_dim,
                label_embed_dim=model_config.label_embed_dim,
                gcn_hidden_dim=fusion_dim // 2,
                num_attention_heads=model_config.label_attention_heads,
                gcn_num_layers=model_config.gcn_num_layers,
                dropout=model_config.gcn_dropout,
                emotion_labels=model_config.emotion_labels,
                cooccurrence=cooc,
                mutual_exclusion=mutex,
            )
        else:
            self.label_association = None

        # ---- 4. 分类输出头：模块② 情感环形表示 或 基础版多标记头 ----
        if self.use_circular_head:
            # 环形表示要求标签为 8 类 Mikels 基础情感
            if getattr(model_config, "use_mikels_basic", False):
                from .config import MIKELS_COOCCURRENCE, MIKELS_MUTUAL_EXCLUSION
                corr_matrix = build_label_correlation_matrix(
                    model_config.emotion_labels,
                    MIKELS_COOCCURRENCE, MIKELS_MUTUAL_EXCLUSION,
                )
            else:
                corr_matrix = build_label_correlation_matrix(
                    model_config.emotion_labels,
                    EMOTION_COOCCURRENCE, EMOTION_MUTUAL_EXCLUSION,
                )

            self.classification_head = CircularMultiLabelHead(
                input_dim=fusion_dim,
                num_labels=model_config.num_emotions,
                emotion_labels=model_config.emotion_labels,
                hidden_dim=model_config.classifier_hidden_dims[0],
                dropout=model_config.fusion_dropout,
                radius=getattr(model_config, "circular_radius", 1.0),
                label_correlation_matrix=corr_matrix,
            )
            # 环形损失（含与 KL 的联合权重）
            self.circular_loss = ProgressiveCircularLoss(
                mu=getattr(model_config, "circular_mu", 0.5),
                angle_mode=getattr(model_config, "circular_angle_mode", "circular"),
            )
            # 情感环形映射器（构造监督信号用）
            self.circle_mapper = EmotionCircleMapper(num_labels=model_config.num_emotions)
        else:
            self.classification_head = MultiLabelClassificationHead(
                input_dim=fusion_dim,
                num_labels=model_config.num_emotions,
                emotion_labels=model_config.emotion_labels,
                hidden_dims=model_config.classifier_hidden_dims,
                classifier_type="shared_attention",
                dropout=model_config.fusion_dropout,
                label_smoothing=model_config.label_smoothing,
            )
            self.circular_loss = None
            self.circle_mapper = None

        self.num_emotions = model_config.num_emotions
        self.emotion_labels = model_config.emotion_labels

        self._print_model_info()

    def _print_model_info(self):
        """打印模型信息摘要"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"MultiLabelEmotionModel 模型信息")
        print(f"{'='*60}")
        print(f"  情感类别数: {self.num_emotions}")
        print(f"  情感标签: {self.emotion_labels}")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  冻结参数: {total_params - trainable_params:,}")
        print(f"  ---- 创新模块 ----")
        print(f"  模块① 冲突感知融合: {'启用' if self.use_conflict_fusion else '禁用（基础版融合）'}")
        print(f"  模块② 情感环形分类头: {'启用' if self.use_circular_head else '禁用（基础版多标记头）'}")
        print(f"  模块③ VL-Adapter: {'启用' if self.use_vl_adapter else '禁用'}"
              f"{'（已挂载 CLIP）' if self._adapter_attached else ''}")
        print(f"  标签关联模块: {'启用' if self.label_association else '禁用'}")
        print(f"{'='*60}\n")

    def encode_text_labels(
        self,
        emotion_labels: List[str],
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        编码情感标签为文本特征

        Args:
            emotion_labels: 情感标签列表
            device: 目标设备

        Returns:
            text_features: (num_labels, text_dim)
        """
        text_out = self.encoder.encode_text(emotion_labels, device=device)
        return text_out["global_feature"]  # (num_labels, text_dim)

    def forward(
        self,
        pixel_values: torch.Tensor,
        emotion_labels: List[str] = None,
        return_attention: bool = False,
        return_intermediate: bool = False,
        texts: List[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        完整前向传播

        Args:
            pixel_values: (B, 3, H, W) 预处理后的图像
            emotion_labels: 情感标签列表（默认使用配置中的标签集）
            return_attention: 是否返回注意力权重（用于可解释性分析）
            return_intermediate: 是否返回所有中间特征
            texts: 可选的原始文本输入（图文对数据集，用于冲突感知融合的双路径建模）

        Returns:
            dict:
                - logits: (B, num_labels) 原始 logits
                - probabilities: (B, num_labels) [0,1] 置信度
                - predictions: (B, num_labels) 0/1 预测
                - circle_vector: (B, L, 3) 情感环形三维表示（模块②启用时）
                - attention_maps: (B, H, L, P) 可选，视觉注意力图
                - intermediate: dict 可选，所有中间表示
        """
        if emotion_labels is None:
            emotion_labels = self.emotion_labels

        device = pixel_values.device
        B = pixel_values.shape[0]

        # ============================================
        # 1. CLIP 双塔编码（VL-Adapter 通过 hook 自动介入）
        # ============================================
        encoder_output = self.encoder.encode_image(
            pixel_values, return_patches=True
        )
        visual_global = encoder_output["global_feature"]    # (B, 512)
        patch_features = encoder_output["patch_features"]    # (B, 49, 768)

        text_output = self.encoder.encode_text(emotion_labels, device=device)
        text_features = text_output["global_feature"].unsqueeze(0).expand(B, -1, -1)
        # (B, num_labels, 512)

        # 若提供原始文本（图文对数据），编码为 token 级特征供双路径冲突注意力使用
        text_token_features = None
        if texts is not None:
            token_out = self.encoder.encode_text_from_raw(
                list(texts), device=device
            )
            text_token_features = token_out.get("token_features", None)
            if text_token_features is not None and text_token_features.shape[0] != B:
                text_token_features = None

        # ============================================
        # 2. 跨模态融合（模块① 冲突感知 或 基础版层次化）
        # ============================================
        if self.use_conflict_fusion:
            fusion_output = self.fusion_module(
                visual_patches=patch_features,
                visual_global=visual_global,
                text_features=text_features,
                text_token_features=text_token_features,
            )
        else:
            fusion_output = self.fusion_module(
                visual_patches=patch_features,
                visual_global=visual_global,
                text_features=text_features,
            )
        fused_features = fusion_output["fused_features"]  # (B, num_labels, 512)

        # ============================================
        # 3. 标签关联建模（可选）
        # ============================================
        label_output = None
        if self.label_association is not None:
            label_output = self.label_association(
                fused_features, return_details=return_intermediate
            )
            fused_features = label_output["enhanced_features"]  # (B, num_labels, 512)

        # ============================================
        # 4. 多标记分类（模块② 情感环形表示 或 基础版）
        # ============================================
        if self.use_circular_head:
            cls_output = self.classification_head(fused_features, return_circle=True)
        else:
            cls_output = self.classification_head(fused_features, return_probs=True)

        result = {
            "logits": cls_output["logits"],
            "probabilities": cls_output["probabilities"],
            "predictions": (
                (cls_output["probabilities"] > 0.5).float()
                if "predictions" not in cls_output
                else cls_output["predictions"]
            ),
        }

        # 环形表示附加输出（模块②）
        if self.use_circular_head:
            result["circle_vector"] = cls_output["circle_vector"]
            result["polarity_probs"] = cls_output["polarity_probs"]
            result["polarity_value"] = cls_output["polarity_value"]
            result["angle"] = cls_output["angle"]
            result["intensity"] = cls_output["intensity"]

        # 冲突感知附加输出（模块①）
        if self.use_conflict_fusion:
            result["conflict_scores"] = fusion_output.get("conflict_scores")
            result["cross_conflict"] = fusion_output.get("cross_conflict")
            result["alignment_weights"] = fusion_output.get("alignment_weights")
            if "contrastive_loss" in fusion_output:
                result["contrastive_loss"] = fusion_output["contrastive_loss"]
            result["text_features_conflict"] = fusion_output.get("text_features")
            result["visual_features_conflict"] = fusion_output.get("visual_features")

        if return_attention:
            result["attention_maps"] = fusion_output.get("attention_maps", None)

        if return_intermediate:
            result["intermediate"] = {
                "encoder_output": encoder_output,
                "text_features": text_features,
                "fusion_output": fusion_output,
                "cls_output": cls_output,
            }
            if label_output is not None:
                result["intermediate"]["label_output"] = label_output

        return result

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_type: str = "bce",
        circular_outputs: Dict[str, torch.Tensor] = None,
        contrastive_loss: torch.Tensor = None,
        loss_weights: Dict[str, float] = None,
    ) -> torch.Tensor:
        """
        计算损失（多目标联合损失）

        基础版: L = 分类损失（BCE / 非对称 / Focal）
        完整版: L = λ1·L_cls + λ2·L_PC + λ3·L_contrastive
          - L_cls: 分类损失（非对称损失，解决标签不平衡）
          - L_PC:  渐进式环形损失（模块②，刻画情感的极性-类型-强度）
          - L_contrastive: 冲突对比损失（模块①，抑制模态内语义干扰）

        Args:
            logits: (B, num_labels) 预测 logits
            targets: (B, num_labels) 真实多热标签向量
            loss_type: "bce" | "asymmetric" | "focal"
            circular_outputs: 分类头输出（模块②启用时传入，用于环形损失）
            contrastive_loss: 冲突对比损失（模块①启用时传入）
            loss_weights: 各损失项权重 {"cls": 1.0, "circular": 0.5, "contrastive": 0.3}

        Returns:
            loss: 标量损失
        """
        weights = {"cls": 1.0, "circular": 0.5, "contrastive": 0.3}
        if loss_weights:
            weights.update(loss_weights)

        # ---- 分类损失（主损失）----
        if self.use_circular_head:
            # 环形分类头输出 logits，用相同的多标记损失
            cls_loss = self._multilabel_loss(logits, targets, loss_type)
        else:
            cls_loss = self.classification_head.compute_loss(
                logits=logits, targets=targets, loss_type=loss_type,
            )

        total = weights["cls"] * cls_loss

        # ---- 渐进式环形损失（模块②）----
        if self.use_circular_head and circular_outputs is not None and self.circular_loss is not None:
            # 由多热标签构造环形监督信号 (p, θ, r)
            with torch.no_grad():
                target_vec = self.circle_mapper.distribution_to_vector(targets.float())
                B, L = targets.shape
                target_polarity = target_vec["polarity"].unsqueeze(-1).expand(B, L)
                target_angle = target_vec["angle"].unsqueeze(-1).expand(B, L)
                target_intensity = target_vec["intensity"].unsqueeze(-1).expand(B, L)

            pc = self.circular_loss(circular_outputs, {
                "polarity": target_polarity,
                "angle": target_angle,
                "intensity": target_intensity,
            })
            total = total + weights["circular"] * pc["pc_loss"]

        # ---- 冲突对比损失（模块①）----
        if contrastive_loss is not None:
            total = total + weights["contrastive"] * contrastive_loss

        return total

    def _multilabel_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_type: str = "bce",
    ) -> torch.Tensor:
        """多标记分类损失（供环形分类头使用，与基础版公式一致）"""
        import torch.nn.functional as F

        if loss_type == "bce":
            return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")

        probs = torch.sigmoid(logits)

        if loss_type == "asymmetric":
            gamma_pos, gamma_neg, clip = 1.0, 4.0, 0.05
            pos_loss = -((1 - probs) ** gamma_pos) * torch.log(probs + 1e-8) * targets
            probs_neg = probs.clamp(max=1 - clip)
            neg_loss = -((probs_neg) ** gamma_neg) * torch.log(1 - probs_neg + 1e-8) * (1 - targets)
            return (pos_loss + neg_loss).mean()

        if loss_type == "focal":
            gamma, alpha = 2.0, 0.25
            pt = probs * targets + (1 - probs) * (1 - targets)
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            return (-alpha_t * ((1 - pt) ** gamma) * torch.log(pt + 1e-8)).mean()

        raise ValueError(f"未知的损失类型: {loss_type}")

    @torch.no_grad()
    def predict(
        self,
        pixel_values: torch.Tensor,
        emotion_labels: List[str] = None,
        threshold: float = None,
    ) -> Dict[str, torch.Tensor]:
        """
        推理接口

        Args:
            pixel_values: (B, 3, H, W) 图像
            emotion_labels: 情感标签列表
            threshold: 自定义决策阈值（默认使用模型学习到的阈值）

        Returns:
            dict:
                - probabilities: (B, num_labels) 置信度
                - predictions: (B, num_labels) 0/1 预测
                - top_emotions: 每个样本的 top-k 情感列表
        """
        self.eval()
        outputs = self.forward(pixel_values, emotion_labels)
        self.train()

        result = {
            "probabilities": outputs["probabilities"],
            "predictions": outputs["predictions"],
        }

        if threshold is not None:
            result["predictions"] = (outputs["probabilities"] > threshold).float()

        return result

    def get_attention_maps(
        self,
        pixel_values: torch.Tensor,
        emotion_labels: List[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        获取注意力热图（用于模型可解释性分析）

        Returns:
            dict with attention_maps: (B, H, num_labels, num_patches)
        """
        outputs = self.forward(
            pixel_values,
            emotion_labels=emotion_labels,
            return_attention=True,
        )
        return {"attention_maps": outputs["attention_maps"]}


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    # Windows 控制台默认 GBK 编码无法输出 emoji，强制 UTF-8
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("完整模型集成测试")
    print("=" * 60)

    import sys
    import os
    # 直接运行时脚本目录不在包路径中，需手动加入
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import ModelConfig

    # 创建轻量测试配置
    config = ModelConfig(
        freeze_visual=True,   # 测试时冻结以减少内存
        freeze_text=True,
        num_emotions=12,
    )

    # 初始化模型
    model = MultiLabelEmotionModel(config)

    # 模拟输入
    B = 2
    dummy_images = torch.randn(B, 3, 224, 224)
    test_labels = config.emotion_labels

    print(f"\n[前向传播测试]")
    print(f"  输入图像: {dummy_images.shape}")
    print(f"  标签数量: {len(test_labels)}")

    with torch.no_grad():
        outputs = model(
            dummy_images,
            emotion_labels=test_labels,
            return_attention=True,
            return_intermediate=True,
        )

    print(f"\n输出:")
    print(f"  logits:        {outputs['logits'].shape}")
    print(f"  probabilities: {outputs['probabilities'].shape}")
    print(f"  predictions:   {outputs['predictions'].shape}")

    # 检查概率范围
    probs = outputs["probabilities"]
    print(f"  概率范围: [{probs.min():.3f}, {probs.max():.3f}]")

    # 模拟多热标签
    dummy_targets = torch.zeros(B, len(test_labels))
    dummy_targets[0, [0, 3, 10]] = 1.0  # joy, fear, excitement
    dummy_targets[1, [1, 5, 6]] = 1.0   # sadness, disgust, love

    # 测试损失计算
    for loss_type in ["bce", "asymmetric", "focal"]:
        loss = model.compute_loss(outputs["logits"], dummy_targets, loss_type)
        print(f"  {loss_type} loss: {loss.item():.4f}")

    # 测试推理接口
    preds = model.predict(dummy_images)
    print(f"\n推理结果示例 (样本0):")
    for i, label in enumerate(test_labels):
        print(f"  {label:12s}: prob={preds['probabilities'][0, i]:.3f}, "
              f"pred={int(preds['predictions'][0, i])}")

    # 测试注意力图
    if outputs.get("attention_maps") is not None:
        attn = outputs["attention_maps"]
        print(f"\n注意力图: {attn.shape} (B, H, L, patches)")

    print("\n✅ 完整模型集成测试通过！")
