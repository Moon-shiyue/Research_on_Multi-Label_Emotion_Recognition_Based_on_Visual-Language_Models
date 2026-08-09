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

from .base_encoder import CLIPEncoder
from .fusion_module import HierarchicalAttentionFusion
from .classification_head import MultiLabelClassificationHead
from .label_association import LabelAssociationModule


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

        # ---- 2. 层次化注意力融合模块 ----
        visual_dim = self.encoder.visual_encoder.hidden_size  # 768 (ViT-B/32)
        text_dim = model_config.text_feature_dim  # 512
        fusion_dim = model_config.fusion_hidden_dim  # 512

        self.fusion_module = HierarchicalAttentionFusion(
            visual_dim=visual_dim,
            text_dim=text_dim,
            hidden_dim=fusion_dim,
            num_heads=model_config.fusion_num_heads,
            dropout=model_config.fusion_dropout,
            num_layers=model_config.fusion_num_layers,
            fusion_output_dim=fusion_dim,
        )

        # ---- 3. 标签关联建模模块 ----
        if model_config.use_label_association:
            from .config import EMOTION_COOCCURRENCE, EMOTION_MUTUAL_EXCLUSION
            self.label_association = LabelAssociationModule(
                num_labels=model_config.num_emotions,
                feature_dim=fusion_dim,
                label_embed_dim=model_config.label_embed_dim,
                gcn_hidden_dim=fusion_dim // 2,
                num_attention_heads=model_config.label_attention_heads,
                gcn_num_layers=model_config.gcn_num_layers,
                dropout=model_config.gcn_dropout,
                emotion_labels=model_config.emotion_labels,
                cooccurrence=EMOTION_COOCCURRENCE,
                mutual_exclusion=EMOTION_MUTUAL_EXCLUSION,
            )
        else:
            self.label_association = None

        # ---- 4. 多标记分类输出头 ----
        self.classification_head = MultiLabelClassificationHead(
            input_dim=fusion_dim,
            num_labels=model_config.num_emotions,
            emotion_labels=model_config.emotion_labels,
            hidden_dims=model_config.classifier_hidden_dims,
            classifier_type="shared_attention",
            dropout=model_config.fusion_dropout,
            label_smoothing=model_config.label_smoothing,
        )

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
    ) -> Dict[str, torch.Tensor]:
        """
        完整前向传播

        Args:
            pixel_values: (B, 3, H, W) 预处理后的图像
            emotion_labels: 情感标签列表（默认使用配置中的统一标签集）
            return_attention: 是否返回注意力权重（用于可解释性分析）
            return_intermediate: 是否返回所有中间特征

        Returns:
            dict:
                - logits: (B, num_labels) 原始 logits
                - probabilities: (B, num_labels) [0,1] 置信度
                - predictions: (B, num_labels) 0/1 预测
                - attention_maps: (B, H, L, P) 可选，视觉注意力图
                - intermediate: dict 可选，所有中间表示
        """
        if emotion_labels is None:
            emotion_labels = self.emotion_labels

        device = pixel_values.device
        B = pixel_values.shape[0]

        # ============================================
        # 1. CLIP 双塔编码
        # ============================================
        encoder_output = self.encoder.encode_image(
            pixel_values, return_patches=True
        )
        visual_global = encoder_output["global_feature"]    # (B, 512)
        patch_features = encoder_output["patch_features"]    # (B, 49, 768)

        text_output = self.encoder.encode_text(emotion_labels, device=device)
        text_features = text_output["global_feature"].unsqueeze(0).expand(B, -1, -1)
        # (B, num_labels, 512)

        # ============================================
        # 2. 层次化注意力融合
        # ============================================
        fusion_output = self.fusion_module(
            visual_patches=patch_features,
            visual_global=visual_global,
            text_features=text_features,
        )
        fused_features = fusion_output["fused_features"]  # (B, num_labels, 512)

        # ============================================
        # 3. 标签关联建模（可选）
        # ============================================
        if self.label_association is not None:
            label_output = self.label_association(fused_features, return_details=return_intermediate)
            fused_features = label_output["enhanced_features"]  # (B, num_labels, 512)

        # ============================================
        # 4. 多标记分类
        # ============================================
        cls_output = self.classification_head(fused_features, return_probs=True)

        result = {
            "logits": cls_output["logits"],
            "probabilities": cls_output["probabilities"],
            "predictions": cls_output["predictions"],
        }

        if return_attention:
            result["attention_maps"] = fusion_output.get("attention_maps", None)

        if return_intermediate:
            result["intermediate"] = {
                "encoder_output": encoder_output,
                "text_features": text_features,
                "fusion_output": fusion_output,
            }
            if self.label_association is not None:
                result["intermediate"]["label_output"] = label_output

        return result

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_type: str = "bce",
    ) -> torch.Tensor:
        """
        计算损失

        Args:
            logits: (B, num_labels) 预测 logits
            targets: (B, num_labels) 真实多热标签向量
            loss_type: "bce" | "asymmetric" | "focal"

        Returns:
            loss: 标量损失
        """
        return self.classification_head.compute_loss(
            logits=logits,
            targets=targets,
            loss_type=loss_type,
        )

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
    print("=" * 60)
    print("完整模型集成测试")
    print("=" * 60)

    from .config import ModelConfig

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
