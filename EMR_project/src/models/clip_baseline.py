"""
CLIP 基线模型

架构:
    1. CLIP 图像编码器 (ViT-B/32 或 ViT-L/14) → image_emb (512/768 dim)
    2. CLIP 文本编码器 (Transformer) → text_emb (512/768 dim)
    3. 多模态融合模块 → fused_emb (256 dim)
    4. 多标签预测头 (MLP) → 8 维 logits (sigmoid)

支持三种融合策略:
    - concat:           直接拼接 [image_emb, text_emb] → MLP → output
    - cross_attention:  图像/文本特征做交叉注意力后再融合
    - gated:            门控融合，学习图像和文本各自的权重

两种文本输入模式:
    - prompt:     使用固定的 prompt 模板 "a photo expressing {emotion}"
    - caption:    使用数据集自带的文本描述（如 Flickr30k 的 caption）
"""
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel, CLIPTextModel, CLIPConfig


# ============================================================
# 多模态融合模块
# ============================================================

class ConcatFusion(nn.Module):
    """简单拼接融合"""

    def __init__(self, image_dim: int, text_dim: int, projection_dim: int, dropout: float = 0.3):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, projection_dim)
        self.text_proj = nn.Linear(text_dim, projection_dim)
        self.fusion = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(projection_dim),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        img = self.image_proj(image_emb)
        txt = self.text_proj(text_emb)
        fused = torch.cat([img, txt], dim=-1)
        return self.fusion(fused)


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合。

    Image 作为 Query，Text 作为 Key/Value（模仿 ViT 的 cross-attention）。
    让视觉特征"关注"文本中的情感语义。
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)    # image query
        self.k_proj = nn.Linear(dim, dim)    # text key
        self.v_proj = nn.Linear(dim, dim)    # text value
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        B = image_emb.shape[0]

        # 检查维度匹配
        if image_emb.shape[-1] != text_emb.shape[-1]:
            raise ValueError(
                f"Image dim ({image_emb.shape[-1]}) != Text dim ({text_emb.shape[-1]}). "
                f"请确保 CLIP 图像和文本编码器输出维度一致。"
            )

        # Multi-head attention
        q = self.q_proj(image_emb).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(text_emb).view(B, self.num_heads, self.head_dim)
        v = self.v_proj(text_emb).view(B, self.num_heads, self.head_dim)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, -1)
        out = self.out_proj(out)

        # Residual + FFN
        out = self.norm1(image_emb + out)
        out = self.norm2(out + self.ffn(out))

        return out


class GatedFusion(nn.Module):
    """
    门控融合: 学习图像和文本的权重。

    gate = sigmoid(W * [image_emb, text_emb])
    fused = gate * image_proj + (1 - gate) * text_proj
    """

    def __init__(self, image_dim: int, text_dim: int, projection_dim: int, dropout: float = 0.3):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, projection_dim)
        self.text_proj = nn.Linear(text_dim, projection_dim)

        self.gate = nn.Sequential(
            nn.Linear(image_dim + text_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.Sigmoid(),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        img_proj = self.image_proj(image_emb)
        txt_proj = self.text_proj(text_emb)

        gate_input = torch.cat([image_emb, text_emb], dim=-1)
        gate = self.gate(gate_input)

        fused = gate * img_proj + (1 - gate) * txt_proj
        return fused


# ============================================================
# 多标签预测头
# ============================================================

class MultiLabelHead(nn.Module):
    """
    多标签预测头。

    输入: 融合后的特征 (projection_dim,)
    输出: 8 个独立的情感概率 (通过 sigmoid)

    额外包含一个简单的标签共现建模层（label graph）：
    在最后一层之前学习标签间的 pairwise 关系。
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int = 8,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        use_label_correlation: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.use_label_correlation = use_label_correlation
        self.num_labels = num_labels

        # Build feature extractor (all layers except the final classifier)
        feature_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            feature_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.features = nn.Sequential(*feature_layers)
        self.feature_dim = in_dim

        # Final classifier layer
        if use_label_correlation:
            # Learnable label correlation matrix
            self.label_corr = nn.Parameter(torch.eye(num_labels) * 0.5)
            # Takes features + label-aware features
            self.classifier = nn.Linear(in_dim, num_labels)
        else:
            self.label_corr = None
            self.classifier = nn.Linear(in_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)

        if self.use_label_correlation and self.label_corr is not None:
            # Raw logits
            raw_logits = self.classifier(feats)
            # Label relation regularization: logits += logits @ label_corr
            logits = raw_logits + raw_logits @ self.label_corr
        else:
            logits = self.classifier(feats)

        return logits


# ============================================================
# CLIP 多标记情感识别模型
# ============================================================

class CLIPMultiLabelEmotionModel(nn.Module):
    """
    CLIP 多标记情感识别模型。

    支持功能:
        - 冻结/微调图像编码器
        - 冻结/微调文本编码器
        - 三种多模态融合方式
        - 两种文本输入模式（prompt模板 / 原始caption）
        - 可解释性：返回各模态的特征，支持模态贡献度分析
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        num_emotions: int = 8,
        fusion_method: str = "concat",
        projection_dim: int = 256,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        freeze_image_encoder: bool = False,
        freeze_text_encoder: bool = False,
        use_label_correlation: bool = True,
        text_prompt_template: str = "a photo expressing {emotion}",
    ):
        """
        Args:
            clip_model_name: HuggingFace CLIP 模型名称
            num_emotions: 情感类别数（默认 8）
            fusion_method: concat / cross_attention / gated
            projection_dim: 融合空间维度
            hidden_dims: 预测头隐藏层维度
            dropout: Dropout 率
            freeze_image_encoder: 是否冻结图像编码器
            freeze_text_encoder: 是否冻结文本编码器
            use_label_correlation: 是否使用标签关系建模
            text_prompt_template: 文本 prompt 模板
        """
        super().__init__()
        print(f"[Model] Loading CLIP: {clip_model_name} ...")

        # 加载 CLIP
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # 获取维度
        self.image_embed_dim = self.clip.config.vision_config.hidden_size
        self.text_embed_dim = self.clip.config.text_config.hidden_size
        print(f"[Model] Image dim: {self.image_embed_dim}, Text dim: {self.text_embed_dim}")

        self.num_emotions = num_emotions
        self.fusion_method = fusion_method
        self.text_prompt_template = text_prompt_template
        self.clip_model_name = clip_model_name

        # 冻结/解冻
        self._set_trainable(freeze_image_encoder, freeze_text_encoder)

        # 多模态融合
        if fusion_method == "concat":
            self.fusion = ConcatFusion(self.image_embed_dim, self.text_embed_dim, projection_dim, dropout)
            fused_dim = projection_dim
        elif fusion_method == "gated":
            self.fusion = GatedFusion(self.image_embed_dim, self.text_embed_dim, projection_dim, dropout)
            fused_dim = projection_dim
        elif fusion_method == "cross_attention":
            # cross_attention 需要相同的维度
            common_dim = max(self.image_embed_dim, self.text_embed_dim)
            self.img_to_common = nn.Linear(self.image_embed_dim, common_dim) if self.image_embed_dim != common_dim else nn.Identity()
            self.txt_to_common = nn.Linear(self.text_embed_dim, common_dim) if self.text_embed_dim != common_dim else nn.Identity()
            self.fusion = CrossAttentionFusion(common_dim, num_heads=8, dropout=dropout)
            fused_dim = common_dim
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")

        # 多标签预测头
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.classifier = MultiLabelHead(
            input_dim=fused_dim,
            num_labels=num_emotions,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_label_correlation=use_label_correlation,
        )

        # 存储中间特征（用于可解释性分析）
        self._image_emb = None
        self._text_emb = None
        self._fused_emb = None

    def _set_trainable(self, freeze_image: bool, freeze_text: bool):
        """设置 CLIP 编码器的可训练性"""

        def _set_requires_grad(module, requires_grad: bool):
            for param in module.parameters():
                param.requires_grad = requires_grad

        if hasattr(self.clip, 'vision_model'):
            _set_requires_grad(self.clip.vision_model, not freeze_image)
        if hasattr(self.clip, 'text_model'):
            _set_requires_grad(self.clip.text_model, not freeze_text)

        # Log
        img_trainable = sum(p.numel() for p in self.clip.vision_model.parameters() if p.requires_grad) if hasattr(self.clip, 'vision_model') else 0
        txt_trainable = sum(p.numel() for p in self.clip.text_model.parameters() if p.requires_grad) if hasattr(self.clip, 'text_model') else 0
        print(f"[Model] Trainable params — Image: {img_trainable:,}, Text: {txt_trainable:,}")

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        return outputs.pooler_output  # (B, image_embed_dim)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """编码文本"""
        outputs = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output  # (B, text_embed_dim)

    def _generate_emotion_prompt(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成情感 prompt 文本的编码。
        对 8 个情感各生成一个 prompt，返回 8 个文本嵌入。
        """
        from src.datasets.label_mapping import BASIC_EMOTIONS

        prompts = [self.text_prompt_template.format(emotion=e) for e in BASIC_EMOTIONS]
        tokens = self.processor(
            text=prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            text_embs = self.encode_text(tokens.input_ids, tokens.attention_mask)  # (8, text_dim)

        return text_embs  # (8, text_dim)

    def forward(
        self,
        pixel_values: torch.Tensor,
        text_inputs: Optional[Union[torch.Tensor, List[str]]] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (B, 3, H, W) 图像张量
            text_inputs:
                - None: 自动用 prompt 模板生成文本
                - List[str]: 原始文本列表
                - Tuple[Tensor, Tensor]: (input_ids, attention_mask)
            return_features: 是否返回中间特征

        Returns:
            dict with:
                - logits: (B, num_emotions) 原始 logits
                - probs:  (B, num_emotions) sigmoid 概率
                - image_emb: (B, image_dim) 可解释性用
                - text_emb:  (B, text_dim) 可解释性用
                - fused_emb: (B, fused_dim) 可解释性用
        """
        B = pixel_values.shape[0]
        device = pixel_values.device

        # === 1) 图像编码 ===
        image_emb = self.encode_image(pixel_values)  # (B, image_dim)

        # === 2) 文本编码 ===
        if isinstance(text_inputs, list) and len(text_inputs) > 0 and isinstance(text_inputs[0], str):
            # 原始文本列表
            tokens = self.processor(
                text=text_inputs, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            text_emb = self.encode_text(tokens.input_ids, tokens.attention_mask)
        elif text_inputs is None:
            # 用 prompt 模板: "a photo expressing joy and sadness"
            # 为每个样本生成 prompt
            emotion_prompts = []
            from src.datasets.label_mapping import BASIC_EMOTIONS
            # 通用 prompt: "a photo expressing emotion"
            default_prompt = self.text_prompt_template.format(emotion="emotion")
            tokens = self.processor(
                text=[default_prompt] * B, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            text_emb = self.encode_text(tokens.input_ids, tokens.attention_mask)
        else:
            # 已 tokenize 的输入
            input_ids, attention_mask = text_inputs
            text_emb = self.encode_text(input_ids, attention_mask)

        # === 3) 多模态融合 ===
        if self.fusion_method == "cross_attention":
            image_emb_aligned = self.img_to_common(image_emb)
            text_emb_aligned = self.txt_to_common(text_emb)
            fused_emb = self.fusion(image_emb_aligned.unsqueeze(1), text_emb_aligned.unsqueeze(1)).squeeze(1)
        else:
            fused_emb = self.fusion(image_emb, text_emb)

        # === 4) 多标签预测 ===
        logits = self.classifier(fused_emb)  # (B, num_emotions)
        probs = torch.sigmoid(logits)

        # 存储中间特征
        self._image_emb = image_emb.detach()
        self._text_emb = text_emb.detach()
        self._fused_emb = fused_emb.detach()

        output = {
            "logits": logits,
            "probs": probs,
        }

        if return_features:
            output["image_emb"] = image_emb
            output["text_emb"] = text_emb
            output["fused_emb"] = fused_emb

        return output

    def get_modality_contributions(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        可解释性分析: 分别用纯图像、纯文本、融合特征预测，计算各模态贡献度。

        Returns:
            dict with image_only_probs, text_only_probs, fused_probs
        """
        B = pixel_values.shape[0]
        device = pixel_values.device

        with torch.no_grad():
            # 图像编码
            image_emb = self.encode_image(pixel_values)

            # 文本编码（prompt）
            default_prompt = self.text_prompt_template.format(emotion="emotion")
            tokens = self.processor(
                text=[default_prompt] * B, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            text_emb = self.encode_text(tokens.input_ids, tokens.attention_mask)

            # 融合
            if self.fusion_method == "cross_attention":
                img_aligned = self.img_to_common(image_emb)
                txt_aligned = self.txt_to_common(text_emb)
                fused_emb = self.fusion(img_aligned.unsqueeze(1), txt_aligned.unsqueeze(1)).squeeze(1)
            else:
                fused_emb = self.fusion(image_emb, text_emb)

            # 三种预测
            fused_probs = torch.sigmoid(self.classifier(fused_emb))

            # 只用图像特征（消融文本）
            img_only_fused = self.fusion(image_emb, torch.zeros_like(text_emb))
            img_only_probs = torch.sigmoid(self.classifier(img_only_fused))

            # 只用文本特征（消融图像）
            txt_only_fused = self.fusion(torch.zeros_like(image_emb), text_emb)
            txt_only_probs = torch.sigmoid(self.classifier(txt_only_fused))

        return {
            "image_only_probs": img_only_probs,
            "text_only_probs": txt_only_probs,
            "fused_probs": fused_probs,
        }


# ============================================================
# 简单特征拼接基线（不微调 CLIP）
# ============================================================

class SimpleFeatureConcatBaseline(nn.Module):
    """
    轻量级基线: 不微调 CLIP，仅训练一个 MLP 分类头。

    1. 用冻结的 CLIP 提取图像特征
    2. 用冻结的 CLIP 提取文本特征 (prompt)
    3. 拼接 → MLP → 8 维输出

    这个版本适合 CPU 训练和快速实验。
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        num_emotions: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        print(f"[SimpleBaseline] Loading CLIP: {clip_model_name} ...")

        self.clip = CLIPModel.from_pretrained(clip_model_name)
        for param in self.clip.parameters():
            param.requires_grad = False  # 冻结全部 CLIP

        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        self.image_dim = self.clip.config.vision_config.hidden_size
        self.text_dim = self.clip.config.text_config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(self.image_dim + self.text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_emotions),
        )

        self.num_emotions = num_emotions

        # 统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SimpleBaseline] Total params: {total_params:,} | Trainable: {trainable_params:,}")

    def forward(
        self,
        pixel_values: torch.Tensor,
        text_inputs: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        B = pixel_values.shape[0]
        device = pixel_values.device

        with torch.no_grad():
            # 图像编码
            image_emb = self.clip.vision_model(pixel_values=pixel_values).pooler_output

            # 文本编码
            if text_inputs is None or (isinstance(text_inputs, list) and not any(text_inputs)):
                text_inputs = ["a photo expressing emotion"] * B

            tokens = self.processor(
                text=text_inputs, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            text_emb = self.clip.text_model(
                input_ids=tokens.input_ids, attention_mask=tokens.attention_mask
            ).pooler_output

        # 拼接 → 分类
        concat_emb = torch.cat([image_emb, text_emb], dim=-1)
        logits = self.classifier(concat_emb)
        probs = torch.sigmoid(logits)

        return {"logits": logits, "probs": probs, "image_emb": image_emb, "text_emb": text_emb}


# ============================================================
# 模型工厂函数
# ============================================================

def build_model(config) -> nn.Module:
    """
    根据配置构建模型。

    Args:
        config: Config 对象

    Returns:
        模型实例
    """
    model_cfg = config.model
    train_cfg = config.training

    # 根据训练模式决定是否冻结编码器
    # CPU 训练建议使用 SimpleFeatureConcatBaseline
    if config.device == "cpu":
        print("[Model] 检测到 CPU 环境，建议使用 SimpleFeatureConcatBaseline")
        print("[Model] 使用哪种模型取决于 full_finetune 设置")

    # 判断是否需要完整微调
    full_finetune = not (model_cfg.freeze_image_encoder and model_cfg.freeze_text_encoder)

    if full_finetune:
        model = CLIPMultiLabelEmotionModel(
            clip_model_name=model_cfg.clip_model_name,
            num_emotions=model_cfg.num_emotions,
            fusion_method=model_cfg.fusion_method,
            projection_dim=model_cfg.projection_dim,
            hidden_dims=model_cfg.hidden_dims.copy(),
            dropout=model_cfg.dropout,
            freeze_image_encoder=model_cfg.freeze_image_encoder,
            freeze_text_encoder=model_cfg.freeze_text_encoder,
            text_prompt_template=model_cfg.text_prompt_template,
        )
    else:
        model = SimpleFeatureConcatBaseline(
            clip_model_name=model_cfg.clip_model_name,
            num_emotions=model_cfg.num_emotions,
            hidden_dim=model_cfg.hidden_dims[0] if model_cfg.hidden_dims else 256,
            dropout=model_cfg.dropout,
        )

    return model
