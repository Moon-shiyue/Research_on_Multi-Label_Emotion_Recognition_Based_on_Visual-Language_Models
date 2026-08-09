"""
基础编码器模块
基于 CLIP (ViT-B/32) 的视觉编码器与文本编码器封装

提供统一的特征提取接口，支持：
- 视觉特征提取：图像 → 512 维特征向量
- 文本特征提取：情感标签文本 → 512 维特征向量
- 冻结/微调控制
- 梯度检查点支持（节省显存）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel, CLIPTextModel
import warnings


class VisualEncoder(nn.Module):
    """
    CLIP 视觉编码器封装 (ViT-B/32)

    提取图像的情感相关视觉特征，输出统一维度的特征向量。

    架构:
        CLIP ViT-B/32 (frozen/unfrozen) → Linear Projection → L2 Norm → 512-dim feature

    输入:
        - pixel_values: (B, 3, 224, 224) 预处理后的图像张量
        - patch_embeddings: 可选，直接传入 patch 嵌入

    输出:
        - visual_features: (B, 512) 全局视觉特征向量
        - patch_features: (B, 50, 768) patch 级别的视觉特征（用于细粒度注意力）
          (1 [CLS] + 49 patches, ViT-B/32 对 224×224 产生 7×7=49 patches)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        projection_dim: int = 512,
        freeze: bool = False,
        use_grad_checkpoint: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.freeze = freeze

        # 加载 CLIP 视觉塔
        try:
            self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"无法加载 CLIP 视觉模型 '{model_name}'。"
                f"请确保已安装 transformers 且网络可访问 Hugging Face。\n"
                f"原始错误: {e}"
            )

        self.hidden_size = self.vision_model.config.hidden_size  # 768 for ViT-B/32

        # 视觉特征投影层：将 CLIP 输出映射到统一维度
        self.visual_projection = nn.Sequential(
            nn.Linear(self.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

        # 梯度检查点
        if use_grad_checkpoint and not freeze:
            self.vision_model.gradient_checkpointing_enable()

        # 冻结控制
        if freeze:
            self._freeze_encoder()

        self._init_weights()

    def _freeze_encoder(self):
        """冻结 CLIP 视觉编码器参数"""
        for param in self.vision_model.parameters():
            param.requires_grad = False

    def _init_weights(self):
        """初始化投影层权重"""
        for module in self.visual_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_patches: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            pixel_values: (B, 3, H, W) 图像张量
            return_patches: 是否返回 patch 级特征

        Returns:
            字典包含:
                - global_feature: (B, projection_dim) 全局视觉特征
                - patch_features: (B, num_patches, hidden_size) patch 特征 (if return_patches)
                - cls_token: (B, hidden_size) CLS token 原始输出
        """
        # CLIP 视觉编码
        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        # 获取特征
        pooler_output = outputs.pooler_output      # (B, hidden_size)
        last_hidden = outputs.last_hidden_state     # (B, seq_len, hidden_size)

        # 投影到统一维度
        global_feature = self.visual_projection(pooler_output)  # (B, projection_dim)

        result = {
            "global_feature": global_feature,
            "cls_token": pooler_output,
        }

        if return_patches:
            # last_hidden[:, 1:, :] 去除 CLS token，保留 patch tokens
            result["patch_features"] = last_hidden[:, 1:, :]  # (B, 49, 768)

        return result


class TextEncoder(nn.Module):
    """
    CLIP 文本编码器封装

    将情感标签的自然语言描述编码为语义特征向量。

    架构:
        CLIP Text Transformer (frozen/unfrozen) → Linear Projection → L2 Norm → 512-dim feature

    支持的输入形式:
        - 原始文本列表: ["a photo expressing joy", ...]
        - 预计算的 token: input_ids + attention_mask

    输入:
        - input_ids: (B, seq_len) tokenized 文本
        - attention_mask: (B, seq_len) 注意力掩码

    输出:
        - text_features: (B, projection_dim) 文本特征向量
        - token_features: (B, seq_len, hidden_size) token 级特征
    """

    # 情感标签的 prompt 模板
    EMOTION_PROMPTS = {
        "joy": "an image expressing joy and happiness",
        "sadness": "an image expressing sadness and sorrow",
        "anger": "an image expressing anger and rage",
        "fear": "an image expressing fear and terror",
        "surprise": "an image expressing surprise and shock",
        "disgust": "an image expressing disgust and revulsion",
        "love": "an image expressing love and affection",
        "peace": "an image expressing peace and tranquility",
        "amusement": "an image expressing amusement and fun",
        "awe": "an image expressing awe and wonder",
        "contentment": "an image expressing contentment and satisfaction",
        "excitement": "an image expressing excitement and thrill",
    }

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        projection_dim: int = 512,
        freeze: bool = False,
        use_grad_checkpoint: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.freeze = freeze

        # 加载 CLIP 文本塔
        try:
            self.text_model = CLIPTextModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"无法加载 CLIP 文本模型 '{model_name}'。\n"
                f"原始错误: {e}"
            )

        self.hidden_size = self.text_model.config.hidden_size  # 512 for CLIP text

        # 文本特征投影层
        self.text_projection = nn.Sequential(
            nn.Linear(self.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

        # 梯度检查点
        if use_grad_checkpoint and not freeze:
            self.text_model.gradient_checkpointing_enable()

        if freeze:
            self._freeze_encoder()

        self._init_weights()

    def _freeze_encoder(self):
        """冻结 CLIP 文本编码器参数"""
        for param in self.text_model.parameters():
            param.requires_grad = False

    def _init_weights(self):
        """初始化投影层权重"""
        for module in self.text_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_emotion_prompts(
        self, emotion_labels: List[str]
    ) -> List[str]:
        """
        将情感标签转换为自然语言 prompt

        Args:
            emotion_labels: 情感标签列表，如 ["joy", "sadness"]

        Returns:
            prompt 文本列表
        """
        prompts = []
        for label in emotion_labels:
            prompt = self.EMOTION_PROMPTS.get(
                label, f"an image expressing {label}"
            )
            prompts.append(prompt)
        return prompts

    def tokenize(
        self,
        texts: List[str],
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
        """
        将文本列表 tokenize

        Args:
            texts: 文本列表
            device: 目标设备

        Returns:
            {"input_ids": ..., "attention_mask": ...}
        """
        tokens = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,  # CLIP 的最大文本长度
        )
        if device is not None:
            tokens = {k: v.to(device) for k, v in tokens.items()}
        return tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: (B, seq_len) token IDs
            attention_mask: (B, seq_len) 注意力掩码
            return_tokens: 是否返回 token 级特征

        Returns:
            字典包含:
                - global_feature: (B, projection_dim) 全局文本特征
                - token_features: (B, seq_len, hidden_size) token 特征 (if return_tokens)
        """
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # CLIP 文本模型用 EOS token 的表示作为整体特征
        pooler_output = outputs.pooler_output  # (B, hidden_size)
        last_hidden = outputs.last_hidden_state  # (B, seq_len, hidden_size)

        global_feature = self.text_projection(pooler_output)  # (B, projection_dim)

        result = {
            "global_feature": global_feature,
        }

        if return_tokens:
            result["token_features"] = last_hidden

        return result


class CLIPEncoder(nn.Module):
    """
    CLIP 双塔编码器统一封装

    同时管理视觉编码器和文本编码器，提供统一的特征提取接口。
    对外暴露简洁的 API，隐藏底层 CLIP 模型管理的复杂性。

    典型用法:
        encoder = CLIPEncoder()
        visual_out = encoder.encode_image(pixel_values)
        text_out = encoder.encode_text(["joy", "sadness", ...])
        text_out_from_raw = encoder.encode_text_from_raw(["a happy photo", ...])
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        projection_dim: int = 512,
        freeze_visual: bool = False,
        freeze_text: bool = False,
        use_grad_checkpoint: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.projection_dim = projection_dim

        # 初始化双编码器
        self.visual_encoder = VisualEncoder(
            model_name=model_name,
            projection_dim=projection_dim,
            freeze=freeze_visual,
            use_grad_checkpoint=use_grad_checkpoint,
        )

        self.text_encoder = TextEncoder(
            model_name=model_name,
            projection_dim=projection_dim,
            freeze=freeze_text,
            use_grad_checkpoint=use_grad_checkpoint,
        )

        self.visual_dim = projection_dim
        self.text_dim = projection_dim

        print(f"[CLIPEncoder] 初始化完成")
        print(f"  - 视觉编码器: {model_name} (frozen={freeze_visual})")
        print(f"  - 文本编码器: {model_name} (frozen={freeze_text})")
        print(f"  - 投影维度: {projection_dim}")

        # 统计参数量
        visual_params = sum(p.numel() for p in self.visual_encoder.parameters())
        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        trainable_visual = sum(p.numel() for p in self.visual_encoder.parameters() if p.requires_grad)
        trainable_text = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        print(f"  - 视觉编码器参数: {visual_params:,} (可训练: {trainable_visual:,})")
        print(f"  - 文本编码器参数: {text_params:,} (可训练: {trainable_text:,})")

    def encode_image(
        self,
        pixel_values: torch.Tensor,
        return_patches: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """编码图像，提取视觉特征"""
        return self.visual_encoder(pixel_values, return_patches=return_patches)

    def encode_text(
        self,
        emotion_labels: List[str],
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
        """根据情感标签列表编码文本特征（自动生成 prompt）"""
        prompts = self.text_encoder.get_emotion_prompts(emotion_labels)
        tokens = self.text_encoder.tokenize(prompts, device=device)
        return self.text_encoder(**tokens)

    def encode_text_from_raw(
        self,
        texts: List[str],
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
        """根据原始文本编码文本特征"""
        tokens = self.text_encoder.tokenize(texts, device=device)
        return self.text_encoder(**tokens)

    def forward(
        self,
        pixel_values: torch.Tensor,
        emotion_labels: List[str],
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
        """
        联合前向传播：同时编码图像和文本

        Args:
            pixel_values: (B, 3, H, W) 图像
            emotion_labels: 情感标签列表
            device: 设备

        Returns:
            dict:
                - visual_features: (B, projection_dim)
                - text_features: (num_labels, projection_dim)
                - patch_features: (B, num_patches, hidden_size)
        """
        visual_out = self.encode_image(pixel_values, return_patches=True)
        text_out = self.encode_text(emotion_labels, device=device)

        return {
            "visual_features": visual_out["global_feature"],
            "text_features": text_out["global_feature"],
            "patch_features": visual_out.get("patch_features", None),
        }


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CLIPEncoder 模块测试")
    print("=" * 60)

    # 测试初始化
    encoder = CLIPEncoder(
        projection_dim=512,
        freeze_visual=True,  # 测试时冻结以节省显存
        freeze_text=True,
    )

    # 测试图像编码（使用随机输入模拟）
    dummy_image = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        visual_out = encoder.encode_image(dummy_image)

    print(f"\n视觉编码输出:")
    print(f"  global_feature: {visual_out['global_feature'].shape}")
    if "patch_features" in visual_out:
        print(f"  patch_features: {visual_out['patch_features'].shape}")

    # 测试文本编码
    test_labels = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
    with torch.no_grad():
        text_out = encoder.encode_text(test_labels)

    print(f"\n文本编码输出:")
    print(f"  global_feature: {text_out['global_feature'].shape}")
    print(f"  标签数量: {len(test_labels)}")

    print("\n✅ 基础编码器测试通过！")
