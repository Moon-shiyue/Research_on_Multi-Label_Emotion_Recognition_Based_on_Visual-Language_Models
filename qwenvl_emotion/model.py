"""
Qwen2.5-VL LoRA 微调模型
Qwen-VL Multi-Label Emotion Recognition Model

核心思路：
  利用 Qwen2.5-VL 原生的多模态理解能力，通过 LoRA 轻量微调，
  让模型学会以结构化的 JSON 格式输出多标记情感识别结果。

与 CLIP 方案的关键区别：
  ┌──────────────────────────────────────────────────────────┐
  │ CLIP 方案                                                │
  │   Image → ViT encoder → 512d                             │
  │   Text → CLIP Text encoder → 512d                        │
  │   → 手写跨模态融合 → 手写标签关联 → 分类头                  │
  │   需要自己设计所有中间模块                                  │
  ├──────────────────────────────────────────────────────────┤
  │ Qwen-VL 方案                                             │
  │   Image + Prompt → Qwen2.5-VL (LoRA) → JSON Output        │
  │   模型内部自动完成：编码、融合、推理、标签关联               │
  │   只需要设计 Prompt 和解析输出                             │
  └──────────────────────────────────────────────────────────┘

LoRA 原理：
  原始权重 W ∈ R^{d×k}
  LoRA: W' = W + ΔW = W + B·A,   B∈R^{d×r}, A∈R^{r×k}, r≪min(d,k)
  仅训练 B 和 A，冻结 W
  参数量：r=16 时约为原始模型的 ~1-2%
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
import json
import re
import warnings
import sys

from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)


class QwenVLEmotionModel(nn.Module):
    """
    Qwen2.5-VL + LoRA 多标记情感识别模型

    使用方式:
        # 初始化
        model = QwenVLEmotionModel(config)
        model.setup_lora()           # 注入 LoRA
        model.prepare_for_training() # 冻结基座，仅训练 LoRA

        # 训练（生成式）
        for batch in dataloader:
            inputs = model.prepare_inputs(
                images=batch["pixel_values"],
                labels=batch["emotion_labels"],
            )
            loss = model(**inputs).loss
            loss.backward()

        # 推理
        results = model.predict(image)
        # {"joy": 0.92, "sadness": 0.03, ...}
    """

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            from .config import default_model_config
            config = default_model_config

        self.config = config

        # 量化配置（可选）
        quant_config = None
        if config.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif config.load_in_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        # 确定数据类型
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

        print(f"[QwenVLEmotionModel] 加载模型: {config.model_name}")
        print(f"  - 数据类型: {config.torch_dtype}")
        print(f"  - 4bit量化: {config.load_in_4bit}")
        print(f"  - 8bit量化: {config.load_in_8bit}")

        # 加载模型
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=torch_dtype,
            quantization_config=quant_config,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )

        # 配置 tokenizer（Qwen-VL 的 padding 策略）
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        self.lora_configured = False
        self.emotion_labels = config.emotion_labels
        self.num_emotions = config.num_emotions

        print(f"  - 模型加载完成")
        print(f"  - 参数量: {sum(p.numel() for p in self.model.parameters())/1e9:.2f}B")

    def setup_lora(self):
        """注入 LoRA 适配器"""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.lora_configured = True

        # 统计 LoRA 参数
        lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"\n[LoRA 注入完成]")
        print(f"  - LoRA rank: {self.config.lora_r}")
        print(f"  - 可训练参数: {lora_params:,} ({100*lora_params/total_params:.2f}%)")
        print(f"  - 总参数: {total_params:,}")

    def prepare_for_training(self):
        """准备训练：确保只有 LoRA 参数可训练"""
        if not self.lora_configured:
            self.setup_lora()

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # 仅解冻 LoRA 参数
        for name, param in self.model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[训练准备] 可训练参数: {trainable:,}")

    def _build_training_prompt(
        self,
        emotion_labels_list: List[str],
    ) -> str:
        """
        构建训练用的 Prompt（含 ground truth 输出格式）

        采用思维链策略，教模型按结构输出
        """
        labels_str = ", ".join(emotion_labels_list)
        emotions_str = ", ".join(self.emotion_labels)

        prompt = f"""分析这张图片表达的情感。

可选情感: {emotions_str}

请严格按以下JSON格式输出（只输出JSON，不要其他文字）：
{{"emotions": {{"joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0, "love": 0.0, "peace": 0.0, "amusement": 0.0, "awe": 0.0, "contentment": 0.0, "excitement": 0.0}}}}"""
        return prompt

    def _build_answer(
        self,
        emotion_multihot: torch.Tensor,  # (num_emotions,)
    ) -> str:
        """构建训练用的 ground truth 回答 JSON"""
        scores = {}
        for i, label in enumerate(self.emotion_labels):
            scores[label] = float(emotion_multihot[i].item())

        answer = json.dumps({"emotions": scores}, ensure_ascii=False)
        return answer

    def prepare_inputs(
        self,
        images: torch.Tensor,           # (B, 3, H, W) 或 PIL Images 列表
        emotion_multihot: torch.Tensor,  # (B, num_emotions) 多热标签
    ) -> Dict:
        """
        准备训练的模型输入（图像 + 文本 prompt → 文本 answer）

        使用 Qwen2-VL 的对话格式构建消息
        """
        B = images.shape[0] if isinstance(images, torch.Tensor) else len(images)

        # 如果输入是 tensor，转为 PIL images
        if isinstance(images, torch.Tensor):
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()
            # 反归一化（CLIP 均值/标准差）
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            images_denorm = images.cpu() * std + mean
            images_denorm = torch.clamp(images_denorm, 0, 1)
            pil_images = [to_pil(img) for img in images_denorm]
        else:
            pil_images = images

        # 构建消息列表
        messages_list = []
        answers = []

        for i in range(B):
            # 获取该样本的正情感标签名称
            positive_labels = [
                self.emotion_labels[j]
                for j in range(self.num_emotions)
                if emotion_multihot[i, j] > 0.5
            ]

            prompt = self._build_training_prompt(positive_labels)
            answer = self._build_answer(emotion_multihot[i])

            # Qwen2-VL 对话格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_images[i]},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer},
                    ],
                },
            ]
            messages_list.append(messages)
            answers.append(answer)

        # 使用 processor 处理
        # Qwen-VL processor 需要特殊的聊天模板
        texts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in messages_list
        ]

        inputs = self.processor(
            text=texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        # 移动到模型设备
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        return inputs

    @torch.no_grad()
    def predict(
        self,
        images,                          # PIL Image 或 tensor (B, 3, H, W)
        prompt_strategy: str = None,
        max_new_tokens: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        推理接口：输入图片，输出多标记情感概率

        Args:
            images: 单张 PIL Image 或 batch tensor
            prompt_strategy: "direct" | "cot" | "contrastive"
            max_new_tokens: 最大生成长度

        Returns:
            dict:
                - probabilities: (B, num_emotions) [0,1] 置信度
                - predictions: (B, num_emotions) 0/1 预测
                - raw_output: 原始 JSON 字符串
                - description: (仅 CoT) 图片描述文本
        """
        from .config import PROMPT_COT, PROMPT_DIRECT

        self.model.eval()

        prompt_strategy = prompt_strategy or self.config.prompt_strategy
        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        # 选择 prompt
        if prompt_strategy == "cot":
            prompt = PROMPT_COT
        elif prompt_strategy == "direct":
            prompt = PROMPT_DIRECT
        else:
            prompt = PROMPT_DIRECT

        # 处理输入
        if isinstance(images, torch.Tensor):
            B = images.shape[0]
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            images_denorm = images.cpu() * std + mean
            images_denorm = torch.clamp(images_denorm, 0, 1)
            pil_images = [to_pil(img) for img in images_denorm]
        else:
            B = 1
            pil_images = [images] if not isinstance(images, list) else images

        # 逐个推理（VLM 逐个处理更稳定）
        all_probs = []
        all_preds = []
        all_raw = []
        all_desc = []

        for pil_img in pil_images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                images=[pil_img],
                return_tensors="pt",
                padding=True,
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,       # 贪婪解码，保证确定性输出
                temperature=None,
            )

            # 解码输出（去除输入部分）
            input_len = inputs["input_ids"].shape[1]
            output_ids = generated_ids[0, input_len:]
            raw_text = self.processor.tokenizer.decode(output_ids, skip_special_tokens=True)

            # 解析 JSON
            probs, description = self._parse_output(raw_text)
            preds = (probs > 0.5).float()

            all_probs.append(probs)
            all_preds.append(preds)
            all_raw.append(raw_text)
            all_desc.append(description)

        return {
            "probabilities": torch.stack(all_probs) if B > 1 else all_probs[0].unsqueeze(0),
            "predictions": torch.stack(all_preds) if B > 1 else all_preds[0].unsqueeze(0),
            "raw_output": all_raw,
            "description": all_desc,
        }

    def _parse_output(self, raw_text: str) -> Tuple[torch.Tensor, str]:
        """
        解析模型输出的 JSON，提取情感分数

        Returns:
            probs: (num_emotions,) 概率张量
            description: 图片描述文本（CoT 模式）
        """
        probs = torch.zeros(self.num_emotions)

        # 尝试提取 JSON
        description = ""

        # 方法1：提取 ```json ``` 代码块
        json_match = re.search(r'```json\s*(.*?)\s*```', raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 方法2：提取 { } 包围的最外层 JSON
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = raw_text

        try:
            data = json.loads(json_str)

            # 处理 {"emotions": {...}} 格式
            if "emotions" in data:
                scores = data["emotions"]
            elif "scores" in data:
                scores = data["scores"]
            else:
                scores = data

            # 提取各情感分数
            for i, label in enumerate(self.emotion_labels):
                if label in scores:
                    val = scores[label]
                    if isinstance(val, (int, float)):
                        probs[i] = float(val)

            # 提取描述
            if "description" in data:
                description = str(data["description"])
            if "analysis" in data:
                description = str(data["analysis"])

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # 兜底：按关键词匹配
            print(f"  ⚠ JSON 解析失败，使用关键词回退: {e}")
            for i, label in enumerate(self.emotion_labels):
                if label.lower() in raw_text.lower():
                    probs[i] = 0.5  # 弱信号

        return probs, description

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_type: str = "bce",
    ) -> torch.Tensor:
        """
        计算多标记分类损失（评估用）

        注意：生成式训练时 loss 由模型内部计算。
        此方法用于评估时比较预测和真实标签。
        """
        import torch.nn.functional as F

        probs = torch.sigmoid(logits)

        if loss_type == "bce":
            return F.binary_cross_entropy_with_logits(logits, labels)
        elif loss_type == "asymmetric":
            gamma_pos, gamma_neg = 1.0, 4.0
            pos_loss = -((1 - probs) ** gamma_pos) * torch.log(probs + 1e-8) * labels
            probs_neg = probs.clamp(max=1 - 0.05)
            neg_loss = -((probs_neg) ** gamma_neg) * torch.log(1 - probs_neg + 1e-8) * (1 - labels)
            return (pos_loss + neg_loss).mean()
        else:
            return F.binary_cross_entropy_with_logits(logits, labels)

    def save_lora(self, path: str):
        """保存 LoRA 适配器权重"""
        if self.lora_configured:
            self.model.save_pretrained(path)
            print(f"LoRA 权重已保存至: {path}")
        else:
            warnings.warn("LoRA 未配置，保存完整模型...")
            self.model.save_pretrained(path)

    def load_lora(self, path: str):
        """加载 LoRA 适配器权重"""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, path)
        print(f"LoRA 权重已从 {path} 加载")

    def merge_and_save(self, path: str):
        """合并 LoRA 到基座模型并保存（推理部署用）"""
        if self.lora_configured:
            merged = self.model.merge_and_unload()
            merged.save_pretrained(path)
            self.processor.save_pretrained(path)
            print(f"合并模型已保存至: {path}")
        else:
            self.model.save_pretrained(path)


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    # Windows 控制台默认 GBK 编码无法输出 emoji，强制 UTF-8
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Qwen2.5-VL 情感识别模型 — 配置测试")
    print("=" * 60)

    import sys
    import os
    # 直接运行时脚本目录不在包路径中，需手动加入
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from qwenvl_emotion.config import QwenVLConfig, PROMPT_COT

    print(f"\n默认配置:")
    cfg = QwenVLConfig()
    for field_name in ["model_name", "lora_r", "lora_alpha", "prompt_strategy"]:
        print(f"  {field_name}: {getattr(cfg, field_name)}")

    print(f"\nCoT Prompt 预览:")
    print(PROMPT_COT[:300] + "...")

    print(f"\n情感标签: {cfg.emotion_labels}")
    print(f"标签数量: {cfg.num_emotions}")

    print("\n---")
    print("⚠ 实际模型加载需要 GPU + 网络（下载 Qwen2.5-VL 权重）")
    print("请在有 GPU 的服务器上运行:")
    print("  from qwenvl_emotion import QwenVLEmotionModel")
    print("  model = QwenVLEmotionModel()")
    print("  model.setup_lora()")
    print("✅ 配置测试通过！")
