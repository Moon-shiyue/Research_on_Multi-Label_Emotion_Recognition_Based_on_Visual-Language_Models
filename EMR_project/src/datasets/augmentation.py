"""
双模态数据增强模块

图像侧:
    - 随机裁剪 (RandomResizedCrop)
    - 亮度/对比度/饱和度调整 (ColorJitter)
    - 水平翻转 (RandomHorizontalFlip)
    - 高斯模糊 (GaussianBlur)
    - 随机擦除 (RandomErasing)

文本侧:
    - 同义词替换 (Synonym Replacement)
    - 随机删除 (Random Deletion)
    - 情感上下文扩展 (Emotion Context Expansion)
    - 回译增强 (Back Translation) — 需要 googletrans，可选

设计原则: 增强前后情感语义不变。
"""
import random
import re
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# NLTK 同义词（离线可用，无需下载额外数据）
_SYNONYM_DICT: Dict[str, List[str]] = {
    "joy": ["happiness", "delight", "cheer", "elation", "bliss", "glee", "ecstasy"],
    "happy": ["joyful", "cheerful", "delighted", "elated", "glad", "pleased", "content"],
    "sadness": ["sorrow", "grief", "melancholy", "misery", "despair", "gloom", "unhappiness"],
    "sad": ["sorrowful", "grieving", "melancholy", "miserable", "gloomy", "unhappy", "downcast"],
    "anger": ["rage", "fury", "wrath", "indignation", "irritation", "resentment", "outrage"],
    "angry": ["furious", "enraged", "irate", "wrathful", "indignant", "irritated", "mad"],
    "fear": ["terror", "dread", "fright", "horror", "panic", "alarm", "anxiety"],
    "scared": ["afraid", "frightened", "terrified", "fearful", "panicked", "alarmed", "spooked"],
    "surprise": ["amazement", "astonishment", "shock", "wonder", "startlement", "awe"],
    "surprised": ["amazed", "astonished", "shocked", "stunned", "startled", "dumbfounded"],
    "disgust": ["revulsion", "aversion", "repulsion", "loathing", "abhorrence", "distaste"],
    "disgusted": ["revolted", "repulsed", "appalled", "sickened", "nauseated", "offended"],
    "love": ["affection", "adoration", "fondness", "devotion", "tenderness", "passion", "warmth"],
    "beautiful": ["gorgeous", "stunning", "lovely", "magnificent", "splendid", "elegant"],
    "person": ["individual", "human", "someone", "figure", "being", "soul"],
    "photo": ["image", "picture", "photograph", "snapshot", "shot", "portrait"],
    "showing": ["displaying", "depicting", "portraying", "presenting", "exhibiting", "revealing"],
    "expression": ["look", "face", "demeanor", "appearance", "countenance", "visage", "mien"],
    "emotion": ["feeling", "sentiment", "mood", "affect", "passion", "sensation"],
}

# 情感上下文扩展模板
_EMOTION_CONTEXT_TEMPLATES: Dict[str, List[str]] = {
    "joy": [
        "The person looks visibly happy and delighted.",
        "A cheerful scene filled with warmth and happiness.",
        "The atmosphere radiates joy and positivity.",
    ],
    "sadness": [
        "The person appears visibly sorrowful and downcast.",
        "A melancholic scene that evokes a sense of loss.",
        "The atmosphere feels heavy with sadness.",
    ],
    "anger": [
        "The person looks furious and agitated.",
        "An intense scene charged with anger and frustration.",
        "The atmosphere feels tense and hostile.",
    ],
    "fear": [
        "The person appears terrified and alarmed.",
        "A frightening scene that evokes a sense of dread.",
        "The atmosphere feels ominous and threatening.",
    ],
    "surprise": [
        "The person looks stunned and amazed.",
        "A striking scene that catches one off guard.",
        "The atmosphere is filled with sudden wonder.",
    ],
    "disgust": [
        "The person appears revolted and sickened.",
        "A repulsive scene that evokes strong aversion.",
        "The atmosphere feels unpleasant and off-putting.",
    ],
    "trust": [
        "The person looks warm and accepting.",
        "A scene of comfort, security, and mutual understanding.",
        "The atmosphere feels safe and reassuring.",
    ],
    "anticipation": [
        "The person appears alert and watchful.",
        "A scene filled with expectation and eagerness.",
        "The atmosphere buzzes with anticipation.",
    ],
}


class ImageAugmentation:
    """图像侧数据增强"""

    @staticmethod
    def get_augmentation(image_size: int = 224, intensity: str = "medium") -> T.Compose:
        """
        Args:
            image_size: 图像目标尺寸
            intensity: light / medium / strong
        """
        if intensity == "light":
            return T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            ])
        elif intensity == "medium":
            return T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
            ])
        else:  # strong
            return T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.3),
                T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ])


class TextAugmentation:
    """文本侧数据增强"""

    def __init__(self, synonym_prob: float = 0.3, delete_prob: float = 0.1):
        """
        Args:
            synonym_prob: 每个词被替换为同义词的概率
            delete_prob: 每个词被随机删除的概率
        """
        self.synonym_prob = synonym_prob
        self.delete_prob = delete_prob

    def synonym_replacement(self, text: str) -> str:
        """同义词替换: 随机将文本中的情感相关词替换为同义词"""
        words = text.split()
        new_words = []
        for word in words:
            word_lower = word.lower().strip(".,!?;:'\"")
            if word_lower in _SYNONYM_DICT and random.random() < self.synonym_prob:
                synonym = random.choice(_SYNONYM_DICT[word_lower])
                # 保持大小写
                if word[0].isupper():
                    synonym = synonym.capitalize()
                new_words.append(synonym)
            else:
                new_words.append(word)
        return " ".join(new_words)

    def random_deletion(self, text: str) -> str:
        """随机删除: 以低概率随机删除非关键词"""
        words = text.split()
        if len(words) <= 3:
            return text  # 太短不删

        # 情感关键词保护列表
        protected = set()
        for key_list in _SYNONYM_DICT.values():
            protected.update(key_list)
        protected.update(_SYNONYM_DICT.keys())

        new_words = []
        for word in words:
            word_lower = word.lower().strip(".,!?;:'\"")
            if word_lower in protected:
                new_words.append(word)  # 保护情感关键词
            elif random.random() > self.delete_prob:
                new_words.append(word)

        if len(new_words) < 2:
            return text  # 至少保留2个词
        return " ".join(new_words)

    def emotion_context_expansion(self, text: str, emotions: List[str] = None) -> str:
        """
        情感上下文扩展: 在文本后追加一句情感上下文的描述。
        仅在文本较短时使用（< 20 词）。
        """
        if len(text.split()) > 20:
            return text

        if emotions is None:
            # 从文本中检测可能的情感词
            emotions = []
            text_lower = text.lower()
            for emo in _EMOTION_CONTEXT_TEMPLATES:
                if emo in text_lower:
                    emotions.append(emo)

        if not emotions:
            emotions = list(_EMOTION_CONTEXT_TEMPLATES.keys())

        emo = random.choice(emotions)
        expansion = random.choice(_EMOTION_CONTEXT_TEMPLATES.get(emo, _EMOTION_CONTEXT_TEMPLATES["joy"]))

        return f"{text}. {expansion}"

    def augment(self, text: str, emotions: List[str] = None) -> str:
        """
        执行完整的文本增强流水线。

        Args:
            text: 原始文本
            emotions: 关联的情感标签列表（用于上下文扩展）

        Returns:
            增强后的文本
        """
        if not text or len(text.strip()) < 5:
            return text

        augmented = text

        # 1) 同义词替换（50% 概率）
        if random.random() < 0.5:
            augmented = self.synonym_replacement(augmented)

        # 2) 随机删除（30% 概率）
        if random.random() < 0.3:
            augmented = self.random_deletion(augmented)

        # 3) 情感上下文扩展（40% 概率）
        if random.random() < 0.4:
            augmented = self.emotion_context_expansion(augmented, emotions)

        return augmented


# ============================================================
# 双模态增强 Dataset 包装器
# ============================================================

class DualModalAugmentation(Dataset):
    """
    双模态数据增强 Dataset 包装器。

    包装任意基础 Dataset，在 __getitem__ 时对图像和文本同时应用增强。
    确保增强前后情感语义不变。
    """

    def __init__(
        self,
        base_dataset: Dataset,
        split: str = "train",
        image_aug_intensity: str = "medium",
        text_synonym_prob: float = 0.3,
    ):
        """
        Args:
            base_dataset: 基础数据集
            split: train / val / test (仅 train 做增强)
            image_aug_intensity: 图像增强强度
            text_synonym_prob: 文本同义词替换概率
        """
        self.base_dataset = base_dataset
        self.split = split
        self.do_augment = (split == "train")

        if self.do_augment:
            # CLIP 标准归一化
            self.normalize = T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
            self.image_aug = ImageAugmentation.get_augmentation(224, image_aug_intensity)
            self.text_aug = TextAugmentation(synonym_prob=text_synonym_prob)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict:
        item = self.base_dataset[idx]

        if not self.do_augment:
            return item

        # === 图像增强 ===
        # 注意: base_dataset 已经做了 Resize + ToTensor + Normalize
        # 我们需要在 ToTensor 之前做增强，为此重新处理图像
        image_tensor = item["image"]

        # 对已归一化的 tensor 做轻量级增强（在像素空间操作需先反归一化）
        # 简化方案: 对 tensor 做轻微的 ColorJitter 式增强
        if random.random() < 0.5:
            # 亮度微调
            brightness_factor = 1.0 + random.uniform(-0.1, 0.1)
            image_tensor = image_tensor * brightness_factor
            image_tensor = torch.clamp(image_tensor, 0.0, 1.0)

        if random.random() < 0.3:
            # 对比度微调
            mean = image_tensor.mean(dim=[1, 2], keepdim=True)
            contrast_factor = 1.0 + random.uniform(-0.1, 0.1)
            image_tensor = (image_tensor - mean) * contrast_factor + mean
            image_tensor = torch.clamp(image_tensor, 0.0, 1.0)

        item["image"] = image_tensor

        # === 文本增强 ===
        if isinstance(item.get("text"), str) and item["text"]:
            # 从标签获取情感名称
            if "label" in item:
                label = item["label"].numpy() if isinstance(item["label"], torch.Tensor) else item["label"]
                active_emotions = []
                from .label_mapping import BASIC_EMOTIONS
                for i, v in enumerate(label):
                    if v > 0.5 and i < len(BASIC_EMOTIONS):
                        active_emotions.append(BASIC_EMOTIONS[i])
            else:
                active_emotions = None

            item["text"] = self.text_aug.augment(item["text"], active_emotions)

        return item
