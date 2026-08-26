"""
统一数据集加载模块

支持数据集:
    - Emotion6:   图像 + 6类情感分布
    - Flickr30k:  图像 + 5条文本描述（无情感标签，用于对比学习 / 伪标签生成）
    - GAPED:      图像 + 情感分类 + 效价/唤醒度
    - ArtPhoto:   图像 + 8类情感分类
    - FI:         图像 + 8类情感分类

输出统一格式: (image_tensor, text_prompt, multi_hot_label)
"""
import os
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms as T

from .label_mapping import (
    BASIC_EMOTIONS, NUM_EMOTIONS, EMOTION_TO_IDX,
    map_dataset_label, emotions_to_multihot,
    label_distribution_to_multihot, get_text_prompts_for_emotions,
    multihot_to_names,
)


# ============================================================
# 基础变换
# ============================================================

def get_base_transforms(image_size: int = 224, is_train: bool = True):
    """获取基础图像变换（不含增强）"""
    if is_train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                       std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                       std=[0.26862954, 0.26130258, 0.27577711]),
        ])


def pil_loader(path: str) -> Image.Image:
    """安全加载图像，处理各种格式问题"""
    with Image.open(path) as img:
        img = img.convert("RGB")
    return img


# ============================================================
# Emotion6 Dataset
# 格式: 每张图片对应一个 6+1 维情感分布向量(anger, disgust, fear, joy, sadness, surprise, neutral)
# 通常以 .mat 或 .csv 存储
# ============================================================

class Emotion6Dataset(Dataset):
    """
    Emotion6 数据集加载器。

    支持两种格式:
        1) 目录结构: emotion6_root/
              images/           ← 所有图片
              labels.csv        ← image_name, anger, disgust, fear, joy, sadness, surprise, neutral
        2) 按类别子文件夹:
              anger/ disgust/ fear/ joy/ sadness/ surprise/ neutral/
    """

    EMOTION6_ORDER = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        transform: Optional[Callable] = None,
        label_threshold: float = 0.2,
        use_text_prompt: bool = True,
    ):
        """
        Args:
            root: Emotion6 数据集根目录
            split: train / val / test
            image_size: 图像尺寸
            transform: 自定义图像变换
            label_threshold: LDL分布→multi-hot的阈值
            use_text_prompt: 是否生成文本描述
        """
        self.root = Path(root)
        self.split = split
        self.label_threshold = label_threshold
        self.use_text_prompt = use_text_prompt

        if transform is None:
            self.transform = get_base_transforms(image_size, is_train=(split == "train"))
        else:
            self.transform = transform

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        """加载所有样本，自动检测存储格式"""
        samples = []

        # === 方式 1: CSV 标签文件 ===
        csv_path = self.root / "labels.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            images_dir = self.root / "images"
            if not images_dir.exists():
                images_dir = self.root

            for _, row in df.iterrows():
                img_name = str(row.iloc[0])
                img_path = images_dir / img_name
                if not img_path.exists():
                    img_path = self.root / img_name  # fallback

                if img_path.exists():
                    distribution = row.iloc[1:8].values.astype(np.float32)
                    samples.append({
                        "image_path": str(img_path),
                        "distribution": distribution,
                    })

        # === 方式 2: 按类别子文件夹 ===
        if not samples:
            for emo_name in self.EMOTION6_ORDER:
                emo_dir = self.root / emo_name
                if emo_dir.exists():
                    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
                        for img_path in emo_dir.glob(ext):
                            # 单标签 → 分布中该情感=1.0，其余=0
                            dist = np.zeros(7, dtype=np.float32)
                            dist[self.EMOTION6_ORDER.index(emo_name)] = 1.0
                            samples.append({
                                "image_path": str(img_path),
                                "distribution": dist,
                            })

        # === 方式 3: 结构化的 JSON 标签文件 ===
        if not samples:
            json_path = self.root / "labels.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    labels_data = json.load(f)
                for item in labels_data:
                    img_path = self.root / item.get("image", item.get("filename", ""))
                    if not img_path.exists():
                        img_path = self.root / "images" / item.get("image", item.get("filename", ""))
                    if img_path.exists():
                        dist = np.array(item.get("labels", item.get("distribution", [])), dtype=np.float32)
                        if len(dist) == 7:
                            samples.append({
                                "image_path": str(img_path),
                                "distribution": dist,
                            })

        return samples

    def _distribution_to_multihot(self, dist_7: np.ndarray) -> np.ndarray:
        """
        将 Emotion6 的 7 维分布(含neutral) 映射到我们的 8 维 multi-hot。
        Emotion6: [anger, disgust, fear, joy, sadness, surprise, neutral]
        我们的体系: [joy, sadness, anger, fear, surprise, disgust, trust, anticipation]
        """
        mapping = {
            0: 2,   # anger → anger(2)
            1: 5,   # disgust → disgust(5)
            2: 3,   # fear → fear(3)
            3: 0,   # joy → joy(0)
            4: 1,   # sadness → sadness(1)
            5: 4,   # surprise → surprise(4)
            6: -1,  # neutral → 丢弃
        }

        vec = np.zeros(NUM_EMOTIONS, dtype=np.float32)
        for src_idx, dst_idx in mapping.items():
            if dst_idx >= 0 and dist_7[src_idx] > self.label_threshold:
                vec[dst_idx] = dist_7[src_idx]  # 保留强度信息
        return vec

    def _generate_text_prompt(self, multihot: np.ndarray) -> str:
        """根据标签生成文本描述"""
        active_emotions = multihot_to_names(multihot, threshold=self.label_threshold)
        if not active_emotions:
            return "a photo with neutral expression"
        return get_text_prompts_for_emotions(
            [EMOTION_TO_IDX[e] for e in active_emotions]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # 加载图像
        try:
            image = pil_loader(sample["image_path"])
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        # 标签转换
        multihot = self._distribution_to_multihot(sample["distribution"])
        multihot_binary = (multihot > self.label_threshold).astype(np.float32)

        # 文本提示
        if self.use_text_prompt:
            text_prompt = self._generate_text_prompt(multihot_binary)
        else:
            text_prompt = ""

        return {
            "image": image,
            "text": text_prompt,
            "label": torch.from_numpy(multihot_binary),
            "label_distribution": torch.from_numpy(multihot),
            "dataset": "emotion6",
        }


# ============================================================
# GAPED Dataset
# 格式: 图像 + 情感类别标签 + valence/arousal/dominance
# ============================================================

class GAPEDDataset(Dataset):
    """
    GAPED (Geneva Affective Picture Database) 数据集加载器。

    支持格式:
        1) 目录结构: gaped_root/
              images/
              labels.csv   ← filename, category, valence, arousal
        2) JSON 标注文件
    """

    GAPED_CATEGORIES = ["anger", "disgust", "fear", "sadness", "joy", "surprise", "neutral"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        transform: Optional[Callable] = None,
        use_text_prompt: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.use_text_prompt = use_text_prompt

        if transform is None:
            self.transform = get_base_transforms(image_size, is_train=(split == "train"))
        else:
            self.transform = transform

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        samples = []

        # === 方式 1: CSV 标签文件 ===
        csv_path = self.root / "labels.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            images_dir = self.root / "images"
            if not images_dir.exists():
                images_dir = self.root

            for _, row in df.iterrows():
                img_name = str(row.iloc[0])
                img_path = images_dir / img_name
                if not img_path.exists():
                    img_path = self.root / img_name

                if img_path.exists():
                    # 情感类别
                    category = str(row.iloc[1]).lower().strip() if len(row) > 1 else "neutral"
                    # valence, arousal (可选)
                    valence = float(row.iloc[2]) if len(row) > 2 else 0.0
                    arousal = float(row.iloc[3]) if len(row) > 3 else 0.0

                    samples.append({
                        "image_path": str(img_path),
                        "category": category,
                        "valence": valence,
                        "arousal": arousal,
                    })

        # === 方式 2: 按类别子文件夹 ===
        if not samples:
            for cat in self.GAPED_CATEGORIES:
                cat_dir = self.root / cat
                if cat_dir.exists():
                    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                        for img_path in cat_dir.glob(ext):
                            samples.append({
                                "image_path": str(img_path),
                                "category": cat,
                                "valence": 0.0,
                                "arousal": 0.0,
                            })

        # === 方式 3: JSON 文件 ===
        if not samples:
            json_path = self.root / "labels.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    labels_data = json.load(f)
                for item in labels_data:
                    img_name = item.get("image", item.get("filename", ""))
                    img_path = self.root / "images" / img_name
                    if not img_path.exists():
                        img_path = self.root / img_name
                    if img_path.exists():
                        samples.append({
                            "image_path": str(img_path),
                            "category": item.get("category", item.get("emotion", "neutral")),
                            "valence": item.get("valence", 0.0),
                            "arousal": item.get("arousal", 0.0),
                        })

        return samples

    def _category_to_multihot(self, category: str) -> np.ndarray:
        """将 GAPED 类别映射到 8 维 multi-hot"""
        vec = map_dataset_label("gaped", category)
        return vec

    def _generate_text_prompt(self, multihot: np.ndarray) -> str:
        active = multihot_to_names(multihot)
        if not active:
            return "a photo with neutral expression"
        return get_text_prompts_for_emotions([EMOTION_TO_IDX[e] for e in active])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        try:
            image = pil_loader(sample["image_path"])
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        multihot = self._category_to_multihot(sample["category"])

        if self.use_text_prompt:
            text_prompt = self._generate_text_prompt(multihot)
        else:
            text_prompt = ""

        return {
            "image": image,
            "text": text_prompt,
            "label": torch.from_numpy(multihot),
            "valence": sample["valence"],
            "arousal": sample["arousal"],
            "dataset": "gaped",
        }


# ============================================================
# Flickr30k Dataset（辅助数据 — 无情感标签）
# 用于: 预训练图像-文本对齐 / CLIP微调 / 伪标签生成
# ============================================================

class Flickr30kDataset(Dataset):
    """
    Flickr30k 数据集加载器。

    格式: flickr30k-images.tar (图像) + results_20130124.token (标注)

    results_20130124.token 格式:
        image_name#0\tcaption text
        image_name#1\tcaption text
        ...

    每张图片有 5 条英文描述，无情感标签。
    可用于:
        1) 对比学习预训练（图像-文本匹配）
        2) 用已有模型打伪标签后作为弱监督数据
        3) 作为纯文本/图像编码器的辅助训练数据
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        transform: Optional[Callable] = None,
        captions_per_image: int = 1,      # 每张图返回几条描述
    ):
        self.root = Path(root)
        self.split = split
        self.captions_per_image = captions_per_image

        if transform is None:
            self.transform = get_base_transforms(image_size, is_train=(split == "train"))
        else:
            self.transform = transform

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        samples = []

        # 找标注文件
        token_path = None
        for p in self.root.rglob("results_20130124.token"):
            token_path = p
            break
        if token_path is None:
            for p in self.root.glob("*.token"):
                token_path = p
                break

        if token_path is None:
            print(f"[Flickr30k] 未找到标注文件(results_20130124.token)，请确认数据集路径: {self.root}")
            return samples

        # 解析标注
        df = pd.read_csv(token_path, sep='\t', header=None, names=['image', 'caption'])

        # 按图片名分组（去掉 #N 后缀）
        df['image_name'] = df['image'].str.replace(r'#\d+$', '', regex=True)

        # 找图片目录
        images_dir = self.root / "flickr30k_images"
        if not images_dir.exists():
            # 尝试常见位置
            for d in self.root.iterdir():
                if d.is_dir() and any(d.glob("*.jpg")):
                    images_dir = d
                    break

        # 按图片分组
        grouped = df.groupby('image_name')
        for img_name, group in grouped:
            img_path = images_dir / img_name
            if not img_path.exists():
                # 搜索
                candidates = list(images_dir.rglob(img_name))
                if candidates:
                    img_path = candidates[0]
                else:
                    continue

            captions = group['caption'].tolist()
            samples.append({
                "image_path": str(img_path),
                "captions": captions,
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        try:
            image = pil_loader(sample["image_path"])
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        # 随机选 caption
        captions = sample["captions"]
        if self.captions_per_image >= len(captions):
            selected = captions
        else:
            selected = random.sample(captions, self.captions_per_image)
        text = " ".join(selected)

        # Flickr30k 无情感标签 → 返回全零（或不在训练中使用 label）
        return {
            "image": image,
            "text": text,
            "label": torch.zeros(NUM_EMOTIONS, dtype=torch.float32),
            "dataset": "flickr30k",
        }


# ============================================================
# ArtPhoto / FI Dataset（图像情感分类数据集）
# 格式: 按类别子文件夹: amusement/ anger/ awe/ contentment/ disgust/ excitement/ fear/ sadness/
# ============================================================

class ArtPhotoDataset(Dataset):
    """
    ArtPhoto / FI (Flickr Images) 数据集加载器。

    格式: 按 8 个情感类别组织子文件夹。
    两个数据集结构相同，通过 dataset_type 区分（映射规则不同）。
    """

    VALID_SPLITS = ['train', 'val', 'test']

    def __init__(
        self,
        root: str,
        dataset_type: str = "artphoto",   # "artphoto" or "fi"
        split: str = "train",
        image_size: int = 224,
        transform: Optional[Callable] = None,
        use_text_prompt: bool = True,
    ):
        self.root = Path(root)
        self.dataset_type = dataset_type.lower()
        self.split = split
        self.use_text_prompt = use_text_prompt

        if transform is None:
            self.transform = get_base_transforms(image_size, is_train=(split == "train"))
        else:
            self.transform = transform

        self.samples = self._load_samples()

    def _get_category_dirs(self) -> List[str]:
        """检测存在的类别子文件夹"""
        categories = ["amusement", "anger", "awe", "contentment",
                      "disgust", "excitement", "fear", "sadness"]
        existing = []
        for cat in categories:
            cat_dir = self.root / cat
            if cat_dir.exists():
                existing.append(cat)
        return existing if existing else categories

    def _load_samples(self) -> List[Dict]:
        samples = []

        # === 按子文件夹加载 ===
        for cat in self._get_category_dirs():
            cat_dir = self.root / cat
            if not cat_dir.exists():
                continue
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.webp"]:
                for img_path in cat_dir.glob(ext):
                    samples.append({
                        "image_path": str(img_path),
                        "category": cat,
                    })

        # === 如果是 CSV 格式 ===
        if not samples:
            csv_path = self.root / "labels.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    img_name = str(row.iloc[0])
                    img_path = self.root / img_name
                    if img_path.exists():
                        samples.append({
                            "image_path": str(img_path),
                            "category": str(row.iloc[1]) if len(row) > 1 else "amusement",
                        })

        # 打乱后按比例划分 train/val/test
        rng = random.Random(42)
        rng.shuffle(samples)
        n = len(samples)
        if self.split == "train":
            samples = samples[:int(n * 0.7)]
        elif self.split == "val":
            samples = samples[int(n * 0.7):int(n * 0.85)]
        else:
            samples = samples[int(n * 0.85):]

        return samples

    def _category_to_multihot(self, category: str) -> np.ndarray:
        """将 ArtPhoto/FI 的类别映射到 8 维 multi-hot"""
        vec = map_dataset_label(self.dataset_type, category)
        # 对于 awe（复合情感）需要特殊处理
        if category.lower() == "awe":
            vec = emotions_to_multihot(["awe"])
        return vec

    def _generate_text_prompt(self, multihot: np.ndarray) -> str:
        active = multihot_to_names(multihot)
        if not active:
            return "a photo with neutral expression"
        return get_text_prompts_for_emotions([EMOTION_TO_IDX[e] for e in active])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        try:
            image = pil_loader(sample["image_path"])
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        multihot = self._category_to_multihot(sample["category"])

        if self.use_text_prompt:
            text_prompt = self._generate_text_prompt(multihot)
        else:
            text_prompt = ""

        return {
            "image": image,
            "text": text_prompt,
            "label": torch.from_numpy(multihot),
            "dataset": self.dataset_type,
        }


# ============================================================
# 数据集合并与划分工具
# ============================================================

def build_unified_dataset(
    config,  # Config object from src.config
    datasets_to_include: List[str] = None,
    split: str = "train",
    use_augmentation: bool = None,
) -> Dataset:
    """
    构建统一的数据集，合并多个数据源。

    Args:
        config: 全局配置
        datasets_to_include: 要包含的数据集列表，如 ["emotion6", "gaped", "artphoto", "fi"]
        split: train / val / test
        use_augmentation: 是否在训练集上使用增强

    Returns:
        合并后的 Dataset（可能包装了增强）
    """
    if datasets_to_include is None:
        datasets_to_include = ["emotion6", "gaped", "artphoto"]

    if use_augmentation is None:
        use_augmentation = (split == "train")

    from .augmentation import DualModalAugmentation

    datasets = []
    for ds_name in datasets_to_include:
        try:
            ds = _create_single_dataset(config, ds_name, split)
            if ds is not None and len(ds) > 0:
                datasets.append(ds)
                print(f"  [{ds_name}] {split}: {len(ds)} samples")
        except Exception as e:
            print(f"  [{ds_name}] 加载失败: {e}")

    if not datasets:
        raise RuntimeError(f"没有成功加载任何数据集! 请检查路径配置。")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    # 训练集包装增强
    if use_augmentation:
        combined = DualModalAugmentation(combined, split=split)

    return combined


def _create_single_dataset(config, ds_name: str, split: str) -> Optional[Dataset]:
    """根据数据集名称创建对应的 Dataset 实例"""
    data_cfg = config.data
    model_cfg = config.model

    if ds_name == "emotion6":
        root = data_cfg.emotion6_root
        if not os.path.exists(root):
            print(f"  [Emotion6] 路径不存在: {root}")
            return None
        return Emotion6Dataset(
            root=root, split=split, image_size=data_cfg.image_size,
        )

    elif ds_name == "gaped":
        root = data_cfg.gaped_root
        if not os.path.exists(root):
            print(f"  [GAPED] 路径不存在: {root}")
            return None
        return GAPEDDataset(
            root=root, split=split, image_size=data_cfg.image_size,
        )

    elif ds_name == "artphoto":
        root = data_cfg.artphoto_root
        if not os.path.exists(root):
            print(f"  [ArtPhoto] 路径不存在: {root}")
            return None
        return ArtPhotoDataset(
            root=root, dataset_type="artphoto", split=split,
            image_size=data_cfg.image_size,
        )

    elif ds_name == "fi":
        root = data_cfg.fi_root
        if not os.path.exists(root):
            print(f"  [FI] 路径不存在: {root}")
            return None
        return ArtPhotoDataset(
            root=root, dataset_type="fi", split=split,
            image_size=data_cfg.image_size,
        )

    elif ds_name == "flickr30k":
        root = data_cfg.flickr30k_root
        if not os.path.exists(root):
            print(f"  [Flickr30k] 路径不存在: {root}")
            return None
        return Flickr30kDataset(
            root=root, split=split, image_size=data_cfg.image_size,
        )

    else:
        print(f"  未知数据集: {ds_name}")
        return None


def create_dataloaders(
    config,
    datasets_to_include: List[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    一键创建 train / val / test 三个 DataLoader。

    Returns:
        train_loader, val_loader, test_loader
    """
    train_cfg = config.training

    print("=" * 60)
    print("构建数据集...")
    print("=" * 60)

    train_ds = build_unified_dataset(config, datasets_to_include, split="train", use_augmentation=True)
    val_ds = build_unified_dataset(config, datasets_to_include, split="val", use_augmentation=False)
    test_ds = build_unified_dataset(config, datasets_to_include, split="test", use_augmentation=False)

    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """自定义 batch 整理函数"""
        images = torch.stack([item["image"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])

        # 文本列表，不 stack
        texts = [item.get("text", "") for item in batch]

        return {
            "image": images,
            "text": texts,
            "label": labels,
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=0,   # CPU 训练设为 0，Windows 下多进程容易出问题
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"\nTrain: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds)} samples, {len(val_loader)} batches")
    print(f"Test:  {len(test_ds)} samples, {len(test_loader)} batches")
    print("=" * 60)

    return train_loader, val_loader, test_loader
