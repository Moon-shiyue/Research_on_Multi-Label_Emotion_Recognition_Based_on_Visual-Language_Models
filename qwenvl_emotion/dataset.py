"""
Qwen2.5-VL 方案 — 数据集加载
与 CLIP 方案使用相同的数据结构，仅图像预处理参数不同
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import csv
from typing import Optional, List, Tuple, Dict
import numpy as np


def get_qwenvl_transforms(
    image_size: Tuple[int, int] = (448, 448),
    is_train: bool = True,
) -> transforms.Compose:
    """
    Qwen2.5-VL 推荐的图像预处理
    - 分辨率更高 (448 而非 CLIP 的 224)
    - 使用 Qwen 官方推荐的标准化参数
    """
    # ImageNet 标准归一化（Qwen-VL 使用此参数）
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


class QwenVLEmotionDataset(Dataset):
    """
    Qwen-VL 多标记情感数据集

    与 CLIP 方案的 MultiLabelEmotionDataset 共享相同的数据格式：
      data_root/
        images/       ← 图像文件
        labels.csv    ← 多热编码标签

    额外输出：
      - pil_image: 原始 PIL Image（Qwen-VL processor 需要）
      - prompt: 训练 prompt 文本
      - answer: ground truth JSON 输出
    """

    def __init__(
        self,
        data_root: str,
        emotion_labels: List[str],
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (448, 448),
        label_file: Optional[str] = None,
    ):
        self.data_root = data_root
        self.emotion_labels = emotion_labels
        self.num_labels = len(emotion_labels)
        self.split = split
        self.label_to_idx = {label: i for i, label in enumerate(emotion_labels)}

        self.image_dir = os.path.join(data_root, "images")
        # 仅用于 tensor 模式；Qwen-VL 推理直接用 PIL
        self.transform = transform or get_qwenvl_transforms(image_size, is_train=(split == "train"))
        self.image_size = image_size

        self.samples = self._load_labels(label_file)

    def _load_labels(self, label_file=None):
        if label_file is None:
            for fname in ["labels.csv", "labels.json", f"{self.split}.csv", f"{self.split}.json"]:
                fpath = os.path.join(self.data_root, fname)
                if os.path.exists(fpath):
                    label_file = fpath
                    break
        if label_file is None:
            raise FileNotFoundError(f"在 {self.data_root} 下未找到标签文件")

        samples = []
        if label_file.endswith(".csv"):
            with open(label_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get("filename", row.get("image", ""))
                    multihot = torch.zeros(self.num_labels)
                    for col in reader.fieldnames:
                        if col in self.label_to_idx:
                            if row.get(col, "").strip() in ("1", "1.0", "True", "true"):
                                multihot[self.label_to_idx[col]] = 1.0
                    samples.append({"image": filename, "labels": multihot})
        elif label_file.endswith(".json"):
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.items() if isinstance(data, dict) else [(d["filename"], d["emotions"]) for d in data]
            for filename, emotions in items:
                multihot = torch.zeros(self.num_labels)
                for emo in emotions:
                    if emo in self.label_to_idx:
                        multihot[self.label_to_idx[emo]] = 1.0
                samples.append({"image": filename, "labels": multihot})

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.image_dir, sample["image"])

        pil_image = Image.open(image_path).convert("RGB")

        return {
            "pixel_values": self.transform(pil_image),  # tensor 模式
            "pil_image": pil_image,                       # PIL 模式（给 processor 用）
            "labels": sample["labels"],
            "image_name": sample["image"],
        }


def create_qwenvl_dataloaders(
    data_roots: List[str],
    emotion_labels: List[str],
    batch_size: int = 4,
    image_size: Tuple[int, int] = (448, 448),
    num_workers: int = 2,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """创建 Qwen-VL 方案的 DataLoader（逻辑同 CLIP 方案）"""
    from torch.utils.data import ConcatDataset, random_split

    all_datasets = []
    for root in data_roots:
        if not os.path.exists(root):
            print(f"⚠ 跳过不存在的目录: {root}")
            continue
        ds = QwenVLEmotionDataset(root, emotion_labels, "train", image_size=image_size)
        all_datasets.append(ds)

    if not all_datasets:
        raise ValueError("没有找到有效的数据集目录！")

    full_dataset = ConcatDataset(all_datasets)

    total = len(full_dataset)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True) if val_size > 0 else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True) if test_size > 0 else None

    return train_loader, val_loader, test_loader
