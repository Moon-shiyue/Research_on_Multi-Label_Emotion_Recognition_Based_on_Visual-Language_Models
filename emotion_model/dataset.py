"""
数据集加载与预处理模块
支持多标记情感识别任务的图像-标签数据加载

支持的数据集:
  - Emotion6: http://chenlab.ece.cornell.edu/downloads.html
  - ArtPhoto: https://www.imageemotion.org/
  - GAPED: https://www.unige.ch/cisa/research/materials-and-online-research/research-material/
  - Flickr30k (情感标注版)

数据格式要求:
  每个数据集目录下需要:
    images/          — 图像文件
    labels.csv       — 标签文件 (columns: filename, emotion1, emotion2, ...)
                       或
    labels.json      — 标签文件 {filename: [emotion1, emotion2, ...]}

  其中 emotion 列值为 0/1 表示该情感标签是否存在。
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import csv
from typing import Optional, Dict, List, Tuple, Union
import numpy as np


# ============================================================
# 图像预处理
# ============================================================

def get_default_transforms(
    image_size: Tuple[int, int] = (224, 224),
    is_train: bool = True,
) -> transforms.Compose:
    """
    获取默认的图像预处理 pipeline

    使用 CLIP 官方推荐的预处理参数（均值/标准差）。
    CLIP 使用特定的归一化值，而非 ImageNet 标准值。
    """
    # CLIP 使用的均值和标准差
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)

    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])


# ============================================================
# 多标记情感数据集
# ============================================================

class MultiLabelEmotionDataset(Dataset):
    """
    多标记情感识别数据集

    支持加载多种格式的标签文件 (CSV / JSON)，
    自动处理不同数据集的标签映射。

    CSV 格式示例:
        filename,joy,sadness,anger,fear,surprise,disgust
        img001.jpg,1,0,0,0,0,0
        img002.jpg,0,1,0,0,0,0

    JSON 格式示例:
        {
            "img001.jpg": ["joy"],
            "img002.jpg": ["sadness", "fear"],
        }
    """

    def __init__(
        self,
        data_root: str,
        emotion_labels: List[str],
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (224, 224),
        label_file: Optional[str] = None,
        label_format: Optional[str] = None,  # "csv" | "json" | "auto"
    ):
        """
        Args:
            data_root: 数据集根目录
            emotion_labels: 统一情感标签列表
            split: "train" / "val" / "test"
            transform: 自定义图像预处理
            image_size: 图像尺寸
            label_file: 标签文件路径（默认在 data_root 下查找）
            label_format: 标签文件格式
        """
        self.data_root = data_root
        self.emotion_labels = emotion_labels
        self.num_labels = len(emotion_labels)
        self.split = split
        self.label_to_idx = {label: i for i, label in enumerate(emotion_labels)}

        # 图像目录
        self.image_dir = os.path.join(data_root, "images")

        # 图像预处理
        self.transform = transform or get_default_transforms(
            image_size, is_train=(split == "train")
        )

        # 加载标签
        self.samples = self._load_labels(label_file, label_format)

        print(f"[MultiLabelEmotionDataset] {split} split 加载完成")
        print(f"  - 数据根目录: {data_root}")
        print(f"  - 样本数: {len(self.samples)}")
        print(f"  - 标签数: {self.num_labels}")

    def _load_labels(
        self,
        label_file: Optional[str],
        label_format: Optional[str],
    ) -> List[Dict]:
        """
        加载标签文件

        Returns:
            List[Dict]: [{"image": "img001.jpg", "labels": tensor([1,0,0,...])}, ...]
        """
        # 自动检测标签文件
        if label_file is None:
            label_file = self._find_label_file()

        # 自动检测格式
        if label_format is None:
            if label_file.endswith(".csv"):
                label_format = "csv"
            elif label_file.endswith(".json"):
                label_format = "json"
            else:
                raise ValueError(f"无法自动识别标签文件格式: {label_file}")

        # 解析标签
        if label_format == "csv":
            raw_labels = self._load_csv_labels(label_file)
        elif label_format == "json":
            raw_labels = self._load_json_labels(label_file)
        else:
            raise ValueError(f"不支持的标签格式: {label_format}")

        # 统一转换为多热编码格式
        samples = []
        for filename, emotion_list in raw_labels:
            multihot = torch.zeros(self.num_labels)
            for emotion in emotion_list:
                if emotion in self.label_to_idx:
                    multihot[self.label_to_idx[emotion]] = 1.0

            samples.append({
                "image": filename,
                "labels": multihot,
                "emotion_list": emotion_list,
            })

        return samples

    def _find_label_file(self) -> str:
        """自动查找标签文件"""
        for fname in ["labels.csv", "labels.json", f"{self.split}.csv", f"{self.split}.json"]:
            fpath = os.path.join(self.data_root, fname)
            if os.path.exists(fpath):
                return fpath
        raise FileNotFoundError(f"在 {self.data_root} 下未找到标签文件")

    def _load_csv_labels(self, filepath: str) -> List[Tuple[str, List[str]]]:
        """加载 CSV 格式的标签"""
        samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename", row.get("image", ""))
                # 提取所有值为 "1" 的标签名
                emotions = [
                    col for col in reader.fieldnames
                    if col not in ("filename", "image", "split")
                    and row.get(col, "").strip() in ("1", "1.0", "True", "true")
                ]
                samples.append((filename, emotions))
        return samples

    def _load_json_labels(self, filepath: str) -> List[Tuple[str, List[str]]]:
        """加载 JSON 格式的标签"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return [(k, v) for k, v in data.items()]
        elif isinstance(data, list):
            return [(item["filename"], item["emotions"]) for item in data]
        else:
            raise ValueError(f"不支持的 JSON 格式: {type(data)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # 加载图像
        image_path = os.path.join(self.image_dir, sample["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        # 预处理
        pixel_values = self.transform(image)

        return {
            "pixel_values": pixel_values,
            "labels": sample["labels"],
            "image_name": sample["image"],
        }


# ============================================================
# 数据集合并与划分
# ============================================================

class ConcatMultiLabelDataset(Dataset):
    """
    合并多个数据集

    自动处理不同数据集的标签对齐问题。
    """

    def __init__(
        self,
        datasets: List[MultiLabelEmotionDataset],
    ):
        self.datasets = datasets
        self.cumulative_sizes = self._cumsum([len(d) for d in datasets])

        print(f"[ConcatMultiLabelDataset] 合并 {len(datasets)} 个数据集")
        print(f"  - 总样本数: {len(self)}")

    @staticmethod
    def _cumsum(sequence):
        r = []
        s = 0
        for n in sequence:
            r.append(s + n)
            s += n
        return r

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = 0
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                dataset_idx = i
                break

        if dataset_idx > 0:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        else:
            sample_idx = idx

        return self.datasets[dataset_idx][sample_idx]


def create_dataloaders(
    data_roots: List[str],
    emotion_labels: List[str],
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    创建训练/验证/测试 DataLoader

    支持从多个数据集目录加载数据并自动划分。

    Args:
        data_roots: 数据集根目录列表
        emotion_labels: 统一情感标签
        batch_size: 批次大小
        image_size: 图像尺寸
        num_workers: 数据加载线程
        val_split: 验证集比例
        test_split: 测试集比例
        seed: 随机种子

    Returns:
        train_loader, val_loader, test_loader
    """
    # 加载所有数据集（训练增强版 + 评估版）
    train_datasets = []
    eval_datasets = []
    for root in data_roots:
        if not os.path.exists(root):
            print(f"⚠ 数据集目录不存在，跳过: {root}")
            continue

        train_datasets.append(MultiLabelEmotionDataset(
            data_root=root,
            emotion_labels=emotion_labels,
            split="train",
            image_size=image_size,
            transform=get_default_transforms(image_size, is_train=True),
        ))
        eval_datasets.append(MultiLabelEmotionDataset(
            data_root=root,
            emotion_labels=emotion_labels,
            split="val",
            image_size=image_size,
            transform=get_default_transforms(image_size, is_train=False),
        ))

    if not train_datasets:
        raise ValueError("没有找到任何有效的数据集目录！")

    # 合并数据集
    train_full = ConcatMultiLabelDataset(train_datasets)
    eval_full = ConcatMultiLabelDataset(eval_datasets)

    # 划分训练/验证/测试（两份数据集共享同一索引划分）
    from torch.utils.data import random_split, Subset
    total = len(train_full)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size

    indices = list(range(total))
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = random_split(
        indices, [train_size, val_size, test_size],
        generator=generator,
    )

    train_ds = Subset(train_full, train_idx)          # 训练增强
    val_ds = Subset(eval_full, val_idx)               # 无增强（稳定评估）
    test_ds = Subset(eval_full, test_idx)             # 无增强（稳定评估）

    # 创建 DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    ) if val_size > 0 else None

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    ) if test_size > 0 else None

    print(f"\nDataLoader 创建完成:")
    print(f"  - 训练集: {train_size} 样本, {len(train_loader)} batches")
    if val_loader:
        print(f"  - 验证集: {val_size} 样本, {len(val_loader)} batches")
    if test_loader:
        print(f"  - 测试集: {test_size} 样本, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("数据集模块测试")
    print("=" * 60)

    from config import MIKELS_BASIC_EMOTIONS

    # 创建模拟数据
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp()
    print(f"\n临时目录: {tmpdir}")

    # 创建模拟图像目录
    os.makedirs(os.path.join(tmpdir, "images"), exist_ok=True)

    # 创建模拟图像（纯色块）
    for i in range(10):
        img = Image.new("RGB", (224, 224), color=(np.random.randint(0, 255),
                                                    np.random.randint(0, 255),
                                                    np.random.randint(0, 255)))
        img.save(os.path.join(tmpdir, "images", f"img_{i:03d}.jpg"))

    # 创建模拟 CSV 标签
    labels_csv = os.path.join(tmpdir, "labels.csv")
    with open(labels_csv, "w", encoding="utf-8") as f:
        header = ["filename"] + MIKELS_BASIC_EMOTIONS[:6]  # 仅用前6个
        f.write(",".join(header) + "\n")
        for i in range(10):
            row = [f"img_{i:03d}.jpg"]
            for _ in range(6):
                row.append(str(np.random.randint(0, 2)))
            f.write(",".join(row) + "\n")

    # 测试数据集加载
    print("\n[数据集加载测试]")
    ds = MultiLabelEmotionDataset(
        data_root=tmpdir,
        emotion_labels=MIKELS_BASIC_EMOTIONS[:6],
        split="train",
    )
    print(f"  样本数: {len(ds)}")

    # 测试获取样本
    sample = ds[0]
    print(f"  pixel_values: {sample['pixel_values'].shape}")
    print(f"  labels: {sample['labels']}")
    print(f"  image_name: {sample['image_name']}")

    # 测试 DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"\n[DataLoader 测试]")
    print(f"  batch pixel_values: {batch['pixel_values'].shape}")
    print(f"  batch labels: {batch['labels'].shape}")
    print(f"  batch image_names: {batch['image_name']}")

    # 清理
    shutil.rmtree(tmpdir)

    print("\n✅ 数据集模块测试通过！")
