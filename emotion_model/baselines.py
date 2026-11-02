"""
CLIP 基线实验（B1 / B2 / B3 / B4）
基于视觉语言模型的多标记情感识别研究

用途
----
提供与完整模型**同一划分、同一指标**的对照基线，用于回答唯一有意义的问题：
    相比「不训练 / 只训一个线性层」，我们那 1900 万可训练参数带来了多少增益？

本模块实现的组别
----------------
    B1  零样本 top-1         CLIP 图文相似度取最大，不训练，0 参数
    B2  零样本 + 阈值         B1 的相似度上做阈值搜索，0 参数
    B3  线性探测             冻结 CLIP 图像特征 + 单个 Linear(768→8)，约 6.2K 参数
    B4  MLP 探测             冻结 CLIP 图像特征 + Linear(768→256)+GELU+Linear(256→8)，约 205K 参数

未实现的组别（原因）
--------------------
    B5  特征拼接 + MLP       需要样本**配文**；当前数据集只有标签，无文本
    B6  逐元素融合           同上
    B7  交叉注意力融合       同上（这一组才是「同样用注意力但没有冲突感知」的公平对照，
                              待数据负责人交付配文后必须补上）
    B8  全量微调             算力与时间主要花在这里，价值低于 B3/B7，暂缓

数据约定
--------
输入为 fixed_splits.json（字段：filename / label / sha256 / pixel_sha256）
图像按 filename 在 image_dir 下平铺查找；label 为单标签字符串。
由于 ArtPhoto 是单标签数据集，多热编码后每行仅一个 1，因此 F1 的 macro 口径
与「8 类单标签分类」一致——这正是与 ArtPhoto_调参后完整模型_20261102 报告
（macro F1 0.1673 @阈值0.5 / 0.2940 @阈值0.11）可直接并列的原因。

阈值协议（关键）
----------------
B2/B3/B4 的输出都是**未归一化分数**，写死 0.5 会严重低估 F1。
本模块统一在 **验证集** 上搜索全局阈值，再**只应用一次**到测试集，测试集不参与选择。
同时报告 @0.5 的数字，以便看出协议差异带来的差距。

用法
----
    # 全部跑（默认 B1 B2 B3 B4）
    python -m emotion_model.baselines \
        --image-dir ../testImages_artphoto \
        --splits    ../ArtPhoto_调参后完整模型_20261102/reports/fixed_splits.json \
        --out       ./results/baselines_artphoto

    # 只跑零样本
    python -m emotion_model.baselines --runs b1 b2 ...

首次运行会把 CLIP 图像特征缓存为 features_<split>.npz，后续重跑秒级。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Windows 控制台默认 GBK 编码无法输出特殊符号，强制 UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

from .config import MIKELS_BASIC_EMOTIONS
from .dataset import get_default_transforms
from .utils import compute_metrics, compute_per_label_metrics, set_seed
DEFAULT_LABELS: List[str] = list(MIKELS_BASIC_EMOTIONS)


# ============================================================
# 数据加载
# ============================================================

def load_fixed_splits(split_file: str | Path) -> Dict[str, List[Dict]]:
    """读取 fixed_splits.json，返回 {split: [{filename, label, ...}]}。"""
    with open(split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    for name in ("train", "val", "test"):
        if name not in splits:
            raise ValueError(f"fixed_splits.json 缺少必需的划分 '{name}'")
    return splits


def rows_to_multihot(rows: Sequence[Dict], labels: Sequence[str]) -> np.ndarray:
    """把单标签行转换为 (N, num_labels) 多热编码。"""
    index = {name: i for i, name in enumerate(labels)}
    out = np.zeros((len(rows), len(labels)), dtype=np.float32)
    for i, row in enumerate(rows):
        label = row["label"]
        if label not in index:
            raise ValueError(
                f"标签 '{label}' 不在统一情感体系中 {list(labels)}；"
                f"请检查 labels 顺序或数据标签拼写"
            )
        out[i, index[label]] = 1.0
    return out


class SplitImageDataset(torch.utils.data.Dataset):
    """按 filename 平铺读取图像；返回 (pixel_values, multihot, filename)。"""

    def __init__(
        self,
        image_dir: str | Path,
        rows: Sequence[Dict],
        labels: Sequence[str],
        image_size: Tuple[int, int] = (224, 224),
    ):
        self.image_dir = Path(image_dir)
        self.rows = list(rows)
        self.labels = list(labels)
        self.transform = get_default_transforms(image_size=image_size, is_train=False)

        missing = [r["filename"] for r in self.rows
                   if not (self.image_dir / r["filename"]).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} 张图像在 {self.image_dir} 下找不到，例如：{missing[:5]}"
            )
        self.targets = torch.from_numpy(rows_to_multihot(self.rows, self.labels))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        with Image.open(self.image_dir / row["filename"]) as img:
            pixel_values = self.transform(img.convert("RGB"))
        return pixel_values, self.targets[idx], row["filename"]


# ============================================================
# CLIP 特征提取与缓存
# ============================================================

@torch.no_grad()
def extract_image_features(
    clip_model,
    loader,
    device: torch.device,
    desc: str = "",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    提取 CLIP 视觉塔的 pooler_output（768 维，ViT-B/32）。

    刻意使用 **CLIP 原生 pooler 输出**，而不是完整模型里的 visual_projection
    （那是本项目自加的可训练投影层）——基线必须是「不带我们任何模块」的 CLIP。

    Returns:
        features (N, 768), targets (N, 8), filenames list
    """
    feats, targets, names = [], [], []
    total = len(loader)
    t0 = time.time()
    for i, (pixel_values, target, filename) in enumerate(loader, 1):
        out = clip_model.vision_model(pixel_values=pixel_values.to(device))
        feats.append(out.pooler_output.float().cpu().numpy())
        targets.append(target.numpy())
        names.extend(filename if isinstance(filename, (list, tuple)) else [filename])
        if i % 10 == 0 or i == total:
            elapsed = time.time() - t0
            print(f"\r  {desc} 批次 {i}/{total}  {elapsed:.1f}s", end="", flush=True)
    print()
    return (
        np.concatenate(feats, axis=0),
        np.concatenate(targets, axis=0),
        list(names),
    )


def build_features(
    image_dir: str | Path,
    splits: Dict[str, List[Dict]],
    labels: Sequence[str],
    clip_model_name: str,
    cache_dir: Optional[str | Path],
    batch_size: int = 16,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """为 train/val/test 提取（或从缓存加载）CLIP 图像特征。"""
    device = device or torch.device("cpu")
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    needed = {}
    for split in ("train", "val", "test"):
        cached = cache_dir / f"features_{split}.npz" if cache_dir else None
        if cached and cached.is_file():
            data = np.load(cached, allow_pickle=False)
            needed[split] = {
                "features": data["features"],
                "targets": data["targets"],
                "filenames": [str(x) for x in data["filenames"]],
            }
            print(f"  [缓存] {split}: {data['features'].shape} ← {cached.name}")
        else:
            needed[split] = None

    if all(v is not None for v in needed.values()):
        return needed  # type: ignore[return-value]

    from transformers import CLIPModel

    print(f"\n加载 CLIP（基线专用，全部冻结）: {clip_model_name}")
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    for split in ("train", "val", "test"):
        if needed[split] is not None:
            continue
        ds = SplitImageDataset(image_dir, splits[split], labels)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=0
        )
        feats, targets, names = extract_image_features(
            clip_model, loader, device, desc=f"{split:<5}"
        )
        needed[split] = {"features": feats, "targets": targets, "filenames": names}
        if cache_dir:
            np.savez(
                cache_dir / f"features_{split}.npz",
                features=feats, targets=targets,
                filenames=np.array(names, dtype=str),
            )

    del clip_model
    return needed  # type: ignore[return-value]


@torch.no_grad()
def build_text_features(
    clip_model_name: str,
    labels: Sequence[str],
    device: torch.device,
    prompt_ensemble: bool = True,
) -> np.ndarray:
    """
    构建 8 个情感标签的 CLIP 文本特征（512 维，已 L2 归一化）。

    与 emotion_model/base_encoder.py 的 EMOTION_PROMPTS 保持一致，
    便于基线数字与完整模型使用同一套语义锚点。
    """
    from transformers import CLIPModel, CLIPProcessor
    from .base_encoder import TextEncoder

    model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    templates: List[str] = []
    owners: List[int] = []
    for i, label in enumerate(labels):
        base = TextEncoder.EMOTION_PROMPTS.get(
            label, f"an image expressing {label}"
        )
        variants = [base]
        if prompt_ensemble:
            variants += [
                f"a photo of a person feeling {label}",
                f"a picture that conveys {label}",
            ]
        templates.extend(variants)
        owners.extend([i] * len(variants))

    tokens = processor(
        text=templates, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    # 直接走 text_model + text_projection，而不是 get_text_features：
    # transformers 5.x 起 get_text_features 返回 BaseModelOutputWithPooling 而非张量，
    # 且不同版本是否已投影不一致。底层组合在 4.x / 5.x 语义稳定。
    with torch.no_grad():
        text_out = model.text_model(**tokens)
        pooled = text_out.pooler_output            # (T, 512)
        emb = model.text_projection(pooled)        # (T, 512) 图文公共空间
    emb = emb.float().cpu().numpy()
    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)

    # prompt ensemble：同一标签的多个模板取平均后再归一化
    owners_arr = np.array(owners)
    out = np.zeros((len(labels), emb.shape[1]), dtype=np.float32)
    for i in range(len(labels)):
        out[i] = emb[owners_arr == i].mean(axis=0)
    out = out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8)

    del model
    return out.astype(np.float32)


@torch.no_grad()
def project_image_features(
    clip_model_name: str,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    用 CLIP 原生 visual_projection 把 768 维 pooler 输出投影到 512 维图文公共空间。

    零样本图文匹配必须在这个公共空间里做余弦相似度——直接拿 768 维 pooler
    输出与 512 维文本特征算内积是没有意义的。
    """
    from transformers import CLIPModel

    model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    proj = model.visual_projection
    out = proj(torch.from_numpy(features).to(device)).float().cpu().numpy()
    out = out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8)
    del model
    return out.astype(np.float32)


# ============================================================
# 阈值搜索与评估
# ============================================================

def macro_f1_binary(scores: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    """按给定阈值二值化后计算 macro F1。"""
    from sklearn.metrics import f1_score
    pred = (scores >= threshold).astype(np.float32)
    return float(f1_score(targets, pred, average="macro", zero_division=0))


def search_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    n_candidates: int = 200,
) -> Tuple[float, float]:
    """
    在**验证集**上搜索最大化 macro F1 的全局阈值。

    候选点为验证集全部分数的分位数（避免逐点扫描的开销），
    再在最优分位点附近细分一次。返回 (最佳阈值, 该阈值下的 macro F1)。

    注意：必须在验证集上调用，测试集不得参与。
    """
    flat = scores.ravel()
    lo, hi = float(flat.min()), float(flat.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.5, macro_f1_binary(scores, targets, 0.5)

    best_t, best_f1 = float(np.quantile(flat, 0.5)), -1.0
    grid = np.quantile(flat, np.linspace(0.0, 1.0, n_candidates))
    for t in np.unique(grid):
        f1 = macro_f1_binary(scores, targets, float(t))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    # 局部细分
    span = (hi - lo) / max(n_candidates - 1, 1)
    for t in np.linspace(best_t - span, best_t + span, 21):
        f1 = macro_f1_binary(scores, targets, float(t))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    return best_t, best_f1


def top1_accuracy(scores: np.ndarray, targets: np.ndarray) -> float:
    """多标签数据上退化为「取最大分标签」的 top-1 正确率。"""
    pred = scores.argmax(axis=1)
    gold = targets.argmax(axis=1)
    return float((pred == gold).mean())


def top1_macro_f1(scores: np.ndarray, targets: np.ndarray) -> float:
    """只保留 top-1 预测标签后计算的 macro F1。"""
    from sklearn.metrics import f1_score
    pred = np.zeros_like(targets)
    pred[np.arange(len(scores)), scores.argmax(axis=1)] = 1.0
    return float(f1_score(targets, pred, average="macro", zero_division=0))


def top1_decision(scores: np.ndarray) -> np.ndarray:
    """argmax 决策（每样本恰好一个正标签），用于不与阈值混淆的场景。"""
    pred = np.zeros_like(scores, dtype=np.float32)
    pred[np.arange(len(scores)), scores.argmax(axis=1)] = 1.0
    return pred


@dataclass
class BaselineResult:
    name: str
    description: str
    n_params: int
    threshold: float
    threshold_source: str
    metrics: Dict[str, float]
    metrics_at_half: Dict[str, float]
    per_label: Dict[str, Dict[str, float]]
    avg_predicted_labels: float
    applies_threshold: bool = True

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "n_trainable_params": self.n_params,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "applies_threshold": self.applies_threshold,
            "avg_predicted_labels": self.avg_predicted_labels,
            "metrics": self.metrics,
            "metrics_at_threshold_0.5": self.metrics_at_half,
            "per_label": self.per_label,
        }


def evaluate_scores(
    name: str,
    description: str,
    n_params: int,
    val_scores: np.ndarray,
    val_targets: np.ndarray,
    test_scores: np.ndarray,
    test_targets: np.ndarray,
    labels: Sequence[str],
    threshold_source: str = "val_search",
) -> BaselineResult:
    """统一的评估入口：验证集搜阈值 → 只用一次到测试集。"""
    if threshold_source == "val_search":
        threshold, val_f1 = search_threshold(val_scores, val_targets)
        threshold_note = f"验证集搜索（验证集 macro F1={val_f1:.4f}）"
    else:  # fixed
        threshold = 0.5
        threshold_note = "固定 0.5（不做搜索）"

    pred = (test_scores >= threshold).astype(np.float32)
    metrics = compute_metrics(pred, test_scores, test_targets, threshold=0.5)
    metrics["top1_accuracy"] = top1_accuracy(test_scores, test_targets)
    metrics["top1_macro_f1"] = top1_macro_f1(test_scores, test_targets)
    metrics["avg_predicted_labels"] = float(pred.sum(axis=1).mean())

    pred_half = (test_scores >= 0.5).astype(np.float32)
    metrics_half = compute_metrics(pred_half, test_scores, test_targets, threshold=0.5)
    metrics_half["top1_accuracy"] = metrics["top1_accuracy"]
    metrics_half["top1_macro_f1"] = metrics["top1_macro_f1"]
    metrics_half["avg_predicted_labels"] = float(pred_half.sum(axis=1).mean())

    per_label = compute_per_label_metrics(pred, test_targets, list(labels))

    return BaselineResult(
        name=name,
        description=description,
        n_params=n_params,
        threshold=threshold,
        threshold_source=threshold_note,
        metrics=metrics,
        metrics_at_half=metrics_half,
        per_label=per_label,
        avg_predicted_labels=float(pred.sum(axis=1).mean()),
    )


# ============================================================
# B1 / B2：CLIP 零样本
# ============================================================

def run_zero_shot(
    label_features: Dict[str, Dict[str, np.ndarray]],
    labels: Sequence[str],
    image_features_768: Dict[str, np.ndarray],
    clip_model_name: str,
    device: torch.device,
) -> List[BaselineResult]:
    """
    B1 零样本 top-1（不训练、无阈值）
    B2 零样本 + 阈值（在验证集上搜索）

    分数 = 图像特征与情感文本特征的余弦相似度（∈[-1,1]）。
    多标签打分用**饱和度无关的余弦**本身：越大越可能属于该情感。
    """
    text_emb = build_text_features(clip_model_name, labels, device)

    scores: Dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        img_512 = project_image_features(
            clip_model_name, image_features_768[split], device
        )
        scores[split] = img_512 @ text_emb.T  # (N, 8) 余弦相似度

    results: List[BaselineResult] = []

    # B1：只看 top-1，不做任何阈值决策
    # 注意：B1 刻意不设阈值。CLIP 余弦相似度整体落在 0.2~0.35 区间，
    # 任何固定阈值（包括 0.5）都会把全部标签判负，macro F1 恒为 0 而无信息量。
    # 因此 B1 只报告阈值无关的排序类指标 + top-1，F1 标为 n/a。
    b1_targets = label_features["test"]["targets"]
    b1_metrics = compute_metrics(
        top1_decision(scores["test"]), scores["test"], b1_targets, threshold=0.5
    )
    b1_metrics["top1_accuracy"] = top1_accuracy(scores["test"], b1_targets)
    b1_metrics["top1_macro_f1"] = top1_macro_f1(scores["test"], b1_targets)
    b1_metrics["avg_predicted_labels"] = 1.0
    results.append(
        BaselineResult(
            name="B1",
            description="CLIP 零样本 top-1（图文余弦相似度取最大，不训练、不设阈值）",
            n_params=0,
            threshold=float("nan"),
            threshold_source="无（只做 argmax 决策，报告排序类指标与 top-1）",
            metrics=b1_metrics,
            metrics_at_half=dict(b1_metrics),
            per_label=compute_per_label_metrics(
                top1_decision(scores["test"]), b1_targets, list(labels),
            ),
            avg_predicted_labels=1.0,
            applies_threshold=False,
        )
    )

    # B2：验证集搜阈值后应用
    results.append(
        evaluate_scores(
            name="B2",
            description="CLIP 零样本 + 验证集阈值搜索（仍不训练）",
            n_params=0,
            val_scores=scores["val"],
            val_targets=label_features["val"]["targets"],
            test_scores=scores["test"],
            test_targets=label_features["test"]["targets"],
            labels=labels,
        )
    )

    return results


# ============================================================
# B3 / B4：线性探测 / MLP 探测
# ============================================================

class LinearProbe(nn.Module):
    """B3：冻结特征上的单个线性层。"""

    def __init__(self, in_dim: int = 768, num_labels: int = 8):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPProbe(nn.Module):
    """B4：冻结特征上的一个隐藏层 MLP。"""

    def __init__(self, in_dim: int = 768, hidden: int = 256, num_labels: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _standardize(
    train_feats: np.ndarray, others: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """按训练集统计量做标准化（基线不允许使用验证/测试集统计量）。"""
    mu = train_feats.mean(axis=0, keepdims=True)
    sd = train_feats.std(axis=0, keepdims=True)
    sd = np.maximum(sd, 1e-6)
    out = {k: (v - mu) / sd for k, v in others.items()}
    return (train_feats - mu) / sd, out


def train_probe(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[nn.Module, int, float]:
    """
    训练探测头；**按验证集 mAP 选轮次**（阈值无关指标，避免与阈值搜索重复用同一信息）。
    返回 (模型, 最佳轮次, 最佳验证 mAP)。
    """
    from sklearn.metrics import average_precision_score

    set_seed(seed)
    device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = torch.from_numpy(x_train).float()
    yt = torch.from_numpy(y_train).float()
    xv = torch.from_numpy(x_val).float()
    n = len(xt)

    best_state, best_epoch, best_map = None, 1, -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(xv)).numpy()
        try:
            val_map = average_precision_score(y_val, val_prob, average="macro")
        except ValueError:
            val_map = -1.0
        if val_map > best_map:
            best_map = val_map
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % 50 == 0 or epoch == 1):
            from sklearn.metrics import f1_score
            f1 = f1_score(y_val, (val_prob >= 0.5).astype(np.float32),
                          average="macro", zero_division=0)
            print(f"      epoch {epoch:3d}  val_mAP={val_map:.4f}  "
                  f"val_macroF1@0.5={f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_map


@torch.no_grad()
def probe_scores(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """输出概率（sigmoid 后），(N, 8)。"""
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(x).float())).numpy()


def run_probes(
    label_features: Dict[str, Dict[str, np.ndarray]],
    labels: Sequence[str],
    image_features_768: Dict[str, np.ndarray],
    seeds: Sequence[int] = (42, 2026, 7),
    epochs: int = 200,
    verbose: bool = False,
) -> List[BaselineResult]:
    """
    B3 线性探测 / B4 MLP 探测。

    多随机种子重复，报告均值（标准差另存），避免单次随机初始化带来的偶然性。
    """
    raw = {
        "train": image_features_768["train"],
        "val": image_features_768["val"],
        "test": image_features_768["test"],
    }
    x_train, rest = _standardize(raw["train"], {"val": raw["val"], "test": raw["test"]})
    y_train = label_features["train"]["targets"]
    y_val = label_features["val"]["targets"]
    y_test = label_features["test"]["targets"]

    specs = [
        ("B3", "冻结 CLIP 特征 + 线性层（Linear 768→8）", LinearProbe, 6_152),
        ("B4", "冻结 CLIP 特征 + MLP（768→256→8，含 dropout 0.1）", MLPProbe, 198_920),
    ]

    results: List[BaselineResult] = []
    for name, desc, factory, expected_params in specs:
        print(f"\n  {name}: {desc}")
        all_seed_results = []
        n_params = None
        for i, seed in enumerate(seeds):
            print(f"    种子 {seed} ...", end="", flush=True)
            model = factory()
            if n_params is None:
                n_params = sum(p.numel() for p in model.parameters())
            model, best_epoch, best_map = train_probe(
                model, x_train, y_train, rest["val"], y_val,
                epochs=epochs, seed=seed, verbose=verbose,
            )
            val_scores = probe_scores(model, rest["val"])
            test_scores = probe_scores(model, rest["test"])
            res = evaluate_scores(
                name=f"{name}@{seed}",
                description=desc,
                n_params=n_params,
                val_scores=val_scores,
                val_targets=y_val,
                test_scores=test_scores,
                test_targets=y_test,
                labels=labels,
            )
            all_seed_results.append(res)
            print(f" 最佳轮次 {best_epoch:3d}  验证集 mAP {best_map:.4f}  "
                  f"→ 测试集 macroF1 {res.metrics['f1_macro']:.4f}")

        # 汇总为「多种子均值」
        agg = BaselineResult(
            name=name,
            description=f"{desc}；{len(seeds)} 个随机种子的均值 ± 标准差",
            n_params=n_params or expected_params,
            threshold=float(np.mean([r.threshold for r in all_seed_results])),
            threshold_source=(
                f"每种子独立在验证集搜索阈值后应用；"
                f"阈值均值 {np.mean([r.threshold for r in all_seed_results]):.4f} "
                f"± {np.std([r.threshold for r in all_seed_results]):.4f}"
            ),
            metrics={
                k: float(np.mean([r.metrics[k] for r in all_seed_results]))
                for k in all_seed_results[0].metrics
            },
            metrics_at_half={
                k: float(np.mean([r.metrics_at_half[k] for r in all_seed_results]))
                for k in all_seed_results[0].metrics_at_half
            },
            per_label={
                label: {
                    key: float(np.mean([r.per_label[label][key]
                                        for r in all_seed_results]))
                    for key in all_seed_results[0].per_label[label]
                }
                for label in labels
            },
            avg_predicted_labels=float(
                np.mean([r.avg_predicted_labels for r in all_seed_results])
            ),
        )
        agg.metrics["f1_macro_std"] = float(
            np.std([r.metrics["f1_macro"] for r in all_seed_results])
        )
        agg.metrics["f1_micro_std"] = float(
            np.std([r.metrics["f1_micro"] for r in all_seed_results])
        )
        agg.metrics["mAP_std"] = float(
            np.std([r.metrics["mAP"] for r in all_seed_results])
        )
        results.append(agg)

    return results


# ============================================================
# 参考基准（恒定基线，非常容易被忽略但极其重要）
# ============================================================

def run_reference_baselines(
    label_features: Dict[str, Dict[str, np.ndarray]],
    labels: Sequence[str],
) -> List[BaselineResult]:
    """
    恒定预测基线：不做任何学习，永远输出同一个答案。
    用途：任何模型的 macro F1 必须明显高于这些数字才有意义。
    """
    y_train = label_features["train"]["targets"]
    y_test = label_features["test"]["targets"]
    n, k = y_test.shape

    results: List[BaselineResult] = []

    # R1 永远猜训练集最频繁类
    freq = y_train.sum(axis=0)
    maj = int(freq.argmax())
    scores = np.zeros((n, k), dtype=np.float32)
    scores[:, maj] = 1.0
    results.append(
        _constant_result("R1", f"永远预测训练集最频繁类（{labels[maj]}）",
                         scores, label_features["test"]["targets"], labels)
    )

    # R2 均匀随机（期望值 1/8）
    rng = np.random.RandomState(42)
    scores = np.zeros((n, k), dtype=np.float32)
    scores[np.arange(n), rng.randint(0, k, size=n)] = 1.0
    results.append(
        _constant_result("R2", "均匀随机猜一个类",
                         scores, label_features["test"]["targets"], labels)
    )

    # R3 预测全部标签
    scores = np.ones((n, k), dtype=np.float32)
    results.append(
        _constant_result("R3", "预测全部 8 个标签均为正",
                         scores, label_features["test"]["targets"], labels)
    )

    return results


def _constant_result(
    name: str,
    description: str,
    scores: np.ndarray,
    targets: np.ndarray,
    labels: Sequence[str],
) -> BaselineResult:
    metrics = compute_metrics(scores.copy(), scores.copy(), targets, threshold=0.5)
    metrics["top1_accuracy"] = top1_accuracy(scores, targets)
    metrics["top1_macro_f1"] = top1_macro_f1(scores, targets)
    metrics["avg_predicted_labels"] = float((scores >= 0.5).sum(axis=1).mean())
    return BaselineResult(
        name=name,
        description=description,
        n_params=0,
        threshold=0.5,
        threshold_source="恒定预测，无阈值决策",
        metrics=metrics,
        metrics_at_half=dict(metrics),
        per_label=compute_per_label_metrics(
            (scores >= 0.5).astype(np.float32), targets, list(labels)
        ),
        avg_predicted_labels=float((scores >= 0.5).sum(axis=1).mean()),
    )


# ============================================================
# 完整模型对照（读取已交付的预测文件，套用同一阈值协议）
# ============================================================

def load_full_model_scores(
    npz_val: str | Path,
    npz_test: str | Path,
    n_params: int,
    labels: Sequence[str],
) -> BaselineResult:
    """
    载入完整模型的验证/测试预测概率，用与 B2/B3/B4 **完全相同**的协议评估。

    存在的意义：交付报告里完整模型 macro F1 = 0.1673(阈值0.5) / 0.2940(阈值0.11)，
    而基线若用另一套协议就不可比。本函数保证两边用同一搜索方式、同一指标实现。

    数据来源（ArtPhoto_调参后完整模型_20261102/reports/）:
        val_lr3e-05.npz       → probabilities (117, 8), targets, filenames
        holdout_predictions.npz → probabilities (118, 8), targets, predictions, filenames
    """
    val_npz = np.load(npz_val, allow_pickle=False)
    test_npz = np.load(npz_test, allow_pickle=False)

    val_scores = val_npz["probabilities"].astype(np.float32)
    val_targets = val_npz["targets"].astype(np.float32)
    test_scores = test_npz["probabilities"].astype(np.float32)
    test_targets = test_npz["targets"].astype(np.float32)

    if val_scores.shape[1] != len(labels):
        raise ValueError(
            f"完整模型预测有 {val_scores.shape[1]} 列，与标签体系 {len(labels)} 不一致"
        )

    return evaluate_scores(
        name="FULL",
        description=(
            "完整模型（CLIP + 冲突感知融合 + 环形双头 + VL-Adapter，"
            "读取已交付的 holdout_predictions.npz，套用同一阈值协议）"
        ),
        n_params=n_params,
        val_scores=val_scores,
        val_targets=val_targets,
        test_scores=test_scores,
        test_targets=test_targets,
        labels=labels,
    )


# ============================================================
# 报告输出
# ============================================================

def format_table(results: Sequence[BaselineResult]) -> str:
    """生成 Markdown 对照表。"""
    head = (
        "| 组别 | 说明 | 可训练参数 | 阈值 | macro F1 | micro F1 | mAP | AUC | "
        "top-1 | 平均预测标签数 |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for r in results:
        m = r.metrics
        if not r.applies_threshold:
            th = "argmax"
            f1_macro = "n/a（不设阈值）"
        else:
            th = "—" if not np.isfinite(r.threshold) else f"{r.threshold:.3f}"
            f1_macro = f"**{m['f1_macro']:.4f}**"
        rows.append(
            f"| **{r.name}** | {r.description} | {r.n_params:,} | {th} | "
            f"{f1_macro} | {m['f1_micro']:.4f} | {m['mAP']:.4f} | "
            f"{m['auc_macro']:.4f} | {m['top1_accuracy']:.4f} | "
            f"{r.avg_predicted_labels:.2f} |"
        )
    return head + "\n".join(rows)


def write_report(
    results: Sequence[BaselineResult],
    out_dir: Path,
    meta: Dict,
) -> None:
    """把结果写成 results.json / RESULTS.md。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": meta,
        "results": [r.to_dict() for r in results],
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines: List[str] = []
    lines.append("# CLIP 基线实验结果\n")
    lines.append(f"- 数据集：{meta.get('image_dir')}")
    lines.append(f"- 划分文件：{meta.get('splits_file')}")
    lines.append(
        f"- 样本量：train {meta['n_train']} / val {meta['n_val']} / test {meta['n_test']}"
    )
    lines.append(f"- CLIP 模型：{meta.get('clip_model_name')}")
    lines.append(f"- 标签体系：{', '.join(meta['labels'])}")
    lines.append(
        "\n> 阈值协议：B2/B3/B4 的阈值在**验证集**上搜索后只应用一次到测试集；"
        "测试集不参与任何选择。\n"
    )
    lines.append("## 主结果\n")
    lines.append(format_table(results))
    lines.append("")

    lines.append("## 阈值 0.5 下的结果（说明协议差异）\n")
    lines.append("| 组别 | macro F1@0.5 | micro F1@0.5 | 平均预测标签数@0.5 |")
    lines.append("|---|---:|---:|---:|")
    for r in results:
        m = r.metrics_at_half
        lines.append(
            f"| {r.name} | {m['f1_macro']:.4f} | {m['f1_micro']:.4f} | "
            f"{m.get('avg_predicted_labels', r.avg_predicted_labels):.2f} |"
        )
    lines.append("")

    probes = [r for r in results if r.name in ("B3", "B4", "FULL")]
    if probes:
        lines.append("## 逐类 F1（测试集，阈值已搜索）\n")
        header = "| 标签 | " + " | ".join(r.name for r in probes) + " |"
        sep = "|---|" + "---:|" * len(probes)
        lines.append(header)
        lines.append(sep)
        for label in meta["labels"]:
            cells = []
            for r in probes:
                cells.append(f"{r.per_label[label]['f1']:.3f}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## 未纳入的组别\n")
    lines.append(
        "| 组别 | 状态 | 原因 |\n|---|---|---|\n"
        "| B5 特征拼接 + MLP | 未实现 | 需要样本配文；当前数据只有图像与标签 |\n"
        "| B6 逐元素融合 | 未实现 | 同上 |\n"
        "| B7 交叉注意力融合 | 未实现 | 同上。**该组才是「同样用注意力但无冲突感知」的公平对照，"
        "待配文数据交付后必须补做** |\n"
        "| B8 全量微调 | 未实现 | 算力开销最大、信息量低于 B3/B7，暂缓 |\n"
    )

    (out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 命令行入口
# ============================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CLIP 基线实验（B1 零样本 / B2 零样本+阈值 / B3 线性探测 / B4 MLP 探测）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image-dir", required=True, help="图像所在目录（按 filename 平铺）")
    p.add_argument("--splits", required=True, help="fixed_splits.json 路径")
    p.add_argument("--out", default="results/baselines", help="结果输出目录")
    p.add_argument("--cache-dir", default=None,
                   help="CLIP 特征缓存目录（默认 <out>/features，可复用避免重复提取）")
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32",
                   help="CLIP 模型名或本地路径")
    p.add_argument("--labels", default=None,
                   help="逗号分隔的标签顺序；默认使用 MIKELS 8 类")
    p.add_argument("--runs", nargs="+", default=["b1", "b2", "b3", "b4", "ref"],
                   choices=["b1", "b2", "b3", "b4", "ref", "full"],
                   help="要运行的组别；ref 为恒定预测参考基线，full 为完整模型对照")
    p.add_argument("--full-val-npz", default=None,
                   help="完整模型的验证集预测 npz（full 组必需），如 reports/val_lr3e-05.npz")
    p.add_argument("--full-test-npz", default=None,
                   help="完整模型的测试集预测 npz（full 组必需），如 reports/holdout_predictions.npz")
    p.add_argument("--full-params", type=int, default=19_095_572,
                   help="完整模型的可训练参数量（用于表格展示）")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 7],
                   help="B3/B4 的随机种子（多种子取均值）")
    p.add_argument("--probe-epochs", type=int, default=200, help="探测头训练轮数")
    p.add_argument("--batch-size", type=int, default=16, help="特征提取批大小")
    p.add_argument("--verbose", action="store_true", help="打印探测头训练过程")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(42)
    device = torch.device("cpu")

    labels = (
        [s.strip() for s in args.labels.split(",")] if args.labels else DEFAULT_LABELS
    )
    image_dir = Path(args.image_dir).resolve()
    splits_file = Path(args.splits).resolve()
    out_dir = Path(args.out).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else out_dir / "features"

    print("=" * 78)
    print("CLIP 基线实验")
    print("=" * 78)
    print(f"  图像目录 : {image_dir}")
    print(f"  划分文件 : {splits_file}")
    print(f"  标签体系 : {labels}")
    print(f"  运行组别 : {', '.join(a.upper() for a in args.runs)}")
    print(f"  输出目录 : {out_dir}")

    splits = load_fixed_splits(splits_file)
    for split in ("train", "val", "test"):
        found = sum(1 for r in splits[split]
                    if (image_dir / r["filename"]).is_file())
        print(f"  {split:<5}: {len(splits[split]):4d} 条，图像存在 {found}")
        if found != len(splits[split]):
            raise FileNotFoundError(
                f"{split} 划分有 {len(splits[split]) - found} 张图像在 {image_dir} 缺失"
            )

    print("\n[1/4] 提取 CLIP 图像特征（ViT-B/32 pooler，768 维）")
    features = build_features(
        image_dir, splits, labels, args.clip_model,
        cache_dir=cache_dir, batch_size=args.batch_size, device=device,
    )
    label_features = {
        split: {
            "targets": rows_to_multihot(splits[split], labels),
            "filenames": [r["filename"] for r in splits[split]],
        }
        for split in ("train", "val", "test")
    }
    image_features = {k: v["features"] for k, v in features.items()}
    del features

    results: List[BaselineResult] = []

    need_zs = any(r in ("b1", "b2") for r in args.runs)
    if need_zs:
        print("\n[2/4] 零样本评估（B1 / B2）")
        if "b1" in args.runs and "b2" in args.runs:
            results.extend(
                run_zero_shot(label_features, labels, image_features,
                              args.clip_model, device)
            )
        else:
            zs = run_zero_shot(label_features, labels, image_features,
                               args.clip_model, device)
            keep = {"b1": "B1", "b2": "B2"}
            results.extend([r for r in zs if r.name in
                            {keep[r] for r in args.runs if r in keep}])

    need_probe = any(r in ("b3", "b4") for r in args.runs)
    if need_probe:
        print("\n[3/4] 探测头评估（B3 线性 / B4 MLP）")
        probes = run_probes(
            label_features, labels, image_features,
            seeds=args.seeds, epochs=args.probe_epochs, verbose=args.verbose,
        )
        wanted = {"b3": "B3", "b4": "B4"}
        keep = {wanted[r] for r in args.runs if r in wanted}
        results.extend([r for r in probes if r.name in keep])

    if "ref" in args.runs:
        print("\n[4/4] 恒定预测参考基线（R1 / R2 / R3）")
        results.extend(run_reference_baselines(label_features, labels))

    if "full" in args.runs:
        if not (args.full_val_npz and args.full_test_npz):
            print(
                "\n[!] full 组需要同时提供 --full-val-npz 与 --full-test-npz，已跳过。\n"
                "    例：--full-val-npz ../ArtPhoto_调参后完整模型_20261102/reports/val_lr3e-05.npz "
                "--full-test-npz ../ArtPhoto_调参后完整模型_20261102/reports/holdout_predictions.npz"
            )
        else:
            print("\n[+] 完整模型对照（同一阈值协议）")
            try:
                full_res = load_full_model_scores(
                    Path(args.full_val_npz).resolve(),
                    Path(args.full_test_npz).resolve(),
                    n_params=args.full_params,
                    labels=labels,
                )
                results.append(full_res)
                print(f"    完整模型 macro F1 = {full_res.metrics['f1_macro']:.4f}"
                      f"（阈值 {full_res.threshold:.3f}，来自验证集搜索）")
            except Exception as exc:  # 文件缺失/列数不符等
                print(f"    [!] 载入完整模型预测失败，已跳过：{exc}")

    if not results:
        print("\n没有运行任何组别，退出。")
        return 1

    results.sort(key=lambda r: (r.name[0] != "B", r.name))

    meta = {
        "image_dir": str(image_dir),
        "splits_file": str(splits_file),
        "clip_model_name": args.clip_model,
        "labels": list(labels),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "seeds": list(args.seeds),
        "probe_epochs": args.probe_epochs,
        "metric_protocol": (
            "阈值在验证集上搜索后仅应用一次到测试集；测试集不参与任何选择。"
            "探测头轮次按验证集 mAP 选择（阈值无关指标）。"
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_report(results, out_dir, meta)

    print("\n" + "=" * 78)
    print("结果汇总（测试集，阈值来自验证集搜索）")
    print("=" * 78)
    print(format_table(results))
    print(f"\n已写入:\n  {out_dir / 'results.json'}\n  {out_dir / 'RESULTS.md'}")

    # 与已交付完整模型并列（数值取自 ArtPhoto_调参后完整模型_20261102 报告）
    if any(r.name.startswith("B3") for r in results):
        print("\n参考：完整模型（ArtPhoto_调参后完整模型_20261102，19,095,572 可训练参数）")
        print("  测试集 macro F1 = 0.1673（阈值 0.5） / 0.2940（阈值 0.11）")
        print("  测试集 top-1 准确率 = 0.2458")
        print("  → 与上表 B3/B4 对比即可判断架构增益是否成立")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
