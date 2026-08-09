"""
工具函数模块
包含训练辅助函数、评估指标、日志工具等
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score, roc_auc_score, hamming_loss,
    coverage_error, label_ranking_loss, label_ranking_average_precision_score,
)
from datetime import datetime
import time
import json
import os


# ============================================================
# 多标记分类评估指标
# ============================================================

def compute_metrics(
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    计算多标记分类的全面评估指标

    Args:
        predictions: (N, num_labels) 二值预测
        probabilities: (N, num_labels) 预测概率
        targets: (N, num_labels) 真实标签（多热编码）
        threshold: 决策阈值

    Returns:
        metrics dict:
            - 基于阈值的指标: accuracy, precision, recall, f1 (micro/macro/samples)
            - 基于排序的指标: mAP, AUC, coverage_error, ranking_loss
            - 标签不平衡指标: hamming_loss
    """
    # 转为 numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # 确保二值格式
    predictions = (predictions > 0.5).astype(np.float32)
    targets = targets.astype(np.float32)

    metrics = {}

    # ---- 基于样本的指标 ----
    try:
        metrics["accuracy_subset"] = accuracy_score(targets, predictions)
    except:
        metrics["accuracy_subset"] = float("nan")

    metrics["hamming_loss"] = hamming_loss(targets, predictions)

    # ---- F1 分数 (micro/macro/samples) ----
    for avg in ["micro", "macro", "samples"]:
        try:
            metrics[f"f1_{avg}"] = f1_score(targets, predictions, average=avg, zero_division=0)
        except:
            metrics[f"f1_{avg}"] = float("nan")

    # ---- Precision & Recall ----
    for avg in ["micro", "macro"]:
        try:
            metrics[f"precision_{avg}"] = precision_score(
                targets, predictions, average=avg, zero_division=0
            )
            metrics[f"recall_{avg}"] = recall_score(
                targets, predictions, average=avg, zero_division=0
            )
        except:
            metrics[f"precision_{avg}"] = float("nan")
            metrics[f"recall_{avg}"] = float("nan")

    # ---- 基于概率的指标 ----
    try:
        metrics["mAP"] = average_precision_score(targets, probabilities, average="macro")
    except:
        metrics["mAP"] = float("nan")

    try:
        metrics["auc_macro"] = roc_auc_score(targets, probabilities, average="macro")
    except:
        metrics["auc_macro"] = float("nan")

    # 每标签 AUC
    try:
        per_label_auc = []
        for i in range(targets.shape[1]):
            if len(np.unique(targets[:, i])) > 1:  # 需要至少两个类别
                per_label_auc.append(roc_auc_score(targets[:, i], probabilities[:, i]))
        metrics["auc_per_label_mean"] = np.mean(per_label_auc) if per_label_auc else float("nan")
    except:
        metrics["auc_per_label_mean"] = float("nan")

    # ---- 排序指标 ----
    try:
        metrics["coverage_error"] = coverage_error(targets, probabilities)
    except:
        metrics["coverage_error"] = float("nan")

    try:
        metrics["ranking_loss"] = label_ranking_loss(targets, probabilities)
    except:
        metrics["ranking_loss"] = float("nan")

    return metrics


def compute_per_label_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    计算每个情感标签的独立指标

    Returns:
        {label: {"precision": ..., "recall": ..., "f1": ..., "support": ...}}
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    predictions = (predictions > 0.5).astype(np.float32)

    per_label = {}
    for i, name in enumerate(label_names):
        per_label[name] = {
            "precision": precision_score(targets[:, i], predictions[:, i], zero_division=0),
            "recall": recall_score(targets[:, i], predictions[:, i], zero_division=0),
            "f1": f1_score(targets[:, i], predictions[:, i], zero_division=0),
            "support": int(targets[:, i].sum()),
        }

    return per_label


# ============================================================
# 训练辅助工具
# ============================================================

class AverageMeter:
    """跟踪和计算滑动平均值"""

    def __init__(self, name: str = "", fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = f"{{name}} {{val{self.fmt}}} ({{avg{self.fmt}}})"
        return fmtstr.format(name=self.name, val=self.val, avg=self.avg)


class EarlyStopping:
    """早停机制"""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False

        if mode == "max":
            self.monitor_op = lambda a, b: a > b + min_delta
            self.best_score_default = float("-inf")
        else:
            self.monitor_op = lambda a, b: a < b - min_delta
            self.best_score_default = float("inf")

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class LRSchedulerWrapper:
    """学习率调度器包装"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        scheduler_type: str = "cosine",
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.scheduler_type = scheduler_type
        self.current_step = 0

    def step(self):
        """更新学习率"""
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self) -> float:
        step = self.current_step

        # Warmup
        if step < self.num_warmup_steps:
            return self.optimizer.defaults["lr"] * (step / max(1, self.num_warmup_steps))

        # 衰减后的步数
        progress = (step - self.num_warmup_steps) / max(
            1, self.num_training_steps - self.num_warmup_steps
        )

        if self.scheduler_type == "cosine":
            return self.optimizer.defaults["lr"] * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.scheduler_type == "linear":
            return self.optimizer.defaults["lr"] * (1 - progress)
        else:  # constant
            return self.optimizer.defaults["lr"]

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


# ============================================================
# 多热标签转换
# ============================================================

def labels_to_multihot(
    label_indices: List[List[int]],
    num_labels: int,
) -> torch.Tensor:
    """
    将标签索引列表转换为多热编码向量

    Args:
        label_indices: [[0, 3], [1, 5, 6], ...] 每个样本的正标签索引
        num_labels: 标签总数

    Returns:
        multihot: (N, num_labels) 多热编码矩阵
    """
    N = len(label_indices)
    multihot = torch.zeros(N, num_labels)
    for i, indices in enumerate(label_indices):
        for idx in indices:
            multihot[i, idx] = 1.0
    return multihot


def multihot_to_labels(
    multihot: torch.Tensor,
    label_names: List[str],
    threshold: float = 0.5,
) -> List[List[str]]:
    """
    将多热编码转换回标签名称列表

    Args:
        multihot: (N, num_labels) 多热编码
        label_names: 标签名称列表
        threshold: 决策阈值

    Returns:
        [[label_name, ...], ...]
    """
    if isinstance(multihot, torch.Tensor):
        multihot = multihot.cpu().numpy()

    result = []
    for row in multihot:
        labels = [label_names[i] for i, v in enumerate(row) if v > threshold]
        result.append(labels)
    return result


# ============================================================
# 日志与保存
# ============================================================

def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """格式化指标为可读字符串"""
    parts = []
    if prefix:
        parts.append(f"[{prefix}]")
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    is_best: bool = False,
):
    """保存训练检查点"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # 保存最新检查点
    torch.save(checkpoint, save_path)

    # 保存最佳模型
    if is_best:
        best_path = save_path.replace(".pt", "_best.pt")
        torch.save(checkpoint, best_path)
        print(f"  ✓ 最佳模型已保存: {best_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = None,
) -> Dict:
    """加载训练检查点"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def set_seed(seed: int = 42):
    """固定随机种子以确保可复现性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """自动检测可用设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("工具模块测试")
    print("=" * 60)

    # 测试指标计算
    N, L = 100, 12
    np.random.seed(42)
    targets = np.random.randint(0, 2, (N, L)).astype(np.float32)
    probs = np.clip(targets + np.random.randn(N, L) * 0.3, 0, 1).astype(np.float32)
    preds = (probs > 0.5).astype(np.float32)

    metrics = compute_metrics(
        torch.tensor(preds), torch.tensor(probs), torch.tensor(targets)
    )

    print("\n评估指标:")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")

    # 测试每标签指标
    label_names = [f"emotion_{i}" for i in range(L)]
    per_label = compute_per_label_metrics(
        torch.tensor(preds), torch.tensor(targets), label_names
    )
    print("\n每标签指标 (前3个):")
    for label, m in list(per_label.items())[:3]:
        print(f"  {label}: f1={m['f1']:.3f}, support={m['support']}")

    # 测试早停
    print("\n早停测试:")
    es = EarlyStopping(patience=3, mode="max")
    scores = [0.5, 0.6, 0.58, 0.59, 0.57, 0.56]
    for i, s in enumerate(scores):
        stopped = es(s, i)
        print(f"  epoch {i}: score={s:.2f}, best={es.best_score:.2f}, "
              f"counter={es.counter}, stop={stopped}")

    print("\n✅ 工具模块测试通过！")
