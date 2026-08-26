"""
多标记情感识别评估指标

指标列表:
    1. Subset Accuracy (Exact Match Ratio)
    2. Hamming Loss
    3. Micro / Macro F1
    4. Per-class Precision, Recall, F1
    5. AUC-ROC (per-class + macro)
    6. KL Divergence (for LDL labels)
    7. Cosine Similarity (between predicted and true distribution)
    8. Jaccard Similarity
    9. Average Precision
    10. Ranking-based metrics (Coverage, One-error, Ranking loss)

适用场景:
    - 多标记分类 (multi-hot labels)
    - 标签分布学习 (LDL labels)
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    import torch

from sklearn.metrics import (
    accuracy_score, hamming_loss,
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    jaccard_score,
    coverage_error, label_ranking_loss, label_ranking_average_precision_score,
)
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr

from src.datasets.label_mapping import BASIC_EMOTIONS, NUM_EMOTIONS


# ============================================================
# 指标计算
# ============================================================

def compute_all_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    distributions: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    计算所有评估指标。

    Args:
        probs:  (N, C) 预测概率
        targets: (N, C) 二值标签
        threshold: 概率→二值决策的阈值
        distributions: (N, C) LDL 分布标签（可选）

    Returns:
        指标字典
    """
    N, C = probs.shape

    # 二值化预测
    preds = (probs >= threshold).astype(np.float32)

    metrics = {}

    # === 1. 基于二值预测的指标 ===

    # Subset Accuracy（精确匹配率）
    metrics["exact_match"] = float(accuracy_score(targets, preds))

    # Hamming Loss
    metrics["hamming_loss"] = float(hamming_loss(targets, preds))
    metrics["hamming_accuracy"] = 1.0 - metrics["hamming_loss"]

    # Micro F1（全局计算）
    metrics["f1_micro"] = float(f1_score(targets, preds, average="micro", zero_division=0))
    metrics["precision_micro"] = float(precision_score(targets, preds, average="micro", zero_division=0))
    metrics["recall_micro"] = float(recall_score(targets, preds, average="micro", zero_division=0))

    # Macro F1（每类平均）
    metrics["f1_macro"] = float(f1_score(targets, preds, average="macro", zero_division=0))
    metrics["precision_macro"] = float(precision_score(targets, preds, average="macro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(targets, preds, average="macro", zero_division=0))

    # Weighted F1
    metrics["f1_weighted"] = float(f1_score(targets, preds, average="weighted", zero_division=0))

    # === 2. Per-class F1 ===
    per_class_f1 = f1_score(targets, preds, average=None, zero_division=0)
    for i, name in enumerate(BASIC_EMOTIONS):
        if i < len(per_class_f1):
            metrics[f"f1_{name}"] = float(per_class_f1[i])
            metrics[f"precision_{name}"] = float(precision_score(targets, preds, average=None, zero_division=0)[i])
            metrics[f"recall_{name}"] = float(recall_score(targets, preds, average=None, zero_division=0)[i])

    # === 3. 基于概率的指标 ===

    # AUC-ROC
    try:
        metrics["auc_macro"] = float(roc_auc_score(targets, probs, average="macro"))
        metrics["auc_micro"] = float(roc_auc_score(targets, probs, average="micro"))
    except ValueError:
        metrics["auc_macro"] = 0.0
        metrics["auc_micro"] = 0.0

    # Average Precision (per-class)
    try:
        ap = average_precision_score(targets, probs, average=None)
        metrics["ap_macro"] = float(np.mean(ap))
        for i, name in enumerate(BASIC_EMOTIONS):
            if i < len(ap):
                metrics[f"ap_{name}"] = float(ap[i])
    except ValueError:
        metrics["ap_macro"] = 0.0

    # === 4. Ranking-based 指标 ===

    # Coverage Error
    metrics["coverage_error"] = float(coverage_error(targets, probs))

    # Label Ranking Loss
    metrics["ranking_loss"] = float(label_ranking_loss(targets, probs))

    # Label Ranking Average Precision (LRAP)
    try:
        metrics["lrap"] = float(label_ranking_average_precision_score(targets, probs))
    except ValueError:
        metrics["lrap"] = 0.0

    # === 5. 分布相似度（如果有 LDL 标签） ===

    if distributions is not None:
        # 卡方散度（Jensen-Shannon）
        js_divs = []
        for i in range(N):
            p = probs[i] / (probs[i].sum() + 1e-8)
            q = distributions[i] / (distributions[i].sum() + 1e-8)
            js_divs.append(jensenshannon(p, q) ** 2)
        metrics["js_divergence"] = float(np.mean(js_divs))

        # Cosine Similarity
        cos_sims = []
        for i in range(N):
            p, q = probs[i], distributions[i]
            cos_sim = np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q) + 1e-8)
            cos_sims.append(cos_sim)
        metrics["cosine_similarity"] = float(np.mean(cos_sims))

        # Pearson Correlation
        pearsons = []
        for i in range(N):
            if np.std(probs[i]) > 1e-6 and np.std(distributions[i]) > 1e-6:
                r, _ = pearsonr(probs[i], distributions[i])
                pearsons.append(r)
        metrics["pearson_r"] = float(np.mean(pearsons)) if pearsons else 0.0

    # === 6. Jaccard Similarity ===
    try:
        metrics["jaccard_macro"] = float(jaccard_score(targets, preds, average="macro", zero_division=0))
        metrics["jaccard_micro"] = float(jaccard_score(targets, preds, average="micro", zero_division=0))
    except Exception:
        metrics["jaccard_macro"] = 0.0
        metrics["jaccard_micro"] = 0.0

    return metrics


def compute_metrics_batch(
    probs_batches: List[np.ndarray],
    targets_batches: List[np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    从多个 batch 的累积结果计算指标。

    适用于验证和测试阶段的批量累积。
    """
    probs = np.concatenate(probs_batches, axis=0)
    targets = np.concatenate(targets_batches, axis=0)
    return compute_all_metrics(probs, targets, threshold)


def find_optimal_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    num_thresholds: int = 50,
    metric: str = "f1_micro",
) -> Tuple[float, float]:
    """
    搜索最优二值化阈值。

    Args:
        probs: (N, C) 预测概率
        targets: (N, C) 二值标签
        num_thresholds: 搜索格点数
        metric: 优化目标指标

    Returns:
        (best_threshold, best_value)
    """
    thresholds = np.linspace(0.1, 0.9, num_thresholds)
    best_threshold = 0.5
    best_value = 0.0

    for t in thresholds:
        m = compute_all_metrics(probs, targets, threshold=t)
        v = m.get(metric, 0.0)
        if v > best_value:
            best_value = v
            best_threshold = t

    return best_threshold, best_value


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """
    格式化指标为表格字符串。

    Args:
        metrics: compute_all_metrics 的输出

    Returns:
        格式化后的多行字符串
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"{'Metric':<30} {'Value':>10}")
    lines.append("-" * 70)

    # 关键指标
    key_metrics = [
        "f1_micro", "f1_macro", "f1_weighted",
        "precision_micro", "recall_micro",
        "exact_match", "hamming_loss",
        "auc_macro", "ap_macro",
        "coverage_error", "ranking_loss", "lrap",
        "jaccard_macro",
    ]
    for key in key_metrics:
        if key in metrics:
            lines.append(f"{key:<30} {metrics[key]:>10.4f}")

    # Per-class F1
    lines.append("-" * 70)
    lines.append("Per-class F1:")
    for i, name in enumerate(BASIC_EMOTIONS):
        key = f"f1_{name}"
        if key in metrics:
            lines.append(f"  {name:<15} {metrics[key]:>10.4f}")

    # 分布指标
    dist_keys = ["js_divergence", "cosine_similarity", "pearson_r"]
    if any(k in metrics for k in dist_keys):
        lines.append("-" * 70)
        lines.append("Distribution metrics:")
        for key in dist_keys:
            if key in metrics:
                lines.append(f"  {key:<20} {metrics[key]:>10.4f}")

    lines.append("=" * 70)
    return "\n".join(lines)


# ============================================================
# 累计指标计算器（用于训练中逐步累积）
# ============================================================

class MetricsAccumulator:
    """在多个 batch 上逐步累积预测和目标，最后统一计算指标。"""

    def __init__(self):
        self.probs_list: List[np.ndarray] = []
        self.targets_list: List[np.ndarray] = []
        self.losses: List[float] = []

    def update(self, probs, targets, loss: float = None):
        """添加一个 batch 的结果"""
        import torch
        self.probs_list.append(probs.detach().cpu().numpy() if isinstance(probs, torch.Tensor) else probs)
        self.targets_list.append(targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else targets)
        if loss is not None:
            self.losses.append(loss)

    def compute(self, threshold: float = 0.5) -> Dict[str, float]:
        """计算所有累积数据的指标"""
        if not self.probs_list:
            return {}
        probs = np.concatenate(self.probs_list, axis=0)
        targets = np.concatenate(self.targets_list, axis=0)
        metrics = compute_all_metrics(probs, targets, threshold)
        if self.losses:
            metrics["avg_loss"] = float(np.mean(self.losses))
        return metrics

    def reset(self):
        """重置累积器"""
        self.probs_list.clear()
        self.targets_list.clear()
        self.losses.clear()
