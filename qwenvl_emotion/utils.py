"""
Qwen2.5-VL 方案 — 工具函数
共享评估指标（与 CLIP 方案完全一致）+ VLM 专用工具
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    f1_score, precision_score, recall_score, hamming_loss,
    average_precision_score, roc_auc_score,
)


def compute_metrics(predictions, probabilities, targets, threshold=0.5) -> Dict[str, float]:
    """多标记评估指标（与 CLIP 方案完全一致，确保对比公平）"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    predictions = (predictions > 0.5).astype(np.float32)
    targets = targets.astype(np.float32)

    metrics = {}
    metrics["hamming_loss"] = hamming_loss(targets, predictions)

    for avg in ["micro", "macro", "samples"]:
        try:
            metrics[f"f1_{avg}"] = f1_score(targets, predictions, average=avg, zero_division=0)
        except:
            metrics[f"f1_{avg}"] = float("nan")

    try:
        metrics["mAP"] = average_precision_score(targets, probabilities, average="macro")
    except:
        metrics["mAP"] = float("nan")
    try:
        metrics["auc_macro"] = roc_auc_score(targets, probabilities, average="macro")
    except:
        metrics["auc_macro"] = float("nan")

    return metrics


class AverageMeter:
    """滑动平均（与 CLIP 方案相同）"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """早停（与 CLIP 方案相同）"""
    def __init__(self, patience=5, mode="max", verbose=True):
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        if (self.mode == "max" and score > self.best_score) or \
           (self.mode == "min" and score < self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


def extract_json_from_text(text: str) -> dict:
    """从模型输出文本中提取 JSON"""
    import re, json
    # 尝试 ```json ``` 代码块
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # 尝试裸 JSON
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    return {}


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
