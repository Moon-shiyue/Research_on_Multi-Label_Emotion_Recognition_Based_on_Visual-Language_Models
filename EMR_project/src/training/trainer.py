"""
训练器模块

功能:
    - 完整训练循环（train / validate / test）
    - 学习率调度 (cosine / linear / step)
    - 早停 (early stopping)
    - 模型检查点保存与恢复
    - 训练日志 (TensorBoard 格式 + 文本)
    - 梯度裁剪
    - 混合精度训练（GPU 可选）
"""
import os
import sys
import time
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts,
    LinearLR, StepLR, ReduceLROnPlateau,
    SequentialLR, LambdaLR,
)

from ..evaluation.metrics import (
    compute_all_metrics, format_metrics_table, MetricsAccumulator,
    find_optimal_threshold,
)
from .losses import get_loss_function


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        """
        Args:
            patience: 容忍 epoch 数
            min_delta: 最小改善量
            mode: max（指标越大越好）或 min（指标越小越好）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.should_stop = False

    def __call__(self, score: float, epoch: int) -> bool:
        """返回 True 表示应该停止"""
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        improved = (
            (self.mode == "max" and score > self.best_score + self.min_delta) or
            (self.mode == "min" and score < self.best_score - self.min_delta)
        )

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        self.should_stop = self.counter >= self.patience
        return self.should_stop

    def state_dict(self) -> Dict:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "counter": self.counter,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state: Dict):
        self.patience = state["patience"]
        self.min_delta = state["min_delta"]
        self.mode = state["mode"]
        self.counter = state["counter"]
        self.best_score = state["best_score"]
        self.best_epoch = state["best_epoch"]


class Trainer:
    """
    CLIP 多标记情感识别训练器。

    使用方式:
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,  # Config object
        test_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config

        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # 配置
        self.train_cfg = config.training
        self.model_cfg = config.model

        # 优化器
        self.optimizer = self._build_optimizer()

        # 学习率调度器
        self.scheduler = self._build_scheduler()
        self.scheduler_type = config.training.lr_scheduler  # Store for later use

        # 损失函数
        self.criterion = get_loss_function(
            self.train_cfg.loss_type,
            asymmetric_gamma_neg=self.train_cfg.asymmetric_gamma_neg,
            asymmetric_gamma_pos=self.train_cfg.asymmetric_gamma_pos,
            focal_alpha=self.train_cfg.focal_alpha,
            focal_gamma=self.train_cfg.focal_gamma,
        )

        # 早停
        self.early_stopping = EarlyStopping(
            patience=self.train_cfg.early_stopping_patience,
            mode="max",  # 以 F1 为目标，越大越好
        )

        # 训练状态
        self.current_epoch = 0
        self.best_epoch = 0
        self.best_val_score = 0.0
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []

        # 输出路径
        self.output_dir = Path(self.train_cfg.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler() if self.train_cfg.use_amp else None

        # 日志
        self.log_file = self.log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        self._log(f"Trainer initialized.")
        self._log(f"  Device: {self.device}")
        self._log(f"  Optimizer: {self.optimizer.__class__.__name__}")
        self._log(f"  Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else 'None'}")
        self._log(f"  Loss: {self.criterion.__class__.__name__}")
        self._log(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """构建优化器，支持分层学习率"""
        cfg = self.train_cfg

        # 分层参数
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(key in name for key in ["clip.vision_model", "clip.text_model", "clip."]):
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {"params": head_params, "lr": cfg.learning_rate},
            {"params": backbone_params, "lr": cfg.learning_rate * cfg.backbone_lr_ratio},
        ]

        if cfg.optimizer.lower() == "adamw":
            return AdamW(param_groups, weight_decay=cfg.weight_decay, eps=cfg.adam_epsilon)
        elif cfg.optimizer.lower() == "sgd":
            return SGD(param_groups, momentum=0.9, weight_decay=cfg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def _build_scheduler(self):
        """构建学习率调度器"""
        cfg = self.train_cfg
        total_steps = len(self.train_loader) * cfg.num_epochs
        warmup_steps = cfg.warmup_steps
        if warmup_steps == 0 and cfg.warmup_ratio > 0:
            warmup_steps = int(total_steps * cfg.warmup_ratio)

        scheduler_type = cfg.lr_scheduler.lower()

        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)
        elif scheduler_type == "cosine_warm_restart":
            scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=total_steps // 3)
        elif scheduler_type == "linear":
            scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.01,
                               total_iters=total_steps - warmup_steps)
        elif scheduler_type == "step":
            scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        else:
            return None

        # 如果有 warmup，在前面加一个线性 warmup
        if warmup_steps > 0:
            warmup = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
            scheduler = SequentialLR(self.optimizer, schedulers=[warmup, scheduler],
                                   milestones=[warmup_steps])

        return scheduler

    def _log(self, message: str):
        """写入日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + "\n")

    def _train_one_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        accumulator = MetricsAccumulator()

        total_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            texts = batch.get("text", None)

            # 前向传播
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(pixel_values=images, text_inputs=texts)
                    logits = outputs["logits"]
                    loss = self.criterion(logits, labels)
            else:
                outputs = self.model(pixel_values=images, text_inputs=texts)
                logits = outputs["logits"]
                loss = self.criterion(logits, labels)

            # 反向传播
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.train_cfg.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.train_cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.max_grad_norm)
                self.optimizer.step()

            # 学习率步进
            if self.scheduler is not None and self.scheduler_type != "plateau":
                self.scheduler.step()

            # 记录
            total_loss += loss.item()
            accumulator.update(
                torch.sigmoid(logits.detach()),
                labels.detach(),
                loss.item(),
            )

            # 日志
            if batch_idx % self.train_cfg.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                elapsed = time.time() - start_time
                self._log(
                    f"  Epoch {self.current_epoch:3d} | "
                    f"Batch {batch_idx:4d}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Elapsed: {elapsed:.0f}s"
                )

        # Epoch 统计
        avg_loss = total_loss / len(self.train_loader)
        train_metrics = accumulator.compute()
        train_metrics["loss"] = avg_loss

        elapsed = time.time() - start_time
        self._log(f"  Train Epoch {self.current_epoch:3d} finished | Loss: {avg_loss:.4f} | Time: {elapsed:.0f}s")

        return train_metrics

    @torch.no_grad()
    def _validate(self, loader: DataLoader, split_name: str = "Val") -> Dict[str, float]:
        """验证/测试"""
        self.model.eval()
        accumulator = MetricsAccumulator()

        total_loss = 0.0

        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            texts = batch.get("text", None)

            outputs = self.model(pixel_values=images, text_inputs=texts)
            logits = outputs["logits"]
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            accumulator.update(
                torch.sigmoid(logits),
                labels,
                loss.item(),
            )

        avg_loss = total_loss / max(len(loader), 1)
        metrics = accumulator.compute()
        metrics["loss"] = avg_loss

        # 打印摘要
        self._log(f"\n{'='*60}")
        self._log(f"  {split_name} Results (Epoch {self.current_epoch}):")
        self._log(f"  Loss: {avg_loss:.4f}")
        self._log(f"  F1 Micro: {metrics.get('f1_micro', 0):.4f} | "
                  f"F1 Macro: {metrics.get('f1_macro', 0):.4f} | "
                  f"AUC: {metrics.get('auc_macro', 0):.4f}")
        self._log(f"  Exact Match: {metrics.get('exact_match', 0):.4f} | "
                  f"Hamming Loss: {metrics.get('hamming_loss', 0):.4f}")
        self._log(f"{'='*60}\n")

        return metrics

    def train(self) -> Dict[str, any]:
        """
        执行完整训练流程。

        Returns:
            包含训练历史和最佳模型信息的字典
        """
        self._log(f"\n{'#'*60}")
        self._log(f"# 开始训练 — {self.config.experiment_name}")
        self._log(f"{'#'*60}\n")

        for epoch in range(1, self.train_cfg.num_epochs + 1):
            self.current_epoch = epoch

            # === 训练 ===
            train_metrics = self._train_one_epoch()
            self.train_history.append(train_metrics)

            # === 验证 ===
            if epoch % self.train_cfg.eval_interval == 0:
                val_metrics = self._validate(self.val_loader, "Val")
                self.val_history.append(val_metrics)

                # Plateaus 调度器在验证后更新
                if self.scheduler is not None and self.scheduler_type == "plateau":
                    self.scheduler.step(val_metrics.get("f1_micro", 0))

                # 检查是否最佳
                val_score = val_metrics.get(self.train_cfg.early_stopping_metric, 0)
                is_best = val_score > self.best_val_score

                if is_best:
                    self.best_val_score = val_score
                    self.best_epoch = epoch
                    self._save_checkpoint("best_model.pt", val_metrics)

                # 早停
                if self.early_stopping(val_score, epoch):
                    self._log(f"Early stopping at epoch {epoch}!")
                    self._log(f"Best score: {self.early_stopping.best_score:.4f} at epoch {self.early_stopping.best_epoch}")
                    break

        # === 结束 ===
        self._log(f"\n{'#'*60}")
        self._log(f"# 训练完成!")
        self._log(f"# Best epoch: {self.best_epoch}, Best {self.train_cfg.early_stopping_metric}: {self.best_val_score:.4f}")
        self._log(f"{'#'*60}\n")

        # 加载最佳模型
        self._load_checkpoint("best_model.pt")

        # 测试
        test_results = None
        if self.test_loader is not None:
            test_results = self._validate(self.test_loader, "Test")

        # 保存最终结果
        self._save_results(train_metrics, test_results)

        return {
            "best_epoch": self.best_epoch,
            "best_val_score": self.best_val_score,
            "train_history": self.train_history,
            "val_history": self.val_history,
            "test_results": test_results,
        }

    def _save_checkpoint(self, filename: str, metrics: Dict[str, float] = None):
        """保存检查点"""
        path = self.checkpoint_dir / filename
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_val_score": self.best_val_score,
            "best_epoch": self.best_epoch,
            "early_stopping": self.early_stopping.state_dict(),
            "metrics": metrics,
            "config": {
                "model": self.model_cfg,
            },
        }
        torch.save(checkpoint, path)
        self._log(f"  Checkpoint saved: {path}")

    def _load_checkpoint(self, filename: str):
        """加载检查点"""
        path = self.checkpoint_dir / filename
        if not path.exists():
            self._log(f"  Checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._log(f"  Model loaded from: {path} (epoch {checkpoint['epoch']})")

    def _save_results(self, final_train_metrics: Dict, test_results: Optional[Dict]):
        """保存最终结果到 JSON"""
        results = {
            "experiment_name": self.config.experiment_name,
            "best_epoch": self.best_epoch,
            "best_val_score": self.best_val_score,
            "val_history": self.val_history,
            "train_history": self.train_history[-1] if self.train_history else {},
            "test_results": test_results,
            "config": {
                "model_name": self.model_cfg.clip_model_name,
                "fusion_method": self.model_cfg.fusion_method,
                "loss_type": self.train_cfg.loss_type,
                "learning_rate": self.train_cfg.learning_rate,
                "batch_size": self.train_cfg.batch_size,
                "num_epochs": self.train_cfg.num_epochs,
            },
        }

        path = self.log_dir / "results.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        self._log(f"Results saved to: {path}")

        # 同时保存一份完整的指标表格
        if test_results is not None:
            table = format_metrics_table(test_results)
            table_path = self.log_dir / "test_metrics.txt"
            with open(table_path, 'w', encoding='utf-8') as f:
                f.write(table)
            self._log(f"Metrics table saved to: {table_path}")


# ============================================================
# 便捷训练函数
# ============================================================

def train_baseline(config, model=None) -> Dict:
    """
    一键训练基线模型。

    Args:
        config: Config 配置对象
        model: 如果提供，使用该模型；否则自动构建

    Returns:
        训练结果字典
    """
    from ..models.clip_baseline import build_model
    from ..datasets.dataset_loader import create_dataloaders

    # 1) 构建数据
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # 2) 构建模型
    if model is None:
        model = build_model(config)

    # 3) 训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
    )

    results = trainer.train()

    return results
