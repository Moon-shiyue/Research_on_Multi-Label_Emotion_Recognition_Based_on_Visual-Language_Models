"""
CLIP 双塔融合方案 — 训练脚本
Multi-Label Emotion Recognition Training Script (CLIP Baseline)

使用方式:
    # 从项目根目录运行（推荐）:
    python -m emotion_model.train --data_root ./data --epochs 50

    # 或直接运行（脚本会自行处理导入路径）:
    python emotion_model/train.py --data_root ./data --epochs 50

    # 多数据集混合训练:
    python -m emotion_model.train --data_root ./data/ArtPhoto ./data/Emotion6 ./data/GAPED

    # GPU 服务器后台训练:
    #   tmux new -s train
    #   conda activate emotion
    #   python -m emotion_model.train --data_root /path/to/data --output ./output --epochs 50
    #   Ctrl+B D 断开

功能:
    - 完整训练循环（BCE / Asymmetric / Focal 三种损失可选）
    - 冻结/微调 CLIP 编码器
    - 混合精度训练（AMP）
    - 余弦学习率 + Warmup
    - 早停 + 最佳模型保存
    - 训练/验证/测试全指标评估（f1/mAP/AUC 等）
"""

import torch
import torch.nn as nn
import os
import sys
import argparse
import json
from datetime import datetime
from tqdm import tqdm

# 确保可以直接运行（python emotion_model/train.py）以及作为包运行（python -m emotion_model.train）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_model.config import (
    ModelConfig, TrainingConfig,
    UNIFIED_EMOTIONS,
)
from emotion_model.dataset import create_dataloaders
from emotion_model.full_model import MultiLabelEmotionModel
from emotion_model.utils import (
    compute_metrics, AverageMeter, EarlyStopping,
    LRSchedulerWrapper, save_checkpoint, set_seed, get_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP 双塔融合 — 多标记情感识别训练")
    parser.add_argument("--data_root", type=str, required=True, nargs="+",
                        help="数据集根目录（可多个，空格分隔）")
    parser.add_argument("--output", type=str, default="./output",
                        help="输出目录（checkpoint 和指标）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--loss_type", type=str, default="asymmetric",
                        choices=["bce", "asymmetric", "focal"],
                        help="损失函数类型")
    parser.add_argument("--classifier_type", type=str, default="shared_attention",
                        choices=["shared_attention", "label_specific"])
    parser.add_argument("--freeze_visual", action="store_true",
                        help="冻结 CLIP 视觉编码器")
    parser.add_argument("--freeze_text", action="store_true",
                        help="冻结 CLIP 文本编码器")
    parser.add_argument("--no_label_association", action="store_true",
                        help="禁用标签关联模块（消融实验）")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的 checkpoint 路径")
    parser.add_argument("--eval_only", action="store_true",
                        help="仅评估不训练")
    parser.add_argument("--innovation", action="store_true",
                        help="启用申请书三大核心创新模块（冲突感知融合 + 情感环形分类头 + VL-Adapter）")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["baseline", "full", "w/o_conflict_fusion",
                                 "w/o_circular_head", "w/o_vl_adapter", "w/o_contrastive"],
                        help="消融实验配置名（优先级高于 --innovation）")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, loss_type, train_config):
    """训练一个 epoch"""
    model.train()
    loss_meter = AverageMeter()

    use_circular = getattr(model, "use_circular_head", False)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        # 移动数据到设备
        pixel_values = batch["pixel_values"].to(model.device)
        targets = batch["labels"].to(model.device)

        # 前向传播
        outputs = model(pixel_values, emotion_labels=model.emotion_labels)
        # 多目标联合损失：分类 + 环形（模块②）+ 冲突对比（模块①）
        loss = model.compute_loss(
            outputs["logits"], targets, loss_type,
            circular_outputs=outputs if use_circular else None,
            contrastive_loss=outputs.get("contrastive_loss"),
        )

        # 混合精度反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.max_grad_norm
        )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loss_meter.update(loss.item())
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}",
                           "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    return loss_meter.avg


@torch.no_grad()
def evaluate(model, data_loader, device):
    """评估：收集所有预测并计算多标记指标"""
    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []

    for batch in tqdm(data_loader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(device)
        targets = batch["labels"].to(device)

        outputs = model(pixel_values, emotion_labels=model.emotion_labels)
        all_probs.append(outputs["probabilities"].cpu())
        all_preds.append(outputs["predictions"].cpu())
        all_targets.append(targets.cpu())

    if not all_probs:
        return {}

    probs = torch.cat(all_probs)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    metrics = compute_metrics(preds, probs, targets)
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 配置（支持基础版 / 创新版 / 消融实验配置）
    if args.ablation:
        from emotion_model.config import create_ablation_configs
        model_config = create_ablation_configs()[args.ablation]
        print(f"  使用消融配置: {args.ablation}")
    elif args.innovation:
        from emotion_model.config import create_innovation_config
        model_config = create_innovation_config()
        print(f"  使用创新版配置（三大核心模块）")
    else:
        model_config = ModelConfig(
            freeze_visual=args.freeze_visual,
            freeze_text=args.freeze_text,
            use_label_association=not args.no_label_association,
        )
    train_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
    )

    device = get_device()
    print("=" * 60)
    print("CLIP 双塔融合 — 多标记情感识别训练")
    print("=" * 60)
    print(f"  设备: {device}")
    print(f"  数据: {args.data_root}")
    print(f"  标签体系: {model_config.num_emotions} 类 {model_config.emotion_labels}")
    print(f"  损失: {args.loss_type}")
    print(f"  冻结视觉/文本: {model_config.freeze_visual}/{model_config.freeze_text}")
    print(f"  模块① 冲突感知融合: {'启用' if getattr(model_config, 'use_conflict_fusion', False) else '禁用'}")
    print(f"  模块② 情感环形分类头: {'启用' if getattr(model_config, 'use_circular_head', False) else '禁用'}")
    print(f"  模块③ VL-Adapter: {'启用' if getattr(model_config, 'use_vl_adapter', False) else '禁用'}")
    print(f"  输出: {args.output}")

    # 加载模型
    print("\n[1/4] 加载模型...")
    model = MultiLabelEmotionModel(model_config)
    model.device = device
    model = model.to(device)

    # 恢复 checkpoint
    start_epoch = 1
    if args.resume:
        from emotion_model.utils import load_checkpoint
        ckpt = load_checkpoint(model, args.resume, device=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  已恢复: epoch {ckpt.get('epoch', 0)}")

    # 数据（标签体系跟随模型配置：创新版为 8 类基础情感）
    print("\n[2/4] 加载数据...")
    if model_config.emotion_labels is not UNIFIED_EMOTIONS:
        print(f"  ⚠ 提示：当前使用 {model_config.num_emotions} 类标签体系"
              f"（{model_config.emotion_labels}），")
        print(f"     请确保数据集的标签文件包含对应标签列。")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_roots=args.data_root,
        emotion_labels=model_config.emotion_labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

    # 优化器与调度器
    print("\n[3/4] 配置优化器...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = LRSchedulerWrapper(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps,
        scheduler_type="cosine",
    )

    # 训练
    if not args.eval_only:
        print("\n[4/4] 开始训练...")
        print(f"  总步数: {total_steps}")

        early_stopper = EarlyStopping(
            patience=train_config.early_stopping_patience,
            mode="max",
        )
        best_f1 = 0.0
        history = []

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, epoch,
                args.loss_type, train_config,
            )

            if val_loader:
                metrics = evaluate(model, val_loader, device)
                f1 = metrics.get("f1_macro", 0)
                print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, "
                      f"val_f1_macro={f1:.4f}, val_mAP={metrics.get('mAP', 0):.4f}")

                history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

                # 保存最佳模型
                if f1 > best_f1:
                    best_f1 = f1
                    save_checkpoint(
                        model, optimizer, epoch, metrics,
                        os.path.join(args.output, "checkpoint.pt"),
                        is_best=True,
                    )
                    with open(os.path.join(args.output, "best_metrics.json"), "w") as f:
                        json.dump(metrics, f, indent=2)

                if early_stopper(f1, epoch):
                    print(f"  早停于 epoch {epoch}")
                    break
            else:
                print(f"  Epoch {epoch}: train_loss={train_loss:.4f}")
                save_checkpoint(
                    model, optimizer, epoch, {"train_loss": train_loss},
                    os.path.join(args.output, "checkpoint.pt"),
                )

        # 保存训练历史
        with open(os.path.join(args.output, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # 最终测试评估
    if test_loader:
        print("\n加载最佳模型进行评估...")
        best_path = os.path.join(args.output, "checkpoint_best.pt")
        if os.path.exists(best_path):
            from emotion_model.utils import load_checkpoint
            load_checkpoint(model, best_path, device=device)

        test_metrics = evaluate(model, test_loader, device)
        print("\n" + "=" * 60)
        print("测试集结果")
        print("=" * 60)
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        with open(os.path.join(args.output, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)

    print(f"\n完成！输出保存在: {args.output}")


if __name__ == "__main__":
    main()
