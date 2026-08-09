"""
Qwen2.5-VL LoRA 微调训练脚本
———————————————
使用方式：
  python -m qwenvl_emotion.train --data_root ./data --epochs 5 --lr 2e-4

服务器上运行（后台）：
  tmux new -s train
  conda activate emotion
  python -m qwenvl_emotion.train --data_root /path/to/data --output ./output
  # Ctrl+B D 断开
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import os
import sys
import argparse
import json
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwenvl_emotion.config import (
    QwenVLConfig, QwenVLTrainingConfig,
    UNIFIED_EMOTIONS, PROMPT_DIRECT, PROMPT_COT,
)
from qwenvl_emotion.model import QwenVLEmotionModel
from qwenvl_emotion.dataset import QwenVLEmotionDataset, create_qwenvl_dataloaders
from qwenvl_emotion.utils import compute_metrics, AverageMeter, EarlyStopping, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL LoRA 微调")
    parser.add_argument("--data_root", type=str, required=True,
                        help="数据集根目录（可多次指定，空格分隔）", nargs="+")
    parser.add_argument("--output", type=str, default="./qwenvl_output",
                        help="输出目录")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="模型名称")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--prompt_strategy", type=str, default="direct",
                        choices=["direct", "cot", "contrastive"])
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="4bit 量化（显存不足时开启）")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="8bit 量化")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的 checkpoint 路径")
    parser.add_argument("--eval_only", action="store_true",
                        help="仅评估不训练")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, scheduler, epoch, args):
    """训练一个 epoch（生成式训练）"""
    model.model.train()
    loss_meter = AverageMeter()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        # 准备输入
        inputs = model.prepare_inputs(
            images=batch["pil_image"],
            emotion_multihot=batch["labels"],
        )

        # 前向传播（loss 由模型内部计算）
        outputs = model.model(**inputs)
        loss = outputs.loss

        # 梯度累积
        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.model.parameters(), args.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss_meter.update(loss.item() * args.gradient_accumulation_steps)
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}",
                           "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    return loss_meter.avg


@torch.no_grad()
def evaluate(model, dataloader, prompt_strategy="direct"):
    """评估：用 predict 方法逐样本推理，收集指标"""
    model.model.eval()

    all_probs = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        pil_images = batch["pil_image"]
        targets = batch["labels"]

        for i, pil_img in enumerate(pil_images):
            try:
                result = model.predict(pil_img, prompt_strategy=prompt_strategy)
                probs = result["probabilities"]  # (1, num_emotions)
                all_probs.append(probs.squeeze(0))
                all_targets.append(targets[i])
            except Exception as e:
                print(f"  ⚠ 推理失败: {e}")
                all_probs.append(torch.zeros(model.num_emotions))
                all_targets.append(targets[i])

    if not all_probs:
        return {"f1_macro": 0.0}

    probs_t = torch.stack(all_probs)
    targets_t = torch.stack(all_targets)
    preds_t = (probs_t > 0.5).float()

    metrics = compute_metrics(preds_t, probs_t, targets_t)
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    # 配置
    model_config = QwenVLConfig(
        model_name=args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        prompt_strategy=args.prompt_strategy,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    train_config = QwenVLTrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("Qwen2.5-VL LoRA 微调 — 多标记情感识别")
    print("=" * 60)
    print(f"  模型: {args.model}")
    print(f"  LoRA: r={args.lora_r}, α={args.lora_alpha}")
    print(f"  Prompt: {args.prompt_strategy}")
    print(f"  学习率: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch_size}")
    print(f"  数据: {args.data_root}")
    print(f"  输出: {args.output}")

    # 加载模型
    print("\n[1/4] 加载模型...")
    model = QwenVLEmotionModel(model_config)
    model.setup_lora()

    # 数据
    print("\n[2/4] 加载数据...")
    # 支持多数据源
    data_roots = args.data_root if isinstance(args.data_root, list) else [args.data_root]
    train_loader, val_loader, test_loader = create_qwenvl_dataloaders(
        data_roots=data_roots,
        emotion_labels=UNIFIED_EMOTIONS,
        batch_size=args.batch_size,
    )

    # 优化器
    print("\n[3/4] 配置优化器...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.03)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # 训练
    if not args.eval_only:
        print("\n[4/4] 开始训练...")
        print(f"  总步数: {total_steps}, 预热步数: {warmup_steps}")

        early_stopper = EarlyStopping(patience=5, mode="max")
        best_f1 = 0.0

        for epoch in range(1, args.epochs + 1):
            # 训练
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, epoch, train_config
            )

            # 验证
            if val_loader:
                metrics = evaluate(model, val_loader, args.prompt_strategy)
                f1 = metrics.get("f1_macro", 0)
                print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, "
                      f"val_f1_macro={f1:.4f}, val_mAP={metrics.get('mAP', 0):.4f}")

                # 保存最佳模型
                if f1 > best_f1:
                    best_f1 = f1
                    model.save_lora(os.path.join(args.output, "best_lora"))
                    with open(os.path.join(args.output, "best_metrics.json"), "w") as f:
                        json.dump(metrics, f, indent=2)

                if early_stopper(f1, epoch):
                    print(f"  早停于 epoch {epoch}")
                    break
            else:
                print(f"  Epoch {epoch}: train_loss={train_loss:.4f}")
                model.save_lora(os.path.join(args.output, f"lora_epoch_{epoch}"))

    # 最终评估
    if test_loader and val_loader:
        print("\n加载最佳 LoRA 权重进行评估...")
        best_path = os.path.join(args.output, "best_lora")
        if os.path.exists(best_path):
            model.load_lora(best_path)

        test_metrics = evaluate(model, test_loader, args.prompt_strategy)
        print("\n" + "=" * 60)
        print("测试集结果")
        print("=" * 60)
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

        with open(os.path.join(args.output, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)

    # 保存最终 LoRA
    model.save_lora(os.path.join(args.output, "final_lora"))
    print(f"\n完成！输出保存在: {args.output}")


if __name__ == "__main__":
    main()
