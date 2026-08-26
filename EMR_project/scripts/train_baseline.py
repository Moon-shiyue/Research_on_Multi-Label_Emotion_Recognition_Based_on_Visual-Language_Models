"""
CLIP 多标记情感识别 — 基线训练脚本

用法:
    # CPU 训练（轻量级基线）
    python scripts/train_baseline.py --device cpu --model simple

    # GPU 训练（完整微调）
    python scripts/train_baseline.py --device cuda --model full --fusion concat

    # 仅做标签统计分析
    python scripts/train_baseline.py --analyze_only

    # 指定数据集
    python scripts/train_baseline.py --datasets emotion6,gaped,artphoto
"""
import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import random


def set_seed(seed: int = 42):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Multi-Label Emotion Recognition Baseline")

    # === 模型 ===
    parser.add_argument("--model", type=str, default="simple",
                        choices=["simple", "full"],
                        help="simple=冻结CLIP+MLP(轻量), full=微调CLIP+融合(完整)")
    parser.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP 模型名称 (HF hub)")
    parser.add_argument("--fusion", type=str, default="concat",
                        choices=["concat", "cross_attention", "gated"],
                        help="多模态融合方式")

    # === 数据 ===
    parser.add_argument("--datasets", type=str, default="emotion6,gaped,artphoto",
                        help="使用的数据集，逗号分隔")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--data_root", type=str,
                        default="C:/Users/32934/Desktop/数据集")

    # === 训练 ===
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="训练设备")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--backbone_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # === 损失 ===
    parser.add_argument("--loss", type=str, default="bce",
                        choices=["bce", "asymmetric", "focal", "focal_bce", "label_smooth"],
                        help="损失函数类型")
    parser.add_argument("--asl_gamma_neg", type=float, default=4.0)
    parser.add_argument("--asl_gamma_pos", type=float, default=1.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    # === 其他 ===
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                        default="C:/Users/32934/Desktop/EMR_project/outputs")
    parser.add_argument("--exp_name", type=str, default="baseline",
                        help="实验名称")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (Windows 建议 0)")
    parser.add_argument("--analyze_only", action="store_true",
                        help="仅分析标签分布，不训练")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # 导入项目模块
    from src.config import Config, DataConfig, ModelConfig, TrainingConfig
    from src.datasets.dataset_loader import create_dataloaders, build_unified_dataset
    from src.datasets.label_stats import analyze_all_datasets
    from src.models.clip_baseline import build_model, SimpleFeatureConcatBaseline
    from src.training.trainer import Trainer

    # ============================================================
    # 构建配置
    # ============================================================
    config = Config()

    # 数据配置
    config.data.data_root = args.data_root
    config.data.image_size = args.image_size

    # 模型配置
    config.model.clip_model_name = args.clip_name
    config.model.fusion_method = args.fusion

    if args.model == "simple":
        # 轻量级：冻结 CLIP，仅训练分类头
        config.model.freeze_image_encoder = True
        config.model.freeze_text_encoder = True

    # 训练配置
    config.device = args.device
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.backbone_lr_ratio = args.backbone_lr_ratio
    config.training.weight_decay = args.weight_decay
    config.training.loss_type = args.loss
    config.training.asymmetric_gamma_neg = args.asl_gamma_neg
    config.training.asymmetric_gamma_pos = args.asl_gamma_pos
    config.training.focal_alpha = args.focal_alpha
    config.training.focal_gamma = args.focal_gamma
    config.training.output_dir = args.output_dir
    config.experiment_name = args.exp_name

    # CPU 训练时减小 batch_size
    if args.device == "cpu":
        config.training.batch_size = min(args.batch_size, 8)
        print(f"[Info] CPU 训练，batch_size 调整为 {config.training.batch_size}")

    # ============================================================
    # 标签统计分析
    # ============================================================
    print("\n" + "=" * 70)
    print("  阶段 1: 标签分布统计")
    print("=" * 70)

    try:
        stats_results = analyze_all_datasets(config)
    except Exception as e:
        print(f"[Warn] 标签统计分析失败: {e}")
        print("[Warn] 继续执行训练...（如果数据集路径尚未配置，请先下载数据集）")

    if args.analyze_only:
        print("\n[Info] 仅分析模式，跳过训练。")
        return

    # ============================================================
    # 构建数据加载器
    # ============================================================
    print("\n" + "=" * 70)
    print("  阶段 2: 构建数据加载器")
    print("=" * 70)

    dataset_list = [d.strip() for d in args.datasets.split(",")]
    print(f"使用数据集: {dataset_list}")

    train_loader, val_loader, test_loader = create_dataloaders(
        config, datasets_to_include=dataset_list
    )

    # ============================================================
    # 构建模型
    # ============================================================
    print("\n" + "=" * 70)
    print("  阶段 3: 构建模型")
    print("=" * 70)

    model = build_model(config)

    # 打印模型结构概要
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters:    {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable ratio:      {trainable_params/total_params:.2%}")

    # ============================================================
    # 训练
    # ============================================================
    print("\n" + "=" * 70)
    print("  阶段 4: 训练")
    print("=" * 70)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
    )

    results = trainer.train()

    # ============================================================
    # 输出最终结果
    # ============================================================
    print("\n" + "=" * 70)
    print("  训练完成!")
    print("=" * 70)

    if results.get("test_results"):
        test_m = results["test_results"]
        print(f"\n  Test Set Results:")
        print(f"  {'-'*40}")
        print(f"  F1 Micro:   {test_m.get('f1_micro', 0):.4f}")
        print(f"  F1 Macro:   {test_m.get('f1_macro', 0):.4f}")
        print(f"  AUC Macro:  {test_m.get('auc_macro', 0):.4f}")
        print(f"  AP Macro:   {test_m.get('ap_macro', 0):.4f}")
        print(f"  Hamming Loss: {test_m.get('hamming_loss', 0):.4f}")
        print(f"  Exact Match:  {test_m.get('exact_match', 0):.4f}")
        print(f"\n  Per-class F1:")
        from src.datasets.label_mapping import BASIC_EMOTIONS
        for i, name in enumerate(BASIC_EMOTIONS):
            f1 = test_m.get(f"f1_{name}", "N/A")
            print(f"    {name:<15}: {f1}" if isinstance(f1, str) else f"    {name:<15}: {f1:.4f}")

    print(f"\n  Best epoch: {results.get('best_epoch', 'N/A')}")
    print(f"  Checkpoint:  {args.output_dir}/checkpoints/best_model.pt")
    print(f"  Logs:        {args.output_dir}/logs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
