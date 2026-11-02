# -*- coding: utf-8 -*-
"""
固定划分消融实验 — 定位「完整模型不如线性层」的原因。

背景
----
CLIP 基线实验（emotion_model/baselines.py）在 ArtPhoto 上得到：

    B3 线性探测（6,152 参数）  macro F1 = 0.4051
    B4 MLP  探测（198,920 参数）macro F1 = 0.4675
    FULL 完整模型（19,095,572 参数）macro F1 = 0.2955  ← 反而不如线性层

本脚本用**同一份 fixed_splits.json、同一套训练超参**，只改变两个因素，
判断差距到底来自哪里：

    因素 A：是否安装 VL-Adapter（模块③）
    因素 B：辅助损失权重（环形 / 一致性 / 对比 三项）

实验组
------
    e0_reproduce       复现原配置（应≈0.2955，用于确认脚本等价性）
    e1_no_adapter      去掉 VL-Adapter，损失权重不变
    e2_reweight        保留 VL-Adapter，只调损失权重
    e3_both            两者同时改

用法
----
    python -m emotion_model.fixed_split_experiment \
        --image-dir ../testImages_artphoto \
        --splits ../ArtPhoto_调参后完整模型_20261102/reports/fixed_splits.json \
        --out ../results/ablation_artphoto \
        --experiments e0_reproduce e1_no_adapter e2_reweight e3_both \
        --epochs 12 --lr 3e-5
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_model.config import create_full_config, MIKELS_BASIC_EMOTIONS
from emotion_model.dataset import get_default_transforms
from emotion_model.full_model import MultiLabelEmotionModel
from emotion_model.utils import compute_metrics, set_seed, LRSchedulerWrapper


LABELS = list(MIKELS_BASIC_EMOTIONS)

# 原始损失权重（emotion_model/full_model.py:507）
ORIGINAL_WEIGHTS = {"cls": 1.0, "angle": 0.1, "circular": 0.5,
                    "consistency": 0.3, "contrastive": 0.3}

# 重加权方案：让分类项从 8% 提到 60% 以上
# 依据：实测训练集上 cls=0.17、circular=1.21、consistency=3.25、contrastive=1.07，
# 乘以下列权重后占比约 cls 0.60 / circular 0.18 / consistency 0.10 / 其它 0.12
REWEIGHTED_WEIGHTS = {"cls": 1.0, "angle": 0.02, "circular": 0.05,
                      "consistency": 0.008, "contrastive": 0.05}


class FixedSplitDataset(Dataset):
    """按 fixed_splits.json 的 filename 直接读图，不做任何重新划分。"""

    def __init__(self, rows, image_dir, is_train, augment=True):
        self.rows = rows
        self.image_dir = Path(image_dir)
        # augment=False 时训练集也只做 Resize，用于检验数据增强是否为瓶颈
        self.transform = get_default_transforms(is_train=(is_train and augment))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(self.image_dir / row["filename"]) as im:
            image = self.transform(im.convert("RGB"))
        target = torch.zeros(len(LABELS))
        target[LABELS.index(row["label"])] = 1
        return image, target, row["filename"]


def load_splits(split_file):
    data = json.loads(Path(split_file).read_text(encoding="utf-8"))
    # 兼容两种结构：{"train": [...], ...} 或 {"splits": {"train": [...]}}
    return data["splits"] if "splits" in data else data


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs_all, targets_all, names_all = [], [], []
    for images, targets, names in loader:
        out = model(images.to(device), emotion_labels=LABELS)
        probs_all.append(out["probabilities"].cpu())
        targets_all.append(targets)
        names_all.extend(names)
    probs = torch.cat(probs_all)
    targets = torch.cat(targets_all)
    return probs, targets, names_all


def search_threshold(scores, targets, n_candidates=200):
    """在验证集上搜索二值化阈值，最大化 macro F1。只在验证集调用。"""
    from sklearn.metrics import f1_score
    flat = scores.reshape(-1)
    lo, hi = float(np.min(flat)), float(np.max(flat))
    if hi <= lo:
        return 0.5
    candidates = np.linspace(lo, hi, n_candidates)
    best_th, best_f1 = 0.5, -1.0
    for th in candidates:
        pred = (scores >= th).astype(np.float32)
        f1 = f1_score(targets, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
    return best_th


def evaluate_with_protocol(scores, targets, threshold):
    """用给定阈值算全套指标（含不依赖阈值的 mAP / AUC）。"""
    pred = (scores >= threshold).astype(np.float32)
    metrics = compute_metrics(pred, scores, targets)
    true_top1 = targets.argmax(1)
    pred_top1 = scores.argmax(1)
    metrics["top1_accuracy"] = float((true_top1 == pred_top1).mean())
    from sklearn.metrics import f1_score
    metrics["top1_macro_f1"] = float(
        f1_score(true_top1, pred_top1, labels=list(range(len(LABELS))),
                 average="macro", zero_division=0)
    )
    return metrics


def run_one(name, weights, use_adapter, splits, image_dir, args, device,
            use_conflict_fusion=True, use_circular_head=True,
            fusion_hidden_dim=None, unfreeze_layer_norm=False):
    """跑一组实验，返回结果 dict。

    fusion_hidden_dim 为 None 时沿用 config.py 的默认值（512）。
    unfreeze_layer_norm 为 True 时解冻主干 LayerNorm 仿射参数（CLIP-Adapter 做法）。
    """
    print("\n" + "=" * 72)
    print(f"实验 {name}")
    print(f"  VL-Adapter: {'启用' if use_adapter else '禁用'}")
    print(f"  模块① 冲突感知融合: {'启用' if use_conflict_fusion else '禁用（退回层次化注意力融合）'}")
    print(f"  模块② 环形分类头  : {'启用' if use_circular_head else '禁用（退回通用多标记头）'}")
    print(f"  融合隐层维度: {fusion_hidden_dim if fusion_hidden_dim else '默认 512'}")
    print(f"  解冻主干 LayerNorm: {'是（CLIP-Adapter 做法）' if unfreeze_layer_norm else '否（主干完全冻结）'}")
    print(f"  损失权重  : {weights}")
    print("=" * 72)

    cfg_overrides = dict(use_vl_adapter=use_adapter,
                         use_conflict_fusion=use_conflict_fusion,
                         use_circular_head=use_circular_head,
                         adapter_unfreeze_layer_norm=unfreeze_layer_norm)
    if fusion_hidden_dim:
        cfg_overrides["fusion_hidden_dim"] = fusion_hidden_dim
    config = create_full_config(**cfg_overrides)

    set_seed(args.seed)

    train_ds = FixedSplitDataset(splits["train"], image_dir, is_train=True)
    val_ds = FixedSplitDataset(splits["val"], image_dir, is_train=False)
    test_ds = FixedSplitDataset(splits["test"], image_dir, is_train=False)

    # GPU 上 num_workers>0 才划算；Windows 下 spawn 有额外开销，用 0 更稳
    nw = args.num_workers
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw)

    model = MultiLabelEmotionModel(config).to(device)
    model.device = device

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  可训练参数: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = LRSchedulerWrapper(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
        scheduler_type="cosine",
    )

    history = []
    best = None
    best_state = None
    best_val_probs = None
    patience = 0
    t0 = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_examples = 0.0, 0
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(images, emotion_labels=LABELS)
            loss = model.compute_loss(
                out["logits"], targets, "asymmetric",
                circular_outputs=out,
                contrastive_loss=out.get("contrastive_loss"),
                loss_weights=weights,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name}: 损失非有限值，停止")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(images)
            n_examples += len(images)

        val_probs, val_targets, _ = evaluate(model, val_loader, device)
        val_np = val_probs.numpy()
        val_t = val_targets.numpy()
        th = search_threshold(val_np, val_t)
        val_metrics = evaluate_with_protocol(val_np, val_t, th)

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_examples, 1),
            "val_threshold": th,
            "val_f1_macro": val_metrics["f1_macro"],
            "val_mAP": val_metrics["mAP"],
            "val_auc": val_metrics["auc_macro"],
        }
        history.append(row)

        marker = ""
        if best is None or val_metrics["f1_macro"] > best["val_f1_macro"]:
            marker = "  <- best"
            best = row.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_probs = val_np
            patience = 0
        else:
            patience += 1

        print(f"  epoch {epoch:02d}  loss={row['train_loss']:.4f}  "
              f"th={th:.3f}  val_macroF1={row['val_f1_macro']:.4f}  "
              f"val_mAP={row['val_mAP']:.4f}{marker}")

        if patience >= args.patience:
            print(f"  早停于 epoch {epoch}")
            break

    elapsed = time.monotonic() - t0

    # 用最佳轮次的验证集阈值，只应用一次到测试集
    model.load_state_dict(best_state)
    test_probs, test_targets, test_names = evaluate(model, test_loader, device)
    test_np = test_probs.numpy()
    test_t = test_targets.numpy()
    test_metrics = evaluate_with_protocol(test_np, test_t, best["val_threshold"])

    print(f"  测试集: macroF1={test_metrics['f1_macro']:.4f}  "
          f"mAP={test_metrics['mAP']:.4f}  auc={test_metrics['auc_macro']:.4f}  "
          f"top1={test_metrics['top1_accuracy']:.4f}  ({elapsed:.0f}s)")

    result = {
        "name": name,
        "use_vl_adapter": use_adapter,
        "use_conflict_fusion": use_conflict_fusion,
        "use_circular_head": use_circular_head,
        "unfreeze_layer_norm": bool(use_adapter and unfreeze_layer_norm),
        "fusion_hidden_dim": config.fusion_hidden_dim,
        "loss_weights": weights,
        "n_trainable_params": n_trainable,
        "best_epoch": best["epoch"],
        "val_threshold": best["val_threshold"],
        "val": best,
        "test": test_metrics,
        "seconds": elapsed,
        "history": history,
    }

    # 保存预测，便于后续复用同一套评估
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "val_predictions.npz",
                        probabilities=best_val_probs, targets=val_t)
    np.savez_compressed(out_dir / "test_predictions.npz",
                        probabilities=test_np, targets=test_t,
                        filenames=np.asarray(test_names))
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )

    del model, optimizer, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="固定划分消融实验")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--out", default="../results/ablation_artphoto")
    p.add_argument("--experiments", nargs="+",
                   default=["e0_reproduce", "e1_no_adapter", "e2_reweight", "e3_both"])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / cpu（缺省自动选择）")
    args = p.parse_args(argv)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("固定划分消融实验")
    print("=" * 72)
    print(f"  设备      : {device}")
    if device.type == "cuda":
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  torch     : {torch.__version__}")
    print(f"  图像目录  : {args.image_dir}")
    print(f"  划分文件  : {args.splits}")
    print(f"  实验组    : {', '.join(args.experiments)}")

    splits = load_splits(args.splits)
    for sp in ("train", "val", "test"):
        print(f"  {sp:<5}: {len(splits[sp])} 条")

    # 实验定义表: (损失权重, VL-Adapter, 模块① 冲突融合, 模块② 环形头, 融合隐层维度, 解冻主干 LayerNorm)
    # 解冻 LayerNorm 一栏默认 False，以保证 e0~e10 与历史结果逐位一致；
    # 只有 e11/e12 显式打开，用于单独度量「解冻 LN」这一因素。
    defs = {
        "e0_reproduce":  (ORIGINAL_WEIGHTS,   True,  True,  True,  None, False),
        "e1_no_adapter": (ORIGINAL_WEIGHTS,   False, True,  True,  None, False),
        "e2_reweight":   (REWEIGHTED_WEIGHTS, True,  True,  True,  None, False),
        "e3_both":       (REWEIGHTED_WEIGHTS, False, True,  True,  None, False),
        # 模块① 换回带文本的层次化注意力融合（消融基线路径），其余不变
        "e4_no_conflict": (ORIGINAL_WEIGHTS,  True,  False, True,  None, False),
        # 模块② 换成通用多标记分类头，其余不变
        "e5_no_circular": (ORIGINAL_WEIGHTS,  True,  True,  False, None, False),
        # 一次性关掉模块①②③，只留 CLIP + 通用头 —— 项目自带的最强基线
        "e6_plain":       (ORIGINAL_WEIGHTS,  False, False, False, None, False),
        # ---- 容量缩减组（m02767「先跑短期试试」）----
        # e7: 只把融合隐层 512→128，其余与 e4 完全一致（模块①关、适配器开、环形头开）
        "e7_e4_dim128":   (ORIGINAL_WEIGHTS,  True,  False, True,  128,  False),
        # e8: 更激进，512→64
        "e8_e4_dim64":    (ORIGINAL_WEIGHTS,  True,  False, True,  64,   False),
        # e9: 不开适配器（因为适配器会把参数拉回千万级），dim=128，看纯缩减上限
        "e9_lean_dim128": (ORIGINAL_WEIGHTS,  False, False, True,  128,  False),
        # e10: 融合隐层砍到 32，融合模块仅 12 万参数（接近容量扫描最优的 20 万）
        "e10_e4_dim32":   (ORIGINAL_WEIGHTS,  True,  False, True,  32,   False),
        # ---- 解冻主干 LayerNorm 组（m03051「先做 A」）----
        # e11: 与 e0 完全一致，仅额外解冻主干 LayerNorm（+65,536 参数）
        "e11_ln_e0":      (ORIGINAL_WEIGHTS,  True,  True,  True,  None, True),
        # e12: 与 e4 完全一致（当前最优结构），仅额外解冻主干 LayerNorm
        "e12_ln_e4":      (ORIGINAL_WEIGHTS,  True,  False, True,  None, True),
    }

    Path(args.out).mkdir(parents=True, exist_ok=True)
    results = []
    for exp in args.experiments:
        if exp not in defs:
            print(f"[!] 未知实验组 {exp}，跳过")
            continue
        weights, use_adapter, use_cf, use_circ, fdim, unfreeze_ln = defs[exp]
        try:
            results.append(run_one(exp, weights, use_adapter, splits,
                                   args.image_dir, args, device,
                                   use_conflict_fusion=use_cf,
                                   use_circular_head=use_circ,
                                   fusion_hidden_dim=fdim,
                                   unfreeze_layer_norm=unfreeze_ln))
        except Exception as exc:
            print(f"[!] 实验 {exp} 失败：{exc}")
            results.append({"name": exp, "error": str(exc)})

    # 汇总表
    ok = [r for r in results if "error" not in r]
    if ok:
        print("\n" + "=" * 72)
        print("汇总（测试集，阈值来自验证集搜索）")
        print("=" * 72)
        print("| 实验 | VL-Adapter | 模块① | 模块② | 解冻LN | 融合维度 | 可训练参数 | 阈值 | macro F1 | micro F1 | mAP | AUC | top-1 |")
        print("|---|---|---|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|")
        for r in ok:
            t = r["test"]
            print(f"| {r['name']} | {'✓' if r['use_vl_adapter'] else '✗'} | "
                  f"{'✓' if r.get('use_conflict_fusion', True) else '✗'} | "
                  f"{'✓' if r.get('use_circular_head', True) else '✗'} | "
                  f"{'✓' if r.get('unfreeze_layer_norm', False) else '✗'} | "
                  f"{r.get('fusion_hidden_dim', 512)} | "
                  f"{r['n_trainable_params']:,} | {r['val_threshold']:.3f} | "
                  f"**{t['f1_macro']:.4f}** | {t['f1_micro']:.4f} | "
                  f"{t['mAP']:.4f} | {t['auc_macro']:.4f} | {t['top1_accuracy']:.4f} |")

        print("\n对照（来自 emotion_model/baselines.py，同一划分）：")
        print("  B3 线性探测（6,152 参数）   macro F1 = 0.4051")
        print("  B4 MLP 探测（198,920 参数） macro F1 = 0.4675")
        print("  B1 零样本 top-1             0.3814")

        (Path(args.out) / "summary.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=float),
            encoding="utf-8",
        )
        print(f"\n结果已写入: {Path(args.out) / 'summary.json'}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
