"""
标签分布统计与分析模块

功能:
    1. 统计全数据集的标签分布（正样本数、占比）
    2. 识别长尾标签
    3. 统计复合情感样本（多标签样本）的占比
    4. 计算标签共现矩阵
    5. 可视化标签分布与共现关系
    6. 导出统计报告
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非 GUI 后端
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from .label_mapping import (
    BASIC_EMOTIONS, NUM_EMOTIONS, IDX_TO_EMOTION,
    EMOTION_CN, COMPOUND_EMOTION_MAP, multihot_to_names,
)


class LabelStatistics:
    """
    标签统计器。

    使用方式:
        stats = LabelStatistics(dataset)
        stats.compute()
        stats.print_report()
        stats.save_plots("outputs/")
    """

    def __init__(self, dataset, dataset_name: str = "unknown"):
        """
        Args:
            dataset: PyTorch Dataset（会遍历所有样本）
            dataset_name: 数据集名称
        """
        self.dataset = dataset
        self.dataset_name = dataset_name

        # 统计量
        self.total_samples: int = 0
        self.label_counts: np.ndarray = None         # (8,) 每个情感的正样本数
        self.label_distribution: np.ndarray = None    # (8,) 每个情感的正样本占比
        self.cooccurrence_matrix: np.ndarray = None   # (8, 8) 标签共现矩阵
        self.multi_label_counts: Counter = Counter()  # 多标签样本数分布
        self.compound_emotion_stats: Dict = {}         # 复合情感统计
        self.single_label_dist: Dict = {}              # 单标签分布

    def compute(self, max_samples: int = None):
        """遍历数据集，计算所有统计量"""
        N = len(self.dataset)
        if max_samples is not None:
            N = min(N, max_samples)

        self.total_samples = N
        self.label_counts = np.zeros(NUM_EMOTIONS, dtype=np.int64)
        cooccurrence = np.zeros((NUM_EMOTIONS, NUM_EMOTIONS), dtype=np.int64)
        multi_label_counter = Counter()
        single_labels = Counter()

        print(f"[LabelStats] 分析 {self.dataset_name}: {N} 个样本...")

        for i in range(N):
            item = self.dataset[i]
            label = item.get("label", None)
            if label is None:
                continue

            # 转为 numpy
            if hasattr(label, 'numpy'):
                label = label.numpy()
            label = np.asarray(label, dtype=np.float32)

            # 二值化（>0.5 视为正标签）
            binary = (label > 0.5).astype(int)
            active_indices = np.where(binary > 0)[0]
            num_active = len(active_indices)

            # 计数
            self.label_counts += binary
            multi_label_counter[num_active] += 1

            # 单标签分布
            if num_active == 1:
                single_labels[IDX_TO_EMOTION[active_indices[0]]] += 1
            elif num_active == 0:
                single_labels["neutral/none"] += 1

            # 共现矩阵
            for a in active_indices:
                for b in active_indices:
                    cooccurrence[a, b] += 1

        # 计算比率
        self.label_distribution = self.label_counts / N
        self.cooccurrence_matrix = cooccurrence
        self.multi_label_counts = multi_label_counter
        self.single_label_dist = dict(single_labels)

        print(f"[LabelStats] 分析完成。")
        return self

    def get_long_tail_labels(self, tail_threshold: float = 0.1) -> List[Tuple[str, float]]:
        """
        识别长尾标签。

        Args:
            tail_threshold: 占比低于此值视为长尾

        Returns:
            [(emotion_name, ratio), ...] 按占比升序排列
        """
        long_tail = []
        for i in range(NUM_EMOTIONS):
            ratio = self.label_distribution[i]
            if ratio < tail_threshold:
                long_tail.append((BASIC_EMOTIONS[i], ratio))
        long_tail.sort(key=lambda x: x[1])
        return long_tail

    def get_multi_label_ratio(self) -> float:
        """返回多标签样本占比（标签数 > 1）"""
        total_with_labels = sum(self.multi_label_counts.values())
        if total_with_labels == 0:
            return 0.0
        multi = sum(v for k, v in self.multi_label_counts.items() if k > 1)
        return multi / total_with_labels

    def get_top_cooccurrences(self, top_k: int = 10) -> List[Tuple[str, str, int]]:
        """
        返回最频繁的标签共现对。

        Returns:
            [(emotion_a, emotion_b, count), ...] 排序
        """
        pairs = []
        for i in range(NUM_EMOTIONS):
            for j in range(i + 1, NUM_EMOTIONS):
                pairs.append((
                    BASIC_EMOTIONS[i],
                    BASIC_EMOTIONS[j],
                    int(self.cooccurrence_matrix[i, j]),
                ))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_k]

    def print_report(self):
        """打印完整的统计报告"""
        print("\n" + "=" * 70)
        print(f"  数据集标签统计报告 — {self.dataset_name}")
        print("=" * 70)

        print(f"\n总样本数: {self.total_samples}")

        # 标签分布
        print(f"\n--- 标签分布 ---")
        print(f"{'情感':<15} {'中文':<8} {'正样本数':>10} {'占比':>10}")
        print("-" * 48)
        for i in range(NUM_EMOTIONS):
            print(f"{BASIC_EMOTIONS[i]:<15} {EMOTION_CN.get(BASIC_EMOTIONS[i], ''):<8} "
                  f"{self.label_counts[i]:>10} {self.label_distribution[i]:>10.2%}")

        # 长尾标签
        long_tail = self.get_long_tail_labels()
        if long_tail:
            print(f"\n--- 长尾标签（占比 < 10%） ---")
            for name, ratio in long_tail:
                print(f"  {name}: {ratio:.2%}")

        # 多标签分布
        print(f"\n--- 多标签分布 ---")
        print(f"  多标签样本占比: {self.get_multi_label_ratio():.2%}")
        for k in sorted(self.multi_label_counts.keys()):
            v = self.multi_label_counts[k]
            pct = v / max(self.total_samples, 1)
            label_desc = f"{k} 个标签" if k > 0 else "无标签(neutral)"
            print(f"  {label_desc}: {v} ({pct:.2%})")

        # 标签共现 Top-5
        print(f"\n--- 标签共现 Top-5 ---")
        top_pairs = self.get_top_cooccurrences(5)
        for e_a, e_b, cnt in top_pairs:
            pct = cnt / max(self.total_samples, 1)
            print(f"  {e_a} + {e_b}: {cnt} ({pct:.2%})")

        print("=" * 70 + "\n")

    def save_plots(self, output_dir: str):
        """保存可视化图表"""
        os.makedirs(output_dir, exist_ok=True)

        # 设置中文字体（尝试）
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass

        # === 图1: 标签分布柱状图 ===
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = sns.color_palette("husl", NUM_EMOTIONS)
        bars = ax.bar(range(NUM_EMOTIONS), self.label_distribution * 100, color=colors)
        ax.set_xticks(range(NUM_EMOTIONS))
        ax.set_xticklabels(BASIC_EMOTIONS, rotation=30, ha='right')
        ax.set_ylabel("Sample Ratio (%)")
        ax.set_title(f"Label Distribution — {self.dataset_name}")
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Long-tail threshold (10%)')
        ax.legend()

        # 在柱子上标注数值
        for bar, ratio in zip(bars, self.label_distribution):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{ratio:.1%}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"label_distribution_{self.dataset_name}.png"), dpi=150)
        plt.close()

        # === 图2: 标签共现热力图 ===
        fig, ax = plt.subplots(figsize=(9, 7))
        # 归一化共现矩阵（除以总样本数）
        cooc_norm = self.cooccurrence_matrix / max(self.total_samples, 1)
        sns.heatmap(
            cooc_norm, annot=True, fmt='.2%', cmap='YlOrRd',
            xticklabels=BASIC_EMOTIONS, yticklabels=BASIC_EMOTIONS,
            square=True, ax=ax, vmin=0, vmax=cooc_norm.max(),
        )
        ax.set_title(f"Label Co-occurrence Matrix — {self.dataset_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cooccurrence_{self.dataset_name}.png"), dpi=150)
        plt.close()

        # === 图3: 多标签样本分布饼图 ===
        fig, ax = plt.subplots(figsize=(7, 7))
        labels_data = {}
        for k, v in self.multi_label_counts.items():
            if k == 0:
                labels_data["No label"] = v
            elif k == 1:
                labels_data["Single label"] = v
            elif k == 2:
                labels_data["2 labels"] = v
            else:
                labels_data[f"{k} labels"] = v

        wedges, texts, autotexts = ax.pie(
            labels_data.values(), labels=labels_data.keys(),
            autopct='%1.1f%%', startangle=90,
            colors=sns.color_palette("pastel", len(labels_data)),
        )
        ax.set_title(f"Multi-label Distribution — {self.dataset_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"multilabel_pie_{self.dataset_name}.png"), dpi=150)
        plt.close()

        print(f"[LabelStats] Plots saved to: {output_dir}")

    def to_dict(self) -> Dict:
        """转为可序列化的字典"""
        return {
            "dataset_name": self.dataset_name,
            "total_samples": self.total_samples,
            "label_counts": self.label_counts.tolist(),
            "label_distribution": self.label_distribution.tolist(),
            "per_class": {
                BASIC_EMOTIONS[i]: {
                    "count": int(self.label_counts[i]),
                    "ratio": float(self.label_distribution[i]),
                }
                for i in range(NUM_EMOTIONS)
            },
            "multi_label_ratio": self.get_multi_label_ratio(),
            "multi_label_distribution": dict(self.multi_label_counts),
            "top_cooccurrences": [
                {"pair": [a, b], "count": cnt}
                for a, b, cnt in self.get_top_cooccurrences(10)
            ],
            "long_tail_labels": [
                {"emotion": name, "ratio": ratio}
                for name, ratio in self.get_long_tail_labels()
            ],
        }

    def save_json(self, path: str):
        """保存统计结果到 JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[LabelStats] Report saved to: {path}")


def analyze_all_datasets(config) -> Dict[str, LabelStatistics]:
    """
    分析所有配置的数据集并生成汇总报告。

    Args:
        config: Config 对象

    Returns:
        {dataset_name: LabelStatistics}
    """
    from .dataset_loader import _create_single_dataset

    dataset_names = ["emotion6", "gaped", "artphoto", "fi"]
    results = {}

    print("\n" + "=" * 70)
    print("  全数据集标签统计")
    print("=" * 70)

    for ds_name in dataset_names:
        try:
            ds = _create_single_dataset(config, ds_name, split="train")
            if ds is not None and len(ds) > 0:
                stats = LabelStatistics(ds, ds_name)
                stats.compute()
                stats.print_report()

                # 保存图表和 JSON
                output_dir = Path(config.data.processed_root) / "label_stats"
                output_dir.mkdir(parents=True, exist_ok=True)
                stats.save_plots(str(output_dir))
                stats.save_json(str(output_dir / f"stats_{ds_name}.json"))

                results[ds_name] = stats
        except Exception as e:
            print(f"  [{ds_name}] 分析失败: {e}")

    # 汇总表
    if results:
        print("\n--- 跨数据集标签分布对比 ---")
        header = f"{'Emotion':<15}"
        for ds_name in results:
            header += f" {ds_name:>12}"
        print(header)
        print("-" * (15 + 14 * len(results)))

        for i in range(NUM_EMOTIONS):
            row = f"{BASIC_EMOTIONS[i]:<15}"
            for ds_name in results:
                ratio = results[ds_name].label_distribution[i]
                row += f" {ratio:>11.2%}"
            print(row)

    return results
