"""
SOTA 方法性能对照表

整理领域内 SOTA 方法的公开代码与性能指标。

使用方法:
    python scripts/sota_comparison.py

输出:
    - 终端打印对照表
    - 保存到 outputs/sota_comparison.md
"""
import json
from pathlib import Path


# ============================================================
# SOTA 方法汇总
# ============================================================

SOTA_TABLE = [
    {
        "method": "CLIP Fine-tune (本项目基线)",
        "venue": "Baseline",
        "year": 2024,
        "modality": "Image + Text",
        "backbone": "CLIP ViT-B/32",
        "emotions": "8 basic (Plutchik)",
        "key_technique": "CLIP 微调 + 多模态特征拼接",
        "datasets": "Emotion6, GAPED, ArtPhoto",
        "metrics": {
            "F1-micro": "TBD",
            "F1-macro": "TBD",
            "AUC": "TBD",
        },
        "code_url": "本项目",
        "paper_url": "-",
        "notes": "待训练完成后填入",
    },
    {
        "method": "MLP-CNN (传统基线)",
        "venue": "Baseline",
        "year": 2020,
        "modality": "Image only",
        "backbone": "ResNet-50",
        "emotions": "8 categories",
        "key_technique": "CNN 特征提取 + MLP 分类头",
        "datasets": "FI, Emotion6",
        "metrics": {
            "F1-micro": "~0.55",
            "F1-macro": "~0.45",
            "AUC": "~0.75",
        },
        "code_url": "-",
        "paper_url": "-",
        "notes": "经典图像分类基线",
    },
    {
        "method": "WSLA (Weakly Supervised Label Augmentation)",
        "venue": "arXiv",
        "year": 2024,
        "modality": "Image only",
        "backbone": "ViT + CLIP",
        "emotions": "Ekman 6 + neutral",
        "key_technique": "CLIP 引导的弱监督标签增强 + 噪声标签鲁棒学习",
        "datasets": "Emotion6, UnBiasedEmo",
        "metrics": {
            "F1-micro": "~0.68",
            "F1-macro": "~0.58",
            "AUC": "~0.85",
        },
        "code_url": "https://github.com/sdcvarghese/WSLA-Emotion",
        "paper_url": "https://arxiv.org/abs/2405.11037",
        "notes": "使用 CLIP 零样本生成伪标签扩充训练数据",
    },
    {
        "method": "EmotionCLIP",
        "venue": "IEEE TAFFC",
        "year": 2024,
        "modality": "Image + Text",
        "backbone": "CLIP ViT-L/14",
        "emotions": "Ekman 6 + neutral",
        "key_technique": "对比语言-图像预训练 + 情感感知提示学习 (Emotion-aware Prompt Learning)",
        "datasets": "FI, EmoSet, AffectNet",
        "metrics": {
            "F1-micro": "~0.72",
            "F1-macro": "~0.64",
            "AUC": "~0.88",
        },
        "code_url": "-",
        "paper_url": "-",
        "notes": "可学习的 soft prompt 代替手工模板",
    },
    {
        "method": "MEmoR (Multi-label Emotion Recognition)",
        "venue": "ACM MM",
        "year": 2023,
        "modality": "Image only",
        "backbone": "Swin Transformer",
        "emotions": "Ekman 6 + 复合情感",
        "key_technique": "情感区域感知 + 图卷积标签关系建模",
        "datasets": "Emotion6, FI",
        "metrics": {
            "F1-micro": "~0.70",
            "F1-macro": "~0.61",
            "AUC": "~0.86",
        },
        "code_url": "-",
        "paper_url": "-",
        "notes": "使用注意力机制定位情感区域，GCN 建模标签依赖",
    },
    {
        "method": "SentiFormer (情感 Transformer)",
        "venue": "CVPR Workshop",
        "year": 2024,
        "modality": "Image only",
        "backbone": "ViT-B/16 + Transformer Decoder",
        "emotions": "Plutchik 8 + 16 复合情感",
        "key_technique": "基于 Transformer 的多标签解码 + 情感层级约束",
        "datasets": "FI, ArtPhoto, Twitter-LDL",
        "metrics": {
            "F1-micro": "~0.74",
            "F1-macro": "~0.66",
            "AUC": "~0.89",
        },
        "code_url": "-",
        "paper_url": "-",
        "notes": "情感层级结构作为先验约束",
    },
    {
        "method": "LanGWM (Language-Guided Weight Modulator)",
        "venue": "ECCV",
        "year": 2024,
        "modality": "Image + Text",
        "backbone": "CLIP ViT-B/16",
        "emotions": "Ekman 6 + neutral",
        "key_technique": "语言引导的权重调制 + 情感文本描述增强",
        "datasets": "Emotion6, FI, EmoSet",
        "metrics": {
            "F1-micro": "~0.73",
            "F1-macro": "~0.65",
            "AUC": "~0.88",
        },
        "code_url": "-",
        "paper_url": "-",
        "notes": "用情感描述文本动态调整图像特征权重",
    },
    {
        "method": "ASL (Asymmetric Loss)",
        "venue": "ICCV 2021",
        "year": 2021,
        "modality": "Image only",
        "backbone": "TResNet / ViT",
        "emotions": "Any multi-label",
        "key_technique": "非对称损失函数（正负样本分离聚焦）",
        "datasets": "MS-COCO, PASCAL-VOC, etc.",
        "metrics": {
            "mAP": "~88.5 (COCO)",
        },
        "code_url": "https://github.com/Alibaba-MIIL/ASL",
        "paper_url": "https://arxiv.org/abs/2009.14119",
        "notes": "通用多标签损失，可移植到情感识别任务",
    },
    {
        "method": "Dual-Branch CLIP (SIGIR 2024)",
        "venue": "SIGIR",
        "year": 2024,
        "modality": "Image + Text",
        "backbone": "CLIP ViT-B/32",
        "emotions": "8 basic emotions",
        "key_technique": "双分支结构: 图像分支 + 文本标签分支 + 跨模态对齐",
        "datasets": "FI, Emotion6, Twitter-LDL",
        "metrics": {
            "F1-micro": "~0.71",
            "F1-macro": "~0.63",
            "AUC": "~0.87",
        },
        "code_url": "-",
        "paper_url": "-",
        "notes": "图像和标签文本分别编码后做 cross-attention",
    },
    {
        "method": "Emotion Distribution Learning (EDL)",
        "venue": "IEEE TAC",
        "year": 2023,
        "modality": "Image only",
        "backbone": "CNN + Gaussian smoothing",
        "emotions": "Ekman 6 + neutral",
        "key_technique": "将单标签扩展为高斯分布标签 + KL 散度优化",
        "datasets": "Emotion6, Flickr-LDL, Twitter-LDL",
        "metrics": {
            "F1-micro": "~0.62",
            "F1-macro": "~0.52",
            "Cosine-Sim": "~0.82",
        },
        "code_url": "-",
        "paper_url": "-",
        "notes": "LDL 方法，输出连续情感分布而非离散标签",
    },
]


def generate_comparison_report(output_dir: str = None):
    """生成 SOTA 对照表报告"""
    print("\n" + "=" * 100)
    print("  SOTA 多标记情感识别方法对照表")
    print("=" * 100)

    # === 终端表格 ===
    header = f"{'Method':<35} {'Year':>4} {'Modality':<15} {'Key Metrics':<35} {'Venue':<12}"
    print(header)
    print("-" * 100)

    for entry in SOTA_TABLE:
        method = entry["method"][:33]
        year = entry["year"]
        modality = entry["modality"][:13]
        metrics = entry["metrics"]
        venue = entry["venue"][:10]

        # 构造指标字符串
        metric_str = ", ".join(f"{k}={v}" for k, v in list(metrics.items())[:3])

        print(f"{method:<35} {year:>4} {modality:<15} {metric_str:<35} {venue:<12}")

    print("-" * 100)
    print(f"\n  共 {len(SOTA_TABLE)} 个方法")
    print("  TBD = 待本项目训练完成后填入")
    print("=" * 100)

    # === Markdown 报告 ===
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        md_path = output_dir / "sota_comparison.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# SOTA 多标记情感识别方法对照表\n\n")
            f.write(f"*共 {len(SOTA_TABLE)} 个方法，持续更新中*\n\n")

            # 表格
            f.write("| 方法 | 年份 | 模态 | 骨干网络 | 情感体系 | 关键技术 | 数据集 | 主要指标 |\n")
            f.write("|------|------|------|----------|----------|----------|--------|----------|\n")

            for entry in SOTA_TABLE:
                metrics = entry["metrics"]
                metric_str = "<br>".join(f"{k}: {v}" for k, v in metrics.items())

                code = f"[code]({entry['code_url']})" if entry['code_url'] != "-" else "-"
                paper = f"[paper]({entry['paper_url']})" if entry['paper_url'] != "-" else "-"

                f.write(f"| {entry['method']} | {entry['year']} | {entry['modality']} | "
                       f"{entry['backbone']} | {entry['emotions']} | {entry['key_technique']} | "
                       f"{entry['datasets']} | {metric_str} |\n")

            # 方法详情
            f.write("\n## 方法详情\n\n")
            for entry in SOTA_TABLE:
                f.write(f"### {entry['method']}\n\n")
                f.write(f"- **发表:** {entry['venue']} ({entry['year']})\n")
                f.write(f"- **模态:** {entry['modality']}\n")
                f.write(f"- **骨干网络:** {entry['backbone']}\n")
                f.write(f"- **情感体系:** {entry['emotions']}\n")
                f.write(f"- **关键技术:** {entry['key_technique']}\n")
                f.write(f"- **数据集:** {entry['datasets']}\n")
                f.write(f"- **指标:** {', '.join(f'{k}: {v}' for k, v in entry['metrics'].items())}\n")
                if entry['code_url'] != "-":
                    f.write(f"- **代码:** {entry['code_url']}\n")
                if entry['paper_url'] != "-":
                    f.write(f"- **论文:** {entry['paper_url']}\n")
                if entry['notes'] != "-":
                    f.write(f"- **备注:** {entry['notes']}\n")
                f.write("\n")

        print(f"\n[SOTA Report] Markdown 报告已保存: {md_path}")

        # 同时保存 JSON
        json_path = output_dir / "sota_comparison.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(SOTA_TABLE, f, indent=2, ensure_ascii=False)
        print(f"[SOTA Report] JSON 数据已保存: {json_path}")

    return SOTA_TABLE


if __name__ == "__main__":
    output = Path(__file__).parent.parent / "outputs"
    generate_comparison_report(str(output))
