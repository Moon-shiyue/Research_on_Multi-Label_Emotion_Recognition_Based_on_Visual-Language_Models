"""
Qwen2.5-VL 方案 — 模块功能验证
"""

import torch
import sys
import os
import json

# Windows 控制台默认 GBK 编码无法输出 emoji，强制 UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwenvl_emotion.config import (
    QwenVLConfig, UNIFIED_EMOTIONS,
    PROMPT_DIRECT, PROMPT_COT, PROMPT_CONTRASTIVE,
)


def verify_config():
    """验证配置完整性"""
    print("=" * 60)
    print("Qwen2.5-VL 方案 — 模块验证")
    print("=" * 60)

    checks = []

    # 1. 配置验证
    print("\n[1] 配置验证")
    cfg = QwenVLConfig()
    c = lambda n, v: checks.append((n, v))
    c("模型名称已设置", cfg.model_name != "")
    c("情感标签数量=12", cfg.num_emotions == 12)
    c("LoRA rank > 0", cfg.lora_r > 0)
    c("LoRA alpha > 0", cfg.lora_alpha > 0)
    c("Prompt策略已设置", cfg.prompt_strategy in ("direct", "cot", "contrastive"))
    for name, ok in checks:
        print(f"  {'✅' if ok else '❌'} {name}")

    # 2. Prompt 模板验证
    print("\n[2] Prompt 模板验证")
    for name, prompt in [("Direct", PROMPT_DIRECT), ("CoT", PROMPT_COT),
                          ("Contrastive", PROMPT_CONTRASTIVE)]:
        ok = len(prompt) > 50 and "JSON" in prompt.upper() or "json" in prompt
        print(f"  {'✅' if ok else '❌'} {name} template ({len(prompt)} chars)")

    # 3. 情感标签中英文对齐
    print("\n[3] 情感标签体系")
    from qwenvl_emotion.config import EMOTION_CN
    for emo in UNIFIED_EMOTIONS:
        has_cn = emo in EMOTION_CN
        print(f"  {'✅' if has_cn else '❌'} {emo} → {EMOTION_CN.get(emo, 'MISSING')}")

    # 4. JSON 解析测试
    print("\n[4] JSON 输出解析测试")
    test_outputs = [
        ('{"emotions": {"joy": 0.9, "sadness": 0.1, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0, "love": 0.0, "peace": 0.0, "amusement": 0.0, "awe": 0.0, "contentment": 0.0, "excitement": 0.0}}',
         True),
        ('```json\n{"emotions": {"joy": 0.8, "sadness": 0.2}}\n```', True),
        ('这是一张快乐的图片 {"emotions": {"joy": 1.0}}', True),
        ('无法解析的文本', False),
    ]
    from qwenvl_emotion.utils import extract_json_from_text
    for text, should_parse in test_outputs:
        parsed = extract_json_from_text(text)
        ok = (len(parsed) > 0) == should_parse
        print(f"  {'✅' if ok else '❌'} '{text[:50]}...' → {len(parsed) > 0}")

    # 5. 数据加载验证（模拟）
    print("\n[5] 数据集创建验证")
    import tempfile, shutil
    from PIL import Image
    import numpy as np
    from qwenvl_emotion.dataset import QwenVLEmotionDataset

    tmpdir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmpdir, "images"), exist_ok=True)

        for i in range(5):
            img = Image.new("RGB", (448, 448), color=tuple(np.random.randint(0, 255, 3)))
            img.save(os.path.join(tmpdir, "images", f"img_{i:03d}.jpg"))

        import csv
        with open(os.path.join(tmpdir, "labels.csv"), "w") as f:
            w = csv.writer(f)
            w.writerow(["filename"] + UNIFIED_EMOTIONS[:6])
            for i in range(5):
                row = [f"img_{i:03d}.jpg"] + [str(np.random.randint(0, 2)) for _ in range(6)]
                w.writerow(row)

        try:
            ds = QwenVLEmotionDataset(tmpdir, UNIFIED_EMOTIONS[:6], "train")
            sample = ds[0]
            print(f"  ✅ 数据集加载成功: {len(ds)} 样本")
            print(f"     pixel_values: {sample['pixel_values'].shape}")
            print(f"     pil_image: {sample['pil_image'].size}")
            print(f"     labels: {sample['labels']}")
        except Exception as e:
            print(f"  ❌ 数据集加载失败: {e}")
    except PermissionError as e:
        print(f"  ⚠ 临时目录不可写，跳过数据加载验证: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # 6. 环境检测
    print("\n[6] 运行环境检测")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # 检查关键依赖
    deps = ["transformers", "peft", "PIL", "sklearn"]
    for dep in deps:
        try:
            __import__(dep)
            print(f"  {dep}: ✅")
        except ImportError:
            print(f"  {dep}: ❌ 未安装")

    print("\n" + "=" * 60)
    print("⚠ 完整模型加载需要 GPU + 下载 Qwen2.5-VL 权重")
    print("在有 GPU 的服务器上运行:")
    print("  from qwenvl_emotion import QwenVLEmotionModel")
    print("  model = QwenVLEmotionModel()")
    print("  model.setup_lora()")
    print("=" * 60)


if __name__ == "__main__":
    verify_config()
