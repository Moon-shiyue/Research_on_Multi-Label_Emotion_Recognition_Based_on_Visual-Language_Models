"""
验证所有模块是否能正常导入和运行
"""
import sys
sys.path.insert(0, r"C:\Users\32934\Desktop\EMR_project")

import numpy as np

# 1. label_mapping
print("=" * 60)
print("测试 label_mapping...")
from src.datasets.label_mapping import (
    BASIC_EMOTIONS, EMOTION_TO_IDX, emotions_to_multihot,
    map_dataset_label, multihot_to_names, compound_to_basic_distribution,
    NUM_EMOTIONS, get_text_prompts_for_emotions,
)
print(f"  Emotions: {BASIC_EMOTIONS}")
print(f"  Joy+sadness: {emotions_to_multihot(['joy', 'sadness'])}")
print(f"  Love (compound): {emotions_to_multihot(['love'])}")
print(f"  Awe distribution: {compound_to_basic_distribution('awe')}")
print(f"  Emotion6→8d: {map_dataset_label('emotion6', 'anger')}")
print(f"  Multihot→names: {multihot_to_names([1,0,1,0,0,0,0,0])}")
print(f"  Text prompts: {get_text_prompts_for_emotions([0, 1])}")
print("  PASSED\n")

# 2. config
print("=" * 60)
print("测试 config...")
from src.config import Config, DataConfig, ModelConfig, TrainingConfig
cfg = Config()
print(f"  Device: {cfg.device}")
print(f"  Image size: {cfg.data.image_size}")
print(f"  Model: {cfg.model.clip_model_name}")
print(f"  Fusion: {cfg.model.fusion_method}")
print(f"  Loss: {cfg.training.loss_type}")
print("  PASSED\n")

# 3. losses
print("=" * 60)
print("测试 losses...")
import torch
from src.training.losses import get_loss_function, AsymmetricLoss, FocalLoss, CombinedLoss
for lt in ["bce", "asymmetric", "focal", "focal_bce", "label_smooth"]:
    loss_fn = get_loss_function(lt)
    logits = torch.randn(4, 8)
    targets = torch.randint(0, 2, (4, 8)).float()
    loss = loss_fn(logits, targets)
    print(f"  {lt}: {loss_fn.__class__.__name__}, loss={loss.item():.4f}")
print("  PASSED\n")

# 4. metrics
print("=" * 60)
print("测试 metrics...")
from src.evaluation.metrics import compute_all_metrics, format_metrics_table, MetricsAccumulator
np.random.seed(42)
N, C = 100, 8
probs = np.random.rand(N, C).astype(np.float32)
targets = (np.random.rand(N, C) > 0.7).astype(np.float32)
m = compute_all_metrics(probs, targets)
print(f"  F1 micro: {m['f1_micro']:.4f}")
print(f"  F1 macro: {m['f1_macro']:.4f}")
print(f"  Hamming Loss: {m['hamming_loss']:.4f}")
print(f"  AUC macro: {m['auc_macro']:.4f}")
print(f"  Coverage Error: {m['coverage_error']:.4f}")
print(f"  LRAP: {m['lrap']:.4f}")

# test accumulator
acc = MetricsAccumulator()
acc.update(torch.randn(10, 8), (torch.rand(10, 8) > 0.5).float(), loss=0.5)
acc.update(torch.randn(10, 8), (torch.rand(10, 8) > 0.5).float(), loss=0.3)
acc_m = acc.compute()
print(f"  Accumulator F1: {acc_m['f1_micro']:.4f}")
print("  PASSED\n")

# 5. augmentation
print("=" * 60)
print("测试 augmentation...")
from src.datasets.augmentation import TextAugmentation, ImageAugmentation
text_aug = TextAugmentation(synonym_prob=0.5)
test_text = "a happy person expressing joy and surprise in a beautiful photo"
aug_text = text_aug.augment(test_text, ["joy", "surprise"])
print(f"  Original: {test_text[:80]}...")
print(f"  Augmented: {aug_text[:80]}...")
print("  PASSED\n")

# 6. CLIP model structure (don't download weights, just check code)
print("=" * 60)
print("测试 model 结构(不加载权重)...")
print("  (跳过 CLIP 权重下载，代码语法正确即可)")
from src.models.clip_baseline import (
    ConcatFusion, CrossAttentionFusion, GatedFusion, MultiLabelHead
)
# Test fusion modules in isolation
concat_fusion = ConcatFusion(512, 512, 256)
dummy_img = torch.randn(2, 512)
dummy_txt = torch.randn(2, 512)
out = concat_fusion(dummy_img, dummy_txt)
print(f"  ConcatFusion: in=(2,512)+(2,512) → out={list(out.shape)}")

gated_fusion = GatedFusion(512, 512, 256)
out = gated_fusion(dummy_img, dummy_txt)
print(f"  GatedFusion: in=(2,512)+(2,512) → out={list(out.shape)}")

head = MultiLabelHead(256, num_labels=8, hidden_dims=[128])
out = head(torch.randn(2, 256))
print(f"  MultiLabelHead: in=(2,256) → out={list(out.shape)}")
print("  PASSED\n")

print("=" * 60)
print("ALL MODULE TESTS PASSED!")
print("=" * 60)
