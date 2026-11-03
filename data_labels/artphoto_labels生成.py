# -*- coding: utf-8 -*-
"""
ArtPhoto 数据集「图片文件名 — 类别标签」清单生成脚本。

ArtPhoto 的类别由文件名前缀给出（如 amusement_0489.jpg），本脚本据此生成
「文件名 — 标签」清单，供训练代码直接读取。

相对原始版本做了三处适配：
  1. 数据目录改为脚本所在目录下的 testImages_artphoto（不写死桌面路径）
  2. 同时输出中文标签与英文标签两份，便于接训练代码
  3. 输出只写文件名而非绝对路径 —— 绝对路径含本机用户名，不适合进公开仓库，
     且 fixed_splits.json 本身也是按文件名索引的，用文件名可直接对应

用法：
    python artphoto_labels生成.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

# 文件名前缀 -> (中文类别标签, 英文标签)
CATEGORY_MAP = {
    "amusement":   ("愉悦", "amusement"),
    "awe":         ("敬畏", "awe"),
    "contentment": ("满足", "contentment"),
    "excitement":  ("兴奋", "excitement"),
    "anger":       ("愤怒", "anger"),
    "disgust":     ("厌恶", "disgust"),
    "fear":        ("恐惧", "fear"),
    "sad":         ("悲伤", "sadness"),   # 注意：悲伤类的前缀是 "sad"，不是 sadness
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

BASE = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE, "testImages_artphoto")
out_zh = os.path.join(BASE, "artphoto_labels.txt")
out_en = os.path.join(BASE, "artphoto_labels_en.txt")

if not os.path.isdir(data_dir):
    raise SystemExit(f"数据目录不存在：{data_dir}")

lines_zh, lines_en, unknown = [], [], []

for name in sorted(os.listdir(data_dir)):
    ext = os.path.splitext(name)[1].lower()
    if ext not in IMAGE_EXTS:
        continue                      # 跳过 .DS_Store / ReadMe.rtf 等
    prefix = name.split("_")[0]       # 第一个下划线之前的部分即类别前缀
    pair = CATEGORY_MAP.get(prefix)
    if pair is None:
        unknown.append(name)
        continue
    zh, en = pair
    # 只写文件名，不写绝对路径：与 fixed_splits.json 的索引方式一致，且不含本机信息
    lines_zh.append(f"{name}\t{zh}")
    lines_en.append(f"{name}\t{en}")

with open(out_zh, "w", encoding="utf-8") as f:
    f.write("\n".join(lines_zh) + "\n")
with open(out_en, "w", encoding="utf-8") as f:
    f.write("\n".join(lines_en) + "\n")

# 统计各类数量，确认八分类是否均衡
counts = {}
for line in lines_en:
    counts[line.rsplit("\t", 1)[1]] = counts.get(line.rsplit("\t", 1)[1], 0) + 1

print(f"共写入 {len(lines_zh)} 条记录")
print(f"  中文标签 -> {out_zh}")
print(f"  英文标签 -> {out_en}")
print("\n各类别数量：")
for k in sorted(counts):
    print(f"  {k:<12} {counts[k]:>4}")

if unknown:
    print("\n以下文件无法识别类别：")
    for u in unknown:
        print("  ", u)
