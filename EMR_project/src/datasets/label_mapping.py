"""
统一标签体系模块

基于 Plutchik 情感轮模型定义 8 类基础情感 + 复合情感映射规则。
将所有数据集的原始标签映射到统一的 8 维向量空间。

8 类基础情感（Plutchik's Wheel of Emotions）:
    0: joy       喜悦
    1: sadness   悲伤
    2: anger     愤怒
    3: fear      恐惧
    4: surprise  惊讶
    5: disgust   厌恶
    6: trust     信任
    7: anticipation  期待

复合情感 = 相邻两种基础情感的混合（按 Plutchik 的 dyad 理论）
"""
import numpy as np
from typing import List, Dict, Optional, Union, Tuple


# ============================================================
# 8 类基础情感定义
# ============================================================
BASIC_EMOTIONS: List[str] = [
    "joy",          # 0
    "sadness",      # 1
    "anger",        # 2
    "fear",         # 3
    "surprise",     # 4
    "disgust",      # 5
    "trust",        # 6
    "anticipation", # 7
]

EMOTION_CN: Dict[str, str] = {
    "joy": "喜悦",
    "sadness": "悲伤",
    "anger": "愤怒",
    "fear": "恐惧",
    "surprise": "惊讶",
    "disgust": "厌恶",
    "trust": "信任",
    "anticipation": "期待",
}

NUM_EMOTIONS: int = len(BASIC_EMOTIONS)
EMOTION_TO_IDX: Dict[str, int] = {e: i for i, e in enumerate(BASIC_EMOTIONS)}
IDX_TO_EMOTION: Dict[int, str] = {i: e for i, e in enumerate(BASIC_EMOTIONS)}


# ============================================================
# 复合情感映射（Plutchik dyads）
# 复合情感 = 两个相邻基础情感的加权组合（权重各 0.5）
# ============================================================
COMPOUND_EMOTION_MAP: Dict[str, List[Tuple[str, float]]] = {
    # 一级 dyad（相邻两个基础情感的混合）
    "love":       [("joy", 0.6), ("trust", 0.4)],
    "submission": [("trust", 0.6), ("fear", 0.4)],
    "awe":        [("fear", 0.5), ("surprise", 0.5)],
    "disapproval":[("surprise", 0.5), ("sadness", 0.5)],
    "remorse":    [("sadness", 0.6), ("disgust", 0.4)],
    "contempt":   [("disgust", 0.5), ("anger", 0.5)],
    "aggression": [("anger", 0.5), ("anticipation", 0.5)],
    "optimism":   [("anticipation", 0.6), ("joy", 0.4)],
    # 二级 dyad（间隔一个情感）
    "guilt":      [("joy", 0.4), ("fear", 0.6)],
    "curiosity":  [("trust", 0.5), ("surprise", 0.5)],
    "despair":    [("sadness", 0.5), ("anger", 0.5)],
    "envy":       [("sadness", 0.4), ("anger", 0.6)],
    "cynicism":   [("disgust", 0.4), ("anticipation", 0.6)],
    "pride":      [("anger", 0.3), ("joy", 0.7)],
    "hope":       [("trust", 0.5), ("anticipation", 0.5)],
    # 三级 dyad（间隔两个或以上）
    "delight":    [("joy", 0.6), ("surprise", 0.4)],
    "sentimentality": [("trust", 0.4), ("sadness", 0.6)],
    "shame":      [("fear", 0.5), ("disgust", 0.5)],
    "outrage":    [("surprise", 0.4), ("anger", 0.6)],
    "morbidness": [("disgust", 0.4), ("anticipation", 0.6)],
    "anxiety":    [("fear", 0.6), ("anticipation", 0.4)],
}

# 别名映射（同义词→标准名称）
EMOTION_ALIASES: Dict[str, str] = {
    "happy": "joy", "happiness": "joy", "joyful": "joy", "cheerful": "joy",
    "sad": "sadness", "sorrow": "sadness", "grief": "sadness", "depressed": "sadness",
    "angry": "anger", "rage": "anger", "fury": "anger", "irritated": "anger",
    "scared": "fear", "frightened": "fear", "terrified": "fear", "anxious": "fear",
    "surprised": "surprise", "amazed": "surprise", "astonished": "surprise", "shocked": "surprise",
    "disgusted": "disgust", "revulsion": "disgust", "aversion": "disgust",
    "trusting": "trust", "acceptance": "trust", "admiration": "trust",
    "anticipating": "anticipation", "expectant": "anticipation", "vigilance": "anticipation",
    "neutral": None,  # 中性情绪→无标签
}


def normalize_emotion_name(name: str) -> Optional[str]:
    """
    将任意情感名称映射到 8 类基础情感之一。
    返回 None 表示该情感不在 8 类体系中（如 neutral）。

    Args:
        name: 原始情感名称（大小写无关）

    Returns:
        标准化的基础情感名，或 None
    """
    name_lower = name.lower().strip()

    # 直接匹配基础情感
    if name_lower in EMOTION_TO_IDX:
        return name_lower

    # 别名匹配
    if name_lower in EMOTION_ALIASES:
        return EMOTION_ALIASES[name_lower]

    # 复合情感匹配（返回 None = 不解构，由调用者处理）
    if name_lower in COMPOUND_EMOTION_MAP:
        return None

    return None


def emotion_to_onehot(emotion_name: str) -> np.ndarray:
    """
    将单个基础情感名称转为 one-hot 向量。

    Args:
        emotion_name: 标准基础情感名

    Returns:
        (8,) numpy array，对应位置为 1
    """
    vec = np.zeros(NUM_EMOTIONS, dtype=np.float32)
    idx = EMOTION_TO_IDX.get(emotion_name.lower().strip())
    if idx is not None:
        vec[idx] = 1.0
    return vec


def emotions_to_multihot(emotion_names: List[str]) -> np.ndarray:
    """
    将多个情感名称转为 multi-hot 向量。

    Args:
        emotion_names: 情感名称列表（可以是基础或复合情感）

    Returns:
        (8,) numpy array
    """
    vec = np.zeros(NUM_EMOTIONS, dtype=np.float32)
    for name in emotion_names:
        name_lower = name.lower().strip()
        # 基础情感直接设置
        if name_lower in EMOTION_TO_IDX:
            vec[EMOTION_TO_IDX[name_lower]] = 1.0
            continue
        # 别名
        if name_lower in EMOTION_ALIASES:
            canon = EMOTION_ALIASES[name_lower]
            if canon is not None:
                vec[EMOTION_TO_IDX[canon]] = 1.0
            continue
        # 复合情感→分解为基础情感
        if name_lower in COMPOUND_EMOTION_MAP:
            for base_emo, weight in COMPOUND_EMOTION_MAP[name_lower]:
                vec[EMOTION_TO_IDX[base_emo]] = max(vec[EMOTION_TO_IDX[base_emo]], weight)
    return vec


def compound_to_basic_distribution(compound_name: str) -> np.ndarray:
    """
    将复合情感分解为 8 维分布向量（各项之和 = 1）。

    Args:
        compound_name: 复合情感名

    Returns:
        (8,) 归一化分布向量
    """
    vec = np.zeros(NUM_EMOTIONS, dtype=np.float32)
    name_lower = compound_name.lower().strip()
    if name_lower in COMPOUND_EMOTION_MAP:
        for base_emo, weight in COMPOUND_EMOTION_MAP[name_lower]:
            vec[EMOTION_TO_IDX[base_emo]] = weight
        # 归一化
        total = vec.sum()
        if total > 0:
            vec /= total
    return vec


def label_distribution_to_multihot(
    distribution: Union[List[float], np.ndarray],
    threshold: float = 0.2
) -> np.ndarray:
    """
    将 LDL（Label Distribution Learning）的分布标签转为 multi-hot。
    适用于 Emotion6 这类提供情感强度分布的数据集。

    Args:
        distribution: 长度为 num_classes 的分布数组
        threshold: 超过此值视为该情感存在

    Returns:
        (8,) multi-hot 向量
    """
    dist = np.asarray(distribution, dtype=np.float32)
    return (dist > threshold).astype(np.float32)


# ============================================================
# 各数据集原始标签 → 8 类基础情感的映射表
# ============================================================

# Emotion6: 6 emotions + neutral
# Ekman's 6 basic emotions
EMOTION6_MAP: Dict[str, str] = {
    "anger":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "joy":      "joy",
    "sadness":  "sadness",
    "surprise": "surprise",
    "neutral":  None,   # 丢弃或作为全零标签
}

# GAPED: Geneva Affective Picture Database
# 原始标签为 valence, arousal, dominance 三维 + 具体情感类别
GAPED_MAP: Dict[str, str] = {
    "anger":     "anger",
    "disgust":   "disgust",
    "fear":      "fear",
    "sadness":   "sadness",
    "joy":       "joy",
    "surprise":  "surprise",
    "neutral":   None,
}

# ArtPhoto: 8 emotion categories from WikiArt
ARTPHOTO_MAP: Dict[str, str] = {
    "amusement":  "joy",
    "anger":      "anger",
    "awe":        None,        # 复合情感 → 分解
    "contentment":"joy",       # 映射为 joy
    "disgust":    "disgust",
    "excitement": "anticipation",
    "fear":       "fear",
    "sadness":    "sadness",
}

# FI (Flickr Images):  8 emotion categories
FI_MAP: Dict[str, str] = {
    "amusement":  "joy",
    "anger":      "anger",
    "awe":        None,
    "contentment":"joy",
    "disgust":    "disgust",
    "excitement": "anticipation",
    "fear":       "fear",
    "sadness":    "sadness",
}


def map_dataset_label(dataset_name: str, original_label: str) -> np.ndarray:
    """
    将任意数据集的原始标签映射为 8 维 multi-hot 向量。

    Args:
        dataset_name: "emotion6" / "gaped" / "artphoto" / "fi"
        original_label: 原始标签字符串

    Returns:
        (8,) multi-hot numpy array
    """
    dataset_name = dataset_name.lower()
    label_lower = original_label.lower().strip()

    mapping = {
        "emotion6": EMOTION6_MAP,
        "gaped": GAPED_MAP,
        "artphoto": ARTPHOTO_MAP,
        "fi": FI_MAP,
    }

    mp = mapping.get(dataset_name, {})
    if label_lower in mp:
        canon = mp[label_lower]
        if canon is None:
            return np.zeros(NUM_EMOTIONS, dtype=np.float32)
        if canon in EMOTION_TO_IDX:
            vec = np.zeros(NUM_EMOTIONS, dtype=np.float32)
            vec[EMOTION_TO_IDX[canon]] = 1.0
            return vec
        # 是复合情感
        return compound_to_basic_distribution(canon)

    # 尝试通用映射
    return emotions_to_multihot([label_lower])


def get_emotion_prompt(emotion_id: int, lang: str = "en") -> str:
    """
    获取某情感对应的文本提示。

    Args:
        emotion_id: 0-7
        lang: "en" / "cn"

    Returns:
        文本提示字符串
    """
    prompts_en = [
        "joy",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "disgust",
        "trust",
        "anticipation",
    ]
    if lang == "cn":
        prompts_cn = ["喜悦", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "信任", "期待"]
        return prompts_cn[emotion_id]
    return prompts_en[emotion_id]


def get_text_prompts_for_emotions(emotion_ids: List[int]) -> str:
    """
    为存在的多个情感生成拼接的文本提示。

    Args:
        emotion_ids: 情感 ID 列表

    Returns:
        "joy and sadness and anger"
    """
    names = [BASIC_EMOTIONS[i] for i in sorted(emotion_ids) if 0 <= i < NUM_EMOTIONS]
    if not names:
        return "neutral emotion"
    return " and ".join(names)


def idx_to_multihot(indices: List[int]) -> np.ndarray:
    """将情感索引列表转为 multi-hot 向量"""
    vec = np.zeros(NUM_EMOTIONS, dtype=np.float32)
    for i in indices:
        if 0 <= i < NUM_EMOTIONS:
            vec[i] = 1.0
    return vec


def multihot_to_names(multihot: Union[np.ndarray, List[float]], threshold: float = 0.5) -> List[str]:
    """将 multi-hot 向量转为情感名称列表"""
    vec = np.asarray(multihot, dtype=np.float32)
    return [BASIC_EMOTIONS[i] for i in range(NUM_EMOTIONS) if vec[i] >= threshold]
