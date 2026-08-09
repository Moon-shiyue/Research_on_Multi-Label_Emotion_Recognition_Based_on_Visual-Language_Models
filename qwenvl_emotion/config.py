"""
Qwen2.5-VL 方案 — 全局配置

与 CLIP 方案共享情感标签体系，使用完全独立的一组超参数
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ============================================================
# 情感标签体系（与 CLIP 方案一致，确保对比公平性）
# ============================================================

UNIFIED_EMOTIONS = [
    "joy", "sadness", "anger", "fear", "surprise", "disgust",
    "love", "peace", "amusement", "awe", "contentment", "excitement"
]

# 中文映射（Qwen 对中文情感理解更强）
EMOTION_CN = {
    "joy": "快乐/喜悦",
    "sadness": "悲伤/难过",
    "anger": "愤怒/生气",
    "fear": "恐惧/害怕",
    "surprise": "惊讶/吃惊",
    "disgust": "厌恶/反感",
    "love": "爱/喜爱",
    "peace": "平静/安宁",
    "amusement": "有趣/好玩",
    "awe": "敬畏/震撼",
    "contentment": "满足/满意",
    "excitement": "兴奋/激动",
}


# ============================================================
# Prompt 模板
# ============================================================

# 策略1：直接分类 — 让模型直接输出多标记概率
PROMPT_DIRECT = """请分析这张图片表达的情感。从以下12种情感中选择所有你认为存在的（可多选）：
joy(快乐), sadness(悲伤), anger(愤怒), fear(恐惧), surprise(惊讶), disgust(厌恶),
love(爱), peace(平静), amusement(有趣), awe(敬畏), contentment(满足), excitement(兴奋)

请以JSON格式输出，每个情感给出0到1之间的置信度：
{"joy": 0.0, "sadness": 0.0, ...}"""

# 策略2：思维链 — 先描述再打分（提高推理质量）
PROMPT_COT = """你是一个情感分析专家。请按以下步骤分析这张图片：

步骤1：用一句话描述图片内容
步骤2：分析图片中的情感氛围
步骤3：对以下12种情感逐项评分（0.0=完全不存在, 1.0=强烈存在）

情感列表：joy, sadness, anger, fear, surprise, disgust, love, peace, amusement, awe, contentment, excitement

请严格按以下JSON格式输出，不要添加其他文字：
{
  "description": "一句话描述",
  "analysis": "一句话情感分析",
  "scores": {
    "joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0,
    "surprise": 0.0, "disgust": 0.0, "love": 0.0, "peace": 0.0,
    "amusement": 0.0, "awe": 0.0, "contentment": 0.0, "excitement": 0.0
  }
}"""

# 策略3：对比式 — 两两比较（最精细但较慢）
PROMPT_CONTRASTIVE = """这张图片更倾向于以下哪种情感？
请对每对情感给出倾向性评分（0=完全偏向左边, 1=完全偏向右边, 0.5=两者都存在或都不存在）：

1. joy vs sadness:
2. anger vs fear:
3. surprise vs disgust:
4. love vs peace:
5. amusement vs awe:
6. contentment vs excitement:

请输出JSON格式的12个独立置信度。"""


# ============================================================
# 模型配置
# ============================================================

@dataclass
class QwenVLConfig:
    """Qwen2.5-VL 微调配置"""

    # ---- 模型 ----
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    # 备选模型:
    #   "Qwen/Qwen2.5-VL-7B-Instruct"     — 更强但需更大显存
    #   "Qwen/Qwen2-VL-2B-Instruct"       — 更轻量
    #   "OpenGVLab/InternVL2-4B"           — 另一选择
    #   "llava-hf/llava-1.5-7b-hf"        — LLaVA

    # ---- 推理精度 ----
    torch_dtype: str = "bfloat16"       # bfloat16/float16/float32
    load_in_4bit: bool = False          # 4bit 量化（7B模型建议开启）
    load_in_8bit: bool = False          # 8bit 量化

    # ---- LoRA 配置 ----
    lora_r: int = 16                    # LoRA rank（越大越强但越慢）
    lora_alpha: int = 32                # LoRA scaling（通常 2×r）
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力投影
        "gate_proj", "up_proj", "down_proj",       # FFN 投影
    ])
    # 如果显存紧张，可以减少目标模块：
    # ["q_proj", "v_proj"]  ← 最小配置，约 0.5% 参数

    # ---- Prompt 策略 ----
    prompt_strategy: str = "cot"        # "direct" | "cot" | "contrastive"
    max_new_tokens: int = 256           # 生成最大 token 数

    # ---- 标签 ----
    num_emotions: int = 12
    emotion_labels: List[str] = field(default_factory=lambda: UNIFIED_EMOTIONS)


@dataclass
class QwenVLTrainingConfig:
    """训练配置"""

    # ---- 优化器 ----
    learning_rate: float = 2e-4         # LoRA 建议稍高
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # ---- 训练循环 ----
    num_epochs: int = 5                 # LoRA 通常 3-5 epoch 即可
    batch_size: int = 4                 # VLM 显存占用大，batch 要小
    gradient_accumulation_steps: int = 8  # 等效 batch = 4×8 = 32

    # ---- 学习率 ----
    lr_scheduler: str = "cosine"

    # ---- 早停 ----
    early_stopping_patience: int = 5

    # ---- 数据 ----
    image_size: Tuple[int, int] = (448, 448)  # Qwen-VL 推荐分辨率
    num_workers: int = 2

    # ---- 日志 ----
    log_interval: int = 10
    eval_interval: int = 200
    save_total_limit: int = 2

    # ---- 混合精度 ----
    use_amp: bool = True


# 默认实例
default_model_config = QwenVLConfig()
default_training_config = QwenVLTrainingConfig()
