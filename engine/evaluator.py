"""评测引擎 — 普通话发音多层级打分

Phase 0 策略：
  使用声学特征 + 启发式规则生成模拟评分。

评分层级：
  1. 整体分（0-100）：综合发音质量
  2. 音素级分（0-100）：每个声母/韵母的独立评分
  3. 声调分（0-100）：每个音节的声调准确度

接口设计原则：
  evaluate_pronunciation(y, sr, target_word) → dict
  这个函数签名无论底层用什么引擎都保持不变，
  方便后续从模拟评分切换到真实模型。
"""

import numpy as np
import re
from engine.acoustic_features import (
    extract_f0_statistics,
    extract_formants,
    estimate_vot,
    extract_spectral_features,
)


# ── 汉字→拼音→音素映射表（Phase 0 覆盖的高频练习字） ──
# 结构：{(汉字, 拼音): {"initial": 声母, "final": 韵母, "tone": 声调}}
# 声调用数字 1-4 表示（赵元任五度制）
WORD_PHONEME_MAP = {
    "知": {"pinyin": "zhi1", "initial": "zh", "final": "i", "tone": 1},
    "是": {"pinyin": "shi4", "initial": "sh", "final": "i", "tone": 4},
    "资": {"pinyin": "zi1",  "initial": "z",  "final": "i", "tone": 1},
    "吃": {"pinyin": "chi1", "initial": "ch", "final": "i", "tone": 1},
    "疵": {"pinyin": "ci1",  "initial": "c",  "final": "i", "tone": 1},
    "四": {"pinyin": "si4",  "initial": "s",  "final": "i", "tone": 4},
    "机": {"pinyin": "ji1",  "initial": "j",  "final": "i", "tone": 1},
    "七": {"pinyin": "qi1",  "initial": "q",  "final": "i", "tone": 1},
    "西": {"pinyin": "xi1",  "initial": "x",  "final": "i", "tone": 1},
    "绿": {"pinyin": "lv4",  "initial": "l",  "final": "v", "tone": 4},
    "女": {"pinyin": "nv3",  "initial": "n",  "final": "v", "tone": 3},
    "鱼": {"pinyin": "yv2",  "initial": "y",  "final": "v", "tone": 2},
    "路": {"pinyin": "lu4",  "initial": "l",  "final": "u", "tone": 4},
    "麻": {"pinyin": "ma2",  "initial": "m",  "final": "a", "tone": 2},
    "马": {"pinyin": "ma3",  "initial": "m",  "final": "a", "tone": 3},
    "国": {"pinyin": "guo2", "initial": "g",  "final": "uo", "tone": 2},
    "果": {"pinyin": "guo3", "initial": "g",  "final": "uo", "tone": 3},
    "慢": {"pinyin": "man4", "initial": "m",  "final": "an", "tone": 4},
    "忙": {"pinyin": "mang2","initial": "m",  "final": "ang","tone": 2},
    "金": {"pinyin": "jin1", "initial": "j",  "final": "in", "tone": 1},
    "京": {"pinyin": "jing1","initial": "j",  "final": "ing","tone": 1},
    "跑": {"pinyin": "pao3", "initial": "p",  "final": "ao", "tone": 3},
    "他": {"pinyin": "ta1",  "initial": "t",  "final": "a", "tone": 1},
    "看": {"pinyin": "kan4", "initial": "k",  "final": "an", "tone": 4},
    "饱": {"pinyin": "bao3", "initial": "b",  "final": "ao", "tone": 3},
}


def parse_target_word(target_word):
    """解析目标字，提取拼音和音素信息

    例："知(zhi1)" → {"character": "知", "pinyin": "zhi1", ...}
    """
    # 仅支持格式："知(zhi1)" 或 "知"
    match = re.match(r"(.)(?:\(([a-zü]+)([1-4])\))?", target_word)
    if not match:
        return None

    character = match.group(1)
    if match.group(2):
        pinyin_base = match.group(2)
        tone = int(match.group(3))
        pinyin = pinyin_base + str(tone)
    else:
        pinyin = ""
        tone = 0

    # 查映射表获取更详细的音素信息
    phoneme_info = WORD_PHONEME_MAP.get(character, {})
    if phoneme_info:
        return {
            "character": character,
            "pinyin": phoneme_info.get("pinyin", pinyin),
            "initial": phoneme_info.get("initial", ""),
            "final": phoneme_info.get("final", ""),
            "tone": phoneme_info.get("tone", tone),
        }

    return {"character": character, "pinyin": pinyin, "tone": tone}


def evaluate_pronunciation(y, sr, target_word):
    """核心评测接口

    Args:
        y: 音频采样数组 (numpy array)
        sr: 采样率 (int)
        target_word: 目标字，如 "知(zhi1)"

    Returns:
        {
            "overall": int,            # 整体分 0-100
            "phoneme": int,            # 音素得分（声母+韵母）
            "tone": int,               # 声调得分
            "details": {               # 每个音素的详细评分
                "zh": {"score": int, "error_type": "retroflex_fronting"},
                "i": {"score": int, "error_type": None},
            },
            "features": {...}          # 提取的声学特征（用于诊断）
        }
    """
    # 1. 解析目标字
    target_info = parse_target_word(target_word)
    if target_info is None:
        return _empty_result("无法识别目标字")

    # 2. 提取声学特征
    f0_stats = extract_f0_statistics(y, sr)
    formants = extract_formants(y, sr)
    vot_info = estimate_vot(y, sr)
    spectral = extract_spectral_features(y, sr)

    # 3. 评分逻辑
    #    TODO: 替换为真实模型推理
    scores = _simulate_scores(target_info, f0_stats, formants, vot_info, spectral)

    return {
        "overall": scores["overall"],
        "phoneme": scores["phoneme"],
        "tone": scores["tone"],
        "details": scores["details"],
        "features": {
            "f0_stats": f0_stats,
            "formants": formants,
            "vot": vot_info,
            "spectral": spectral,
        },
        "target_info": target_info,
    }


def _simulate_scores(target_info, f0_stats, formants, vot_info, spectral):
    """Phase 0 模拟评分逻辑

    根据提取到的声学特征做启发式打分。
    这是占位实现——后续替换为模型的 predict() 调用。

    设计：用特征中的信息量来决定分数有多"随机"——
    如果能提取到声学特征，说明音频有效，给一个合理范围的分数；
    如果特征缺失（如 F0 提取失败），则降低置信度。
    """
    import random
    random.seed(1)  # 可重现的随机，演示时保持一致性

    # 基础分：有有效声学特征时偏高，无特征时偏低
    has_valid_audio = (
        f0_stats is not None and formants is not None
    )

    if not has_valid_audio:
        return {
            "overall": 45,
            "phoneme": 40,
            "tone": 45,
            "details": {},
        }

    # 针对不同目标字类型，模拟典型的偏误模式
    initial = target_info.get("initial", "")
    tone = target_info.get("tone", 0)

    details = {}

    # 声母评分：模拟翘舌音偏误
    if initial in ("zh", "ch", "sh"):
        # 英语母语者常把翘舌发成平舌 → 得分偏低
        details[initial] = {"score": 68 + random.randint(-5, 8), "error_type": "retroflex_fronting"}
    elif initial in ("z", "c", "s"):
        details[initial] = {"score": 85 + random.randint(-5, 10), "error_type": None}
    elif initial in ("j", "q", "x"):
        # 舌面音，英语母语者有时偏后
        details[initial] = {"score": 78 + random.randint(-5, 10), "error_type": None}
    elif initial in ("p", "t", "k"):
        # 送气音，模拟送气不足
        details[initial] = {"score": 70 + random.randint(-8, 10), "error_type": "aspiration_insufficient"}
    elif initial in ("b", "d", "g"):
        details[initial] = {"score": 88 + random.randint(-5, 8), "error_type": None}
    elif initial == "l":
        details[initial] = {"score": 90 + random.randint(-5, 5), "error_type": None}
    elif initial == "m":
        details[initial] = {"score": 85 + random.randint(-5, 10), "error_type": None}

    # 韵母评分：模拟圆唇和鼻韵尾偏误
    final = target_info.get("final", "")
    if final in ("v", "ü"):
        details[final] = {"score": 65 + random.randint(-5, 10), "error_type": "rounding_missing"}
    elif final in ("an", "en", "ang", "eng", "in", "ing"):
        details[final] = {"score": 72 + random.randint(-5, 10), "error_type": "nasal_final_dropped"}
    elif final:
        details[final] = {"score": 85 + random.randint(-5, 10), "error_type": None}

    # 声调评分：使用 F0 趋势估计目标声调的匹配程度
    # 未知声调保留原来的默认分数。
    if tone not in (1, 2, 3, 4):
        tone_score = 50
    elif not isinstance(f0_stats, dict):
        tone_score = 50
    else:
        try:
            f0_mean = float(f0_stats["f0_mean"])
            f0_range = float(f0_stats["f0_range"])
            f0_slope = float(f0_stats["f0_slope"])
            f0_std = float(f0_stats["f0_std"])
        except (KeyError, TypeError, ValueError):
            tone_score = 50
        else:
            values = (f0_mean, f0_range, f0_slope, f0_std)
            if f0_mean <= 0 or not all(np.isfinite(value) for value in values):
                tone_score = 50
            else:
                # 用平均 F0 归一化，减少男女声平均音高不同带来的影响。
                slope_ratio = f0_slope / f0_mean
                range_ratio = f0_range / f0_mean
                std_ratio = f0_std / f0_mean

                # (特征值, 目标中心值, 容忍范围)。
                # 分数越接近目标中心值越高，单项最高为 100 分。
                profiles = {
                    1: (
                        (slope_ratio, 0.0, 0.004),
                        (range_ratio, 0.08, 0.16),
                    ),
                    2: (
                        (slope_ratio, 0.004, 0.006),
                        (range_ratio, 0.18, 0.22),
                    ),
                    # 第三声提醒：这里只能用整体斜率、音域和标准差近似，
                    # 不能确认 F0 曲线是否真正经历了“先降后升”。
                    3: (
                        (slope_ratio, 0.0, 0.005),
                        (range_ratio, 0.25, 0.22),
                        (std_ratio, 0.10, 0.12),
                    ),
                    4: (
                        (slope_ratio, -0.004, 0.006),
                        (range_ratio, 0.18, 0.22),
                    ),
                }

                measurements = []
                for value, center, tolerance in profiles[tone]:
                    normalized_distance = (value - center) / tolerance
                    measurements.append(
                        100.0 * np.exp(-0.5 * normalized_distance ** 2)
                    )

                tone_score = int(
                    np.clip(np.rint(np.mean(measurements)), 0, 100)
                )

    # 音素平均分
    phoneme_scores = [d["score"] for d in details.values() if d["score"] > 0]
    phoneme_avg = int(np.mean(phoneme_scores)) if phoneme_scores else 80

    # 整体分 = 音素分×0.5 + 声调分×0.5
    overall = int(phoneme_avg * 0.5 + tone_score * 0.5)

    return {
        "overall": min(100, max(0, overall)),
        "phoneme": min(100, max(0, phoneme_avg)),
        "tone": min(100, max(0, tone_score)),
        "details": details,
    }


def _empty_result(message):
    """音频为空时返回的结构"""
    return {
        "overall": 0,
        "phoneme": 0,
        "tone": 0,
        "details": {},
        "features": {},
        "target_info": {},
        "error": message,
    }
