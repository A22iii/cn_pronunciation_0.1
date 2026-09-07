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
import os
import pickle
from pathlib import Path
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
    "资": {"pinyin": "zi1",  "initial": "z",  "final": "i", "tone": 1},
    "吃": {"pinyin": "chi1", "initial": "ch", "final": "i", "tone": 1},
    "疵": {"pinyin": "ci1",  "initial": "c",  "final": "i", "tone": 1},
    "机": {"pinyin": "ji1",  "initial": "j",  "final": "i", "tone": 1},
    "七": {"pinyin": "qi1",  "initial": "q",  "final": "i", "tone": 1},
    "绿": {"pinyin": "lv4",  "initial": "l",  "final": "v", "tone": 4},
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
    "饱": {"pinyin": "bao3", "initial": "b",  "final": "ao", "tone": 3},
}


def parse_target_word(target_word):
    """解析目标字，提取拼音和音素信息

    例："知(zhī)" → {"character": "知", "pinyin": "zhi1", ...}
    """
    # 支持格式："知(zhī)" 或 "知"
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
        target_word: 目标字，如 "知(zhī)"

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

    scores = _model_inference(
        target_info,
        f0_stats,
        formants,
        vot_info,
        spectral,
    )

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


_MODEL = None
_MODEL_PATH = None


def _model_inference(target_info, f0_stats, formants, vot_info, spectral):
    """Run the configured model and normalize its prediction.

    ``PRONUNCIATION_MODEL_PATH`` may point to a pickle/joblib model exposing
    either ``predict_pronunciation(target_info, features)`` or ``predict(X)``.
    The latter receives one fixed-size NumPy feature vector.
    """
    features = {
        "f0_stats": f0_stats,
        "formants": formants,
        "vot": vot_info,
        "spectral": spectral,
    }
    model = _load_model()
    if model is None:
        return _acoustic_inference(target_info, f0_stats, formants, vot_info, spectral)

    if hasattr(model, "predict_pronunciation"):
        prediction = model.predict_pronunciation(target_info, features)
    else:
        prediction = model.predict(_feature_vector(features).reshape(1, -1))
    return _normalize_prediction(prediction, target_info)


def _load_model():
    """Load the model once per configured path so requests stay cheap."""
    global _MODEL, _MODEL_PATH

    model_path = os.environ.get("PRONUNCIATION_MODEL_PATH")
    if not model_path:
        return None
    if _MODEL is not None and _MODEL_PATH == model_path:
        return _MODEL

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pronunciation model not found: {path}")

    if path.suffix.lower() in {".joblib", ".jl"}:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("joblib is required for a .joblib pronunciation model") from exc
        _MODEL = joblib.load(path)
    else:
        with path.open("rb") as handle:
            _MODEL = pickle.load(handle)
    _MODEL_PATH = model_path
    return _MODEL


def _feature_vector(features):
    """Build the stable numeric input expected by sklearn-style models."""
    f0 = features.get("f0_stats") or {}
    formants = features.get("formants") or {}
    vot = features.get("vot") or {}
    spectral = features.get("spectral") or {}
    values = [
        f0.get("f0_mean"), f0.get("f0_range"), f0.get("f0_min"),
        f0.get("f0_max"), f0.get("f0_slope"), f0.get("f0_std"),
        formants.get("f1"), formants.get("f2"), formants.get("f3"),
        vot.get("vot_ms"), float(bool(vot.get("is_aspirated"))),
        spectral.get("centroid_mean"), spectral.get("centroid_std"),
        spectral.get("bandwidth_mean"),
    ]
    return np.nan_to_num(
        np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )


def _normalize_prediction(prediction, target_info):
    """Convert common model outputs to the evaluator's public schema."""
    if isinstance(prediction, np.ndarray):
        if prediction.size == 0:
            raise ValueError("Pronunciation model returned an empty prediction")
        prediction = prediction[0] if prediction.ndim > 1 else prediction

    if isinstance(prediction, dict):
        overall = prediction.get("overall")
        phoneme = prediction.get("phoneme", overall)
        tone = prediction.get("tone", overall)
        details = prediction.get("details", {})
    else:
        values = np.asarray(prediction, dtype=np.float32).reshape(-1)
        if values.size == 1:
            overall = phoneme = tone = values[0]
        elif values.size >= 3:
            overall, phoneme, tone = values[:3]
        else:
            raise ValueError("Pronunciation model must return one or three scores")
        details = {}

    if overall is None or phoneme is None or tone is None:
        raise ValueError("Pronunciation model output is missing score fields")
    phoneme = _score(phoneme)
    tone = _score(tone)
    overall = _score(overall)
    if not details:
        details = {}
        initial = target_info.get("initial", "")
        final = target_info.get("final", "")
        if initial:
            details[initial] = {"score": phoneme, "error_type": _initial_error(initial, phoneme)}
        if final:
            details[final] = {"score": phoneme, "error_type": _final_error(final, phoneme)}
    return {
        "overall": overall,
        "phoneme": phoneme,
        "tone": tone,
        "details": details,
    }


def _score(value):
    return int(np.clip(round(float(value)), 0, 100))


def _initial_error(initial, score):
    if score >= 80:
        return None
    if initial in ("zh", "ch", "sh"):
        return "retroflex_fronting"
    if initial in ("p", "t", "k"):
        return "aspiration_insufficient"
    return None


def _final_error(final, score):
    if score >= 80:
        return None
    if final == "v":
        return "rounding_missing"
    if final in ("an", "en", "ang", "eng", "in", "ing"):
        return "nasal_final_dropped"
    return None


def _acoustic_inference(target_info, f0_stats, formants, vot_info, spectral):
    """Infer pronunciation scores from the extracted acoustic feature vector.

    This compact prototype-based model is deterministic and operates on the
    feature vector already computed by ``evaluate_pronunciation``.
    """
    feature_groups = (f0_stats, formants, vot_info, spectral)
    if not any(isinstance(group, dict) and group for group in feature_groups):
        return {"overall": 45, "phoneme": 40, "tone": 45, "details": {}}

    initial = target_info.get("initial", "")
    final = target_info.get("final", "")
    details = {}

    if initial:
        score = _infer_initial_score(initial, formants, vot_info, spectral)
        details[initial] = {
            "score": score,
            "error_type": _infer_initial_error(initial, score),
        }

    if final:
        score = _infer_final_score(final, formants, spectral)
        details[final] = {
            "score": score,
            "error_type": _infer_final_error(final, score),
        }

    phoneme_scores = [item["score"] for item in details.values()]
    phoneme = _clamp_score(np.mean(phoneme_scores)) if phoneme_scores else 60
    tone = _infer_tone_score(target_info.get("tone", 0), f0_stats)

    return {
        "overall": _clamp_score((phoneme + tone) * 0.5),
        "phoneme": phoneme,
        "tone": tone,
        "details": details,
    }


def _infer_initial_score(initial, formants, vot_info, spectral):
    """Estimate initial accuracy from place, manner, and aspiration cues."""
    measurements = []
    centroid = _number(spectral, "centroid_mean")
    f2 = _number(formants, "f2")
    f3 = _number(formants, "f3")
    vot = _number(vot_info, "vot_ms")

    if initial in ("zh", "ch", "sh"):
        if centroid is not None:
            measurements.append(_distance_score(centroid, 2200, 1100))
        if f3 is not None:
            measurements.append(_distance_score(f3, 2800, 1100))
    elif initial in ("z", "c", "s"):
        if centroid is not None:
            measurements.append(_distance_score(centroid, 3000, 1400))
        if f2 is not None:
            measurements.append(_distance_score(f2, 1900, 1200))
    elif initial in ("j", "q", "x"):
        if centroid is not None:
            measurements.append(_distance_score(centroid, 2500, 1300))
        if f2 is not None:
            measurements.append(_distance_score(f2, 2200, 900))
    elif initial in ("p", "t", "k") and vot is not None:
        measurements.append(_vot_score(vot, aspirated=True))
    elif initial in ("b", "d", "g") and vot is not None:
        measurements.append(_vot_score(vot, aspirated=False))
    elif initial == "m" and f2 is not None:
        measurements.append(_distance_score(f2, 1400, 1200))
    elif initial == "l" and f2 is not None:
        measurements.append(_distance_score(f2, 1700, 1200))

    return _measurement_average(measurements, default=65)


def _infer_final_score(final, formants, spectral):
    """Estimate final accuracy from vowel formants and nasal spectral energy."""
    measurements = []
    f1 = _number(formants, "f1")
    f2 = _number(formants, "f2")
    centroid = _number(spectral, "centroid_mean")
    bandwidth = _number(spectral, "bandwidth_mean")

    # Broad prototypes reduce sensitivity to speaker sex and recording level.
    profile = {
        "i": ((f2, 2400, 850),),
        "in": ((f2, 2250, 900),),
        "ing": ((f2, 2250, 900),),
        "u": ((f2, 950, 700),),
        "v": ((f2, 1450, 800),),
        "a": ((f1, 700, 600),),
        "an": ((f1, 650, 600), (f2, 1700, 1200)),
        "ang": ((f1, 700, 650),),
        "ao": ((f1, 650, 650),),
        "uo": ((f1, 500, 550), (f2, 1100, 900)),
        "o": ((f1, 500, 550), (f2, 1000, 850)),
    }
    for value, center, tolerance in profile.get(final, ()):
        if value is not None:
            measurements.append(_distance_score(value, center, tolerance))

    if final in ("an", "en", "ang", "eng", "in", "ing"):
        if bandwidth is not None:
            measurements.append(_distance_score(bandwidth, 1700, 1100))
        elif centroid is not None:
            measurements.append(_distance_score(centroid, 1900, 1300))

    return _measurement_average(measurements, default=65)


def _infer_tone_score(tone, f0_stats):
    """Score the target tone against its expected F0 direction and movement."""
    if not isinstance(f0_stats, dict):
        return 50

    mean = _number(f0_stats, "f0_mean")
    slope = _number(f0_stats, "f0_slope")
    f0_range = _number(f0_stats, "f0_range")
    f0_std = _number(f0_stats, "f0_std")
    if mean is None or mean <= 0:
        return 50

    slope_ratio = slope / mean if slope is not None else None
    range_ratio = f0_range / mean if f0_range is not None else None
    std_ratio = f0_std / mean if f0_std is not None else None
    measurements = []
    profiles = {
        1: ((slope_ratio, 0.0, 0.004), (range_ratio, 0.08, 0.16)),
        2: ((slope_ratio, 0.004, 0.006), (range_ratio, 0.18, 0.22)),
        3: ((slope_ratio, 0.0, 0.005), (range_ratio, 0.25, 0.22),
            (std_ratio, 0.10, 0.12)),
        4: ((slope_ratio, -0.004, 0.006), (range_ratio, 0.18, 0.22)),
    }
    for value, center, tolerance in profiles.get(tone, ()):
        if value is not None:
            measurements.append(_distance_score(value, center, tolerance))

    return _measurement_average(measurements, default=55)


def _infer_initial_error(initial, score):
    if score >= 80:
        return None
    if initial in ("zh", "ch", "sh"):
        return "retroflex_fronting"
    if initial in ("p", "t", "k"):
        return "aspiration_insufficient"
    return None


def _infer_final_error(final, score):
    if score >= 80:
        return None
    if final == "v":
        return "rounding_missing"
    if final in ("an", "en", "ang", "eng", "in", "ing"):
        return "nasal_final_dropped"
    return None


def _number(values, key):
    if not isinstance(values, dict):
        return None
    try:
        value = float(values[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _distance_score(value, center, tolerance):
    if value is None:
        return None
    normalized_distance = (float(value) - center) / max(float(tolerance), 1e-9)
    return _clamp_score(100.0 * np.exp(-0.5 * normalized_distance ** 2))


def _vot_score(vot, aspirated):
    if aspirated:
        return _clamp_score(50.0 + (float(vot) - 5.0) * 2.0)
    return _clamp_score(100.0 - max(0.0, float(vot) - 12.0) * 2.5)


def _measurement_average(measurements, default=50):
    values = [float(value) for value in measurements if value is not None]
    return _clamp_score(np.mean(values)) if values else default


def _clamp_score(value):
    return int(np.clip(np.rint(float(value)), 0, 100))


def _simulate_scores(target_info, f0_stats, formants, vot_info, spectral):
    """Phase 0 模拟评分逻辑

    根据提取到的声学特征做启发式打分。
    这是占位实现——后续替换为模型的 predict() 调用。

    设计：用特征中的信息量来决定分数有多"随机"——
    如果能提取到声学特征，说明音频有效，给一个合理范围的分数；
    如果特征缺失（如 F0 提取失败），则降低置信度。
    """
    import random
    random.seed(42)  # 可重现的随机，演示时保持一致性

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

    # ── 针对不同目标字类型，模拟典型的偏误模式 ──
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

    # 声调评分：模拟二声/三声混淆
    tone_score = 90
    if tone == 2 or tone == 3:
        tone_score = 68 + random.randint(-5, 10)
    elif tone == 1:
        tone_score = 82 + random.randint(-5, 10)

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
