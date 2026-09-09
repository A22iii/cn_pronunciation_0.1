"""声学特征提取模块

使用 librosa + parselmouth 提取语音的声学特征，用于：
1. F0 基频曲线（声调可视化和声调诊断）
2. 共振峰 F1/F2/F3（元音质量、舌位推断）
3. 频谱特征（辅音发音部位推断）
4. VOT 送气段检测（送气/不送气区分）

原理说明（给实验语音学专业的学生）：
- F0（基频）：声带振动的基本频率，对应感知上的"音高"。
  普通话四个声调的调值就是 F0 在时间上的变化模式。
- F1/F2/F3（共振峰）：声道共鸣腔的共振频率。
  F1 与舌位高低负相关（舌位越低，F1 越高）；
  F2 与舌位前后正相关（舌位越前，F2 越高）；
  F2 也与圆唇相关（圆唇降低 F2）；
  F3 受舌尖位置影响（翘舌/卷舌降低 F3）。
- 频谱重心（Spectral Centroid）：频谱能量分布的中心频率。
  前移的发音部位（如 z/c/s 比 zh/ch/sh 偏前）通常中心频率更高。
"""

import numpy as np

# librosa 和 parselmouth 是语音处理的核心库
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import parselmouth
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False


def extract_f0_curve(y, sr, target_word="", method="parselmouth"):
    """提取 F0 基频曲线

    获取声调。
    返回用于前端绘图的 (time_array, f0_array) 或 None。
    parselmouth（Praat）方法比 librosa 的 pYIN 更准确，
    尤其对声调语言的基频跟踪效果更好。
    """
    if y is None or len(y) == 0:
        return None

    if method == "parselmouth" and HAS_PARSELMOUTH:
        try:
            sound = parselmouth.Sound(y, sampling_frequency=sr)
            # 将 numpy 数组转换为 Praat 可以处理的对象
            # 音高范围设为 50-400 Hz，覆盖男女声普通话声调范围
            pitch = sound.to_pitch(    # 提取基频
                time_step=0.01,        # 每 10ms 一个点
                pitch_floor=50,        # 最低可检测 F0（男声可能低至 50Hz）
                pitch_ceiling=400,     # 最高可检测 F0（女声可能高至 400Hz）
            )
            times = pitch.xs()
            f0_values = pitch.selected_array['frequency']
            # 返回时间及其对应的基频音高
            # 将 0（无音高）替换为 NaN，避免绘图时出现零值
            f0_values = np.where(f0_values == 0, np.nan, f0_values)

            return {"time": times.tolist(), "f0": f0_values.tolist()}
        except Exception:
            pass  # parselmouth 失败时回退到 librosa

    if HAS_LIBROSA:
        try:
            # pYIN: 概率 YIN 算法，对声调跟踪效果较好
            f0, voiced_flag, voiced_prob = librosa.pyin(
                y.astype(np.float64),
                fmin=50,
                fmax=400,
                sr=sr,
                frame_length=2048,
            )
            times = librosa.times_like(f0, sr=sr)
            f0_values = np.where(voiced_flag, f0, np.nan)
            return {"time": times.tolist(), "f0": f0_values.tolist()}
        except Exception:
            pass

    return None


def extract_f0_statistics(y, sr):
    """提取 F0 的统计特征，用于声调偏误检测

    返回：
    - f0_mean: 平均基频
    - f0_range: 基频范围（max - min）
    - f0_min: 最低基频
    - f0_max: 最高基频
    - f0_slope: 基频变化趋势（正=上升，负=下降，零=平）
    """
    curve_data = extract_f0_curve(y, sr)
    if curve_data is None:
        return None

    f0_values = np.array(curve_data["f0"])
    valid_f0 = f0_values[~np.isnan(f0_values)]
    # 逐元素取反

    if len(valid_f0) < 3:
        return None

    # 线性拟合求斜率 —— 判断声调调理（升/降/平）
    # TO DO 第三声需要求两次斜率
    x = np.arange(len(valid_f0))
    slope = np.polyfit(x, valid_f0, 1)[0] if len(valid_f0) > 2 else 0

    return {
        "f0_mean": float(np.mean(valid_f0)),
        "f0_range": float(np.max(valid_f0) - np.min(valid_f0)),
        "f0_min": float(np.min(valid_f0)),
        "f0_max": float(np.max(valid_f0)),
        "f0_slope": float(slope),  
        "f0_std": float(np.std(valid_f0)),      # 音高变化幅度
    }


def extract_formants(y, sr, n_formants=3):
    """使用 LPC（线性预测编码）提取共振峰 F1/F2/F3

    共振峰是声道共鸣的产物，与发音器官配置直接相关：
    - F1 ↑ ≈ 舌位 ↓（下颌开度大）
    - F2 ↑ ≈ 舌位 → 前
    - F2 ↓ ≈ 圆唇
    - F3 ↓ ≈ 卷舌/翘舌
    """
    if not HAS_PARSELMOUTH:
        return None

    try:
        sound = parselmouth.Sound(y, sampling_frequency=sr)
        # LPC 阶数：经验上 2 * (n_formants + 2) 效果较好
        # 共振峰过少，可能漏掉或合并真是共振峰，过多容易过拟合噪音，产生虚假共振峰
        formant = sound.to_formant_burg(
            time_step=0.01,             # 每隔10ms分析一次
            max_number_of_formants=5,
            maximum_formant=5000,       # 女声/儿童用 5500
            window_length=0.025,        # 分析窗口为25ms
        )

        # 在时间中点提取共振峰
        midpoint = sound.duration / 2
        f1 = formant.get_value_at_time(1, midpoint)
        f2 = formant.get_value_at_time(2, midpoint)
        f3 = formant.get_value_at_time(3, midpoint)

        return {"f1": f1, "f2": f2, "f3": f3}
    except Exception:
        return None


def estimate_vot(y, sr):
    """粗略估计 VOT（浊音起始时间 / Voice Onset Time）

    VOT 是区分送气/不送气塞音的关键声学参数：
    - VOT ≈ 0: 不送气清塞音（b, d, g）
    - VOT ≈ 30-80ms: 送气清塞音（p, t, k）
    - VOT < 0: 浊塞音（英语 b, d, g，普通话中没有）

    方法：检测除阻（burst）后的周期性信号起始点。
    简化版：检测信号能量陡增 → 基频出现的时间差。
    """
    if y is None or len(y) == 0:
        return None

    try:
        # 用短时能量检测除阻点
        frame_length = int(0.005 * sr)  # 5ms 内的采样点数
        hop_length = int(0.001 * sr)    # 1ms 跳

        energy_values = []
        for i in range(0, len(y) - frame_length, hop_length):
            value = np.sum(y[i:(i + frame_length)] ** 2)
            energy_values.append(value)

        energy = np.array(energy_values)

        if len(energy) < 10:
            return None

        # 能量陡增点 ≈ 除阻时刻
        energy_diff = np.diff(energy)
        burst_frame = np.argmax(energy_diff[:min(50, len(energy_diff))])

        # 在 burst 后找基频出现点 ≈ 声带开始振动
        burst_sample = burst_frame * hop_length
        remaining = y[burst_sample:]

        # 如果剩余时间不足 2ms，信息不足， 无法判断 VOT 
        if len(remaining) < int(0.002 * sr):
            return None

        # 用自相关检测周期性信号的出现
        for i in range(0, len(remaining) - int(0.01 * sr), int(0.002 * sr)):
            segment = remaining[i:i + int(0.01 * sr)]
            if len(segment) < int(0.01 * sr):
                break
            # 归一化自相关，检测周期性
            autocorr = np.correlate(segment, segment, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            if len(autocorr) < 2:
                continue
            # 自相关峰值 > 阈值 → 有周期性（声带已振动）
            if autocorr[0] > 0 and np.max(autocorr[1:]) / autocorr[0] > 0.3:
                vot_ms = i / sr * 1000
                return {"vot_ms": vot_ms, "is_aspirated": vot_ms > 25}

        return None
    except Exception:
        return None


def extract_spectral_features(y, sr):
    """提取频谱特征：频谱重心、频谱散布度

    频谱重心受发音部位前后影响：
    - 靠前的辅音（z/c/s，舌尖在齿背）→ 频谱重心偏高
    - 靠后的辅音（zh/ch/sh，舌尖在硬腭）→ 频谱重心偏低
    """
    if y is None or len(y) == 0 or not HAS_LIBROSA:
        return None

    try:
        # 计算短时傅里叶变换
        D = np.abs(librosa.stft(y.astype(np.float64), n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # 频谱重心：频率的加权平均
        centroid = librosa.feature.spectral_centroid(
            S=D, sr=sr, n_fft=2048, hop_length=512
        )[0]

        # 频谱带宽：重心周围的频率分布范围
        bandwidth = librosa.feature.spectral_bandwidth(
            S=D, sr=sr, n_fft=2048, hop_length=512
        )[0]

        return {
            "centroid_mean": float(np.mean(centroid)),
            "centroid_std": float(np.std(centroid)),
            "bandwidth_mean": float(np.mean(bandwidth)),
        }
    except Exception:
        return None


def extract_acoustic_features(y, sr, target_word=""):
    """一站式提取所有声学特征

    这是评测引擎调用声学特征提取的统一入口。
    返回一个字典，包含所有可用于偏误诊断的声学参数。
    """
    features = {
        "f0_curve": extract_f0_curve(y, sr, target_word),
        "f0_stats": extract_f0_statistics(y, sr),
        "formants": extract_formants(y, sr),
        "vot": estimate_vot(y, sr),
        "spectral": extract_spectral_features(y, sr),
    }
    return features
