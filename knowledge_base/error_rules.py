"""偏误规则表 — 英语母语者汉语发音偏误知识库

声学特征异常 → 偏误类型 → 发音器官诊断 三级映射

Phase 0 覆盖 Top-5 高频偏误，后续扩展到 30+
每条规则包含：
  - 偏误描述（中英文）
  - 发音器官层面的诊断
  - 可操作的纠正建议
  - 声学特征标记（用于自动检测）

数据来源参考：
  - SAIT 实验室发音偏误趋势检测（PET）研究成果
  - 英语母语者普通话习得偏误文献综述
  - Ladefoged & Johnson (2011) A Course in Phonetics
"""

ERROR_RULES = {
    # ── 偏误 1：翘舌→平舌（英语母语者最高频偏误） ──
    "retroflex_fronting": {
        "error_id": "en_zh_retroflex_fronting",
        "l1_background": "English",
        "description_cn": "翘舌音发成了平舌音（zh/ch/sh → z/c/s）",
        "description_en": "Retroflex consonants pronounced as alveolar (zh/ch/sh → z/c/s)",
        "articulatory_cn": (
            "舌尖位置偏前 —— 你的舌尖放在了齿背附近，"
            "而发 zh/ch/sh 时需要舌尖向后缩，轻轻抵住或靠近硬腭前部。"
            "英语中没有翘舌音，所以你习惯性地用英语的舌叶音替代。"
        ),
        "articulatory_en": (
            "Tongue tip is too far forward — it should be retracted to the "
            "front of the hard palate, not touching the back of the teeth."
        ),
        "correction_cn": (
            "1. 先发 z（舌尖抵齿背），感受这个位置\n"
            "2. 舌尖沿上颚向后滑动约 1 厘米，直到碰到上颚由平滑变粗糙的边界（硬腭前部）\n"
            "3. 保持舌尖在这个位置，舌面微微隆起，发 zh\n"
            "4. 对着镜子检查：舌尖应该看不到（藏在口腔内部）"
        ),
        "correction_en": (
            "Start with 'z', then slide your tongue tip back along the roof "
            "of your mouth about 1cm until you feel the hard palate. "
            "Keep the tip there and raise the tongue body slightly."
        ),
        # 声学标记：F2/F3 偏低是翘舌音偏前的典型特征
        "acoustic_markers": {
            "f2_lowered": True,
            "f3_lowered": True,
            "spectral_center_of_gravity_forward": True,
        },
        # 发音器官动画参数
        "animation_params": {
            "tongue_tip_x": 0.35,       # 偏误：舌尖靠前（0=最前，1=最后）
            "tongue_tip_x_correct": 0.65,  # 正确：舌尖在硬腭前部
            "tongue_body_y": 0.5,       # 偏误：舌体低平
            "tongue_body_y_correct": 0.6,  # 正确：舌体微隆
            "lip_rounding": 0.1,        # 平舌不需圆唇
            "lip_rounding_correct": 0.2,   # 翘舌有轻微圆唇
        },
        "difficulty_level": 3,
        "related_minimal_pairs": ["zhī-zī", "chī-cí", "shì-sì"],
    },

    # ── 偏误 2：送气不足（英语母语者常将送气清音发成不送气浊音） ──
    "aspiration_insufficient": {
        "error_id": "en_aspiration_insufficient",
        "l1_background": "English",
        "description_cn": "送气不足（p/t/k 发得像 b/d/g）",
        "description_en": "Insufficient aspiration (p/t/k sound like b/d/g)",
        "articulatory_cn": (
            "声带振动开始过早 —— 送气辅音 p/t/k 在除阻后需要有约 30-80ms "
            "的清音送气段（VOT > 0），之后声带才开始振动。你的发音中这段送气太短或缺失。"
            "英语中 /p/ 在词首虽有送气，但在 /s/ 后不送气（如 'spin'），"
            "所以你不太习惯刻意控制送气时长。"
        ),
        "articulatory_en": (
            "Voicing starts too early — aspirated stops need ~30-80ms of "
            "voiceless aspiration (positive VOT) before vocal fold vibration begins."
        ),
        "correction_cn": (
            "1. 把手掌放在嘴前约 5 厘米处\n"
            "2. 发 p 音，手掌应感到一股明显的气流冲出\n"
            "3. 对比：先发 b（几乎无气流），再发 p（强烈气流），感受区别\n"
            "4. 如果感觉不到气流，先夸张地送气（像吹蜡烛），再逐渐收敛到正常力度\n"
            "5. 可以用一张薄纸放在嘴前，看纸张是否被吹动"
        ),
        "correction_en": (
            "Place your hand 5cm in front of your mouth. When saying 'p', "
            "you should feel a strong puff of air. Compare with 'b' (no puff)."
        ),
        "acoustic_markers": {
            "vot_duration_ms_lt": 30,  # VOT < 30ms 表示送气不足
            "f0_onset_early": True,    # 基频过早出现（声带早振）
        },
        "animation_params": {
            "aspiration_duration": 15,          # 偏误：送气段太短
            "aspiration_duration_correct": 60,  # 正确：送气段约 60ms
            "glottis_open_duration": 0.2,       # 偏误：声门早闭
            "glottis_open_duration_correct": 0.6,
        },
        "difficulty_level": 4,
        "related_minimal_pairs": ["pǎo-bǎo", "tā-dā", "kàn-gàn"],
    },

    # ── 偏误 3：圆唇缺失（ü 对英语母语者极难） ──
    "rounding_missing": {
        "error_id": "en_rounding_missing",
        "l1_background": "English",
        "description_cn": "圆唇动作缺失（ü 发成 i 或 u）",
        "description_en": "Lip rounding missing (ü pronounced as i or u)",
        "articulatory_cn": (
            "双唇未撮起 —— 发 ü 时舌位同 i（高前元音），但嘴唇必须撮圆。"
            "你的嘴唇可能保持了展唇（偏 i）或舌头位置向后滑动了（偏 u）。"
            "英语中没有前圆唇元音，这个发音动作对英语母语者完全陌生。"
        ),
        "articulatory_en": (
            "Lips are not rounded — ü requires the tongue position of /i/ "
            "(high-front) but with rounded lips. English has no front rounded vowels."
        ),
        "correction_cn": (
            "1. 先发 i（舌头抵住下齿背，舌面高抬）\n"
            "2. 保持舌头完全不动！这是关键\n"
            "3. 慢慢把嘴唇撮圆，像要吹口哨\n"
            "4. 听到的音应该从 i 变成 ü\n"
            "5. 对着镜子检查：口型是圆的小 O 形，不是扁的"
        ),
        "correction_en": (
            "Say 'ee' (as in 'see'), freeze your tongue, then round your "
            "lips as if to whistle. That's ü."
        ),
        "acoustic_markers": {
            "f2_lowered": True,    # 圆唇会降低 F2
            "f3_lowered": True,    # 圆唇也会影响 F3
        },
        "animation_params": {
            "tongue_body_x": 0.25,     # 舌位高前（同 i）
            "tongue_body_y": 0.85,
            "lip_rounding": 0.1,       # 偏误：展唇
            "lip_rounding_correct": 0.8,  # 正确：撮唇
            "jaw_open": 0.15,
        },
        "difficulty_level": 4,
        "related_minimal_pairs": ["lǜ-lù", "nǚ-nǐ", "yú-yí"],
    },

    # ── 偏误 4：鼻韵尾脱落（英语母语者常丢失 -n/-ng） ──
    "nasal_final_dropped": {
        "error_id": "en_nasal_final_dropped",
        "l1_background": "English",
        "description_cn": "鼻韵尾脱落或弱化（-n/-ng 发得不完整）",
        "description_en": "Nasal coda dropped or weakened (-n/-ng incomplete)",
        "articulatory_cn": (
            "软腭未充分下降 —— 发鼻音 -n 或 -ng 时，软腭需要下降打开鼻腔通道，"
            "让气流同时或主要通过鼻腔流出。你的软腭可能降得不够低，"
            "导致大部分气流仍从口腔流出，听起来像元音鼻化而非完整的鼻韵尾。"
            "英语中元音+鼻音的序列（如 'can'）与普通话鼻韵母不同——"
            "英语的鼻音是独立的辅音韵尾，普通话的鼻韵尾与前面元音的耦合更紧密。"
        ),
        "articulatory_en": (
            "Velum not fully lowered — the soft palate must descend to open "
            "the nasal cavity for -n/-ng. Incomplete lowering produces nasalized "
            "vowels rather than true nasal finals."
        ),
        "correction_cn": (
            "1. 发韵腹元音时保持口腔形状\n"
            "2. 韵尾阶段：用鼻子哼出后半段（像 hum 的感觉）\n"
            "3. 用手指轻捏鼻子两侧，发鼻韵尾时应感受到鼻翼振动\n"
            "4. 对比练习：妈(ma) vs 慢(man) —— man 的结尾应有鼻音振动\n"
            "5. 区分 n 和 ng：n 舌尖抵齿龈，ng 舌根抬至软腭"
        ),
        "correction_en": (
            "Hum through your nose at the end of the syllable. "
            "When you say 'man', your nose should vibrate at the end."
        ),
        "acoustic_markers": {
            "nasal_formant_present": False,    # 缺失鼻音共振峰
            "nasal_duration_ms_lt": 30,        # 鼻音段太短
            "f1_bandwidth_narrow": False,      # F1 带宽未因鼻音耦合而展宽
        },
        "animation_params": {
            "velum_lowering": 0.2,        # 偏误：软腭未降
            "velum_lowering_correct": 0.9,   # 正确：软腭充分下降
            "nasal_airflow": 0.1,          # 偏误：鼻腔气流弱
            "nasal_airflow_correct": 0.8,
        },
        "difficulty_level": 3,
        "related_minimal_pairs": ["màn-máng", "jīn-jīng", "wán-wáng"],
    },

    # ── 偏误 5：二声/三声混淆（非声调语言母语者的核心难点） ──
    "tone2_3_confusion": {
        "error_id": "en_tone2_3_confusion",
        "l1_background": "English",
        "description_cn": "二声与三声混淆（阳平 35 ↔ 上声 214）",
        "description_en": "Confusion between Tone 2 (rising) and Tone 3 (dipping)",
        "articulatory_cn": (
            "声带振动频率控制不准确 —— 英语中没有用音高区别词义的机制，"
            "所以你对声带振动频率的精细控制缺乏训练。\n"
            "二声（˧˥, 35）：声带从中间频率线性上升到高频率，像英语问句 'What?' 的上扬。\n"
            "三声（˨˩˦, 214）：声带先下降到低频（有时伴随紧喉/creaky voice），再上升到中高频率。"
            "你常见的错误是：把两个声调都发成上升调（二声正确，三声错误），"
            "或把三声发成纯降调（丢失了末尾的上升段）。"
        ),
        "articulatory_en": (
            "Vocal fold frequency control is imprecise — Tone 2 requires "
            "a linear rise from mid to high (35→55), while Tone 3 requires "
            "a fall to low then rise to mid-high (21→14)."
        ),
        "correction_cn": (
            "二声（ˊ）：\n"
            "  - 像英语问句 'What?' 或 'Huh?' 的音高上扬\n"
            "  - 用头从中间向上扬的动作辅助感受音高上升\n"
            "  - 关键：是一个平滑持续的上升，中间不下掉\n\n"
            "三声（ˇ）：\n"
            "  - 像英语犹豫时的 'We~ell...' 先降后升\n"
            "  - 头先向下低再抬起，辅助感受音高变化\n"
            "  - 关键：低点要真正够低，有时可以带一点喉化（气泡音）\n\n"
            "对比练习：má(麻) vs mǎ(马)，对着镜子看头部动作"
        ),
        "correction_en": (
            "Tone 2: like asking a question 'What?' — voice rises smoothly.\n"
            "Tone 3: like saying 'We~ell...' hesitantly — voice dips then rises."
        ),
        "acoustic_markers": {
            "f0_contour_type": "rising_only",  # 偏误：只有上升，缺少低凹
            "f0_min_too_high": True,           # 三声最低点不够低
            "f0_range_compressed": True,       # 调域压缩（声调空间偏窄）
        },
        "animation_params": {
            "f0_start": 160,               # Hz
            "f0_mid": 150,                 # 偏误：中间不降或降得不够
            "f0_mid_correct": 90,          # 正确：三声中间降到很低
            "f0_end": 180,
            "creaky_voice": 0.0,           # 偏误：无紧喉
            "creaky_voice_correct": 0.3,   # 正确：三声低点可能略紧喉
        },
        "difficulty_level": 5,
        "related_minimal_pairs": ["má-mǎ", "guó-guǒ", "shí-shǐ"],
    },
}
