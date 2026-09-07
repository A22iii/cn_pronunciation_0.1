"""偏误诊断引擎 — 发音器官级别的诊断文本生成

核心逻辑：
  低分音素 → 查找偏误规则表 → 生成发音器官级诊断文本

诊断包含三个层次：
  1. 问题描述（哪个音错了、错成什么样）
  2. 发音器官诊断（舌头、嘴唇、软腭、声带分别哪里不对）
  3. 纠正建议（具体的、可操作的改进步骤）

设计原则：
  - 诊断文本面向学习者，语言清晰可操作
  - 与发音动画参数联动（diagnosis 中的 animation_params 直接驱动前端动画）
  - 规则表独立于诊断引擎，方便后续扩充
"""

from knowledge_base.error_rules import ERROR_RULES


def diagnose_errors(scores, target_word="", y=None, sr=None):
    """根据评测结果生成发音器官级诊断

    Args:
        scores: evaluate_pronunciation() 的返回结果
        target_word: 目标字
        y: 原始音频（Phase 0 暂未用于诊断，Phase 2 可用于声学特征验证）
        sr: 采样率

    Returns:
        诊断结果字典，包含文字诊断和动画参数
    """
    if not scores or scores.get("error"):
        return {
            "diagnosis_text": "评测失败，请重新录音。",
            "has_errors": True,
            "error_details": [],
        }

    details = scores.get("details", {})
    if not details:
        return {
            "diagnosis_text": "发音良好，未检测到明显偏误。继续保持！",
            "has_errors": False,
            "error_details": [],
        }

    error_details = []
    has_errors = False

    for phoneme, detail in details.items():
        score = detail.get("score", 100)
        error_type = detail.get("error_type")

        # 低于 80 分且有关联偏误规则 → 生成诊断
        if score < 80 and error_type and error_type in ERROR_RULES:
            has_errors = True
            rule = ERROR_RULES[error_type]

            error_details.append({
                "phoneme": phoneme,
                "score": score,
                "error_type": error_type,
                "description": rule["description_cn"],
                "articulatory_diagnosis": rule["articulatory_cn"],
                "correction": rule["correction_cn"],
                "animation_params": rule.get("animation_params", {}),
                "difficulty_level": rule.get("difficulty_level", 3),
                "related_minimal_pairs": rule.get("related_minimal_pairs", []),
            })
        elif score < 80:
            # 低于 80 分但没有匹配到具体规则 → 给出通用诊断
            has_errors = True
            error_details.append({
                "phoneme": phoneme,
                "score": score,
                "error_type": "unknown",
                "description": f"{phoneme} 发音不够标准",
                "articulatory_diagnosis": f"发音器官动作与标准发音存在偏差，建议仔细模仿标准发音的口型和舌位。",
                "correction": "请先播放标准音，仔细听辨差异，然后对着镜子模仿发音，注意舌头和嘴唇的位置。",
                "animation_params": {},
                "difficulty_level": 2,
                "related_minimal_pairs": [],
            })

    # 生成层级诊断文本
    diagnosis_text = _format_diagnosis_text(error_details, has_errors)

    return {
        "diagnosis_text": diagnosis_text,
        "has_errors": has_errors,
        "error_details": error_details,
    }


def _format_diagnosis_text(error_details, has_errors):
    """将错误详情格式化为可读的诊断文本

    按音素分层组织：
      第一层：音素名和得分
      第二层：发音器官诊断（核心）
      第三层：纠正建议（可操作）
    """
    if not has_errors:
        return "发音良好，未检测到明显偏误。继续保持！"

    lines = []

    # 根据错误数量和严重程度给出总体评价
    avg_score = sum(d["score"] for d in error_details) / len(error_details)
    if avg_score < 60:
        lines.append("需要重点练习，多个音素存在明显偏误。\n")
    elif avg_score < 75:
        lines.append("有一定基础，部分音素需要针对纠正。\n")
    else:
        lines.append("整体不错，个别音素微调即可。\n")

    for i, err in enumerate(error_details, 1):
        # 用分隔线区分不同音素的诊断
        lines.append("─" * 40)
        lines.append(f"【诊断 {i}】音素：{err['phoneme']}（得分：{err['score']}/100）")

        if err["error_type"] != "unknown":
            lines.append(f"\n问题：{err['description']}")
            lines.append(f"\n发音器官诊断：\n  {err['articulatory_diagnosis']}")
            lines.append(f"\n纠正建议：\n  {err['correction']}")

            # 推荐最小对立体练习
            if err["related_minimal_pairs"]:
                pairs = "、".join(err["related_minimal_pairs"])
                lines.append(f"\n推荐对比练习：{pairs}")
        else:
            lines.append(f"\n{err['description']}")
            lines.append(f"\n{err['articulatory_diagnosis']}")

        lines.append("")

    # 学习建议
    lines.append("─" * 40)
    lines.append("学习建议：")
    lines.append("1. 先听标准发音，想象发音器官的位置和运动")
    lines.append("2. 对着镜子练习，观察自己的口型是否与标准一致")
    lines.append("3. 录音对比自己的发音与标准发音，反复调整")
    lines.append("4. 每个音至少连续 3 次达到 80 分以上再进入下一个")

    return "\n".join(lines)


def get_correction_instruction(error_type, lang="cn"):
    """获取特定偏误类型的纠正指导（简短版，用于实时反馈卡片）

    与 diagnose_errors 的区别：返回单条精简指令，适合在跟读练习中快速展示。
    """
    rule = ERROR_RULES.get(error_type)
    if not rule:
        return "请仔细模仿标准发音。"

    if lang == "cn":
        return rule["correction_cn"]
    else:
        return rule["correction_en"]
