"""
对外汉语智能发音纠错系统 — Phase 0 演示框架
==============================================
技术栈：Gradio + librosa + parselmouth

"""

import gradio as gr
import numpy as np
import os
import tempfile
from pathlib import Path

from engine.evaluator import evaluate_pronunciation, parse_target_word
from engine.diagnosis import diagnose_errors
from engine.acoustic_features import extract_f0_curve

# ── 练习字表 ──
# 按偏误类型分组
PRACTICE_WORDS = {
    "翘舌音 (zh/ch/sh)": ["知(zhi1)", "吃(chi1)", "是(shi4)"],
    "平舌音 (z/c/s)": ["资(zi1)", "疵(ci1)", "四(si4)"],
    "舌面音 (j/q/x)": ["机(ji1)", "七(qi1)", "西(xi1)"],
    "圆唇音 (ü)": ["绿(lv4)", "女(nv3)", "鱼(yv2)"],
    "送气音 (p/t/k)": ["跑(pao3)", "他(ta1)", "看(kan4)"],
    "鼻韵尾 (-n/-ng)": ["慢(man4)", "忙(mang2)", "金(jin1)", "京(jing1)"],
    "声调对比 (二声vs三声)": ["麻(ma2)", "马(ma3)", "国(guo2)", "果(guo3)"],
}

# 扁平化为下拉菜单选项
ALL_WORDS = []
for group, words in PRACTICE_WORDS.items():
    for w in words:
        ALL_WORDS.append(f"{w}")

REFERENCE_AUDIO_DIR = Path(__file__).resolve().parent / "example"


def load_reference_audio(target_word):
    """根据目标字加载对应的标准发音 WAV 文件。"""
    target_info = parse_target_word(target_word)
    if not target_info:
        return None, "无法识别目标字，暂时无法加载标准发音。"

    pinyin = target_info.get("pinyin", "")
    if not pinyin:
        return None, "该目标字没有可用的拼音信息。"

    audio_path = REFERENCE_AUDIO_DIR / f"{pinyin}.wav"
    if not audio_path.is_file():
        return None, f"暂无标准发音音频：{audio_path.name}"

    return str(audio_path), f"已加载标准发音：{pinyin}"


# ── 样式定制 ──
CUSTOM_CSS = """
.gradio-container {
    max-width: 960px !important;
    margin: auto !important;
}
.header-title {
    text-align: center;
    padding: 1rem 0;
}
.header-title h1 {
    font-size: 1.8rem;
    color: #c41e3a;
    margin-bottom: 0.2rem;
}
.header-title p {
    color: #666;
    font-size: 0.9rem;
}
.score-good { color: #22c55e; font-weight: bold; }
.score-medium { color: #eab308; font-weight: bold; }
.score-bad { color: #ef4444; font-weight: bold; }
.diagnosis-box {
    font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
    line-height: 1.8;
    white-space: pre-wrap;
}
.footer-text {
    text-align: center;
    color: #999;
    font-size: 0.8rem;
    margin-top: 1rem;
}
"""


def score_color(score):
    """根据分数返回对应的 CSS 类名"""
    if score is None or score == 0:
        return "score-bad"
    if score >= 80:
        return "score-good"
    elif score >= 60:
        return "score-medium"
    return "score-bad"


def format_score_display(score):
    """格式化评分显示，带颜色标记"""
    if score is None or score == 0:
        return "—"
    return f"{score}"


def process_audio(target_word, audio_input):
    """核心处理函数：录音 → 评测 → 诊断 → 结果

    这是整个系统的流程入口，被绑定的 submit 按钮触发。
    """
    # ── 输入验证 ──
    if audio_input is None:
        return (
            0, 0, 0,
            "请先点击录音按钮，朗读上方显示的汉字。\n\n提示：使用电脑麦克风或手机浏览器均可录音。",
            None,
            ""
        )

    sr, y = audio_input

    # 处理立体声 → 单声道
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # 检查音频是否有效（长度 > 0.3 秒，避免空录音）
    duration = len(y) / sr
    if duration < 0.3:
        return (
            0, 0, 0,
            "录音太短，请至少朗读 1 秒。\n\n请重新点击录音按钮，大声清晰地朗读目标汉字。",
            None,
            ""
        )

    # ── 1. 评测打分 ──
    scores = evaluate_pronunciation(y, sr, target_word)

    # ── 2. 偏误诊断 ──
    diagnosis = diagnose_errors(scores, target_word, y, sr)

    # ── 3. 声调曲线 ──
    f0_data = extract_f0_curve(y, sr, target_word)

    # ── 4. 音频保存（本地临时文件，用于调试） ──
    audio_path = None
    try:
        import soundfile as sf
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y, sr, subtype='PCM_16')
        audio_path = tmp.name
    except Exception:
        pass

    # ── 5. 组装诊断信息卡片 ──
    info_lines = []
    target_info = scores.get("target_info", {})

    if target_info:
        info_lines.append(f"【练习字】{target_info.get('character', '?')}")
        info_lines.append(f"【拼音】{target_info.get('pinyin', '?')}")
        info_lines.append(f"【标准声母】{target_info.get('initial', '?')}　"
                         f"【标准韵母】{target_info.get('final', '?')}　"
                         f"【标准声调】第{target_info.get('tone', '?')}声")
        info_lines.append(f"【录音时长】{duration:.1f} 秒")
        info_lines.append("")

    # 添加声学特征摘要（如果有）
    features = scores.get("features", {})
    if features.get("f0_stats"):
        f0s = features["f0_stats"]
        info_lines.append(f"【F0 范围】{f0s['f0_mean']:.0f} ± {f0s['f0_std']:.0f} Hz")
    if features.get("formants"):
        fmts = features["formants"]
        if fmts.get("f1") and fmts.get("f2"):
            info_lines.append(f"【F1/F2】{fmts['f1']:.0f} / {fmts['f2']:.0f} Hz")

    feature_summary = "\n".join(info_lines) if info_lines else ""

    return (
        scores.get("overall", 0),
        scores.get("phoneme", 0),
        scores.get("tone", 0),
        diagnosis.get("diagnosis_text", "评测完成"),
        f0_data,
        feature_summary,
    )


# ── 构建 Gradio 界面 ──
def create_demo():
    """创建 Gradio 应用"""

    with gr.Blocks(title="对外汉语智能发音纠错系统") as demo:

        # ── 标题区 ──
        gr.HTML("""
        <div class="header-title">
            <h1>汉语发音学习</h1>
        </div>
        """)

        gr.Markdown("""
        ### 使用说明
        1. 从下拉菜单中选择一个练习字
        2. 点击麦克风图标录音（朗读上方汉字，约 1-2 秒）
        3. 点击 **"开始评测"** 按钮，查看评分和发音器官诊断
        """)

        gr.Markdown("---")

        with gr.Row(equal_height=False):
            # ── 左侧：控制面板 ──
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("#### 录音控制")

                target_word = gr.Dropdown(
                    choices=ALL_WORDS,
                    label="选择练习字",
                    value="知(zhi1)",
                    interactive=True,
                )

                reference_btn = gr.Button(
                    "播放正确发音",
                    variant="secondary",
                )

                reference_audio = gr.Audio(
                    value=str(REFERENCE_AUDIO_DIR / "zhi1.wav"),
                    label="标准发音",
                    type="filepath",
                    interactive=False,
                    autoplay=False,
                    elem_id="reference-audio",
                )

                reference_status = gr.Markdown(
                    "已加载标准发音：zhi1"
                )

                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="点击麦克风图标开始录音",
                )

                submit_btn = gr.Button(
                    "开始评测",
                    variant="primary",
                    size="lg",
                )

                # 练习分组提示
                gr.Markdown("""
                **练习分组：**
                - 翘舌音：知 吃 是
                - 平舌音：资 疵 四
                - 舌面音：机 七 西
                - 圆唇音：绿 女 鱼
                - 送气音：跑 他 看
                - 鼻韵尾：慢 忙 金 京
                - 声调对比：麻/马 国/果
                """)

            # ── 右侧：结果面板 ──
            with gr.Column(scale=2, min_width=400):
                gr.Markdown("#### 评测结果")

                # 三分评分行
                with gr.Row():
                    overall_score = gr.Number(
                        label="总分",
                        value=0,
                        precision=0,
                    )
                    phoneme_score = gr.Number(
                        label="音素得分",
                        value=0,
                        precision=0,
                    )
                    tone_score = gr.Number(
                        label="声调得分",
                        value=0,
                        precision=0,
                    )

                # 声学特征信息
                feature_info = gr.Textbox(
                    label="录音分析",
                    value="等待录音...",
                    lines=3,
                    interactive=False,
                )

                # 声调曲线图
                f0_plot = gr.LinePlot(
                    x="time",
                    y="f0",
                    x_title="时间 (秒)",
                    y_title="基频 F0 (Hz)",
                    title="声调曲线 (音高变化轨迹)",
                    height=200,
                )

                gr.Markdown("#### 发音器官诊断报告")

                diagnosis_text = gr.Textbox(
                    label="",
                    placeholder="朗读后这里会显示：舌头位置、唇形、送气等方面的具体问题和纠正建议...",
                    lines=14,
                    max_lines=20,
                    interactive=False,
                    elem_classes=["diagnosis-box"],
                )

        gr.Markdown("---")

        # ── 底部信息 ──
        gr.HTML("""
        <div class="footer-text">
            <p>Phase 0 Demo </p>
        </div>
        """)

        # ── 事件绑定 ──
        target_word.change(
            fn=load_reference_audio,
            inputs=[target_word],
            outputs=[reference_audio, reference_status],
        )

        reference_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            js="""
            () => {
                const audio = document.querySelector("#reference-audio audio");
                if (!audio) return;
                audio.currentTime = 0;
                const playPromise = audio.play();
                if (playPromise !== undefined) {
                    playPromise.catch(() => {});
                }
            }
            """,
        )

        submit_btn.click(
            fn=process_audio,
            inputs=[target_word, audio_input],
            outputs=[
                overall_score,
                phoneme_score,
                tone_score,
                diagnosis_text,
                f0_plot,
                feature_info,
            ],
        )

    return demo


# ── 入口 ──
if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="red"),
    )
