"""
对外汉语智能发音纠错系统 — Phase 0 演示框架
==============================================
技术栈：Gradio + librosa + parselmouth
目标：1 周内跑通"录音 → 评测 → 诊断"闭环，向导师演示

运行方式：
    cd demo_project
    pip install -r requirements.txt
    python app.py

然后浏览器打开 http://localhost:7860

"""

import gradio as gr
import numpy as np
import os
import tempfile

from engine.evaluator import evaluate_pronunciation
from engine.diagnosis import diagnose_errors
from engine.acoustic_features import extract_f0_curve

# ── 练习字表 ──
# 按偏误类型分组，每组的第一个是标准音，其余是易混淆音
PRACTICE_WORDS = {
    "翘舌音 (zh/ch/sh)": ["知(zhī)", "吃(chī)", "是(shì)"],
    "平舌音 (z/c/s)": ["资(zī)", "疵(cī)", "四(sì)"],
    "舌面音 (j/q/x)": ["机(jī)", "七(qī)", "西(xī)"],
    "圆唇音 (ü)": ["绿(lǜ)", "女(nǚ)", "鱼(yú)"],
    "送气音 (p/t/k)": ["跑(pǎo)", "他(tā)", "看(kàn)"],
    "鼻韵尾 (-n/-ng)": ["慢(màn)", "忙(máng)", "金(jīn)", "京(jīng)"],
    "声调对比 (二声vs三声)": ["麻(má)", "马(mǎ)", "国(guó)", "果(guǒ)"],
}

# 扁平化为下拉菜单选项
ALL_WORDS = []
for group, words in PRACTICE_WORDS.items():
    for w in words:
        ALL_WORDS.append(f"{w}")


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
                    value="知(zhī)",
                    interactive=True,
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
    # share=False: 仅本地访问，适合校内开发调试
    # share=True: 生成公网 Gradio 链接，导师可远程查看
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="red"),
    )
