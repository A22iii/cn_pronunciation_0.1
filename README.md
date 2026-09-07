# 对外汉语智能发音纠错系统 — Phase 0

## 快速开始

```bash
# 1. 进入项目目录
cd demo_project

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动应用
python app.py

# 4. 浏览器打开
# http://localhost:7860
```

## 项目结构

```
demo_project/
├── app.py                      # Gradio 主应用
├── engine/
│   ├── evaluator.py            # 评测引擎（模拟评分）
│   ├── diagnosis.py            # 偏误诊断引擎
│   └── acoustic_features.py    # 声学特征提取
├── knowledge_base/
│   └── error_rules.py          # 偏误规则知识库
├── requirements.txt
└── README.md
```

## 当前覆盖的偏误（Top-5）

| 偏误 | 说明 |
|------|------|
| retroflex_fronting | zh/ch/sh → z/c/s（舌尖偏前） |
| aspiration_insufficient | p/t/k → b/d/g（送气不足） |
| rounding_missing | ü → i/u（圆唇缺失） |
| nasal_final_dropped | -n/-ng 脱落（软腭未降） |
| tone2_3_confusion | 二声/三声混淆 |

## 后续计划

- Phase 1: 迁移至 FastAPI + React，接入实验室 TDNN chain 模型
- Phase 2: 扩充至 30+ 偏误规则，加入 3D 发音动画
- Phase 3: 多母语支持，句子级训练
