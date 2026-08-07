# Anger Detection

中文语音**愤怒检测**端到端工程：DS-CNN 训练 / CASIA 微调 / Binary V2 / ONNX·TFLite 部署 / Streamlit 实时 Demo，并含 TinyCNN 轻量基线。

| 项 | 值 |
|----|-----|
| 当前分支 | **`v2`**（binary anger + src 布局） |
| 上一迭代 | `v1`（五类 + 决策层） |
| 基线快照 | `main` + tag **`v0`** |
| 版本文件 | [`VERSION`](VERSION) |

仓库：https://github.com/Stoicesf/anger-detection

## 路线图

```text
v0 基线 Demo
  → Phase1 决策层 (EMA / 连续触发 / 状态机)     ← 已完成
  → Phase2 真实数据增强                         ← 已完成
  → V2 Binary anger + src 布局                  ← 当前
  → Phase3 量化 / 蒸馏
  → Phase4 FastAPI 部署
```

## 功能概览

| 能力 | 说明 |
|------|------|
| 五类情绪模型 | 中性 / 高兴 / 愤怒 / 悲伤 / 惊讶（ESD 中文） |
| Binary V2 | `artifacts/models/dscnn_anger_v2.onnx`（Demo 优先） |
| CASIA 微调 | `artifacts/models/dscnn_casia_deploy.onnx` |
| 实时决策层 | EMA + 连续命中 + `NORMAL/SUSPECT/ANGER/RECOVER` |
| Streamlit Demo | 默认非愤怒；进入 ANGER 才报警；持续采麦 |
| TinyCNN 基线 | `baselines/tinycnn/`，~6K 参数，Gradio Demo |
| 边缘部署 | int8 TFLite ~28KB，面向 ESP32-S3 |

## 快速开始（实时 Demo）

```powershell
conda activate pytorch12
git clone https://github.com/Stoicesf/anger-detection.git
cd anger-detection
git checkout v2
pip install -e .
pip install -r requirements.txt
streamlit run apps/streamlit_demo/realtime_emotion_demo.py --server.port 8503
```

浏览器打开：http://127.0.0.1:8503

TinyCNN Gradio Demo：

```powershell
pip install -r baselines/tinycnn/requirements.txt
python baselines/tinycnn/src/demo.py
# http://127.0.0.1:7861
```

## 仓库结构

```text
anger-detection/
├─ src/anger_detection/           # 可安装核心库
│  ├─ common.py                   # 路径 / 特征超参
│  ├─ features/                   # Log-Mel
│  ├─ models/                     # DSCNN / binary anger / TF Keras
│  ├─ losses/                     # Focal BCE
│  └─ decision/                   # 状态机 + 实时采麦
├─ apps/streamlit_demo/           # Streamlit UI
├─ scripts/
│  ├─ data/                       # 下载 / 特征 / hard-negative
│  ├─ train/                      # 训练 / 微调 / binary V2
│  ├─ eval/                       # 评测 / 预测
│  └─ export/                     # ONNX→TFLite / 量化
├─ data/
│  ├─ raw/                        # 原始语料（gitignore）
│  ├─ processed/                  # 特征 npy（gitignore）
│  └─ curated/                    # hard_negative / home_anger
├─ artifacts/
│  ├─ models/                     # ONNX / PT / TFLite
│  └─ reports/                    # 指标 JSON / 日志
├─ baselines/tinycnn/
├─ tests/
├─ tools/migrate_to_v2_layout.py  # 目录迁移脚本（可 dry-run）
├─ pyproject.toml
├─ requirements.txt
└─ VERSION
```

本地大数据（**不入库**）：`data/raw/`、`*.npy`、校准集、中间 TFLite 导出目录等，见 [`.gitignore`](.gitignore)。

## 模型文件（已入库）

| 文件 | 用途 |
|------|------|
| `artifacts/models/dscnn_anger_v2.onnx` | **Demo 优先**（binary V2） |
| `artifacts/models/dscnn_casia_deploy.onnx` | CASIA 微调五类 |
| `artifacts/models/dscnn.onnx` / `dscnn_torch_best.pt` | ESD 原训练 |
| `artifacts/models/anger_detection_model_int8.tflite` | ESP32 量化模型 |
| `baselines/tinycnn/checkpoints/best_model.pth` | TinyCNN 最优权重 |

## 依赖与路径注意

1. **必须先** `pip install -e .`，否则 `import anger_detection` 会失败（已去掉 `sys.path` hack）。
2. 训练额外依赖：`pip install -r requirements-train.txt`（或 `pip install -e ".[train]"` + 本机 CUDA torch）。
3. 外部语料根目录：`ANGER_CASIA_ROOT` / `ANGER_CNEV_ROOT`（见 [`ROADMAP.md`](ROADMAP.md)）。
4. 旧路径 `models/`、`output/`、`dataset/`、`demo/` 已分别迁至 `artifacts/models`、`artifacts/reports`、`data/curated`、`apps` + `src/.../decision`。

## 训练 / 评估（V2）

```powershell
python scripts/train/train_anger_binary.py --device cuda --epochs 40 --rebuild-cache
python scripts/eval/eval_anger.py --onnx artifacts/models/dscnn_anger_v2.onnx --grid --realtime
```

更多细节见 [`ROADMAP.md`](ROADMAP.md)。
