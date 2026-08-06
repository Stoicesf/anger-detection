# Anger Detection

中文语音**愤怒检测**端到端工程：DS-CNN 训练 / CASIA 微调 / ONNX·TFLite 部署 / Streamlit 实时 Demo，并附 TinyCNN 轻量基线。

| 项 | 值 |
|----|-----|
| 当前分支 | **`v1`**（迭代开发） |
| 基线快照 | `main` + tag **`v0`** |
| 版本文件 | [`VERSION`](VERSION) |

仓库：https://github.com/Stoicesf/anger-detection

## 路线图

```text
v0 基线 Demo
  → Phase1 决策层 (EMA / 连续触发 / 状态机)     ✅ 已完成
  → Phase2 真实数据增强
  → Phase3 量化 / 蒸馏
  → Phase4 FastAPI 部署
```

## 功能概览

| 能力 | 说明 |
|------|------|
| 五类情绪模型 | 中性 / 高兴 / 愤怒 / 悲伤 / 惊讶（ESD 中文） |
| CASIA 微调 | `models/dscnn_casia_deploy.onnx`（Demo 默认） |
| 实时决策层 (v1) | EMA + 连续命中 + `NORMAL/SUSPECT/ANGER/RECOVER` |
| Streamlit Demo | 默认非愤怒；进入 ANGER 才报警；持续采麦 |
| TinyCNN 基线 | `baselines/tinycnn/`，~6K 参数，Gradio Demo |
| 边缘部署 | int8 TFLite ~28KB，面向 ESP32-S3 |

## 快速开始（主 Demo）

```powershell
conda activate pytorch12
git clone https://github.com/Stoicesf/anger-detection.git
cd anger-detection
git checkout v1
pip install -r requirements.txt
streamlit run demo/realtime_emotion_demo.py --server.port 8503
```

浏览器打开：http://127.0.0.1:8503

- **▶ 启动实时检测**：后台采麦，约每 0.5s 推理  
- 侧边栏可调触发/解除阈值、分析窗口  

TinyCNN Gradio Demo：

```powershell
pip install -r baselines/tinycnn/requirements.txt
python baselines/tinycnn/src/demo.py
# http://127.0.0.1:7861
```

## 仓库结构

```text
anger-detection/
├─ demo/                          # 实时 Demo + 决策层
│  ├─ realtime_emotion_demo.py
│  ├─ live_monitor.py
│  ├─ decision.py
│  └─ test_decision.py
├─ scripts/                       # 数据 / 训练 / 微调 / 导出 / 评测
├─ models/                        # ONNX / PT / TFLite 权重
├─ baselines/
│  └─ tinycnn/                    # 轻量 CNN 基线 + Gradio
├─ data/processed/                # 特征需本地生成（大文件 gitignore）
├─ output/                        # 训练/微调指标 JSON
├─ requirements.txt               # Demo 依赖
├─ requirements-train.txt         # 训练额外依赖
├─ VERSION
└─ run_finetune_and_demo.ps1
```

本地大数据（**不入库**）：`data/raw/`、`*.npy`、校准集、中间 TFLite 导出目录等，见 `.gitignore`。

## 模型文件（已入库）

| 文件 | 用途 |
|------|------|
| `models/dscnn_casia_deploy.onnx` | **主 Demo 默认**（CASIA 微调） |
| `models/dscnn_casia.onnx` / `*.pt` | 留一说话人微调产物 |
| `models/dscnn.onnx` / `dscnn_torch_best.pt` | ESD 原训练 |
| `models/anger_detection_model_int8.tflite` | ESP32 量化模型 |
| `baselines/tinycnn/checkpoints/best_model.pth` | TinyCNN 最优权重 |

## 评测摘要

- ESD holdout（五类）：Acc ≈ **88.7%**
- CASIA 官方 holdout · **愤怒二分类**：
  - **DS-CNN（本仓库）**：Acc **96.7%** · F1 0.90 · ~2 ms  
  - SenseVoice：Acc 95.0% · ~800 ms  
  - TinyCNN 基线：Acc ~84–88% · ~3–5 ms（详见 `baselines/tinycnn/checkpoints/`）

## 训练 / 微调

```powershell
conda activate pytorch12
pip install -r requirements-train.txt

# ESD 中文特征（需自行下载，体积大）
python scripts/download_chinese_data.py
python scripts/prepare_data_chinese.py
python scripts/train_torch.py

# CASIA 微调（需本机 CASIA 路径，见 scripts/finetune_casia.py）
python scripts/finetune_casia.py --device cpu --epochs 35
```

一键脚本：`run_finetune_and_demo.ps1`

## 分支约定

| 分支 / 标签 | 含义 |
|-------------|------|
| `main` + tag `v0` | v0 基线快照（勿在此堆新功能） |
| **`v1`** | 当前迭代分支（PR / 开发请基于此） |

## 许可与数据

训练数据含 ESD 等，请遵守各数据集许可（如 CC-BY-NC）。代码与模型权重按项目约定使用。
