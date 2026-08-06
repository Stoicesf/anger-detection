# Anger Detection v0

中文语音**愤怒检测**端到端工程：DS-CNN 训练 / CASIA 微调 / ONNX 部署 / Streamlit 实时 Demo。

**版本：`v0`（基线）** · 迭代请在分支 `v1` 上进行。

## 功能概览

| 能力 | 说明 |
|------|------|
| 五类情绪模型 | 中性 / 高兴 / 愤怒 / 悲伤 / 惊讶（ESD 中文训练） |
| CASIA 微调 | `dscnn_casia_deploy.onnx`，面向公开中文测试集 |
| 实时 Demo | 默认「非愤怒」，仅愤怒过阈值才报警；麦克风持续监听 |
| 边缘部署 | int8 TFLite ~28KB，面向 ESP32-S3 |

## 快速开始（实时 Demo）

```powershell
conda activate pytorch12
cd "E:\桌面\Anger detection"   # 或你的克隆路径
pip install -r requirements.txt
streamlit run demo/realtime_emotion_demo.py --server.port 8503
```

浏览器打开：http://127.0.0.1:8503  

- **▶ 启动实时检测**：后台持续采麦，约每 0.5s 推理一次  
- 侧边栏可调触发/解除阈值、分析窗口  

## 仓库结构

```text
Anger detection/
├─ demo/
│  ├─ realtime_emotion_demo.py   # Streamlit UI
│  └─ live_monitor.py            # 后台麦流 + 推理线程
├─ scripts/
│  ├─ common.py / prepare_data*.py / train_torch.py
│  ├─ finetune_casia.py          # CASIA(+CNEV) 微调
│  ├─ export_tflite_torch.py / evaluate_tflite.py
│  └─ predict_wav.py / test_mic.py
├─ models/                       # 权重（ONNX / PT / TFLite）
├─ data/processed/               # 特征需本地生成（gitignore）
├─ output/                       # 训练/微调指标 JSON
├─ requirements.txt
├─ VERSION                       # 当前 0.1.0-v0
└─ run_finetune_and_demo.ps1
```

## 模型文件（已纳入仓库）

| 文件 | 用途 |
|------|------|
| `models/dscnn_casia_deploy.onnx` | **Demo 默认**（CASIA 微调部署） |
| `models/dscnn_casia.onnx` / `*.pt` | 留一说话人微调产物 |
| `models/dscnn.onnx` / `dscnn_torch_best.pt` | ESD 原训练 |
| `models/anger_detection_model_int8.tflite` | ESP32 量化模型 |

## 评测摘要（v0）

- ESD holdout（原训练）：Acc ≈ **88.7%**（五类）  
- CASIA 官方 holdout · **愤怒二分类**（与 TinyCNN/SenseVoice 同集）：  
  - Anger Detection：**Acc 96.7%** · F1 0.90 · **~2 ms**  
  - SenseVoice：Acc 95.0% · ~800 ms  
  - TinyCNN：Acc 87.5% · ~5 ms  

详细对比见本地 `E:\桌面\test\eval_results\four_model_bench.json`（不在本仓库）。

## 训练 / 微调

```powershell
conda activate pytorch12
# 下载并准备中文 ESD 特征（体积大，需自行下载）
python scripts/download_chinese_data.py
python scripts/prepare_data_chinese.py
python scripts/train_torch.py

# CASIA 微调（需本机有 CASIA 路径，见 scripts/finetune_casia.py）
python scripts/finetune_casia.py --device cpu --epochs 35
```

## 分支约定

| 分支 / 标签 | 含义 |
|-------------|------|
| `main` + tag `v0` | v0 基线快照 |
| `v1` | **当前迭代分支**（在此开发） |

## 许可与数据

训练数据来源含 ESD 等，请遵守各数据集许可（如 CC-BY-NC）。本仓库代码与模型权重按项目需要自行约定使用范围。
