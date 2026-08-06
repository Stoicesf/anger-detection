# TinyCNN Baseline（愤怒二分类）

轻量 CNN 基线，用于与主工程 DS-CNN 对比。输入为 2s / 16kHz MFCC `(1,40,T)`，参数量约 **6K**。

## 目录

```text
baselines/tinycnn/
├─ src/                 # 训练 / 推理 / Gradio Demo / 导出
├─ checkpoints/         # best_model.pth + 性能报告
└─ requirements.txt
```

**不包含** CASIA 原始 wav（体积大，请自备）。数据准备脚本默认读取本机 CASIA 路径。

## 快速评测 / Demo

```powershell
conda activate pytorch12
cd baselines/tinycnn

# Gradio 实时演示（默认端口 7861）
python src/demo.py

# 用已有权重生成报告（需先准备 dataset/）
python src/report.py
```

## 训练（可选）

```powershell
# 1) 将 CASIA 整理为 angry / non_angry，并 8:1:1 划分
python src/prepare_data.py --casia_root <你的CASIA路径> --out_root dataset

# 2) 训练
python src/train.py --epochs 80 --batch_size 128 --augment

# 3) 测试 / 导出 ONNX
python src/test.py
python src/export.py --make_calib
```

## CASIA holdout 摘要（本仓库 checkpoint）

| 指标 | Val | Test |
|------|-----|------|
| Acc | 90.0% | 84.2% |
| F1 | 0.769 | 0.655 |
| Recall | 100% | 90.0% |

- 最优 epoch：**61** · GPU：RTX 4060 · 训练约 24.5 min  
- 详细报告：`checkpoints/tinycnn-performance-report.md`  
- Demo 默认阈值 **0.65**，待机/静音判为**非愤怒**

## 与主模型对比（同集愤怒二分类，见主 README）

| 模型 | Acc | 备注 |
|------|-----|------|
| DS-CNN（本仓库 Demo） | ~96.7% | CASIA 微调 ONNX |
| TinyCNN（本基线） | ~84–88% | 更小、适合教学/边缘对比 |
