# 迭代路线图（仅中文语音）

> 约束：**只检测中文语音情绪 / 愤怒事件**。不引入英文数据与英文评测。

## 状态

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | CASIA 微调 Demo + ONNX | 完成（tag `v0`） |
| 1 | EMA + 连续触发 + 状态机 | 完成（`src/anger_detection/decision/`） |
| 2 | Hard Negative（中文）+ 增强微调 | 完成（baseline 保留） |
| **V2** | **Binary anger + Focal + tail crop + SE(可选) + src 布局** | **进行中（分支 `v2`）** |
| 3 | INT8 / 蒸馏 | 待做 |
| 4 | FastAPI 多端部署 | 待做 |

## V2 目标错位修复

| 问题 | V1 | V2 |
|------|----|----|
| 训练目标 | 5 类 softmax | binary sigmoid |
| HN 映射 | excited→happy 等 | 全部 → 0 |
| Crop | center+random | center+random+**tail** |
| Loss | CE + 降权 angry | Focal BCE α=0.75 |
| 采样 | shuffle | WeightedRandomSampler |

## V2 产物

```
src/anger_detection/models/model_anger.py
src/anger_detection/losses/focal.py
scripts/train/train_anger_binary.py
scripts/eval/eval_anger.py
artifacts/models/dscnn_anger_v2.pt
artifacts/models/dscnn_anger_v2.onnx
data/curated/home_anger/   # Day4 家长辅导数据
```

Baseline 保留：`scripts/train/train_torch.py` / `scripts/train/finetune_hard_negative.py`。

## 推荐运行

```powershell
conda activate pytorch12
cd "E:\桌面\Anger detection\Anger detection"
pip install -e .
pip install -r requirements.txt

# Day1+2: binary baseline
python scripts/train/train_anger_binary.py --device cuda --epochs 40 --rebuild-cache

# Day3: +SE A/B
python scripts/train/train_anger_binary.py --device cuda --epochs 40 --se --tag v2

# 评估
python scripts/eval/eval_anger.py --onnx artifacts/models/dscnn_anger_v2.onnx --realtime

# Demo（优先加载 v2）
streamlit run apps/streamlit_demo/realtime_emotion_demo.py --server.port 8503
```

外部语料根目录可用环境变量覆盖（默认仍指向本机 CASIA/CNEV）：

```powershell
$env:ANGER_CASIA_ROOT = "E:\桌面\test\CASIA"
$env:ANGER_CNEV_ROOT  = "E:\桌面\test\CNEV_Vocalizations\Core Set"
```

## V2 Day3-A（阈值 / hits / 校准）

```powershell
python scripts/eval/eval_anger.py --onnx artifacts/models/dscnn_anger_v2.onnx --grid --realtime --calib-T
```

产物：`artifacts/reports/eval_anger_v2_grid.json` / `.md`

结论摘要：

- 文件级 LOSO 在 thr=0.60 可达 R≥90%（此前 crop 级 R=0.54 被低估）
- 短 clip 实时流：**无法同时** FAR&lt;1/h 且 detect&gt;85%
- 推荐双档：低误报 `thr=0.65 hits=3`；均衡 `thr=0.65 hits=2`
- Temperature 主要是重映射阈值，收益有限
- SE 仍暂缓，优先真实长负样本测 FAR
