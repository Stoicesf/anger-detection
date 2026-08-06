# 迭代路线图（仅中文语音）

> 约束：**只检测中文语音情绪 / 愤怒事件**。不引入英文数据与英文评测。

## 状态

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | CASIA 微调 Demo + ONNX | 完成（tag `v0`） |
| 1 | EMA + 连续触发 + 状态机 | 完成（`demo/decision.py`） |
| 2 | Hard Negative（中文）+ 增强微调 | 完成（本提交） |
| 3 | INT8 / 蒸馏 | 待做 |
| 4 | FastAPI 多端部署 | 待做 |

## Phase 2 产物

- `dataset/hard_negative/`：中文困难负样本（兴奋/惊讶/大声中性/语速扰动等）
- `scripts/build_hard_negative.py`
- `scripts/finetune_hard_negative.py` → 更新 `models/dscnn_casia_deploy.onnx`
- `scripts/eval_realtime_metrics.py`：FAR / 延迟（zh-CN）

## 推荐运行

```powershell
conda activate pytorch12
cd "E:\桌面\Anger detection"

# 构建中文 hard negative
python scripts/build_hard_negative.py --clean

# 微调（仅中文 CASIA/CNEV/HN）
python scripts/finetune_hard_negative.py --device cpu --epochs 20

# 实时指标
python scripts/eval_realtime_metrics.py --long-negative-minutes 20 --threshold 0.65 --required-hits 2

# Demo
streamlit run demo/realtime_emotion_demo.py --server.port 8503
```

Demo 侧边栏提供三档预设：**低误报 / 均衡 / 高召回**。
