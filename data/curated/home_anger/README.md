# 家庭辅导场景专项数据（V2 Day4）

目录约定：

```
data/curated/home_anger/
  positive/
    mild/
    medium/
    strong/
  negative/
    normal/
    teaching/
    explaining/
    excited/
    loud/
```

标签：

- `positive/**/*.wav` → anger=1
- `negative/**/*.wav` → anger=0

目标比例：positive ≈ 40%，negative ≈ 60%。

训练接入：

```powershell
python scripts/train/train_anger_binary.py --use-home --rebuild-cache --device cuda
```

当前可为空；放入 wav 后再 `--use-home`。
