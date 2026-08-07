# Fine-tune DS-CNN on CASIA then launch realtime demo
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$py = "E:\ANACONDA\envs\pytorch12\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}

Write-Host "==> Editable install ..." -ForegroundColor Cyan
& $py -m pip install -e . -q
if ($LASTEXITCODE -ne 0) { throw "pip install -e . failed" }

Write-Host "==> Fine-tune on CASIA ..." -ForegroundColor Cyan
& $py scripts\train\finetune_casia.py --device cpu --epochs 35 --test-speaker ZhaoZuoxiang
if ($LASTEXITCODE -ne 0) { throw "finetune failed" }

Write-Host "==> Launch demo on http://127.0.0.1:8503" -ForegroundColor Cyan
& $py -m streamlit run apps\streamlit_demo\realtime_emotion_demo.py --server.port 8503 --server.headless true
