@echo off
cd /d "%~dp0..\.."
"E:\ANACONDA\envs\pytorch12\python.exe" "scripts\train\train_torch.py" > "artifacts\reports\train_torch_log.txt" 2>&1
