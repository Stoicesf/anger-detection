@echo off
cd /d "%~dp0..\.."
"E:\ANACONDA\envs\pytorch12\python.exe" "scripts\train\train.py" > "artifacts\reports\train_log.txt" 2>&1
