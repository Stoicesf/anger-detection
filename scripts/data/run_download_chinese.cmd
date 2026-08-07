@echo off
cd /d "%~dp0..\.."
"E:\ANACONDA\envs\pytorch12\python.exe" "scripts\data\download_chinese_data.py" --only esd > "artifacts\reports\download_chinese_log.txt" 2>&1
