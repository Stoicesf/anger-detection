@echo off
cd /d "%~dp0..\.."
"E:\ANACONDA\envs\pytorch12\python.exe" "scripts\data\retry_esd.py" > "artifacts\reports\retry_esd_log.txt" 2>&1
