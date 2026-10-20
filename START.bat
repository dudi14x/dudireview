@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe scripts\fetch_model.py
if errorlevel 1 exit /b 1
set YOLO_CONFIG_DIR=%CD%\data\yolo
.venv\Scripts\python.exe -m floorreview.app
pause
