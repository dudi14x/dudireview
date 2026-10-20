@echo off
setlocal
cd /d "%~dp0"
python -c "import sys; assert (3,10) <= sys.version_info[:2] < (3,13), 'Install 64-bit Python 3.10, 3.11 or 3.12 and add it to PATH.'"
if errorlevel 1 goto failed
if not exist .venv\Scripts\python.exe python -m venv .venv
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe scripts\fetch_model.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip check
if errorlevel 1 goto failed
set YOLO_CONFIG_DIR=%CD%\data\yolo
echo Open http://localhost:8765 in your browser after the server starts.
.venv\Scripts\python.exe -m floorreview.app
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Setup failed. Keep this window open and share the error shown above.
pause
exit /b 1
