@echo off
setlocal
cd /d "%~dp0"
call :find_python
if errorlevel 1 goto missing_python
if not exist .venv-win\Scripts\python.exe %FR_PYTHON_CMD% -m venv .venv-win
if errorlevel 1 goto failed
.venv-win\Scripts\python.exe -c "import sys,struct; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,13) and struct.calcsize('P') == 8 else 1)"
if errorlevel 1 goto bad_venv
.venv-win\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto failed
.venv-win\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
.venv-win\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv-win\Scripts\python.exe scripts\fetch_model.py
if errorlevel 1 goto failed
.venv-win\Scripts\python.exe -m pip check
if errorlevel 1 goto failed
set YOLO_CONFIG_DIR=%CD%\data\yolo
echo Open http://localhost:8765 in your browser after the server starts.
.venv-win\Scripts\python.exe -m floorreview.app
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Setup failed. Keep this window open and share the error shown above.
pause
exit /b 1

:find_python
for %%V in (3.12 3.11 3.10) do (
    py -%%V -c "import struct,sys; sys.exit(0 if struct.calcsize('P') == 8 else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "FR_PYTHON_CMD=py -%%V"
        echo Using 64-bit Python %%V through the Windows launcher.
        exit /b 0
    )
)
python -c "import sys,struct; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,13) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if errorlevel 1 exit /b 1
set "FR_PYTHON_CMD=python"
python --version
exit /b 0

:missing_python
echo.
echo No compatible 64-bit Python installation was found.
echo Install Python 3.12 Windows installer (64-bit) from:
echo https://www.python.org/downloads/release/python-31210/
echo Enable the Python launcher and Add python.exe to PATH, then run START.bat again.
echo Your other Python versions can remain installed.
echo.
echo Current default Python:
python --version
echo Python launcher installations:
py -0p
goto failed

:bad_venv
echo The .venv-win folder uses an unsupported Python version.
echo Rename that folder, then run START.bat again. Do not remove the data folder.
goto failed
