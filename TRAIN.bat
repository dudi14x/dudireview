@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv-win\Scripts\python.exe" (
  echo Run START.bat once to install the application first.
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Usage: TRAIN.bat path\to\manifest.json
  echo See docs\NEURAL_TRAINING.md for the manifest format.
  pause
  exit /b 1
)
.venv-win\Scripts\python.exe -m floorreview.dataset --manifest "%~1" --output datasets\reviewed
if errorlevel 1 goto failed
.venv-win\Scripts\python.exe -m floorreview.neural --data datasets\reviewed --output models\reviewed-plan-unet.pt --epochs 60
if errorlevel 1 goto failed
echo Training complete. Metrics: models\reviewed-plan-unet.metrics.json
echo To use the checkpoint, add this line to .env.local and restart START.bat:
echo FLOORREVIEW_MODEL=models/reviewed-plan-unet.pt
pause
exit /b 0
:failed
echo Training failed. Read the error above. Existing models have not been activated.
pause
exit /b 1
