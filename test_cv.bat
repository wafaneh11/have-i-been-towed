@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Run start_towtrace.bat once first so the environment is installed.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python backend\cv\tow_cv_only.py demo\tow_test.mp4 --debug
pause
