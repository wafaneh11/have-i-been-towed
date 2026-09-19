@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
    py -3 start_towtrace.py
    if not errorlevel 1 exit /b 0
    goto :error
)

where python >nul 2>nul
if errorlevel 1 goto :no_python

python start_towtrace.py
if not errorlevel 1 exit /b 0
goto :error

:no_python
echo.
echo Python 3 was not found.
echo Install Python from https://www.python.org/downloads/ and check
echo "Add python.exe to PATH" during setup, then run this file again.
pause
exit /b 1

:error
echo.
echo TowTrace could not start. Read the message above, then try again.
pause
exit /b 1
