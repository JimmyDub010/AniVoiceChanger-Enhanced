@echo off
setlocal
cd /d "%~dp0"
py -3.10 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Run Launch-Enhanced.cmd.
pause
exit /b 0
:failed
echo Setup failed. Python 3.10 x64 and internet access are required.
pause
exit /b 1
