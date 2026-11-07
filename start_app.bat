@echo off
REM Double-click to start the vehicle recognition website at http://localhost:7860
cd /d "%~dp0"
echo Loading models, please wait about 20 seconds...
start "" cmd /c "timeout /t 25 >nul && start http://localhost:7860"
"C:\Users\Kevin\cvh\.venv\Scripts\python.exe" app.py
pause
