@echo off
cd /d "%~dp0"
echo ================================================
echo  Installing Becky (one time, a few minutes)
echo ================================================
python --version >nul 2>&1
if errorlevel 1 (
  echo.
  echo Python is not installed. Install it from https://www.python.org/downloads/
  echo and TICK "Add python.exe to PATH" on the first screen. Then run this again.
  pause
  exit /b 1
)
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
python -m pip install -r requirements.txt -e .
if errorlevel 1 goto failed
echo.
echo Done. Double-click start.bat to talk to Becky.
pause
exit /b 0
:failed
echo.
echo Something failed above. Copy the red/error text and send it to me.
pause
exit /b 1
