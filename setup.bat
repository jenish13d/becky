@echo off
cd /d "%~dp0"
echo ================================================
echo  Installing Becky (one time, a few minutes)
echo ================================================
python --version >nul 2>&1
if errorlevel 1 (
  echo.
  echo Python is not installed, or is not on your PATH.
  echo Install it from https://www.python.org/downloads/ and TICK "Add python.exe to PATH"
  echo on the first screen. Then double-click setup.bat again.
  echo.
  pause
  exit /b 1
)
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
python -m pip install -r requirements.txt -e .
if errorlevel 1 goto failed
echo.
echo ================================================
echo  Checking that everything works
echo ================================================
python check_setup.py
if errorlevel 1 goto failed
echo.
echo Setup finished. Now double-click start.bat to talk to Becky.
echo (This window closes when you press a key.)
pause
exit /b 0
:failed
echo.
echo Something went wrong. Read the lines marked FAIL or the error above.
echo The file check-report.txt in this folder has the details: send it to me.
echo.
pause
exit /b 1
