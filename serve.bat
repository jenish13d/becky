@echo off
cd /d "%~dp0"
python check_setup.py --quiet
if errorlevel 1 (
  echo.
  echo Becky cannot start yet. Fix the FAIL lines above, or double-click setup.bat.
  echo.
  pause
  exit /b 1
)
echo Starting Becky online. Keep this window open. Press Ctrl+C to stop.
echo.
python -m ouroboros serve
if errorlevel 1 (
  echo.
  echo Becky stopped with an error. Read the message above.
  echo Details are also saved in: %USERPROFILE%\.becky\last-error.txt  - send me that file.
)
pause
