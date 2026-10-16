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
  echo Becky stopped. The message above says why.
  echo Most of the time it is something simple, like a passphrase that was too short. Just run this again.
  echo Only if it says "unexpected error" is a file saved at %USERPROFILE%\.becky\last-error.txt - send me that.
)
pause
