@echo off
cd /d "%~dp0"
python check_setup.py --quiet
if errorlevel 1 (
  echo.
  echo Becky cannot start yet. Fix the FAIL lines above, or double-click setup.bat.
  echo The file check-report.txt in this folder has the details: send it to me.
  echo.
  pause
  exit /b 1
)
python -m ouroboros chat
if errorlevel 1 (
  echo.
  echo Becky stopped. The message above says why.
  echo Most of the time it is something simple, like a passphrase that was too short. Just run this again.
  echo Only if it says "unexpected error" is a file saved at %USERPROFILE%\.becky\last-error.txt - send me that.
  echo.
)
pause
