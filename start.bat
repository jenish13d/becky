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
  echo Becky stopped with an error. Read the message above.
  echo Details are also saved in: %USERPROFILE%\.becky\last-error.txt  - send me that file.
  echo.
)
pause
