@echo off
setlocal
set "SPEED="
if /i "%~1"=="fast" set "SPEED=--fast"
where py >nul 2>nul
if not errorlevel 1 (
  py -3 "%~dp0tools\run_xemu.py" %SPEED%
) else (
  python "%~dp0tools\run_xemu.py" %SPEED%
)
if errorlevel 1 pause
