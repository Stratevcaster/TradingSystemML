@echo off
REM Bootstrap script wrapper for Windows
REM Run this to set up the trading environment (conda or venv+pip)

cd /d "%~dp0"
echo.
echo ============================================================
echo Trading System ML - Environment Setup Bootstrap
echo ============================================================
echo.

py -3 bootstrap.py
if errorlevel 1 (
    echo.
    echo Setup failed. Press any key to exit...
    pause
    exit /b 1
)

echo.
echo Setup complete! Press any key to close this window...
pause
