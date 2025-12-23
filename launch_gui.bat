@echo off
REM Windows batch file to launch the Trading GUI with auto-dependency setup
REM Run this file to open the GUI (dependencies auto-install if missing)

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo Trading GUI - Auto Setup Launcher
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    py -3 --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python not found. Please install Python 3.10+ or add it to PATH.
        pause
        exit /b 1
    )
    set PYTHON=py -3
) else (
    set PYTHON=python
)

echo Python found: %PYTHON%
echo.

REM Run the auto-setup launcher
%PYTHON% launch_gui.py
if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch GUI.
    pause
    exit /b 1
)

endlocal
