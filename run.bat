@echo off

title Mini AI Agent

echo ========================================
echo        MINI AI AGENT
echo ========================================
echo.

cd /d "%~dp0"

echo Checking virtual environment...
echo.

if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    echo.
    echo Please create it first using:
    echo python -m venv venv
    echo.
    pause
    exit /b 1
)

echo Virtual environment found.
echo.

echo Starting Mini AI Agent...
echo.

venv\Scripts\python.exe -m streamlit run app.py

echo.
echo ========================================
echo        Mini AI Agent stopped
echo ========================================
echo.

pause