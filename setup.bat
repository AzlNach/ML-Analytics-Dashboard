@echo off
echo ========================================
echo   ML Analytics Dashboard Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

echo [1/5] Checking Python version...
python --version

REM Create virtual environment
echo.
echo [2/5] Creating Python virtual environment...
if exist venv311 (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q venv311
)
python -m venv venv311

REM Activate virtual environment and install dependencies
echo.
echo [3/5] Installing Python dependencies...
call venv311\Scripts\activate.bat
python -m pip install --upgrade pip
cd backend
pip install -r requirements.txt
cd ..

REM Check if Node.js is installed
echo.
echo [4/5] Checking Node.js...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 16+ from https://nodejs.org
    pause
    exit /b 1
)

node --version

REM Install Node.js dependencies
echo.
echo [5/5] Installing Node.js dependencies...
npm install

REM Create necessary directories
echo.
echo Creating necessary directories...
if not exist backend\models mkdir backend\models
if not exist backend\uploads mkdir backend\uploads
if not exist backend\data mkdir backend\data
if not exist backend\data\cleaned mkdir backend\data\cleaned
if not exist backend\data\reports mkdir backend\data\reports
if not exist logs mkdir logs

echo.
echo ========================================
echo   Setup Complete! 
echo ========================================
echo.
echo To start the application:
echo.
echo 1. Backend (Terminal 1):
echo    venv311\Scripts\activate
echo    cd backend
echo    python app.py
echo.
echo 2. Frontend (Terminal 2):
echo    npm start
echo.
echo Access the application at:
echo Frontend: http://localhost:3000
echo Backend:  http://localhost:5000
echo.
pause
