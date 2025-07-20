@echo off
echo ========================================
echo     ML Analytics Dashboard Starter
echo ========================================
echo.

echo [1/6] Checking Python and Node.js installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

echo [2/6] Setting up backend directory...
cd backend
if not exist models mkdir models

echo [3/6] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies
    echo Please check your Python installation and requirements.txt
    pause
    exit /b 1
)

echo [4/6] Starting Flask backend server...
start "Flask Backend" cmd /k "echo Starting Flask API Server... && python app.py"

echo [5/6] Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

echo [6/6] Setting up frontend...
cd ..
echo Installing Node.js dependencies...
npm install
if errorlevel 1 (
    echo ERROR: Failed to install Node.js dependencies
    echo Please check your Node.js installation and package.json
    pause
    exit /b 1
)

echo.
echo ========================================
echo        Starting React Frontend
echo ========================================
echo.
echo Both servers will be running:
echo Frontend: http://localhost:3000
echo Backend:  http://localhost:5000
echo.
echo Press Ctrl+C to stop the servers
echo.

npm start
