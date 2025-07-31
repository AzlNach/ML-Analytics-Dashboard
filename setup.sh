#!/bin/bash

echo "========================================"
echo "   ML Analytics Dashboard Setup"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    echo "Please install Python 3.11+ from https://python.org"
    exit 1
fi

echo "[1/5] Checking Python version..."
python3 --version

# Create virtual environment
echo
echo "[2/5] Creating Python virtual environment..."
if [ -d "venv311" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf venv311
fi

# Try Python 3.11 first, then fallback to python3
if command -v python3.11 &> /dev/null; then
    echo "Using Python 3.11..."
    python3.11 -m venv venv311
elif command -v python3 &> /dev/null; then
    echo "Using python3..."
    python3 -m venv venv311
else
    echo "ERROR: Python 3.11+ is required but not found"
    echo "Please install Python 3.11+ from https://python.org"
    exit 1
fi

# Activate virtual environment and install dependencies
echo
echo "[3/5] Installing Python dependencies..."
source venv311/bin/activate
python -m pip install --upgrade pip
cd backend
pip install -r requirements.txt
cd ..

# Check if Node.js is installed
echo
echo "[4/5] Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed"
    echo "Please install Node.js 16+ from https://nodejs.org"
    exit 1
fi

node --version

# Install Node.js dependencies
echo
echo "[5/5] Installing Node.js dependencies..."
npm install

# Create necessary directories
echo
echo "Creating necessary directories..."
mkdir -p backend/models
mkdir -p backend/uploads
mkdir -p backend/data/cleaned
mkdir -p backend/data/reports
mkdir -p logs

echo
echo "========================================"
echo "   Setup Complete!"
echo "========================================"
echo
echo "To start the application:"
echo
echo "1. Backend (Terminal 1):"
echo "   source venv311/bin/activate"
echo "   cd backend"
echo "   python app.py"
echo
echo "2. Frontend (Terminal 2):"
echo "   npm start"
echo
echo "Access the application at:"
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:5000"
echo
