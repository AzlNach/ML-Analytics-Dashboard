# ML Analytics Dashboard Setup Script
# PowerShell version for Windows 10/11

param(
    [switch]$SkipPython,
    [switch]$SkipNode,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
ML Analytics Dashboard Setup Script

Usage:
    .\setup.ps1                 # Full setup
    .\setup.ps1 -SkipPython     # Skip Python setup
    .\setup.ps1 -SkipNode       # Skip Node.js setup
    .\setup.ps1 -Help           # Show this help

Requirements:
    - Python 3.11+
    - Node.js 16+
    - Git
"@
    exit 0
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   ML Analytics Dashboard Setup" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to check command existence
function Test-Command {
    param($Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# Function to create directory if not exists
function New-DirectoryIfNotExists {
    param($Path)
    if (!(Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        Write-Host "✓ Created directory: $Path" -ForegroundColor Green
    }
}

try {
    # Check Python
    if (!$SkipPython) {
        Write-Host "[1/5] Checking Python..." -ForegroundColor Blue
        if (!(Test-Command "python")) {
            throw "Python is not installed or not in PATH. Please install Python 3.11+ from https://python.org"
        }
        
        $pythonVersion = python --version 2>&1
        Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
        
        # Create virtual environment
        Write-Host "`n[2/5] Setting up Python environment..." -ForegroundColor Blue
        if (Test-Path "venv311") {
            Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
            Remove-Item "venv311" -Recurse -Force
        }
        
        python -m venv venv311
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create virtual environment"
        }
        Write-Host "✓ Virtual environment created" -ForegroundColor Green
        
        # Activate and install dependencies
        Write-Host "Installing Python dependencies..." -ForegroundColor Blue
        & "venv311\Scripts\Activate.ps1"
        python -m pip install --upgrade pip --quiet
        
        Push-Location "backend"
        pip install -r requirements.txt --quiet
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install Python dependencies"
        }
        Pop-Location
        Write-Host "✓ Python dependencies installed" -ForegroundColor Green
    }
    
    # Check Node.js
    if (!$SkipNode) {
        Write-Host "`n[3/5] Checking Node.js..." -ForegroundColor Blue
        if (!(Test-Command "node")) {
            throw "Node.js is not installed or not in PATH. Please install Node.js 16+ from https://nodejs.org"
        }
        
        $nodeVersion = node --version
        Write-Host "✓ Found Node.js: $nodeVersion" -ForegroundColor Green
        
        # Install Node.js dependencies
        Write-Host "`n[4/5] Installing Node.js dependencies..." -ForegroundColor Blue
        npm install --silent
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install Node.js dependencies"
        }
        Write-Host "✓ Node.js dependencies installed" -ForegroundColor Green
    }
    
    # Create necessary directories
    Write-Host "`n[5/5] Creating project directories..." -ForegroundColor Blue
    $directories = @(
        "backend\models",
        "backend\uploads", 
        "backend\data\cleaned",
        "backend\data\reports",
        "logs"
    )
    
    foreach ($dir in $directories) {
        New-DirectoryIfNotExists $dir
    }
    
    # Success message
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "   Setup Complete! ✓" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To start the application:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Backend (Terminal 1):" -ForegroundColor White
    Write-Host "   venv311\Scripts\Activate.ps1" -ForegroundColor Gray
    Write-Host "   cd backend" -ForegroundColor Gray
    Write-Host "   python app.py" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Frontend (Terminal 2):" -ForegroundColor White
    Write-Host "   npm start" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Access the application at:" -ForegroundColor Yellow
    Write-Host "Frontend: " -NoNewline -ForegroundColor White
    Write-Host "http://localhost:3000" -ForegroundColor Cyan
    Write-Host "Backend:  " -NoNewline -ForegroundColor White
    Write-Host "http://localhost:5000" -ForegroundColor Cyan
    Write-Host ""
    
} catch {
    Write-Host "`nERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Setup failed. Please check the error above." -ForegroundColor Red
    exit 1
}

Write-Host "Press any key to continue..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
