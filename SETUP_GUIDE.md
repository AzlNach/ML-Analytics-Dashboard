# 🚀 Quick Setup Guide - ML Analytics Dashboard

## 📋 Prerequisites
- **Python 3.11+** (Required)
- **Node.js 16+** (Required)
- **Git** (Required)

---

## ⚡ Quick Setup (5 Minutes)

### Step 1: Clone Repository
```bash
git clone https://github.com/AzlNach/ML-Analytics-Dashboard.git
cd ML-Analytics-Dashboard
```

### Step 2: Backend Setup (Python)
```bash
# Create virtual environment
python -m venv venv311

# Activate virtual environment
# Windows:
venv311\Scripts\activate

# Linux/Mac:
source venv311/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt
```

### Step 3: Frontend Setup (React)
```bash
# Return to root directory
cd ..

# Install dependencies
npm install
```

### Step 4: Start Both Servers

**Terminal 1 - Backend:**
```bash
# Activate environment (if not active)
venv311\Scripts\activate

# Start backend
cd backend
python app.py
```

**Terminal 2 - Frontend:**
```bash
# Start frontend
npm start
```

### Step 5: Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000

---

## 🔧 Environment Configuration

### Backend Environment (.env)
Create `backend/.env` file (optional):
```env
FLASK_ENV=development
FLASK_DEBUG=1
FLASK_RUN_PORT=5000
```

### VS Code Setup (Recommended)
1. Open project in VS Code
2. Install Python extension
3. Select Python interpreter: `venv311\Scripts\python.exe`
4. Use built-in tasks:
   - `Ctrl+Shift+P` → "Tasks: Run Task" → "Start Backend Server"

---

## 🐍 Python Environment Details

### Verify Installation
```bash
# Check Python version
python --version  # Should be 3.11+

# Check installed packages
pip list

# Test imports
python -c "import flask, pandas, sklearn, numpy; print('All packages installed successfully!')"
```

### Common Issues & Solutions

**Issue**: `python` command not found
```bash
# Solution: Use python3 or full path
python3 -m venv venv311
# or
C:\Python311\python.exe -m venv venv311
```

**Issue**: Permission denied (Windows)
```bash
# Solution: Run as administrator or use PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Issue**: Package installation fails
```bash
# Solution: Upgrade pip first
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📦 Dependencies Overview

### Backend (Python)
- **Flask**: Web framework
- **Flask-CORS**: Cross-origin requests
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning
- **numpy**: Numerical computing
- **ydata-profiling**: Data analysis

### Frontend (React)
- **React 18**: UI framework
- **lucide-react**: Icons
- **recharts**: Data visualization
- **mathjs**: Mathematical operations

---

## 🗂️ Project Structure After Setup

```
ML-Analytics-Dashboard/
├── venv311/                    # Python virtual environment (created)
├── backend/
│   ├── app.py                 # Flask application
│   ├── requirements.txt       # Python dependencies
│   ├── models/               # Trained ML models (created)
│   └── .env                  # Environment config (optional)
├── src/                      # React components
├── public/                   # Static files
├── training_data/            # Sample datasets
├── node_modules/             # Node.js dependencies (created)
└── package.json              # Node.js config
```

---

## 🔄 Development Workflow

### Daily Development
1. **Activate Python environment**:
   ```bash
   venv311\Scripts\activate
   ```

2. **Start both servers**:
   ```bash
   # Terminal 1 - Backend
   cd backend && python app.py
   
   # Terminal 2 - Frontend  
   npm start
   ```

3. **Make changes and test**

### Adding New Dependencies

**Python packages**:
```bash
# Activate environment first
venv311\Scripts\activate

# Install package
pip install package-name

# Update requirements
pip freeze > backend/requirements.txt
```

**Node.js packages**:
```bash
npm install package-name
```

---

## 🚀 Production Deployment

### Build Frontend
```bash
npm run build
```

### Environment Variables
Set these for production:
```env
FLASK_ENV=production
FLASK_DEBUG=0
NODE_ENV=production
```

---

## 📞 Need Help?

### Troubleshooting Checklist
- [ ] Python 3.11+ installed?
- [ ] Virtual environment activated?
- [ ] All dependencies installed?
- [ ] Both servers running?
- [ ] Ports 3000 and 5000 available?

### Common Commands
```bash
# Reset environment
rm -rf venv311 node_modules
python -m venv venv311
venv311\Scripts\activate
pip install -r backend/requirements.txt
npm install

# Check running processes
netstat -ano | findstr :3000
netstat -ano | findstr :5000
```

### Contact
- **GitHub Issues**: [Create an issue](https://github.com/AzlNach/ML-Analytics-Dashboard/issues)
- **Documentation**: Check `docs/` folder for detailed guides

---

*Happy coding! 🎉*
