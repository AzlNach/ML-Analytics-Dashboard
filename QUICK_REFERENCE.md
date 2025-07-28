# 📋 Quick Reference - ML Analytics Dashboard

## 🚀 Installation Commands

### One-Line Setup
```bash
# Windows (Command Prompt)
setup.bat

# Windows (PowerShell)
.\setup.ps1

# Linux/Mac
chmod +x setup.sh && ./setup.sh
```

### Manual Setup
```bash
# 1. Clone
git clone https://github.com/AzlNach/ML-Analytics-Dashboard.git
cd ML-Analytics-Dashboard

# 2. Python Environment
python -m venv venv311
venv311\Scripts\activate          # Windows
source venv311/bin/activate       # Linux/Mac
cd backend && pip install -r requirements.txt && cd ..

# 3. Node.js Dependencies  
npm install

# 4. Start Servers
# Terminal 1: venv311\Scripts\activate && cd backend && python app.py
# Terminal 2: npm start
```

---

## 🔧 Daily Development

### Start Application
```bash
# Terminal 1 - Backend
venv311\Scripts\activate     # Windows
source venv311/bin/activate  # Linux/Mac
cd backend
python app.py

# Terminal 2 - Frontend
npm start
```

### Access URLs
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **API Health**: http://localhost:5000/health

---

## 📁 Important Files

### ✅ Include in Git
```
src/                 # React source code
backend/app.py       # Flask application
package.json         # Node dependencies
requirements.txt     # Python dependencies
docs/               # Documentation
training_data/      # Sample datasets
README.md           # Project info
```

### ❌ Exclude from Git
```
venv311/            # Virtual environment
node_modules/       # Node dependencies  
backend/models/     # Trained models
.env               # Environment secrets
__pycache__/       # Python cache
*.log              # Log files
```

---

## 🐛 Troubleshooting

### Common Issues

**Error**: `python: command not found`
```bash
# Use python3 or full path
python3 -m venv venv311
```

**Error**: `Permission denied` (Windows)
```powershell
# Run as administrator or enable scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Error**: `Module not found`
```bash
# Reinstall dependencies
pip install -r backend/requirements.txt
npm install
```

**Error**: `Port already in use`
```bash
# Kill process on port
# Windows: netstat -ano | findstr :5000
# Linux/Mac: lsof -ti:5000 | xargs kill
```

### Reset Environment
```bash
# Clean installation
rm -rf venv311 node_modules
python -m venv venv311
venv311\Scripts\activate
pip install -r backend/requirements.txt
npm install
```

---

## 🔍 Useful Commands

### Check Installation
```bash
# Python packages
pip list

# Node packages  
npm list --depth=0

# Test imports
python -c "import flask, pandas, sklearn; print('OK')"

# Check running processes
netstat -ano | findstr :3000
netstat -ano | findstr :5000
```

### Project Info
```bash
# Repository size
git count-objects -vH

# Files being tracked
git ls-files --cached

# Check ignored files
git status --ignored
```

---

## 🎯 VS Code Integration

### Recommended Extensions
- Python
- ES7+ React/Redux/React-Native snippets
- Prettier - Code formatter
- GitLens

### Built-in Tasks
1. Press `Ctrl+Shift+P`
2. Type "Tasks: Run Task"
3. Select "Start Backend Server"

### Python Interpreter
1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose `venv311\Scripts\python.exe`

---

## 📊 Project Structure

```
ML-Analytics-Dashboard/
├── 🚀 setup.bat/.sh/.ps1    # Setup scripts
├── 📋 SETUP_GUIDE.md        # Detailed setup
├── 📁 GIT_UPLOAD_GUIDE.md   # Git guidelines
├── 📦 package.json          # Node config
├── 🐍 backend/
│   ├── app.py              # Flask app
│   ├── requirements.txt    # Python deps
│   └── .env.example       # Config template
├── ⚛️ src/                  # React code
├── 📊 training_data/        # Sample data
├── 📚 docs/                # Documentation
└── 🚫 .gitignore           # Git ignore
```

---

## 📞 Getting Help

### Documentation
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Detailed setup
- [GIT_UPLOAD_GUIDE.md](GIT_UPLOAD_GUIDE.md) - File management
- [docs/](docs/) - Complete documentation

### Support
- **GitHub Issues**: [Create Issue](https://github.com/AzlNach/ML-Analytics-Dashboard/issues)
- **Discussions**: Check repository discussions

### Before Asking for Help
1. Check error messages carefully
2. Verify Python 3.11+ and Node.js 16+ installed
3. Try resetting environment (commands above)
4. Check if ports 3000/5000 are available

---

*Happy coding! 🎉*
