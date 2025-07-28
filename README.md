# 🚀 ML Analytics Dashboard

Sebuah aplikasi web full-stack yang memungkinkan Anda mengupload data CSV dan secara otomatis menghasilkan analisis machine learning dengan akurasi tinggi menggunakan backend Python Flask dan visualisasi interaktif React.

## 📑 Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Quick Start](#-quick-start)
- [Alur Kerja Sistem](#-alur-kerja-sistem)
- [Struktur Project](#-struktur-project)
- [Dokumentasi Lengkap](#-dokumentasi-lengkap)
- [Contributing](#-contributing)

---

## ✨ Fitur Utama

### 🖥️ Frontend (React)
- **📊 Data Visualization**: Grafik korelasi, distribusi, dan trend analysis
- **🎯 Interactive Dashboard**: Interface modern dan responsif dengan multiple tabs
- **🧠 Model Training Interface**: Train multiple ML algorithms dengan UI yang user-friendly
- **🔮 Prediction Interface**: Make predictions menggunakan trained models
- **📈 Real-time Analysis**: Visualisasi hasil ML secara langsung
- **💾 Export Results**: Export hasil analisis dalam format JSON
- **🔄 5-Step Workflow**: Alur kerja interaktif dari upload hingga prediksi

### ⚙️ Backend (Python Flask)
- **🤖 Multiple ML Algorithms**: 
  - Decision Tree (Pohon Keputusan)
  - Random Forest (Ensemble Method)
  - Logistic Regression (Regresi Logistik)
  - Support Vector Machine (SVM)
- **🔍 DBSCAN Clustering**: Pengelompokan data dengan scikit-learn
- **⚠️ Anomaly Detection**: Deteksi anomali menggunakan Isolation Forest
- **🎯 Model Management**: Save, load, dan delete trained models
- **📊 Cross-validation**: Validasi silang untuk evaluasi model yang akurat
- **🔮 Prediction API**: Endpoint untuk prediksi real-time dengan confidence scores
- **📈 Statistical Analysis**: Analisis statistik komprehensif menggunakan YData Profiling
- **📝 Model History**: Track semua training sessions dan performance

### 🔬 Machine Learning Features
- **📚 Training Data Management**: Folder khusus untuk dataset training
- **🧠 Pre-trained Models**: Model tersimpan untuk prediksi cepat
- **📊 Model Evaluation**: Accuracy scoring dan performance metrics
- **🛠️ Data Cleaning**: Otomatis handling missing values, duplicates, dan outliers
- **📋 Data Profiling**: Comprehensive data quality assessment

---

## 🛠 Teknologi yang Digunakan

### Frontend
- **React 18**: Modern JavaScript framework
- **Material-UI / Custom CSS**: Styling framework
- **Chart.js / D3.js**: Data visualization
- **Axios**: HTTP client untuk API calls

### Backend
- **Python 3.11**: Core programming language
- **Flask**: Web framework
- **Flask-CORS**: Cross-origin resource sharing
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **YData Profiling**: Automated EDA
- **Joblib**: Model serialization

### Development Tools
- **VS Code**: IDE dengan task automation
- **Git**: Version control
- **npm**: Package manager untuk frontend
- **pip**: Package manager untuk backend

---

## 🚀 Instalasi dan Setup

### Prerequisites
- **Python 3.11.x** (Wajib - untuk kompatibilitas optimal)
- **Node.js 18+** dan npm
- **Git** untuk version control

### ⚡ Quick Setup (Recommended)

#### Automated Setup
```bash
# 1. Clone repository
git clone https://github.com/AzlNach/ML-Analytics-Dashboard.git
cd ML-Analytics-Dashboard

# 2. Run setup script
# Windows:
setup.bat

# Linux/Mac:
chmod +x setup.sh
./setup.sh

# 3. Start application (2 terminals)
# Terminal 1 - Backend:
venv311\Scripts\activate  # Windows
source venv311/bin/activate  # Linux/Mac
cd backend && python app.py

# Terminal 2 - Frontend:
npm start
```

📖 **For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

#### 5. Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000

### Manual Setup (Alternative)

#### Python Environment Setup
```bash
# Create virtual environment
python3.11 -m venv venv311

# Activate environment
# Windows:
venv311\Scripts\activate.bat
# Linux/Mac:
source venv311/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install backend dependencies
cd backend
pip install -r requirements.txt
```

#### Environment Configuration
1. Copy `.env.example` to `.env` (jika ada)
2. Sesuaikan konfigurasi database atau API keys jika diperlukan

---

## 🔄 Alur Kerja Sistem

Sistem menggunakan **5-Step Interactive Workflow** yang terstruktur:

### **Step 1: Upload Data** 📤
**Aksi Pengguna:**
- Upload file dataset dalam format `.csv`
- File akan diproses dan dimuat ke dalam sistem

**Proses Sistem:**
- Parsing CSV dengan handling untuk quoted values dan special characters
- Validasi struktur data dan deteksi tipe kolom
- Storage dalam state management

**Output:**
- Dataset mentah tersimpan
- Headers kolom tersedia
- Auto-progress ke Step 2

---

### **Step 2: Analisis & Pemahaman Data** 📊
**Trigger:** Otomatis setelah upload berhasil

**Proses Sistem:**
- **Profil Data Awal**: Jumlah baris, kolom, tipe data, ukuran memori
- **Statistik Deskriptif**: Mean, median, std dev untuk numerik; frekuensi untuk kategorikal
- **Kualitas Data**: Deteksi missing values, duplicate rows, outliers (IQR method)
- **Visualisasi**: Histogram, bar charts, correlation heatmap

**Output:**
- Data quality report
- Statistical summary
- Visualization charts
- Recommendations untuk cleaning

---

### **Step 3: Persiapan & Pembersihan Data** 🛠️
**Aksi Pengguna:**
- Pilih strategi handling missing values (remove/fill_mean/fill_mode)
- Pilih handling duplicates (remove/keep)
- Pilih handling outliers (remove/keep/capping)

**Proses Sistem:**
- Apply selected cleaning strategies
- Generate cleaned dataset
- Create downloadable `dataset_cleaned.csv`
- Validate cleaned data quality

**Output:**
- Cleaned dataset
- Cleaning summary report
- Auto-progress ke Step 4

---

### **Step 4: Pemodelan & Evaluasi** 🤖
**Aksi Pengguna:**
- Pilih target column
- Pilih ML algorithm (Decision Tree/Random Forest/Logistic Regression/SVM)
- Set hyperparameters

**Proses Sistem:**
- **Wajib menggunakan cleaned data** dari Step 3
- Data splitting (train/test)
- Model training dengan cross-validation
- Performance evaluation
- Model serialization ke `model.pkl`

**Output:**
- Trained model file
- Training results dengan predictions
- Performance metrics (accuracy, confusion matrix, classification report)
- Auto-progress ke Step 5

---

### **Step 5: Prediksi & Deployment** 🔮
**Aksi Pengguna:**
- Input data baru untuk prediksi
- atau Upload file untuk batch prediction

**Proses Sistem:**
- Load trained model dari Step 4
- Preprocess input data sesuai trained model
- Generate predictions dengan confidence scores
- Format output untuk user consumption

**Output:**
- Individual predictions
- Batch prediction results
- Confidence scores
- Downloadable prediction results

---

## 📁 Struktur Project

```
ML-Analytics-Dashboard/
├── 📁 backend/                    # Python Flask Backend
│   ├── app.py                     # Main Flask application
│   ├── requirements.txt           # Python dependencies
│   └── models/                    # Saved ML models (.pkl files)
├── 📁 src/                        # React Frontend Source
│   ├── App.js                     # Main React component
│   ├── MLAnalyticsDashboard.jsx   # Main dashboard component
│   ├── ModelTrainingComponent.jsx # ML training interface
│   ├── PredictionComponent.jsx    # Prediction interface
│   ├── SimpleDashboard.jsx        # Alternative dashboard
│   └── services/
│       └── api.js                 # API service functions
├── 📁 public/                     # Static frontend files
│   └── index.html                 # Main HTML template
├── 📁 training_data/              # Sample datasets for training
│   ├── iris_dataset.csv
│   ├── customer_behavior.csv
│   ├── house_prices.csv
│   └── employee_performance.csv
├── 📁 venv311/                    # Python virtual environment
├── 📁 .vscode/                    # VS Code configuration
│   └── tasks.json                 # Automated tasks
├── .gitignore                     # Git ignore rules
├── package.json                   # Frontend dependencies
├── setup_python311_environment.bat # Automated Python setup
└── README.md                      # Project documentation
```

---

## 🔌 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### Data Analysis
```bash
POST /analyze
# Upload dan analisis data
Body: FormData dengan file CSV

GET /analyze
# Mendapatkan hasil analisis terakhir
```

#### Data Cleaning
```bash
POST /clean_data
# Membersihkan data dengan strategi yang dipilih
Body: {
  "missing_strategy": "fill_mean|fill_mode|remove_rows",
  "duplicates_strategy": "remove|keep",
  "outliers_strategy": "remove|keep|cap"
}
```

#### Model Training
```bash
POST /train_model
# Train ML model
Body: {
  "algorithm": "decision_tree|random_forest|logistic_regression|svm",
  "target_column": "column_name",
  "parameters": {}
}

GET /models
# List semua trained models

DELETE /models/<model_id>
# Hapus specific model
```

#### Predictions
```bash
POST /predict
# Prediksi individual
Body: {
  "model_id": "model_filename",
  "data": {"feature1": value1, "feature2": value2}
}

POST /predict_batch
# Prediksi batch dari file
Body: FormData dengan file CSV
```

#### Utilities
```bash
GET /health
# Health check

GET /datasets
# List available training datasets
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Python Version Issues
**Problem**: Library compatibility errors
**Solution**: 
```bash
# Pastikan menggunakan Python 3.11
python --version
# Should show Python 3.11.x

# Re-create virtual environment jika perlu
rm -rf venv311
python3.11 -m venv venv311
```

#### 2. Flask Server Not Starting
**Problem**: ModuleNotFoundError atau import errors
**Solution**:
```bash
# Activate virtual environment
venv311\Scripts\activate.bat

# Install dependencies
cd backend
pip install -r requirements.txt

# Test imports
python -c "import flask, pandas, sklearn; print('All imports successful')"
```

#### 3. CORS Issues
**Problem**: Frontend tidak bisa connect ke backend
**Solution**: Pastikan Flask-CORS installed dan configured:
```python
# Dalam app.py
from flask_cors import CORS
CORS(app, origins=['http://localhost:3000'])
```

#### 4. Node.js Issues
**Problem**: npm install gagal
**Solution**:
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules dan reinstall
rm -rf node_modules package-lock.json
npm install
```

#### 5. Model Training Fails
**Problem**: Error saat training model
**Solution**:
- Pastikan data sudah dibersihkan (Step 3)
- Check target column format
- Verify data types compatibility

### Log Locations
- **Backend logs**: Console output dari Flask server
- **Frontend logs**: Browser console (F12)
- **Model files**: `backend/models/` directory

---

## 🤝 Contributing

### Development Workflow
1. Fork repository
2. Create feature branch: `git checkout -b feature/nama-fitur`
3. Commit changes: `git commit -am 'Add fitur baru'`
4. Push branch: `git push origin feature/nama-fitur`
5. Submit Pull Request

### Code Standards
- **Python**: Follow PEP 8 styling
- **JavaScript**: Use ES6+ features, JSX for components
- **Comments**: Write clear, descriptive comments
- **Testing**: Add tests untuk new features

### File Naming Conventions
- **Components**: PascalCase (e.g., `ModelTrainingComponent.jsx`)
- **Utilities**: camelCase (e.g., `dataProcessor.js`)
- **Constants**: UPPER_SNAKE_CASE
- **Models**: descriptive names dengan timestamp

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## � File Management & Git Guidelines

### 📋 Important Documentation
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)**: Detailed setup instructions
- **[GIT_UPLOAD_GUIDE.md](GIT_UPLOAD_GUIDE.md)**: What to upload/ignore in Git
- **[docs/](docs/)**: Complete project documentation

### 🚫 Files NOT to Upload to Git
```
❌ venv311/           # Python virtual environment (100MB+)
❌ node_modules/      # Node.js dependencies (200MB+)
❌ backend/models/    # Trained ML models (binary files)
❌ .env              # Environment variables (may contain secrets)
❌ __pycache__/      # Python cache files
❌ *.log             # Log files
```

### ✅ Files TO Upload to Git
```
✅ src/              # Source code
✅ backend/app.py    # Backend code
✅ package.json      # Dependencies list
✅ requirements.txt  # Python dependencies
✅ docs/             # Documentation
✅ training_data/    # Sample datasets
✅ setup.bat/.sh     # Setup scripts
```

---

## �👨‍💻 Author

**AzlNach**
- GitHub: [@AzlNach](https://github.com/AzlNach)
- Repository: [ML-Analytics-Dashboard](https://github.com/AzlNach/ML-Analytics-Dashboard)

---

## 🙏 Acknowledgments

- **Scikit-learn** team untuk ML algorithms
- **React** community untuk frontend framework
- **Flask** untuk lightweight backend framework
- **YData Profiling** untuk automated EDA capabilities

---

**Happy Analyzing! 🚀📊🤖**
