# 🚀 ML Analytics Dashboard

Sebuah aplikasi web full-stack yang memungkinkan Anda mengupload data CSV dan secara otomatis menghasilkan analisis machine learning dengan akurasi tinggi menggunakan backend Python Flask dan visualisasi interaktif React.

## ✨ Fitur Utama

### Frontend (React)
- **📊 Data Visualization**: Grafik korelasi, distribusi, dan trend analysis
- **🎯 Interactive Dashboard**: Interface modern dan responsif dengan multiple tabs
- **🧠 Model Training Interface**: Train multiple ML algorithms dengan UI yang user-friendly
- **🔮 Prediction Interface**: Make predictions menggunakan trained models
- **📈 Real-time Analysis**: Visualisasi hasil ML secara langsung
- **💾 Export Results**: Export hasil analisis dalam format JSON

### Backend (Python Flask)
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
- **📈 Statistical Analysis**: Analisis statistik komprehensif
- **📝 Model History**: Track semua training sessions dan performanceashboard

Sebuah aplikasi web full-stack yang memungkinkan Anda mengupload data CSV dan secara otomatis menghasilkan analisis machine learning dengan akurasi tinggi menggunakan backend Python dan visualisasi interaktif.

## ✨ Fitur Utama

### Frontend (React)
- **📊 Data Visualization**: Grafik korelasi, distribusi, dan trend analysis
- **🎯 Interactive Dashboard**: Interface modern dan responsif
- **� Real-time Analysis**: Visualisasi hasil ML secara langsung
- **💾 Export Results**: Export hasil analisis dalam format JSON

### Backend (Python Flask)
- **�🔍 DBSCAN Clustering**: Pengelompokan data dengan scikit-learn
- **⚠️ Anomaly Detection**: Deteksi anomali menggunakan Isolation Forest
- **🌳 Decision Tree**: Pohon keputusan dengan feature importance
- **🎯 Model Training**: Latih model dari dataset training
- **🔮 Prediction API**: Endpoint untuk prediksi real-time
- **� Statistical Analysis**: Analisis statistik komprehensif

### Machine Learning Features
- **📚 Training Data Management**: Folder khusus untuk dataset training
- **🧠 Pre-trained Models**: Model tersimpan untuk prediksi cepat
- **� Model Evaluation**: Accuracy scoring dan performance metrics
- **🔄 Cross-validation**: Validasi model untuk akurasi optimal

## 🛠️ Teknologi yang Digunakan

### Frontend
- **React 18** - Framework JavaScript modern
- **Recharts** - Library visualisasi data
- **Lucide React** - Icon components
- **Tailwind CSS** - Styling framework

### Backend  
- **Python 3.8+** - Backend language
- **Flask** - Web framework
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Flask-CORS** - Cross-origin resource sharing

## 🚀 Cara Menjalankan

### Prerequisites
- **Node.js** (versi 14 atau lebih tinggi)
- **Python 3.8+** dengan pip
- **npm** atau yarn

### Quick Start (Windows)
```bash
# Jalankan script otomatis
start.bat
```

### Manual Setup

#### 1. Setup Backend (Python Flask)
```bash
# Masuk ke direktori backend
cd backend

# Buat virtual environment (opsional tapi direkomendasikan)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies Python
pip install -r requirements.txt

# Buat folder untuk model
mkdir models

# Jalankan server Flask
python app.py
```

#### 2. Setup Frontend (React)
```bash
# Install dependencies Node.js
npm install

# Jalankan aplikasi React
npm start
```

### Akses Aplikasi
- **Frontend**: `http://localhost:3000`
- **Backend API**: `http://localhost:5000`

## 📝 Cara Penggunaan

### 1. Training Model dari Dataset
1. Buka tab "Upload & Train"
2. Lihat "Available Training Datasets" 
3. Klik "Train Model" dan masukkan target column
4. Model akan dilatih dan tersimpan otomatis

### 2. Analisis Data Baru
1. Upload file CSV di tab "Upload & Train"
2. Pilih kolom untuk analisis di tab "Overview" 
3. Klik "Run ML Analysis" untuk clustering dan anomaly detection
4. Jelajahi hasil di tab "Clustering", "Anomaly Detection", dan "Decision Tree"

### 3. Prediksi Menggunakan Model Terlatih
1. Setelah model dilatih, masukkan nilai input di form prediksi
2. Klik "Make Prediction" untuk mendapatkan hasil prediksi
3. Lihat confidence score dan hasil prediksi

## 📊 Format Data yang Didukung

- **CSV files** dengan header
- **Data numerik dan kategorikal**
- **Training datasets** di folder `training_data/`
- **Sample data** tersedia:
  - `employee_performance.csv` - Data performa karyawan
  - `house_prices.csv` - Data harga rumah

## 🎯 Algoritma ML yang Diimplementasikan

### DBSCAN Clustering (scikit-learn)
- Pengelompokan berbasis density
- Otomatis mendeteksi jumlah cluster
- Identifikasi noise/outliers
- Parameter yang dapat disesuaikan (eps, min_samples)

### Isolation Forest (scikit-learn)
- Deteksi anomali tanpa supervised learning
- Scoring anomali 0-1
- Visualisasi interaktif dengan scatter plot
- Customizable contamination rate

### Decision Tree (scikit-learn)
- Klasifikasi dengan feature importance
- Information gain splitting
- Model evaluation dengan accuracy metrics
- Exportable tree rules untuk interpretabilitas

### Model Training & Prediction
- **Training dari file**: Gunakan dataset di `training_data/` folder
- **Model persistence**: Automatic saving dengan joblib
- **Real-time prediction**: API endpoint untuk prediksi
- **Cross-validation**: Train/test split untuk evaluasi akurat

## 🏗️ Arsitektur Sistem

```
ML-Analytics-Dashboard/
├── frontend (React)
│   ├── src/
│   │   ├── MLAnalyticsDashboard.jsx
│   │   ├── services/api.js
│   │   └── ...
│   └── package.json
├── backend (Python Flask)
│   ├── app.py
│   ├── requirements.txt
│   └── models/ (trained models)
├── training_data/ (CSV datasets)
│   ├── employee_performance.csv
│   └── house_prices.csv
└── start.bat (Quick start script)
```

## 🔧 API Endpoints

### Data Analysis
- `POST /api/analyze` - Basic statistical analysis
- `POST /api/clustering` - DBSCAN clustering
- `POST /api/anomaly-detection` - Isolation Forest anomaly detection
- `POST /api/decision-tree` - Decision tree building

### Model Management
- `GET /api/training-data` - List available training datasets
- `POST /api/train-from-file` - Train model from training data
- `POST /api/predict` - Make prediction with trained model
- `GET /api/health` - Backend health check

## 🔧 Build untuk Production

```bash
npm run build
```

File production akan tersimpan di folder `build/`

## 🤝 Contributing

1. Fork repository ini
2. Buat branch fitur baru (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👨‍💻 Author

**AzlNach** - [GitHub Profile](https://github.com/AzlNach)

## 🙏 Acknowledgments

- React team untuk framework yang luar biasa
- Recharts untuk library visualisasi yang powerful
- Komunitas open source untuk inspirasi algoritma ML
