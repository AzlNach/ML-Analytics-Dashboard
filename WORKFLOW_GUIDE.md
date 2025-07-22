# 🔄 Panduan Alur Sistem Analisis Data Interaktif

## Tujuan Sistem
Sistem ini menyediakan alur analisis data yang terstruktur dan interaktif untuk membantu pengguna melakukan analisis data lengkap, mulai dari mengunggah dataset mentah hingga mendapatkan hasil prediksi yang akurat melalui 5 langkah yang sistematis.

---

## 📋 Alur 5 Langkah Sistem

### **Langkah 1: Unggah Data** 📤

**Aksi Pengguna:**
- Pengguna mengunggah file dataset dalam format `.csv`
- File akan diproses dan dimuat ke dalam sistem

**Proses Sistem:**
- Menerima dan memuat data ke dalam memori sebagai JavaScript objects
- Parsing CSV dengan handling untuk quoted values dan special characters
- Validasi struktur data dan deteksi tipe kolom

**Output:**
- Dataset mentah tersimpan dalam state `data`
- Headers kolom tersimpan dalam state `columns`
- Status: `workflowStep = 1`

---

### **Langkah 2: Analisis & Pemahaman Data** 📊

**Trigger:** Otomatis setelah upload berhasil

**Proses Sistem:**

#### **Profil Data Awal:**
- Informasi dasar: jumlah baris, kolom, tipe data
- Penggunaan memori dan ukuran dataset

#### **Statistik Deskriptif:**
- **Kolom Numerik:** mean, median, standar deviasi, min, max, kuartil
- **Kolom Kategorikal:** frekuensi, unique values, distribusi

#### **Verifikasi Kualitas Data:**
- **Missing Values:** Deteksi dan hitung nilai yang hilang per kolom
- **Duplicate Rows:** Identifikasi baris duplikat
- **Outliers:** Deteksi pencilan menggunakan metode IQR (Interquartile Range)

#### **Visualisasi Eksplorasi:**
- Histogram untuk kolom numerik
- Bar chart untuk kolom kategorikal
- Correlation heatmap untuk relasi antar kolom numerik

**Output:**
- `analysis` object dengan statistical summary
- `dataQualityReport` object dengan quality metrics
- Status: `workflowStep = 2`
- File virtual: `quality_report.json`

---

### **Langkah 3: Persiapan & Pembersihan Data** 🛠️

**Trigger:** User navigation setelah melihat quality report

**Interaksi Pengguna:**
Sistem menampilkan opsi pembersihan berdasarkan masalah yang terdeteksi:

#### **Penanganan Missing Values:**
- **Hapus baris:** Hilangkan baris yang mengandung nilai kosong
- **Isi dengan Mean:** Untuk kolom numerik
- **Isi dengan Mode:** Untuk kolom kategorikal
- **Isi dengan nilai konstan:** User-defined value

#### **Penanganan Duplicate Rows:**
- **Hapus duplikat:** Hilangkan baris yang identik

#### **Penanganan Outliers:**
- **Hapus outliers:** Hilangkan data di luar batas IQR
- **Keep outliers:** Pertahankan semua data
- **Capping:** Batasi nilai pada percentile tertentu

**Proses Sistem:**
- Apply cleaning operations sesuai pilihan user
- Generate cleaned dataset
- Create CSV content untuk export

**Output:**
- `cleanedData` state dengan dataset yang sudah dibersihkan
- `cleanedDataFile` object: 
  ```javascript
  {
    name: "filename_cleaned.csv",
    content: "CSV content string",
    data: cleanedDataArray
  }
  ```
- Status: `workflowStep = 3`
- File tersimpan: `dataset_cleaned.csv`

---

### **Langkah 4: Pemodelan & Evaluasi** 🤖

**Prasyarat:** Dataset cleaned tersedia (dari Langkah 3)

**Input Data:** Sistem WAJIB menggunakan `cleanedData` sebagai input

**Subfitur yang Tersedia:**

#### **1. Model Training (ModelTrainingComponent):**
- **Algorithm Selection:** Decision Tree, Random Forest, SVM, Logistic Regression
- **Target Column Selection:** User memilih variabel target (Y)
- **Feature Selection:** User memilih fitur-fitur (X)
- **Training Options:** Test size, cross-validation settings

#### **2. Quick Analysis:**
- **Clustering (Unsupervised):** K-Means untuk grouping data
- **Anomaly Detection:** Isolation Forest untuk deteksi outliers

**Proses Training:**
1. Preprocess cleaned data untuk ML algorithms
2. Split data menjadi train/test sets
3. Train selected algorithm
4. Evaluate model performance
5. Save trained model

**Output:**
- `trainedModel` state dengan model information
- `trainedModelFile` object:
  ```javascript
  {
    name: "filename_model.pkl",
    modelId: "unique_model_id",
    accuracy: 0.85,
    created_at: "2025-01-23T..."
  }
  ```
- `trainingResultsFile` object:
  ```javascript
  {
    name: "filename_training_results.csv",
    content: "CSV with predictions",
    data: dataWithPredictions
  }
  ```
- Status: `workflowStep = 4`
- Files tersimpan: `model.pkl` + `dataset_trained_results.csv`

---

### **Langkah 5: Prediksi** 🔮

**Prasyarat:** Model telah dilatih pada Langkah 4

**Input:** Model yang tersimpan (`model.pkl`) dari Langkah 4

**Proses Sistem:**

#### **1. Model Loading:**
- Load trained model dari backend storage
- Retrieve model metadata (features, target, performance)

#### **2. Input Interface:**
- Dynamic form generation berdasarkan model features
- Input validation untuk setiap feature
- Real-time input preprocessing

#### **3. Prediction Process:**
- Preprocess input data sesuai training data structure
- Apply same transformations yang digunakan saat training
- Generate predictions menggunakan loaded model
- Calculate confidence scores jika tersedia

#### **4. Results Display:**
- Prediction value/class
- Confidence score/probability
- Feature importance (jika ada)
- Input summary yang digunakan

**Output:**
- Real-time predictions pada data baru
- Downloadable prediction results
- Model performance metrics
- Status: `workflowStep = 5`

---

## 🔄 Konsistensi Data Flow

### **Data Progression:**
```
Raw Data (Step 1) 
    ↓
Analysis Data (Step 2) [uses raw data]
    ↓
Cleaned Data (Step 3) [produces cleaned dataset]
    ↓
Training Data (Step 4) [MUST use cleaned data]
    ↓
Prediction Data (Step 5) [uses trained model]
```

### **File Generation:**
- **Step 1:** `original_dataset.csv` → loaded into system
- **Step 2:** `quality_report.json` → virtual file with analysis
- **Step 3:** `dataset_cleaned.csv` → downloadable cleaned data
- **Step 4:** `model.pkl` + `dataset_trained_results.csv` → model & results
- **Step 5:** New predictions using saved model

### **State Management:**
```javascript
// Core data states
data              // Raw uploaded data
cleanedData       // Processed data from step 3
trainedModel      // Model info from step 4

// File states
cleanedDataFile      // Step 3 output file
trainedModelFile     // Step 4 model file
trainingResultsFile  // Step 4 results file

// Progress tracking
workflowStep         // Current step (1-5)
dataQualityReport    // Step 2 analysis results
```

---

## 🚀 Implementasi Terkini

### **Fitur yang Sudah Implementasi:**
✅ **Step 1:** File upload dengan CSV parsing  
✅ **Step 2:** Automatic EDA dan quality analysis  
✅ **Step 3:** Interactive data cleaning dengan file generation  
✅ **Step 4:** Model training dengan multiple algorithms  
✅ **Step 5:** Prediction interface dengan trained models  
✅ **Progress tracking:** Visual workflow indicators  
✅ **File management:** Automatic file naming dan download  

### **Perbaikan yang Telah Dilakukan:**
- ✅ Workflow step enforcement (tidak bisa skip steps)
- ✅ Proper data flow (cleaned data untuk modeling)
- ✅ File generation untuk setiap step
- ✅ Visual progress indicators dengan file info
- ✅ Improved UI/UX untuk workflow clarity
- ✅ Proper error handling dan validation

### **Backend Integration:**
- ✅ Flask API untuk ML algorithms
- ✅ Model persistence dalam backend
- ✅ Real-time prediction API
- ✅ Training data management

---

## 🔧 Penggunaan Sistem

### **Langkah Penggunaan:**
1. **Upload:** Pilih dan upload file CSV
2. **Review:** Lihat analysis results dan quality report
3. **Clean:** Pilih cleaning options dan apply
4. **Train:** Select algorithm, target, dan train model
5. **Predict:** Input new data untuk predictions

### **Tips Penggunaan:**
- Pastikan dataset memiliki header columns
- Review quality report sebelum cleaning
- Pilih target column yang sesuai untuk supervised learning
- Validasi predictions dengan real data
- Download intermediate files untuk backup
