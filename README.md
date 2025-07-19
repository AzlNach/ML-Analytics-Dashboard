# 🚀 ML Analytics Dashboard

Sebuah aplikasi web interaktif yang memungkinkan Anda mengupload data CSV dan secara otomatis menghasilkan analisis machine learning dengan visualisasi yang menarik.

## ✨ Fitur Utama

- **📊 Data Visualization**: Grafik korelasi, distribusi, dan trend analysis
- **🔍 DBSCAN Clustering**: Pengelompokan data otomatis dengan visualisasi
- **⚠️ Anomaly Detection**: Deteksi anomali menggunakan Isolation Forest
- **🌳 Decision Tree**: Pohon keputusan interaktif untuk klasifikasi
- **📈 Statistical Analysis**: Analisis statistik dasar untuk semua kolom
- **💾 Export Results**: Export hasil analisis dalam format JSON

## 🛠️ Teknologi yang Digunakan

- **React 18** - Framework JavaScript
- **Recharts** - Library visualisasi data
- **Lucide React** - Icon components
- **MathJS** - Operasi matematika
- **Tailwind CSS** - Styling framework

## 🚀 Cara Menjalankan

### Prerequisites
- Node.js (versi 14 atau lebih tinggi)
- npm atau yarn

### Instalasi
```bash
# Clone repository
git clone https://github.com/AzlNach/ML-Analytics-Dashboard.git

# Masuk ke direktori proyek
cd ML-Analytics-Dashboard

# Install dependencies
npm install

# Jalankan aplikasi
npm start
```

Aplikasi akan berjalan di `http://localhost:3000`

## 📝 Cara Penggunaan

1. **Upload Data**: Klik tab "Upload Data" dan pilih file CSV Anda
2. **Overview**: Lihat statistik dataset dan pilih kolom untuk analisis ML
3. **Visualization**: Jelajahi berbagai grafik dan visualisasi data
4. **Clustering**: Lihat hasil pengelompokan DBSCAN
5. **Anomaly Detection**: Temukan data anomali dengan Isolation Forest
6. **Decision Tree**: Analisis pohon keputusan (jika ada data kategorikal)

## 📊 Format Data yang Didukung

- File CSV dengan header
- Data numerik dan kategorikal
- Contoh format data tersedia di `sample_data.csv`

## 🎯 Algoritma ML yang Diimplementasikan

### DBSCAN Clustering
- Pengelompokan berbasis density
- Otomatis mendeteksi jumlah cluster
- Identifikasi noise/outliers

### Isolation Forest
- Deteksi anomali tanpa supervised learning
- Scoring anomali 0-1
- Visualisasi interaktif

### Decision Tree
- Klasifikasi dengan visualisasi pohon
- Information gain splitting
- Interpretabilitas tinggi

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
