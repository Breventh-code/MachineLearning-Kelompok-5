# Model training dan inferensi SmartBearing CWRU (4 kelas)

**Saat ZIP diunduh, model hasil training belum tersedia.** Workflow:

1. Tempatkan CSV Kaggle `feature_time_48k_2048_load_1.csv` dalam `data/raw/`.
2. Jalankan `python preprocessing.py --data-dir data`, akan menghasilkan `scaler.pkl`, `label_encoder.pkl`, `feature_list.json`, dan memperbarui `feature_schema.json`.
3. ML Engineer 1: latih `RandomForestClassifier` pada 13 kolom fitur `train_scaled.csv` (`label` dikeluarkan dari X) dan ekspor estimator hasil training sebagai `models/random_forest_pipeline.pkl`.
4. ML Engineer 2: latih `KMeans` pada 13 kolom fitur scaled **tanpa label** dan ekspor sebagai `models/kmeans_pipeline.pkl`.
5. Karena scaler sudah diterapkan oleh preprocessing, **jangan menambahkan scaler kedua** ke model yang dilatih dengan `train_scaled.csv`.
6. Backend otomatis memakai `preprocessing.transform_input()` untuk mengubah 9 fitur dari browser menjadi 13 fitur scaled. Label dibaca langsung dari `label_encoder.pkl`.
7. Restart `python app.py`; periksa `/api/status`.

Berkas model `.pkl` serta fitted scaler diabaikan oleh `.gitignore`. Bagikan dengan jalur yang aman sesuai kebutuhan tim. **Hanya load pickle/joblib dari sumber terpercaya.**
