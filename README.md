# SmartBearing CWRU — Web + Preprocessing Terintegrasi

**Versi gabungan** dari backend/frontend Flask dan notebook/script preprocessing CWRU final. Mendukung **Supervised (Random Forest)** dan **Unsupervised (K-Means)**. Web tetap bisa dibuka sebelum model ML ada; API prediksi memberi HTTP 503 sampai model & scaler tersedia, tanpa prediksi tiruan.

## Persyaratan dataset

Sumber: https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets

Dataset yang dipakai di notebook adalah CSV feature-based `feature_time_48k_2048_load_1.csv`. **File CSV asli tidak disertakan**: download dari Kaggle lalu tempatkan ke `data/raw/`. Bukan parser `.mat` mentah.

- Klasifikasi: **4 kelas** `Ball`, `IR`, `Normal`, `OR`.
- Form web: **9 fitur dasar** `max`, `min`, `mean`, `sd`, `rms`, `skewness`, `kurtosis`, `crest`, `form`.
- Backend: menghitung **4 fitur turunan** `peak_to_peak`, `abs_peak`, `energy` (= RMS kuadrat/mean square), `sd_rms_ratio`.
- ML model: menerima **13 fitur yang sudah di-scaling**, urutan `models/feature_list.json`.
- Nama file hasil training: `models/random_forest_pipeline.pkl` dan `models/kmeans_pipeline.pkl` (meski memakai akhiran "pipeline", model boleh berisi estimator saja setelah data scaled).

## Cara menjalankan di Windows (PowerShell)

Dari folder proyek:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Tempatkan CSV Kaggle ke data/raw/ terlebih dahulu
python preprocessing.py --data-dir data
python app.py
```

Jika PowerShell menolak aktivasi, gunakan `.venv\Scripts\python.exe` sebagai awalan tiap perintah, atau Command Prompt. Untuk EDA buka `notebooks/CWRU_EDA_Preprocessing_Final.ipynb`, lalu **Run All** setelah menyediakan CSV.

Buka http://127.0.0.1:5000 — Home / Supervised / Unsupervised / About.

## Output preprocessing

- `data/processed/{cwru_clean,train_scaled,test_scaled,split_manifest}.csv`
- `data/processed/preprocessing_summary.json`
- `models/{scaler,label_encoder}.pkl`
- `models/feature_list.json`
- `models/feature_schema.json` (menggantikan schema lama 8 fitur)

## Penyerahan ke ML Engineer

1. Data Engineer menjalankan preprocessing dan membagikan `train_scaled.csv`, `test_scaled.csv`, `label_encoder.pkl`, dan `feature_list.json`.
2. ML Engineer 1 melatih Random Forest dengan `X=train.drop(columns=['label'])`, `y=train['label']`, mengevaluasi pada test dan menyimpan estimator trained ke `models/random_forest_pipeline.pkl`.
3. ML Engineer 2 melakukan K-Means pada 13 fitur dari data training **tanpa label**, memilih jumlah cluster dengan evaluasi clustering, dan menyimpan model ke `models/kmeans_pipeline.pkl`.
4. Training HARUS menggunakan fitur scaled dari `train_scaled.csv`. **Jangan memfit StandardScaler tambahan** di dalam estimator/pipeline jika memakai file ini; web menerapkan scaler Data Engineer sekali.
5. Backend akan menerjemahkan label menggunakan `label_encoder.pkl`, bukan mapping label lama hard-coded.
6. Jalankan `python -m pytest -q` untuk memeriksa kode. Cek `/api/status` dan tes form web setelah `.pkl` model tersedia.

## Struktur folder

```text
smartbearing-cwru-web/
├── app.py
├── preprocessing.py
├── notebooks/CWRU_EDA_Preprocessing_Final.ipynb
├── backend/{schema.py, validators.py, model_service.py, routes.py}
├── frontend/templates/
├── frontend/static/{css/,js/}
├── data/{raw/,processed/,README.md}
├── models/{feature_schema.json,README.md}  # fitted model .pkl generated later
├── docs/API.md
├── tests/test_app.py
├── requirements.txt
└── .gitignore
```

## Penempatan di repository GitHub yang sudah ada

Arsip memiliki nama folder teratas **`smartbearing-cwru-web/` sama seperti ZIP lama** agar bisa diekstrak di folder induk yang sama, lalu pilih **Replace/Overwrite** untuk file terkait. Jangan menghapus folder `.git` pada proyek lokal. Sebaiknya lakukan backup commit dahulu, periksa `git status` dan `git diff`, lalu commit dan push perubahan dari branch tim.

## Batasan penting

- ZIP **tidak menyertakan CSV dataset asli** atau model `.pkl` hasil training.
- CSV CWRU sudah berupa fitur tabular; aplikasi tidak merekam sensor real-time ataupun mengimpor `.mat` langsung.
- `train_test_split(..., stratify=y)` bekerja pada baris/window, **bukan berdasarkan recording_id asli**. Metrik test tidak otomatis merepresentasikan kemampuan generalisasi ke rekaman baru.
- Ini aplikasi edukasi, bukan alat diagnosis industri tersertifikasi.
- `joblib` / `.pkl` hanya boleh digunakan untuk file yang tepercaya; usahakan lingkungan training dan runtime memakai versi library yang kompatibel.
