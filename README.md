# SmartBearing CWRU — Flask Web Project

Starter **backend + frontend terstruktur** untuk proyek deteksi kerusakan bearing CWRU dengan dua paradigma:

- **Supervised:** klasifikasi fault menggunakan model Random Forest dari ML Engineer 1.
- **Unsupervised:** clustering fitur getaran dengan K-Means dari ML Engineer 2.

Frontend responsif menggunakan HTML, CSS, JavaScript vanilla dan grafik Canvas bawaan browser, tanpa ketergantungan CDN.

## Status isi ZIP

✅ Server Flask, empat halaman web, validasi 8 fitur, dua endpoint prediksi, status model, visualisasi prediksi, halaman error, tes API.

⚠️ **Tidak menyertakan model `.pkl` atau dataset CWRU asli.** Selama model belum dipasang, website tetap terbuka dan form mengirim pesan jelas `503 Model belum tersedia`. Tidak ada prediksi tiruan.

## Menjalankan di Windows (PowerShell)

Buka terminal di folder `smartbearing-cwru-web`, lalu:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Jika PowerShell memblokir aktivasi, gunakan Command Prompt dan `.venv\Scripts\activate.bat` atau jalankan `.venv\Scripts\python.exe app.py` setelah instal dependensi.

Alternatif: jalankan `run_windows.bat`.

Buka **http://127.0.0.1:5000**.

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

## Integrasi dengan 2 model dari tim ML

1. Data Engineer menyiapkan fitur, berurutan:
   `rms`, `std_dev`, `peak_to_peak`, `kurtosis`, `skewness`, `crest_factor`, `signal_energy`, `dominant_frequency`.
2. ML Engineer 1 melatih dan mengekspor **pipeline** yang menerima `pandas.DataFrame` dengan kolom tersebut ke `models/random_forest_pipeline.pkl`.
3. ML Engineer 2 melatih pipeline `StandardScaler → KMeans` ke `models/kmeans_pipeline.pkl`.
4. Periksa **class encoding** di `models/feature_schema.json` dan samakan dengan training. Mapping default hanya usulan.
5. Jika fitur akhir tim berbeda, perbarui **`models/feature_schema.json`** dan samakan dengan kolom model. Pastikan satuan/definisi fitur juga konsisten.
6. Refresh halaman; cek status model di Home atau endpoint `/api/status`.
7. Masukkan delapan fitur yang **benar-benar diekstraksi dari sinyal CWRU**. Jangan masukkan sinyal mentah sebagai fitur dan jangan anggap input numerik sebarang sebagai diagnosis nyata.

> `joblib.load` hanya untuk model buatan tim atau berkas yang dipercaya. Versi library training dan inference sebaiknya cocok. K-Means cluster tidak otomatis setara kelas kerusakan.

## Folder utama

```text
smartbearing-cwru-web/
├── app.py                        # entry point Flask
├── backend/
│   ├── __init__.py               # create_app
│   ├── routes.py                 # routes halaman dan API
│   ├── schema.py                 # metadata fitur
│   ├── validators.py             # validasi JSON
│   └── model_service.py          # load model dan prediksi
├── frontend/
│   ├── templates/                # Jinja2 HTML
│   └── static/
│       ├── css/style.css
│       └── js/                    # interaksi dan grafik
├── models/
│   ├── feature_schema.json
│   └── README.md
├── data/README.md
├── docs/API.md
├── tests/test_app.py
└── requirements.txt
```

## Endpoint

| Endpoint | Metode | Fungsi |
|---|---|---|
| `/` | GET | Home |
| `/supervised` | GET | Form klasifikasi |
| `/unsupervised` | GET | Form clustering |
| `/about` | GET | Informasi proyek |
| `/api/health` | GET | Kesehatan service + status model |
| `/api/status` | GET | Kesiapan 2 model & urutan fitur |
| `/api/predict/supervised` | POST | Klasifikasi Random Forest |
| `/api/predict/unsupervised` | POST | K-Means clustering |

Lihat `docs/API.md` untuk contoh format JSON dan respons.

## Testing

```bash
python -m pytest -q
```

Test bawaan memvalidasi halaman, penolakan input bermasalah dan kontrak respons API dengan mock prediksi. Untuk test model sungguhan, tambahkan tes menggunakan pipeline `.pkl` hasil training.

## Batasan

- Ini proyek pendidikan untuk *data historis CWRU*, bukan monitoring sensor real-time.
- Upload sinyal `.mat` dan feature extraction otomatis dari file belum disertakan; pekerjaan itu menjadi antarmuka pengembangan berikutnya dengan Data Engineer.
- Saat integrasi, cek potensi *data leakage* dari pembagian window sinyal yang berasal dari rekaman sama.
- Gunakan server WSGI yang sesuai saat production, bukan `debug=True`.
