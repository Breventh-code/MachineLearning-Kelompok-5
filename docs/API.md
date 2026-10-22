# SmartBearing REST API — 4 kelas, 9 input → 13 fitur scaled

Jalankan `python app.py` lalu buka `http://127.0.0.1:5000`.

## GET /api/status

Menampilkan kesiapan dua model dan urutan **9 fitur input web**. Model akan berstatus belum siap sebelum CSV diproses dan kedua `.pkl` tersedia.

## POST /api/predict/supervised dan /api/predict/unsupervised

Request (`application/json`) berisi 9 fitur dasar CWRU seperti contoh **format sintetis** berikut (bukan sampel dataset asli):

```json
{
  "features": {
    "max": 0.8,
    "min": -0.7,
    "mean": 0.02,
    "sd": 0.3,
    "rms": 0.31,
    "skewness": 0.08,
    "kurtosis": 3.2,
    "crest": 2.58,
    "form": 1.15
  }
}
```

Backend memvalidasi 9 input, menghitung `peak_to_peak`, `abs_peak`, `energy = rms**2` (mean square), `sd_rms_ratio`, lalu menerapkan `scaler.pkl` menjadi **13 fitur** sesuai `feature_list.json`. Model ML harus sudah dilatih pada CSV scaled 13 fitur tanpa scaler kedua.

- `200`: prediksi berhasil setelah kedua artefak preprocessing dan model tersedia.
- `400`: input JSON atau fitur tidak valid.
- `503`: model atau artefak preprocessing belum tersedia/tidak kompatibel.
- `500`: error dalam preprocessing atau prediksi.

Supervised: `result.prediction`, `result.label`, `result.probabilities`.
Unsupervised (BIRCH): `result.algorithm`, `result.cluster`, `result.label`, `result.cluster_profile`, `result.note`. `result.distances` dipertahankan sebagai array kosong untuk kompatibilitas API lama.

`cluster_profile` memuat komposisi label historis pada 460 data testing, HANYA jika hash model sesuai. Proporsi komposisi bukan probabilitas kerusakan untuk input baru.
Cluster tidak boleh diperlakukan sebagai label kerusakan tanpa analisis tambahan.
