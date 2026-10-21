# Dokumentasi API SmartBearing

Server lokal: `http://127.0.0.1:5000`.

## GET /api/status

Menampilkan kesiapan model dan `feature_order` yang wajib sama dengan fitur pelatihan.

## POST /api/predict/supervised
## POST /api/predict/unsupervised

Kirim `Content-Type: application/json` dan JSON dengan 8 fitur.
**Nilai di bawah hanya contoh struktur request sintetis; bukan cuplikan CWRU dan bukan hasil ekstraksi sebenarnya.**

```json
{
  "features": {
    "rms": 0.1,
    "std_dev": 0.11,
    "peak_to_peak": 0.6,
    "kurtosis": 3.0,
    "skewness": 0.2,
    "crest_factor": 3.4,
    "signal_energy": 20.0,
    "dominant_frequency": 123.0
  }
}
```

- `200`: prediksi sukses (ketika model tepercaya sudah dipasang).
- `400`: format request/fitur tidak valid.
- `503`: file model `.pkl` belum terpasang atau gagal dimuat.
- `500`: kesalahan inferensi, misalnya nama fitur tidak sama dengan training.

Respons sukses supervised berisi `result.prediction`, `result.label`, dan `result.probabilities` jika classifier punya `predict_proba`.

Respons sukses unsupervised berisi `result.cluster`, `result.label`, dan `result.distances` jika model punya `transform`. Jarak adalah jarak dalam ruang fitur setelah transformasi pipeline bila digunakan.

## Contoh cURL

```bash
curl -X POST http://127.0.0.1:5000/api/predict/supervised \
  -H "Content-Type: application/json" \
  -d '{"features":{"rms":0.1,"std_dev":0.11,"peak_to_peak":0.6,"kurtosis":3,"skewness":0.2,"crest_factor":3.4,"signal_energy":20,"dominant_frequency":123}}'
```

Jika file `.pkl` belum tersedia, respons yang benar adalah `503`, bukan hasil prediksi palsu.
