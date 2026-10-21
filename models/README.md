# Integrasi model dari ML Engineer

Tempatkan file asli berikut di folder ini:

1. `random_forest_pipeline.pkl` — **pipeline lengkap** supervised (contoh: `RandomForestClassifier` dan preprocessing jika digunakan).
2. `kmeans_pipeline.pkl` — **pipeline lengkap** unsupervised (contoh: `StandardScaler` + `KMeans`).

**Penting:** kedua pipeline wajib menerima satu baris `pandas.DataFrame` dengan nama dan urutan fitur sesuai `feature_schema.json`. Ubah file schema dan backend bila fitur training berbeda. Jangan hanya mengganti nama `.pkl` tanpa menyesuaikan fiturnya.

Kelas supervised pada `feature_schema.json` memakai mapping contoh `0=Normal`, `1=Inner Race`, `2=Outer Race`, `3=Ball Fault`. **Cek kembali label encoding yang benar dari dataset training** dan sesuaikan mapping jika berbeda.

`pickle`/`joblib` dapat menjalankan kode ketika dibuka: **hanya gunakan model yang dibuat sendiri/dari sumber tepercaya**. Latih atau ekspor ulang model jika versi scikit-learn tidak cocok.

Berkas `.pkl` tidak disertakan dalam template ini karena belum tersedia hasil training dari tim ML.
