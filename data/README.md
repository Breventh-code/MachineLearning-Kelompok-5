# Data folder

- `data/raw/feature_time_48k_2048_load_1.csv` — sumber data CWRU yang diunduh secara sah dari Kaggle (tidak disertakan ZIP).
- `data/processed/train_scaled.csv` — 13 kolom fitur scaled + `label` (target 4 kelas).
- `data/processed/test_scaled.csv` — same schema, held-out random WINDOW split.
- `data/processed/cwru_clean.csv` — data bersih + metadata untuk EDA.
- `data/processed/split_manifest.csv` — metadata pembagian, jangan dimasukkan dalam fitur training.

Jalankan dari root `python preprocessing.py --data-dir data` untuk menghasilkan file processed.
Jangan commit data besar/privat; `.gitignore` sudah mengecualikannya.
**Catatan:** split berdasarkan row/window, bukan recording ID yang independen.
