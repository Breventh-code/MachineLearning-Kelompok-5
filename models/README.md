# Model SmartBearing — Random Forest + BIRCH

- `random_forest_pipeline.pkl`: Supervised Random Forest (tidak diubah oleh patch ini).
- `best_unsupervised_model.pkl`: model final BIRCH 3 cluster, 13 fitur terstandardisasi.
- `scaler.pkl`, `label_encoder.pkl`, `feature_list.json`, `feature_schema.json`: preprocessing yang SAMA dengan training.
- `birch_cluster_profiles.json`: komposisi hasil uji 460 sampel untuk model BIRCH dalam ZIP. Dipakai hanya saat SHA-256 model cocok.
- `unsupervised_algorithm_winners.json`: ringkasan pemilihan dari eksperimen (Agglomerative terbaik eksploratif, BIRCH terbaik untuk prediksi web).

**Tidak lagi memakai:** `kmeans_pipeline.pkl`, `metrics_clustering.json` (artefak lama). Script pemasangan memindahkan salinan file lama ke backup dan menghapusnya dari root project.

Jangan mengganti `scaler.pkl` dengan hasil training lain tanpa melatih ulang model. File `.pkl` hanya boleh dimuat dari sumber tepercaya.
