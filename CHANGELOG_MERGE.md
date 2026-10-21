# Catatan Merge ZIP

- Menambahkan notebook final dan `preprocessing.py` baru.
- Memperbarui schema input web dari 8 -> 9 fitur statistik CWRU.
- Menghubungkan model service ke `transform_input()` agar API memakai 13 fitur yang sudah scaled.
- Menyelaraskan mapping kelas memakai `label_encoder.pkl` hasil preprocessing (4 kelas).
- Memperbarui validasi, dokumen, pengujian dan kebutuhan paket.
- Tidak menyertakan dataset asli atau model ML hasil training.
- Jika repo sudah mengandung file yang berbeda, periksa diff sebelum commit.
