# Bukti Pemisahan Cluster BIRCH — SmartBearing CWRU

- Data testing: 460 baris, 13 fitur scaled.
- Model: BIRCH (threshold=0.8, branching_factor=50, n_clusters=3).
- Jumlah subcluster pada CF Tree: 107.
- Audit jarak: semua hasil prediksi cocok dengan subcluster terdekat = True.

## Distribusi per cluster (hanya evaluasi setelah training)
- Cluster 0: 379 sampel (Ball=138, IR=138, OR=57, Normal=46).
- Cluster 1: 32 sampel (OR=32).
- Cluster 2: 49 sampel (OR=49).

## Bukti profil fitur (median z-score vs cluster baseline)
- Cluster 1 dibanding Cluster 0: kurtosis (Δmedian z=3.015), min (Δmedian z=2.576), abs_peak (Δmedian z=2.511), peak_to_peak (Δmedian z=2.426), crest (Δmedian z=2.397).
- Cluster 2 dibanding Cluster 0: energy (Δmedian z=3.216), rms (Δmedian z=2.916), sd (Δmedian z=2.913), peak_to_peak (Δmedian z=2.718), min (Δmedian z=2.707).

## Bukti penugasan sampel
Setiap sampel dipetakan ke pusat subcluster BIRCH terdekat menurut jarak Euclidean dalam 13 fitur scaled.
Tabel penuh ada di `birch_bukti_jarak_setiap_sampel.csv`, termasuk jarak ke subcluster dari setiap cluster akhir dan margin pilihan.
Kontribusi kuadrat jarak per fitur ada di `birch_bukti_kontribusi_jarak_fitur.csv`.

## Batasan
- Komposisi kelas adalah label referensi untuk evaluasi, bukan input fitting BIRCH.
- Cluster tidak otomatis berarti diagnosis OR atau tingkat kerusakan.
- Fitur turunan mungkin saling berkorelasi; statistik ini deskriptif dan tidak membuktikan kausalitas.
- Dataset dipisah per window/baris dan belum divalidasi antarrekaman independen.
