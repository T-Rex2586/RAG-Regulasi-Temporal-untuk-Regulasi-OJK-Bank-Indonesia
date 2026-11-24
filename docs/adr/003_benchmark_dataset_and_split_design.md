# ADR 003: Benchmark Dataset and Split Design for Temporal Regulatory QA

## Status
Accepted

## Konteks
Sistem Temporal Regulatory QA untuk regulasi perbankan dan pasar keuangan OJK dan BI memerlukan benchmark evaluasi yang representatif, mencakup skenario pertanyaan faktual, multi-hop, konflik versi akibat amandemen regulasi, evaluasi acuan temporal titik waktu (*point-in-time*), serta pertanyaan di luar cakupan (*unanswerable*) untuk menguji kemampuan model dalam melakukan *abstain* dan mitigasi halusinasi.

## Keputusan
1. **Jumlah dan Komposisi Query**:
   - Membangun **125 query** yang terdistribusi secara seimbang ke dalam 5 slice (masing-masing 25 query):
     - `easy`: Pertanyaan faktual berbasis satu pasal regulasi.
     - `multi_hop`: Pertanyaan penalaran lintas pasal (keterkaitan definisi, kewajiban, tata cara, dan sanksi).
     - `konflik_versi`: Pertanyaan mengenai dampak amandemen/perubahan regulasi terhadap aturan sebelumnya.
     - `temporal_point_in_time`: Pertanyaan dengan parameter acuan waktu eksplisit (`as_of`) untuk menguji validitas hukum pada tanggal tertentu.
     - `unanswerable`: Pertanyaan yang tidak diatur dalam korpus (pajak, izin ketenagakerjaan, retribusi daerah, regulasi fiktif masa depan) untuk menguji mekanisme *abstain*.

2. **Keseimbangan Regulator**:
   - Terdiri dari 48 query OJK, 50 query BI, 2 query gabungan (BOTH), dan 25 query out-of-scope (NONE).
   - Seluruh query *answerable* terhubung secara valid dengan `doc_id` dan nomor pasal nyata di korpus `data/parsed/`.

3. **Integritas dan Anti-Kebocoran**:
   - Uji duplikasi eksak dan uji *near-duplicate* berbasis Jaccard similarity karakter 3-gram dengan ambang batas (threshold) 0.85 — seluruh pasangan query lolos dengan nilai 0 duplikat.
   - Pembagian dataset (*stratified split*) dengan proporsi:
     - **Train**: 75 query (60%, 15 query per slice).
     - **Dev**: 25 query (20%, 5 query per slice).
     - **Test**: 25 query (20%, 5 query per slice).
   - Random seed terkunci pada `seed = 42`.
   - File `test.jsonl` dikunci menggunakan hash kriptografi SHA-256 (`3898015c21241292...`) dan dicatat di `benchmark/splits/manifest.json`.

## Konsekuensi
- Benchmark terkunci dan siap digunakan untuk evaluasi baseline B0–B6 tanpa risiko data leakage.
- Dataset train/dev/test tersimpan dalam format standar JSONL di direktori `benchmark/splits/`.
