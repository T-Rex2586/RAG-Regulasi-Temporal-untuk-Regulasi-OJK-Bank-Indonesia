# ADR 002: Chunking Strategies and Hybrid Indexing Implementation

## Status
Accepted

## Konteks
Sistem Temporal Regulatory QA membutuhkan indexing atas 320 dokumen peraturan (POJK, PBI, PADG) yang memiliki struktur hierarkis formal (Bab, Pasal, Ayat, Huruf, Angka). Retrieval harus mampu melayani pencarian teks eksak (BM25) dan semantik (Dense Vector), serta mendukung filtering berdasarkan metadata temporal dan regulasi (`regulator`, `jenis`, `nomor`, `tahun`, `status`, `pasal_nomor`, `bab_nomor`, `strategy`).

## Keputusan
1. **Tiga Strategi Chunking**:
   - `fixed`: Chunking karakter 600 chars dengan overlap 100 chars (baseline pembanding).
   - `heading_aware`: Chunking berdasarkan batas struktural Bab/Pasal dengan batas maksimum 1500 chars, menjaga integritas satu pasal utuh sebisa mungkin.
   - `hierarchical`: Chunking hierarkis parent-child (child max 600 chars) yang mengaitkan teks ayat spesifik dengan konteks bab/pasal induknya.

2. **Mesin Indexing**:
   - **BM25**: Implementasi `rank_bm25` (BM25Okapi) dengan tokenizer bahasa Indonesia berbobot ringan, disimpan dalam format serial `.pkl` ter-versi (`v1`).
   - **Dense Index**: Qdrant Vector Database berjalan di Docker Desktop container (`rag-qdrant`), diakses melalui port gRPC `6334` dengan konfigurasi `prefer_grpc=True`.
   - **Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (dimensi 384, ONNX runtime via fastembed), memberikan throughput stabil dan akurasi semantik bahasa Indonesia yang tinggi.

3. **Integritas dan Reproduksibilitas**:
   - Point ID di Qdrant menggunakan deterministic UUIDv5 berbasis `chunk_id` unik.
   - Pembacaan parsed JSON diurutkan (`sorted`) agar urutan chunking dan chunk_id deterministik antar-run.
   - Metadata payload terindeks secara eksplisit dengan keyword schema (`doc_id`, `regulator`, `jenis`, `nomor`, `tahun`, `status`, `pasal_nomor`, `bab_nomor`, `strategy`).

## Konsekuensi
- Semua 3 strategi (`fixed`, `heading_aware`, `hierarchical`) tersedia secara independen untuk dievaluasi pada baseline B0–B5.
- Latensi retrieval BM25 berada di kisaran 60–90 ms, dense Qdrant di kisaran 100–200 ms.
- Seluruh spesifikasi ukuran dan statistik panjang chunk tercatat dalam run manifest `runs/index_20261117_103201/manifest.json`.
