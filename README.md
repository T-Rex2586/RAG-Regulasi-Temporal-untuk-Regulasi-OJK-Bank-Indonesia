# Temporal Regulatory Question Answering System (OJK & Bank Indonesia)

> **Frontier LLM Research & Engineering**: Sistem Tanya-Jawab Berbasis Regulasi Finansial Dinamis dengan RAG Hibrida, Penalaran Temporal *Point-in-Time*, Verifikasi Klaim Mandiri, dan Siklus *Self-Correction*.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen2.5--1.5B--Instruct-orange.svg)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![Vector DB](https://img.shields.io/badge/Vector%20DB-Qdrant%20(gRPC)-red.svg)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/License-Academic%20Research-green.svg)]()

---

## 📌 Ringkasan Eksekutif

Regulasi perbankan dan jasa keuangan di Indonesia (POJK, SEOJK, PBI, PADG) memiliki karakteristik dinamis: peraturan sering diubah atau dicabut, memiliki batas waktu keberlakuan (*point-in-time validity*), serta saling merujuk antar-pasal. Model bahasa besar komersial biasa rentan mengalami halusinasi parah ketika menjawab pertanyaan regulasi.

Repositori ini mengimplementasikan riset rekayasa sistem QA regulasi hukum temporal komprehensif (14 Fase) dengan basis model terbuka (*open-weight*) `Qwen2.5-1.5B-Instruct` yang dievaluasi secara ketat terhadap benchmark 125 kueri multi-slice dengan *ground truth qrels* berkesepakatan tinggi (**Cohen's Kappa = 0.8970**).

---

## 📊 Tangga Arsitektur & Hasil Evaluasi (Dev Split: 25 Kueri)

Evaluasi dihitung secara deterministik menggunakan **Automated Evaluation Harness** terhadap *gold qrels* bergradasi:

| Baseline | Deskripsi Arsitektur | Doc R@5 | Evidence Rec | Citation Prec | Unsupported Claim | False Refusal | False Confident | Correctness | p50 Latensi |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0** | Direct Zero-Shot LLM | 0.0% | 0.0% | 20.0% | 100.0% | **0.0%** | **100.0%** | 6.9% | 58.80s |
| **B1** | Structured Few-Shot Prompted | 0.0% | 12.0% | 28.0% | 80.0% | 20.0% | 100.0% | 13.8% | 66.94s |
| **B2** | Naive RAG (Dense Qdrant) | 30.0% | 8.0% | 16.0% | 76.0% | **65.0%** | 80.0% | 12.4% | 55.45s |
| **B3** | Hybrid RAG (BM25 + Dense RRF) | **57.5%** | **50.0%** | **44.0%** | 48.0% | **15.0%** | 60.0% | **37.6%** | 102.12s |
| **B4** | **Self-Corrective RAG** | **57.5%** | 32.0% | 36.0% | **40.0%** | 35.0% | **20.0%** | 32.0% | **83.61s** |

### 🔍 Temuan Kunci Riset:
1. **Halusinasi Ekstrem pada B0**: B0 memiliki *False Confident Rate* **100%** (mengarang pasal/bank fiktif untuk pertanyaan di luar domain).
2. **Kekuatan Fusi Hibrida (B3 vs B2)**: Kombinasi BM25 + Dense dengan *Reciprocal Rank Fusion* ($k=60$) mendongkrak **Doc Recall@5 dari 30.0% ke 57.5%** dan memangkas *False Refusal* dari 65.0% ke 15.0%.
3. **Ketahanan Self-Corrective (B4 vs B3)**: *Query Planner* dan *Claim Verifier* memangkas **False Confident Rate dari 60.0% menjadi 20.0%** (penolakan bersih kueri *out-of-domain* pada biaya 0 token).

---

## 🏛️ Struktur Repositori

```text
├── benchmark/               # Benchmark kueri, qrels emas, dan split terkunci
│   ├── queries.jsonl        # 125 kueri (5 slice seimbang)
│   ├── qrels.tsv            # 633 pasangan kueri-pasal teranotasi (Kappa=0.897)
│   ├── annotation_guidelines.md
│   ├── disagreement_log.csv
│   └── splits/              # Stratified Train (75), Dev (25), Test (25 locked)
├── configs/                 # Konfigurasi deklaratif YAML
│   ├── baselines.yaml       # Hyperparameter B0 s.d. B4
│   └── eval.yaml            # Konfigurasi automated evaluation harness
├── data/
│   ├── source_register.csv  # Metadata 320 PDF resmi OJK/BI + hash SHA-256
│   └── parsed/              # 320 dokumen JSON hasil parsing hierarkis
├── docs/adr/                # Architecture Decision Records (ADR 001 s.d. 007)
│   ├── 001_corpus_acquisition_and_rights.md
│   ├── 002_chunking_and_indexing_strategy.md
│   ├── 003_benchmark_dataset_and_split_design.md
│   ├── 004_annotation_protocol_and_inter_annotator_agreement.md
│   ├── 005_baseline_architectures_b0_to_b3.md
│   ├── 006_automated_evaluation_harness_and_metrics.md
│   └── 007_self_corrective_rag_b4.md
├── prompts/                 # Prompt registry berversi (B0, B1, B2, B3, B4)
├── runs/                    # Run manifests, log evaluasi JSONL, & laporan PDF
│   └── b4_20261119_094854/
│       ├── eval_report_dev.pdf   # Laporan evaluasi komparatif resmi (PDF 2 halaman)
│       ├── eval_results_dev.json # Metrik lengkap mesin
│       └── manifest.json         # Audit telemetri dan status verifikasi hash
└── src/
    ├── benchmark/           # Generator & splitter dataset
    ├── eval/                # Harness evaluasi, metrik IR & generasi, PDF generator
    ├── ingest/              # Scraper resmi & parser PDF PyMuPDF
    ├── pipeline/            # Implementasi B0, B1, B2, B3, B4, & generator
    ├── retrieval/           # BM25 Indexer, Dense Indexer FastEmbed, & RRF
    └── verify/              # Claim Verifier & Temporal Point-in-Time Checker
```

---

## 🚀 Panduan Penggunaan

### 1. Prasyarat & Instalasi
- Python 3.11
- Docker Desktop (untuk Qdrant vector database)
```bash
# Clone repositori
git clone https://github.com/T-Rex2586/RAG-Regulasi-Temporal-untuk-Regulasi-OJK-Bank-Indonesia.git
cd RAG-Regulasi-Temporal-untuk-Regulasi-OJK-Bank-Indonesia

# Install dependensi
pip install -r requirements.txt  # atau install torch, transformers, qdrant-client, rank-bm25, fastembed, reportlab
```

### 2. Menjalankan Docker Qdrant
```bash
docker run -d -p 6333:6333 -p 6334:6334 --name rag-qdrant qdrant/qdrant:latest
```

### 3. Menjalankan Pipeline Baseline (B0–B4)
```bash
# Evaluasi Baseline B4 pada Dev Set (25 kueri)
python -m src.pipeline.run_baselines --baselines B4

# Atau jalankan seluruh baseline secara berurutan
python -m src.pipeline.run_baselines --baselines B0 B1 B2 B3 B4
```

### 4. Menjalankan Automated Evaluation Harness & Laporan PDF
```bash
# Evaluasi otomatis terhadap gold qrels
python -m src.eval.harness --run_id b4_20261119_094854 --baselines B0 B1 B2 B3 B4

# Hasilkan laporan resmi dalam format PDF
python -m src.eval.generate_pdf_report --run_id b4_20261119_094854
```

---

## 🔒 Jaminan Anti-Kebocoran (*Anti-Leakage*)

Sistem menerapkan validasi integritas ketat:
- Test split (25 kueri) dikunci dengan hash SHA-256: `3898015c21241292352a8a02b42aa22ab134bf2ffef65e067eb7972c8698c19b`.
- Fungsi `verify_test_lock()` memverifikasi keutuhan file sebelum evaluasi apa pun dimulai.
- Few-shot exemplars murni diambil dari dataset latihan (`train.jsonl`).

---

## 📄 Dokumen Keputusan Arsitektur (ADR)
Riset ini mendokumentasikan setiap keputusan rekayasa secara transparan:
- [ADR 001: Akuisisi Korpus & Hak Guna](docs/adr/001_corpus_acquisition_and_rights.md)
- [ADR 002: Strategi Chunking & Indexing](docs/adr/002_chunking_and_indexing_strategy.md)
- [ADR 003: Desain Benchmark Dataset & Split](docs/adr/003_benchmark_dataset_and_split_design.md)
- [ADR 004: Protokol Anotasi & Kesepakatan Inter-Anotator](docs/adr/004_annotation_protocol_and_inter_annotator_agreement.md)
- [ADR 005: Arsitektur Baseline B0 s.d. B3](docs/adr/005_baseline_architectures_b0_to_b3.md)
- [ADR 006: Automated Evaluation Harness & Formulasi Metrik](docs/adr/006_automated_evaluation_harness_and_metrics.md)
- [ADR 007: Baseline B4 — Self-Corrective RAG](docs/adr/007_self_corrective_rag_b4.md)
