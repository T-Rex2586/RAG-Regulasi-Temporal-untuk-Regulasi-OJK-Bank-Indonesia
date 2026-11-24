# PRD: Temporal Regulatory QA OJK/BI (TBP Frontier LLM)

## 0. Instruksi untuk Gemini

| Aturan | Isi |
|---|---|
| Peran | Engineer pelaksana. Kerjakan per fase, satu fase per sesi, tunggu konfirmasi sebelum lanjut |
| Output tiap fase | Kode siap jalan, config, perintah eksekusi, dan checklist acceptance |
| Gaya kode | Python 3.11, bersih, tanpa komentar tidak perlu, type hints seperlunya |
| Perubahan | Edit aditif minimal, jangan menulis ulang file yang sudah ada |
| Reproducibility | Semua seed, versi model, versi index, dan prompt tercatat di `runs/<id>/manifest.json` |
| Larangan | Jangan bocorkan label/qrels ke prompt; jangan tuning di test; jangan ganti base model antar B0-B5; jangan klaim tanpa angka |
| Jika ragu | Tanya, jangan berasumsi |

## 1. Ringkasan

| Item | Isi |
|---|---|
| Use case | Temporal Regulatory QA regulasi OJK dan BI (POJK, SEOJK, PBI, PADG) |
| Pengguna | Mahasiswa pengerja tunggal, dinilai sebagai riset + rekayasa |
| Masalah | Aturan berubah versi, ada tanggal berlaku, saling rujuk antar pasal, dan banyak pertanyaan di luar cakupan |
| Output sistem | `answer`, `claims[]` (atomik), `citations[]` (dokumen, pasal, ayat), `as_of`, `verdict` (supported/unknown), `abstain_reason` |
| Base model | Qwen2.5-1.5B-Instruct, sama untuk B0-B5 |
| Stack | BM25 (rank_bm25/Pyserini), bge-m3, bge-reranker-v2-m3, Qdrant, FastAPI, Gradio |

## 2. Target Minimum (WAJIB)

| Elemen | Target | Bukti |
|---|---|---|
| Corpus | ≥300 dokumen resmi ojk.go.id / bi.go.id | `source_register.csv` (url, tanggal unduh, hak guna, SHA-256, versi) |
| Benchmark | ≥100 query, split train/dev/test terkunci | `splits/manifest.json` + SHA-256, cek dedup dan near-dup |
| Qrels | Untuk semua query answerable di test; label unanswerable untuk query tanpa evidence | `qrels.tsv` (id, relevance, doc, passage, versi, aturan anotasi) |
| Anotasi | 2 anotator blind, adjudikasi bila beda | `agreement.json`, `disagreement_log.csv` |
| Red-team | 20 kasus: 5 injection, 5 PII/policy leak, 5 tool misuse, 5 safety | `redteam/fixtures.jsonl` (id, goal, surface, baseline, mitigasi) |
| LLM comparison | Base vs LoRA/QLoRA | Raw output per item, versi prompt/model/index, token, latensi |
| Repeatability | ≥3 run independen pada subset test per konfigurasi | `runs/` + varians |
| Serving | UI + API + log/trace + fallback | Screenshot, log, demo langsung |

## 3. Struktur Repo

| Path | Fungsi |
|---|---|
| `data/raw/` | PDF asli |
| `data/parsed/` | Teks per pasal/ayat + metadata |
| `data/source_register.csv` | Register sumber |
| `benchmark/` | queries, qrels, splits, agreement |
| `index/` | BM25 dan Qdrant (versi index) |
| `src/ingest/` | scraper, parser, chunker |
| `src/retrieval/` | bm25, dense, hybrid, rerank |
| `src/pipeline/` | b0 sampai b6 |
| `src/verify/` | claim verifier, temporal checker |
| `src/tools/` | cek_masa_berlaku, cari_pasal, bandingkan_versi |
| `src/train/` | QLoRA |
| `src/eval/` | metrik, harness, run manifest |
| `src/serve/` | FastAPI + Gradio |
| `prompts/` | prompt registry berversi |
| `redteam/` | fixture dan hasil |
| `configs/` | satu YAML per konfigurasi/ablation |
| `docs/adr/` | keputusan desain |

## 4. Fase Kerja

| Fase | Tugas Gemini | Acceptance |
|---|---|---|
| 1. Corpus | Scraper ojk.go.id dan bi.go.id, unduh PDF, hash SHA-256, isi source register, cek robots/hak guna | ≥300 dokumen, register lengkap, tanpa duplikat |
| 2. Parsing | PDF ke teks, deteksi Bab/Pasal/Ayat, ekstrak metadata (jenis, nomor, tahun, tanggal berlaku, status, mencabut/mengubah) | ≥95% dokumen terparse, sampel 30 dicek manual |
| 3. Chunking dan index | Tiga strategi: fixed, heading-aware, hierarchical. Index BM25 dan Qdrant dengan metadata filter | Index terversi, ukuran index dan panjang konteks tercatat |
| 4. Benchmark | Generator query semi-otomatis dari pasal (easy, multi-hop, konflik versi, "berlaku per tanggal X", unanswerable), dikurasi manual. Split train/dev/test, kunci test | ≥100 query, distribusi slice seimbang, dedup lolos |
| 5. Anotasi | Template anotasi blind, skrip agreement (Cohen kappa), log disagreement | Agreement terhitung, adjudikasi terdokumentasi |
| 6. B0-B3 | B0 zero-shot, B1 few-shot (train/dev saja) + schema JSON + aturan abstain, B2 dense + fixed chunk, B3 BM25 + dense (RRF) + reranker + filter metadata | Semua pakai base model sama, hasil dev tersimpan |
| 7. Eval harness | Recall@5/10, nDCG@10, MRR@10, correctness, evidence recall, citation precision, temporal accuracy, unsupported claim rate, false-refusal, token, p50/p95 latensi | Satu perintah menjalankan eval dan menulis manifest |
| 8. B4 | Query planner, skor kecukupan evidence, retry maksimal 2x, claim verifier independen, abstain | Retry bounded, biaya tercatat |
| 9. B5 | QLoRA pada parent B4, data train berizin dan terpisah dari test, simpan adapter dan log training | Config GPU, loss curve, adapter tersimpan, uji OOD |
| 10. B6 | Parent terbaik di dev + tool terstruktur dengan otorisasi dan budget | Tool correctness, unsafe call rate, budget overrun terukur |
| 11. Ablation | A1-A7, satu variabel per percobaan, pilih konfigurasi di dev, laporan di test | Tabel hasil per ablation |
| 12. Red-team | 20 fixture, jalankan pada baseline dan mitigasi | Tabel outcome sebelum/sesudah |
| 13. Serving | FastAPI (`/ask`, `/health`, `/trace/{id}`), Gradio, logging, fallback | Demo berjalan, log dan trace tersedia |
| 14. Laporan | Metodologi, hasil, analisis error per sumber (retrieval, reasoning, tool, source, prompt, adaptation), batasan, ADR | Semua klaim punya definisi dan angka |

## 5. Konfigurasi B0-B6

| ID | Fitur | Pembeda utama |
|---|---|---|
| B0 | Zero-shot, tanpa retrieval | Baseline |
| B1 | Few-shot, schema, aturan abstain | Efek prompt |
| B2 | Dense + fixed chunk + top-k | Efek retrieval |
| B3 | BM25 + dense + RRF + reranker + filter | Efek hybrid dan latensi |
| B4 | Planner, sufficiency, retry, verifier, abstain | Efek self-corrective |
| B5 | B4 + LoRA/QLoRA | Efek adaptasi |
| B6 | Parent terbaik (dev) + tool | Efek agentic, dibandingkan dengan parent identik |

## 6. Ablation

| ID | Variabel | Metrik |
|---|---|---|
| A1 | Fixed vs heading-aware vs hierarchical | Passage recall, ukuran index, panjang konteks, groundedness |
| A2 | Dense vs BM25 vs hybrid vs hybrid+rerank | Recall@5/10, nDCG@10, MRR@10 |
| A3 | Tanpa vs dengan verifier + repair | Unsupported claim, false-refusal |
| A4 | Tanpa vs dengan LoRA pada B4 | Gain, overfit, resource, OOD |
| A5 | B6 tanpa vs dengan tool | Tool correctness, unsafe call, budget overrun, end-task |
| A6 | Retrieval fixed vs adaptif | Kualitas per slice, biaya per kualitas |
| A7 | Budget 2k vs 6k token | Pareto kualitas, p95, biaya |

## 7. Tool B6

| Tool | Argumen | Aturan |
|---|---|---|
| `cek_masa_berlaku` | `doc_id`, `as_of` | Baca-saja, validasi tanggal |
| `cari_pasal` | `doc_id`, `pasal`, `ayat?` | Hanya dokumen di corpus |
| `bandingkan_versi` | `doc_id_a`, `doc_id_b` | Maksimal 2 dokumen per panggilan |

Semua argumen divalidasi skema; panggilan di luar skema ditolak dan dicatat. Batas panggilan per query ditetapkan di config.

## 8. Anti-Kebocoran dan Etika

| Aturan | Implementasi |
|---|---|
| Test terkunci | Hash split diperiksa sebelum setiap eval; test tidak dipakai memilih konfigurasi |
| Tanpa ground truth di prompt | Few-shot hanya dari train/dev |
| Tanpa PII ke API eksternal | Corpus publik saja; jika memakai API eksternal, catat di ADR |
| Klaim | "lebih baik", "aman", "grounded", "agentic" hanya dengan definisi operasional dan angka |

## 9. Definisi Metrik

| Metrik | Definisi |
|---|---|
| Correctness | Jawaban sesuai gold (judge terkalibrasi + cek manual sampel) |
| Evidence recall | Proporsi passage gold yang terambil dan dikutip |
| Citation precision | Proporsi sitasi yang benar-benar mendukung klaim |
| Temporal accuracy | Versi/pasal yang dipilih benar untuk `as_of` |
| Unsupported claim rate | Klaim tanpa dukungan evidence / total klaim |
| False-refusal | Abstain pada query answerable / total answerable |
| False confident | Jawab pasti pada query unanswerable / total unanswerable |

## 10. Risiko

| Risiko | Mitigasi |
|---|---|
| Parsing PDF pasal buruk | Fallback OCR, sampel manual, aturan regex per jenis dokumen |
| Hak guna corpus | Catat di register, gunakan dokumen publik resmi |
| Anotator kedua terbatas | Teman/dosen untuk subset, LLM sebagai anotator kedua, adjudikasi manual, catat di ADR |
| GPU terbatas | QLoRA 4-bit di Colab/Kaggle, simpan bukti konfigurasi |
| Waktu | Pangkas A1/A7 ke subset, 3 tool B6 saja |
