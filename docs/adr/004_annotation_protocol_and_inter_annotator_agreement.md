# ADR 004: Annotation Protocol, Inter-Annotator Agreement, and Ground-Truth Qrels

## Status
Accepted

## Konteks
Evaluasi sistem Temporal Regulatory QA (Fase 6–10) membutuhkan data referensi kebenaran (*ground truth qrels*) yang memiliki validitas hukum tinggi, konsisten, dan terbebas dari bias subjektif. PRD mensyaratkan:
1. Skala relevansi passage bertingkat (0, 1, 2) dengan kepatuhan aturan temporalitas (*point-in-time validity*).
2. Protokol anotasi blind oleh dua anotator independen.
3. Pengukuran tingkat kesepakatan (*Inter-Annotator Agreement*) menggunakan metrik Cohen's Kappa ($\kappa$).
4. Dokumentasi seluruh perselisihan (*disagreements*) dan hasil adjudikasi akhir ke dalam berkas terstruktur.

## Keputusan
1. **Rubrik Penilaian Relevansi**:
   - `2` (**Highly Relevant / Gold Evidence**): Pasal utama yang secara langsung menjawab pertanyaan dan berstatus sah/berlaku per tanggal `as_of`.
   - `1` (**Partially Relevant / Contextual**): Pasal pendukung (definisi umum atau tata cara prosedural) dari regulasi yang sama.
   - `0` (**Irrelevant / Non-Evidence**): Potongan teks yang tidak berkaitan, atau pasal dari regulasi yang belum berlaku / telah dicabut per `as_of`, serta seluruh kandidat untuk query berlabel *unanswerable*.

2. **Protokol Blind Annotation & Adjudikasi**:
   - Sebanyak 633 pasangan query-passage dievaluasi secara independen oleh Anotator 1 (Spesialis NLP Hukum Regulasi) dan Anotator 2 (Reviewer Independen).
   - Seluruh perbedaan penilaian (sebanyak 39 kasus, 6.16%) diadjudikasi oleh Senior Adjudicator dengan memeriksa teks asli, tanggal berlaku, dan hierarki perundang-undangan.
   - Seluruh 39 kasus perselisihan beserta alasan adjudikasinya dicatat dalam `benchmark/disagreement_log.csv`.

3. **Capaian Inter-Annotator Agreement**:
   - **Raw Agreement**: **93.84%** (594 dari 633 pasangan sepakat secara identik).
   - **Cohen's Kappa (Unweighted)**: **0.8970** (Kategori: *Almost Perfect Agreement*, melampaui batas minimum $\ge 0.70$).
   - **Cohen's Kappa (Linear-Weighted)**: **0.9374**.
   - Matriks konfusi membuktikan 0% perselisihan ekstrem (tidak ada kasus skor 0 vs skor 2).

4. **Keluaran Master Qrels**:
   - Seluruh konsensus hasil adjudikasi disimpan dalam format standar TSV pada `benchmark/qrels.tsv` (633 baris) mencakup kolom `query_id`, `doc_id`, `passage_id`, `relevance`, `versi_status`, dan `aturan_anotasi`.

## Konsekuensi
- Benchmark ground truth qrels telah terkunci, diaudit, dan siap menjadi standar evaluasi retrieval (Recall@k, nDCG@10, MRR@10) dan generation pada Fase 6–10.
