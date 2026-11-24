# Pedoman Anotasi Relevansi dan Temporalitas (Annotation Guidelines)
## Proyek: Temporal Regulatory QA OJK & BI (TBP Frontier LLM)

Dokumen ini memuat panduan operasional bagi anotator independen (blind annotators) dan adjudikator dalam menilai relevansi potongan teks regulasi (*passage/chunk*) terhadap pertanyaan (*query*) hukum perbankan dan pasar keuangan.

---

### 1. Skala Skor Relevansi (Relevance Grades)

Setiap pasangan `(query, passage)` dinilai menggunakan skala 3-tingkat:

| Skor | Kategori | Definisi Operasional |
|:---:|---|---|
| **2** | **Highly Relevant / Gold Evidence** | Passage secara langsung menjawab substansi inti query. Memuat fakta hukum definitif, batasan numerik, formula, persyaratan utama, atau amar sanksi yang ditanyakan, serta sah berlaku pada tanggal acuan (`as_of`). |
| **1** | **Partially Relevant / Contextual** | Passage memberikan informasi latar belakang, definisi pendukung, atau rujukan prosedural yang berkaitan erat dengan topik query, namun **belum cukup** untuk menjawab query secara mandiri tanpa konteks pasal lain. |
| **0** | **Irrelevant / Non-Evidence** | Passage tidak membahas substansi yang ditanyakan, berasal dari topik berbeda, atau memuat aturan yang **tidak sah / belum berlaku / telah dicabut** pada tanggal acuan (`as_of`) query. |

---

### 2. Aturan Evaluasi Temporal (*Temporal Validity Rules*)

1. **Query Tanpa Parameter Waktu (`as_of = None`)**:
   - Diuji berdasarkan versi peraturan **terbaru yang berlaku sah** dalam korpus.
   - Jika pasal pada regulasi lama telah dicabut atau diubah oleh regulasi baru, pasal lama diberi skor `0` (atau maksimal `1` jika ditanyakan konteks historisnya).
2. **Query Dengan Parameter Waktu (`as_of = YYYY-MM-DD`)**:
   - Periksa tanggal pengundangan dan tanggal mulai berlaku (*effective date*) dari regulasi terkait.
   - Jika per tanggal `as_of` regulasi amandemen **belum berlaku**, maka passage dari regulasi amandemen tersebut berstatus `BELUM_BERLAKU` dan diberi skor `0` untuk pembuktian kondisi saat itu.
   - Sebaliknya, regulasi awal yang berlaku pada tanggal tersebut menjadi acuan sah (skor `2`).
3. **Query Konflik Versi (*Temporal Amendment*)**:
   - Baik pasal lama maupun pasal baru yang mengubah memiliki relevansi tinggi (skor `2`) karena query secara eksplisit menanyakan perbandingan atau dampak perubahannya.

---

### 3. Aturan Query di Luar Cakupan (*Unanswerable Queries*)

1. Untuk seluruh query berlabel `slice = unanswerable` (seperti ketentuan pajak PPh/PPN, retribusi parkir, izin tenaga kerja asing, baku mutu cerobong pabrik, atau regulasi fiktif masa depan):
   - **Seluruh candidate passage** wajib diberi skor `0` (Irrelevant).
   - Sistem wajib menghasilkan verdict **abstain** (`verdict = unknown`) dengan mencantumkan `abstain_reason` yang valid.
   - Pada berkas `qrels.tsv`, query unanswerable dicatat dengan `doc_id = NONE` dan `passage_id = NONE` berskor `0`.

---

### 4. Protokol Blind Annotation & Adjudikasi

1. **Anotasi Independen**: Dua anotator (Anotator 1 dan Anotator 2) melakukan pelabelan secara terpisah tanpa saling berkomunikasi atau melihat hasil penilaian satu sama lain.
2. **Kalkulasi Inter-Annotator Agreement (IAA)**:
   - Dihitung menggunakan **Cohen's Kappa ($\kappa$)** multikelas (skala 0, 1, 2) dan **Raw Percentage Agreement**.
3. **Penyelesaian Perbedaan (Disagreement Adjudication)**:
   - Setiap item di mana Anotator 1 $\neq$ Anotator 2 dicatat secara rinci dalam `benchmark/disagreement_log.csv`.
   - Senior Adjudicator memeriksa teks regulasi resmi, tanggal berlaku, dan hierarki hukum untuk menetapkan nilai final (*adjudicated gold score*).
   - Nilai konsensus hasil adjudikasi dimasukkan ke dalam `benchmark/qrels.tsv`.
