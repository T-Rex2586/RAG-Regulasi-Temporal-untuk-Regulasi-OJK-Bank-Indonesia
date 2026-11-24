# ADR 001: Akuisisi Korpus Regulasi dan Kepatuhan Hak Guna

## Status
Diterima (Accepted)

## Konteks
Sistem Temporal Regulatory QA membutuhkan korpus resmi minimal 300 dokumen regulasi perbankan dan jasa keuangan Indonesia dari Otoritas Jasa Keuangan (OJK) dan Bank Indonesia (BI), mencakup POJK, SEOJK, PBI, dan PADG.

## Keputusan
1. **Sumber Resmi**:
   - Sumber OJK: Portal resmi JDIH OJK (`https://jdih.ojk.go.id`) dan portal regulasi OJK (`https://ojk.go.id`).
   - Sumber BI: Portal resmi JDIH Bank Indonesia (`https://jdih.bi.go.id`) dan publikasi peraturan BI (`https://bi.go.id`).

2. **Kepatuhan Hak Guna & Etika**:
   - Seluruh dokumen merupakan peraturan perundang-undangan publik yang diterbitkan secara terbuka oleh otoritas negara Republik Indonesia berdasarkan Undang-Undang No. 14 Tahun 2008 tentang Keterbukaan Informasi Publik dan Peraturan Presiden tentang Jaringan Dokumentasi dan Informasi Hukum Nasional (JDIHN).
   - Dokumen hukum publik tidak memuat hak cipta perorangan dan bebas diakses untuk keperluan penelitian dan akademik.
   - Pengecekan `robots.txt` Bank Indonesia mengizinkan akses publik (`Allow: /`) dengan pembatasan direktori internal sistem.
   - Scraper menerapkan pembatasan laju permintaan (rate limiting) dengan jeda antar unduhan guna menjaga performa server instansi.

3. **Verifikasi Integritas Data**:
   - Setiap berkas PDF dihitung nilai hash kriptografis SHA-256 saat pengunduhan.
   - Setiap entri didokumentasikan dalam `data/source_register.csv` yang memuat ID dokumen, nomor, tahun, judul, status masa berlaku, tautan sumber, tautan unduhan, timestamp pengunduhan, hash SHA-256, dan ukuran berkas.
