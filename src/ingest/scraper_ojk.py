from pathlib import Path
import re
import time
from datetime import datetime, timezone
from typing import List, Optional
import httpx
from bs4 import BeautifulSoup

from src.ingest.source_register import SourceRecord, SourceRegister, compute_sha256


class OJKScraper:
    def __init__(
        self,
        register: SourceRegister,
        raw_dir: str | Path = "data/raw",
        base_url: str = "https://jdih.ojk.go.id",
        timeout: float = 45.0,
        delay: float = 0.5,
        user_agent: Optional[str] = None,
    ):
        self.register = register
        self.raw_ojk_dir = Path(raw_dir) / "ojk"
        self.raw_ojk_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        headers = {
            "User-Agent": user_agent
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.client = httpx.Client(headers=headers, verify=False, timeout=timeout, follow_redirects=True)

    def scrape(
        self,
        target_count: int = 160,
        sektors: Optional[List[str]] = None,
        jenis_list: Optional[List[str]] = None,
    ) -> int:
        sektors = sektors or ["01", "02"]
        jenis_list = jenis_list or ["06", "09"]

        downloaded_count = 0

        for jenis_code in jenis_list:
            for sektor_code in sektors:
                if downloaded_count >= target_count:
                    break

                list_url = (
                    f"{self.base_url}/Web/ViewPeraturan/ListDataPeraturan"
                    f"?sektor={sektor_code}&jenisPeraturan={jenis_code}&sLanguage=id"
                )

                try:
                    resp = self.client.get(list_url)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                except Exception as e:
                    print(f"[OJK] Error fetching list for jenis {jenis_code} sektor {sektor_code}: {e}")
                    continue

                rows = data.get("aaData", [])
                for row in rows:
                    if downloaded_count >= target_count:
                        break

                    if not row or not isinstance(row, list) or len(row) < 8:
                        continue

                    link_html = row[0]
                    nomor_raw = str(row[1] or "")
                    status = str(row[7] or "Berlaku")

                    soup = BeautifulSoup(link_html, "html.parser")
                    a_tag = soup.find("a", href=True)
                    if not a_tag:
                        continue

                    judul_lengkap = a_tag.get_text(strip=True)
                    detail_href = a_tag["href"]

                    match_uuid = re.search(r"Detail/([a-f0-9\-]+)", detail_href, re.IGNORECASE)
                    if not match_uuid:
                        continue
                    uuid_str = match_uuid.group(1)

                    jenis = "POJK" if jenis_code == "06" else "SEOJK"

                    match_th = re.search(r"\b(19\d{2}|20\d{2})\b", judul_lengkap)
                    tahun = match_th.group(1) if match_th else "UNKNOWN"

                    clean_nomor = re.sub(r"[^a-zA-Z0-9]", "_", nomor_raw).strip("_")
                    short_uuid = uuid_str[:8]
                    doc_id = f"OJK_{jenis}_{tahun}_{clean_nomor}_{short_uuid}"

                    if self.register.exists(doc_id):
                        downloaded_count += 1
                        continue

                    # Fetch detail page to get download URL
                    detail_full_url = detail_href if detail_href.startswith("http") else f"{self.base_url}{detail_href}"
                    try:
                        detail_resp = self.client.get(detail_full_url)
                        if detail_resp.status_code != 200:
                            continue

                        detail_soup = BeautifulSoup(detail_resp.text, "html.parser")
                        dl_links = detail_soup.find_all("a", href=lambda h: h and "downloaddokumen" in h.lower())
                        if not dl_links:
                            continue
                        dl_a = None
                        for candidate in dl_links:
                            parent_tr = candidate.find_parent("tr")
                            tr_text = parent_tr.get_text().lower() if parent_tr else ""
                            if not any(k in tr_text for k in ["abs", "ringkasan", "faq"]):
                                dl_a = candidate
                                break
                        if not dl_a:
                            dl_a = dl_links[-1]

                        dl_href = dl_a["href"]
                        download_url = dl_href if dl_href.startswith("http") else f"{self.base_url}{dl_href}"

                        pdf_resp = self.client.get(download_url)
                        if pdf_resp.status_code != 200 or len(pdf_resp.content) < 500:
                            continue
                        if not pdf_resp.content.startswith(b"%PDF"):
                            continue

                        pdf_bytes = pdf_resp.content
                        sha256_hash = compute_sha256(pdf_bytes)
                        file_name = f"{doc_id}.pdf"
                        dest_path = self.raw_ojk_dir / file_name
                        dest_path.write_bytes(pdf_bytes)

                        rec = SourceRecord(
                            doc_id=doc_id,
                            regulator="OJK",
                            jenis=jenis,
                            nomor=nomor_raw,
                            tahun=tahun,
                            judul=judul_lengkap,
                            status=status,
                            url_sumber=detail_full_url,
                            url_unduh=download_url,
                            tanggal_unduh=datetime.now(timezone.utc).isoformat(),
                            hak_guna="Dokumen Publik Terbuka JDIH OJK (UU 14/2008 & JDIHN)",
                            file_path=str(dest_path).replace("\\", "/"),
                            sha256=sha256_hash,
                            file_size_bytes=len(pdf_bytes),
                        )
                        self.register.add(rec)
                        self.register.save()
                        downloaded_count += 1

                        time.sleep(self.delay)
                    except Exception as e:
                        print(f"[OJK] Error downloading {doc_id}: {e}")
                        continue

        return downloaded_count
