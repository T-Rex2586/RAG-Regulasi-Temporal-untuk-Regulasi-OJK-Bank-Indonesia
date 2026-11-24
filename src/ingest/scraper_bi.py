from pathlib import Path
import re
import time
from datetime import datetime, timezone
from typing import List, Optional
import httpx
from bs4 import BeautifulSoup

from src.ingest.source_register import SourceRecord, SourceRegister, compute_sha256


class BIScraper:
    def __init__(
        self,
        register: SourceRegister,
        raw_dir: str | Path = "data/raw",
        base_url: str = "https://jdih.bi.go.id",
        timeout: float = 45.0,
        delay: float = 0.5,
        user_agent: Optional[str] = None,
    ):
        self.register = register
        self.raw_bi_dir = Path(raw_dir) / "bi"
        self.raw_bi_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        headers = {
            "User-Agent": user_agent
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.client = httpx.Client(headers=headers, verify=False, timeout=timeout, follow_redirects=True)

    def scrape(self, target_count: int = 160) -> int:
        listing_url = f"{self.base_url}/Web/DaftarPeraturan"
        resp = self.client.get(listing_url)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        cards = soup.find_all("div", class_="card")

        downloaded_count = 0
        for card in cards:
            if downloaded_count >= target_count:
                break

            title_link = card.find("h5", class_="card-title")
            if not title_link:
                continue
            a_tag = title_link.find("a", href=True)
            if not a_tag:
                continue

            title_text = a_tag.get_text(strip=True)
            href = a_tag["href"]

            match_id = re.search(r"/Detail/(\d+)", href)
            if not match_id:
                onclick = card.find("a", onclick=True)
                if onclick:
                    match_id = re.search(r"DownloadPdf\('(\d+)'", onclick["onclick"])
            if not match_id:
                continue

            peraturan_id = match_id.group(1)

            # Jenis & Nomor & Tahun
            jenis = "PBI" if "pbi" in title_text.lower() or "peraturan bank indonesia" in title_text.lower() else "PADG"
            tahun_tag = card.find("i", class_="tahun")
            tahun = tahun_tag.get_text(strip=True) if tahun_tag else ""
            if not tahun:
                match_th = re.search(r"\b(19\d{2}|20\d{2})\b", title_text)
                tahun = match_th.group(1) if match_th else "UNKNOWN"

            match_no = re.search(r"Nomor\s+([0-9/\.\-A-Za-z]+)", title_text, re.IGNORECASE)
            nomor = match_no.group(1) if match_no else peraturan_id

            clean_num = re.sub(r"[^a-zA-Z0-9]", "_", nomor).strip("_")
            doc_id = f"BI_{jenis}_{tahun}_{clean_num}_{peraturan_id}"

            if self.register.exists(doc_id):
                downloaded_count += 1
                continue

            status_tag = card.find("span", class_="badge")
            status = status_tag.get_text(strip=True) if status_tag else "Berlaku"

            ringkasan_tag = card.find("p", class_="ringkasan")
            judul = ringkasan_tag.get_text(strip=True) if ringkasan_tag else title_text

            download_url = f"{self.base_url}/api/WebJDIH/GetFilePeraturan/{peraturan_id}"
            source_url = f"{self.base_url}{href}"

            try:
                pdf_resp = self.client.get(download_url)
                if pdf_resp.status_code != 200 or len(pdf_resp.content) < 500:
                    continue
                if not pdf_resp.content.startswith(b"%PDF"):
                    continue

                pdf_bytes = pdf_resp.content
                sha256_hash = compute_sha256(pdf_bytes)
                file_name = f"{doc_id}.pdf"
                dest_path = self.raw_bi_dir / file_name
                dest_path.write_bytes(pdf_bytes)

                rec = SourceRecord(
                    doc_id=doc_id,
                    regulator="BI",
                    jenis=jenis,
                    nomor=nomor,
                    tahun=tahun,
                    judul=judul,
                    status=status,
                    url_sumber=source_url,
                    url_unduh=download_url,
                    tanggal_unduh=datetime.now(timezone.utc).isoformat(),
                    hak_guna="Dokumen Publik Terbuka JDIH Bank Indonesia (UU 14/2008 & JDIHN)",
                    file_path=str(dest_path).replace("\\", "/"),
                    sha256=sha256_hash,
                    file_size_bytes=len(pdf_bytes),
                )
                self.register.add(rec)
                self.register.save()
                downloaded_count += 1

                time.sleep(self.delay)
            except Exception as e:
                print(f"[BI] Error downloading {doc_id}: {e}")
                continue

        return downloaded_count
