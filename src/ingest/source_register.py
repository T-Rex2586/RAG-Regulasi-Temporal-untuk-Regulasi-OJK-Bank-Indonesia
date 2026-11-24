from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import hashlib
from typing import Dict, List, Optional


@dataclass
class SourceRecord:
    doc_id: str
    regulator: str
    jenis: str
    nomor: str
    tahun: str
    judul: str
    status: str
    url_sumber: str
    url_unduh: str
    tanggal_unduh: str
    hak_guna: str
    file_path: str
    sha256: str
    file_size_bytes: int


CSV_FIELDS = [
    "doc_id",
    "regulator",
    "jenis",
    "nomor",
    "tahun",
    "judul",
    "status",
    "url_sumber",
    "url_unduh",
    "tanggal_unduh",
    "hak_guna",
    "file_path",
    "sha256",
    "file_size_bytes",
]


class SourceRegister:
    def __init__(self, register_path: str | Path):
        self.register_path = Path(register_path)
        self.records: Dict[str, SourceRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.register_path.exists():
            return
        with open(self.register_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = SourceRecord(
                    doc_id=row["doc_id"],
                    regulator=row["regulator"],
                    jenis=row["jenis"],
                    nomor=row["nomor"],
                    tahun=row["tahun"],
                    judul=row["judul"],
                    status=row["status"],
                    url_sumber=row["url_sumber"],
                    url_unduh=row["url_unduh"],
                    tanggal_unduh=row["tanggal_unduh"],
                    hak_guna=row["hak_guna"],
                    file_path=row["file_path"],
                    sha256=row["sha256"],
                    file_size_bytes=int(row["file_size_bytes"]),
                )
                self.records[rec.doc_id] = rec

    def exists(self, doc_id: str) -> bool:
        return doc_id in self.records

    def add(self, record: SourceRecord) -> None:
        self.records[record.doc_id] = record

    def save(self) -> None:
        self.register_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.register_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for rec in self.records.values():
                writer.writerow(asdict(rec))

    def verify_integrity(self) -> Dict[str, any]:
        total = len(self.records)
        missing_files = []
        corrupted_hashes = []
        hashes_seen = set()
        duplicates = []

        for rec in self.records.values():
            p = Path(rec.file_path)
            if not p.exists():
                missing_files.append(rec.doc_id)
                continue
            content = p.read_bytes()
            curr_hash = hashlib.sha256(content).hexdigest()
            if curr_hash != rec.sha256:
                corrupted_hashes.append(rec.doc_id)
            if curr_hash in hashes_seen:
                duplicates.append(rec.doc_id)
            else:
                hashes_seen.add(curr_hash)

        return {
            "total_records": total,
            "unique_hashes": len(hashes_seen),
            "missing_files_count": len(missing_files),
            "corrupted_hashes_count": len(corrupted_hashes),
            "duplicates_count": len(duplicates),
            "is_valid": (
                total >= 300
                and len(missing_files) == 0
                and len(corrupted_hashes) == 0
                and len(duplicates) == 0
            ),
        }


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
