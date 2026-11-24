from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import re
from typing import List, Optional, Tuple, Dict, Any
import pymupdf


@dataclass
class AyatNode:
    ayat_nomor: str
    teks: str


@dataclass
class PasalNode:
    pasal_nomor: str
    bab_nomor: Optional[str]
    bab_judul: Optional[str]
    teks_lengkap: str
    ayat_list: List[AyatNode] = field(default_factory=list)


@dataclass
class BabNode:
    bab_nomor: str
    bab_judul: str


@dataclass
class ParsedDocument:
    doc_id: str
    regulator: str
    jenis: str
    nomor: str
    tahun: str
    judul: str
    status: str
    tanggal_ditetapkan: Optional[str]
    tanggal_berlaku: Optional[str]
    diundangkan_pada: Optional[str]
    mencabut: List[str]
    mengubah: List[str]
    bab_list: List[BabNode]
    pasal_list: List[PasalNode]
    total_pasal: int
    total_ayat: int
    total_karakter: int
    parsed_successfully: bool
    parse_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LegalDocParser:
    def __init__(self):
        self.re_bab = re.compile(
            r"(?m)^(BAB\s+[IVXLCDM]+)\s*\n+([A-Z\s\n,–\-\(\)/]{3,150}?)(?=\n\s*(?:BAB\s+[IVXLCDM]+|Pasal\s+\d+|Bagian|\Z))"
        )
        self.re_pasal_start = re.compile(
            r"(?m)^(?:(?:\n|\r\n)\s*|^\s*)Pasal\s+(\d+[A-Z]?)\s*$", re.IGNORECASE
        )
        self.re_ayat = re.compile(
            r"(?m)^\s*\((\d+)\)\s*(.*?)(?=(?:^\s*\(\d+\)|\Z))", re.DOTALL
        )
        self.re_mulai_berlaku = re.compile(
            r"mulai berlaku\s+(?:pada\s+tanggal|sejak\s+tanggal|pada)\s+([^\.\n;]+)",
            re.IGNORECASE,
        )
        self.re_ditetapkan = re.compile(
            r"Ditetapkan di\s+[^\n]+\s+pada tanggal\s+([^\.\n;]+)", re.IGNORECASE
        )
        self.re_diundangkan = re.compile(
            r"Diundangkan di\s+[^\n]+\s+pada tanggal\s+([^\.\n;]+)", re.IGNORECASE
        )
        self.re_mencabut = re.compile(
            r"(?:mencabut dan menyatakan tidak berlaku|dicabut dan dinyatakan tidak berlaku)\s+([^\.;\n]+(?:(?:\n[^\.;\n]+){1,3}))",
            re.IGNORECASE,
        )
        self.re_mengubah = re.compile(
            r"(?:Perubahan\s+(?:Pertama|Kedua|Ketiga|Keempat|Kelima|Keenam|Ketujuh|atas)?\s*([^\n\.;]+tentang[^\n\.;]+))",
            re.IGNORECASE,
        )

    def extract_text_from_pdf(self, pdf_path: str | Path) -> str:
        doc = pymupdf.open(str(pdf_path))
        text_pages = []
        for page in doc:
            t = page.get_text()
            cleaned_lines = []
            for line in t.split("\n"):
                sline = line.strip()
                if re.match(r"^-\s*\d+\s*-$", sline) or re.match(r"^\d+$", sline):
                    continue
                if "www.ojk.go.id" in sline.lower() or "www.bi.go.id" in sline.lower():
                    continue
                cleaned_lines.append(line)
            text_pages.append("\n".join(cleaned_lines))
        return "\n\n".join(text_pages)

    def parse(
        self,
        pdf_path: str | Path,
        doc_id: str,
        regulator: str,
        jenis: str,
        nomor: str,
        tahun: str,
        judul: str,
        status: str,
    ) -> ParsedDocument:
        raw_text = self.extract_text_from_pdf(pdf_path)
        total_chars = len(raw_text)

        # 1. Metadata extraction
        tgl_ditetapkan = None
        match_dit = self.re_ditetapkan.search(raw_text)
        if match_dit:
            tgl_ditetapkan = match_dit.group(1).strip()

        tgl_diundangkan = None
        match_diu = self.re_diundangkan.search(raw_text)
        if match_diu:
            tgl_diundangkan = match_diu.group(1).strip()

        tgl_berlaku = None
        match_ber = self.re_mulai_berlaku.search(raw_text)
        if match_ber:
            tgl_berlaku = match_ber.group(1).strip()
            if "diundangkan" in tgl_berlaku.lower() and tgl_diundangkan:
                tgl_berlaku = f"{tgl_berlaku} ({tgl_diundangkan})"
        elif tgl_diundangkan:
            tgl_berlaku = f"diundangkan ({tgl_diundangkan})"

        # Revocations & Amendments
        mencabut = [
            c.strip().replace("\n", " ")
            for c in self.re_mencabut.findall(raw_text)
        ]
        mengubah = [
            u.strip().replace("\n", " ")
            for u in self.re_mengubah.findall(raw_text)
        ]

        # 2. Extract Bab
        bab_list: List[BabNode] = []
        bab_spans: List[Tuple[str, str, int, int]] = []
        for m in self.re_bab.finditer(raw_text):
            b_nomor = m.group(1).strip()
            b_judul = re.sub(r"\s+", " ", m.group(2).strip())
            bab_list.append(BabNode(bab_nomor=b_nomor, bab_judul=b_judul))
            bab_spans.append((b_nomor, b_judul, m.start(), m.end()))

        # 3. Extract Pasal
        pasal_matches = list(self.re_pasal_start.finditer(raw_text))
        pasal_list: List[PasalNode] = []

        for i, match in enumerate(pasal_matches):
            p_nomor = match.group(1)
            start_pos = match.end()
            end_pos = (
                pasal_matches[i + 1].start()
                if i + 1 < len(pasal_matches)
                else len(raw_text)
            )

            p_text = raw_text[start_pos:end_pos].strip()
            # Clean trailing Ketentuan Penutup / Tanda Tangan from the last pasal
            if i + 1 == len(pasal_matches):
                penutup_idx = re.search(
                    r"(?m)^(?:Agar setiap orang mengetahuinya|Ditetapkan di|GUBERNUR BANK INDONESIA|DEWAN KOMISIONER)",
                    p_text,
                )
                if penutup_idx:
                    p_text = p_text[: penutup_idx.start()].strip()

            # Assign to Bab
            p_bab_nomor = None
            p_bab_judul = None
            for b_nomor, b_judul, b_start, b_end in bab_spans:
                if b_start < match.start():
                    p_bab_nomor = b_nomor
                    p_bab_judul = b_judul
                else:
                    break

            # Parse Ayat
            ayat_matches = list(self.re_ayat.finditer(p_text))
            ayat_list: List[AyatNode] = []
            if ayat_matches:
                for am in ayat_matches:
                    a_no = f"({am.group(1)})"
                    a_txt = am.group(2).strip()
                    ayat_list.append(AyatNode(ayat_nomor=a_no, teks=a_txt))
            else:
                ayat_list.append(AyatNode(ayat_nomor="pokok", teks=p_text))

            pasal_list.append(
                PasalNode(
                    pasal_nomor=p_nomor,
                    bab_nomor=p_bab_nomor,
                    bab_judul=p_bab_judul,
                    teks_lengkap=p_text,
                    ayat_list=ayat_list,
                )
            )

        total_ayat = sum(len(p.ayat_list) for p in pasal_list)
        parsed_success = len(pasal_list) > 0 or total_chars > 500

        notes = ""
        if len(pasal_list) == 0:
            notes = "Dokumen tidak memiliki struktur Pasal eksplisit (misal format SE/Surat Edaran naratif)."
            # Create a single fallback section
            pasal_list.append(
                PasalNode(
                    pasal_nomor="Naratif",
                    bab_nomor=None,
                    bab_judul=None,
                    teks_lengkap=raw_text,
                    ayat_list=[AyatNode(ayat_nomor="isi", teks=raw_text)],
                )
            )

        return ParsedDocument(
            doc_id=doc_id,
            regulator=regulator,
            jenis=jenis,
            nomor=nomor,
            tahun=tahun,
            judul=judul,
            status=status,
            tanggal_ditetapkan=tgl_ditetapkan,
            tanggal_berlaku=tgl_berlaku,
            diundangkan_pada=tgl_diundangkan,
            mencabut=mencabut,
            mengubah=mengubah,
            bab_list=bab_list,
            pasal_list=pasal_list,
            total_pasal=len(pasal_list),
            total_ayat=total_ayat,
            total_karakter=total_chars,
            parsed_successfully=parsed_success,
            parse_notes=notes,
        )
