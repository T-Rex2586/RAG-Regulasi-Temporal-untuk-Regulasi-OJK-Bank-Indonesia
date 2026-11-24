from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re
from typing import List, Dict, Any, Optional


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    regulator: str
    jenis: str
    nomor: str
    tahun: str
    status: str
    tanggal_berlaku: Optional[str]
    bab_nomor: Optional[str]
    bab_judul: Optional[str]
    pasal_nomor: Optional[str]
    ayat_nomor: Optional[str]
    strategy: str
    text: str
    parent_text: Optional[str] = None
    char_count: int = 0
    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Chunker:
    def __init__(
        self,
        fixed_size: int = 600,
        fixed_overlap: int = 100,
        heading_max: int = 1500,
        hierarchical_child_max: int = 600,
    ):
        self.fixed_size = fixed_size
        self.fixed_overlap = fixed_overlap
        self.heading_max = heading_max
        self.hierarchical_child_max = hierarchical_child_max

    def chunk_fixed(self, doc_data: Dict[str, Any]) -> List[Chunk]:
        chunks: List[Chunk] = []
        doc_id = doc_data["doc_id"]
        regulator = doc_data["regulator"]
        jenis = doc_data["jenis"]
        nomor = doc_data["nomor"]
        tahun = doc_data["tahun"]
        status = doc_data["status"]
        tgl_berlaku = doc_data.get("tanggal_berlaku")

        full_text_parts = [
            f"{regulator} {jenis} Nomor {nomor} Tahun {tahun}",
            doc_data.get("judul", ""),
        ]
        for p in doc_data.get("pasal_list", []):
            full_text_parts.append(f"Pasal {p.get('pasal_nomor')}\n{p.get('teks_lengkap', '')}")

        full_text = "\n\n".join(full_text_parts)
        text_len = len(full_text)

        start = 0
        idx = 0
        step = self.fixed_size - self.fixed_overlap
        if step <= 0:
            step = self.fixed_size

        while start < text_len:
            end = min(start + self.fixed_size, text_len)
            sub_text = full_text[start:end].strip()
            if len(sub_text) > 30:
                c_id = f"{doc_id}_f_{idx:04d}"
                chunks.append(
                    Chunk(
                        chunk_id=c_id,
                        doc_id=doc_id,
                        regulator=regulator,
                        jenis=jenis,
                        nomor=nomor,
                        tahun=tahun,
                        status=status,
                        tanggal_berlaku=tgl_berlaku,
                        bab_nomor=None,
                        bab_judul=None,
                        pasal_nomor=None,
                        ayat_nomor=None,
                        strategy="fixed",
                        text=sub_text,
                        parent_text=None,
                        char_count=len(sub_text),
                        token_count=len(sub_text.split()),
                    )
                )
                idx += 1
            start += step

        return chunks

    def chunk_heading_aware(self, doc_data: Dict[str, Any]) -> List[Chunk]:
        chunks: List[Chunk] = []
        doc_id = doc_data["doc_id"]
        regulator = doc_data["regulator"]
        jenis = doc_data["jenis"]
        nomor = doc_data["nomor"]
        tahun = doc_data["tahun"]
        status = doc_data["status"]
        tgl_berlaku = doc_data.get("tanggal_berlaku")

        for p in doc_data.get("pasal_list", []):
            p_no = p.get("pasal_nomor", "")
            b_no = p.get("bab_nomor")
            b_jud = p.get("bab_judul")
            p_text = p.get("teks_lengkap", "").strip()

            header_prefix = (
                f"[{regulator} {jenis} No {nomor}/{tahun}] "
                f"[{b_no or 'BAB'}: {b_jud or 'Ketentuan'}] "
                f"[Pasal {p_no}] "
            )

            if len(header_prefix) + len(p_text) <= self.heading_max:
                full_chunk_text = f"{header_prefix}\n{p_text}"
                c_id = f"{doc_id}_ha_p{p_no}_00"
                chunks.append(
                    Chunk(
                        chunk_id=c_id,
                        doc_id=doc_id,
                        regulator=regulator,
                        jenis=jenis,
                        nomor=nomor,
                        tahun=tahun,
                        status=status,
                        tanggal_berlaku=tgl_berlaku,
                        bab_nomor=b_no,
                        bab_judul=b_jud,
                        pasal_nomor=p_no,
                        ayat_nomor=None,
                        strategy="heading_aware",
                        text=full_chunk_text,
                        parent_text=None,
                        char_count=len(full_chunk_text),
                        token_count=len(full_chunk_text.split()),
                    )
                )
            else:
                # Split large pasal by ayats or blocks
                ayats = p.get("ayat_list", [])
                if len(ayats) > 1:
                    current_block = []
                    current_len = len(header_prefix)
                    sub_idx = 0
                    for a in ayats:
                        a_line = f"{a.get('ayat_nomor', '')} {a.get('teks', '')}".strip()
                        if current_len + len(a_line) > self.heading_max and current_block:
                            block_text = f"{header_prefix}\n" + "\n".join(current_block)
                            c_id = f"{doc_id}_ha_p{p_no}_{sub_idx:02d}"
                            chunks.append(
                                Chunk(
                                    chunk_id=c_id,
                                    doc_id=doc_id,
                                    regulator=regulator,
                                    jenis=jenis,
                                    nomor=nomor,
                                    tahun=tahun,
                                    status=status,
                                    tanggal_berlaku=tgl_berlaku,
                                    bab_nomor=b_no,
                                    bab_judul=b_jud,
                                    pasal_nomor=p_no,
                                    ayat_nomor=None,
                                    strategy="heading_aware",
                                    text=block_text,
                                    parent_text=None,
                                    char_count=len(block_text),
                                    token_count=len(block_text.split()),
                                )
                            )
                            sub_idx += 1
                            current_block = [a_line]
                            current_len = len(header_prefix) + len(a_line)
                        else:
                            current_block.append(a_line)
                            current_len += len(a_line) + 1

                    if current_block:
                        block_text = f"{header_prefix}\n" + "\n".join(current_block)
                        c_id = f"{doc_id}_ha_p{p_no}_{sub_idx:02d}"
                        chunks.append(
                            Chunk(
                                chunk_id=c_id,
                                doc_id=doc_id,
                                regulator=regulator,
                                jenis=jenis,
                                nomor=nomor,
                                tahun=tahun,
                                status=status,
                                tanggal_berlaku=tgl_berlaku,
                                bab_nomor=b_no,
                                bab_judul=b_jud,
                                pasal_nomor=p_no,
                                ayat_nomor=None,
                                strategy="heading_aware",
                                text=block_text,
                                parent_text=None,
                                char_count=len(block_text),
                                token_count=len(block_text.split()),
                            )
                        )
                else:
                    # Generic text slice
                    step = self.heading_max - 200
                    sub_idx = 0
                    for s_start in range(0, len(p_text), step):
                        s_sub = p_text[s_start : s_start + self.heading_max].strip()
                        full_chunk_text = f"{header_prefix}\n{s_sub}"
                        c_id = f"{doc_id}_ha_p{p_no}_{sub_idx:02d}"
                        chunks.append(
                            Chunk(
                                chunk_id=c_id,
                                doc_id=doc_id,
                                regulator=regulator,
                                jenis=jenis,
                                nomor=nomor,
                                tahun=tahun,
                                status=status,
                                tanggal_berlaku=tgl_berlaku,
                                bab_nomor=b_no,
                                bab_judul=b_jud,
                                pasal_nomor=p_no,
                                ayat_nomor=None,
                                strategy="heading_aware",
                                text=full_chunk_text,
                                parent_text=None,
                                char_count=len(full_chunk_text),
                                token_count=len(full_chunk_text.split()),
                            )
                        )
                        sub_idx += 1

        return chunks

    def chunk_hierarchical(self, doc_data: Dict[str, Any]) -> List[Chunk]:
        chunks: List[Chunk] = []
        doc_id = doc_data["doc_id"]
        regulator = doc_data["regulator"]
        jenis = doc_data["jenis"]
        nomor = doc_data["nomor"]
        tahun = doc_data["tahun"]
        status = doc_data["status"]
        tgl_berlaku = doc_data.get("tanggal_berlaku")

        for p in doc_data.get("pasal_list", []):
            p_no = p.get("pasal_nomor", "")
            b_no = p.get("bab_nomor")
            b_jud = p.get("bab_judul")
            p_text = p.get("teks_lengkap", "").strip()

            parent_context = (
                f"[{regulator} {jenis} No {nomor}/{tahun}] "
                f"[{b_no or 'BAB'}: {b_jud or 'Ketentuan'}] "
                f"[Pasal {p_no}]\n{p_text}"
            )

            ayats = p.get("ayat_list", [])
            for a_idx, a in enumerate(ayats):
                a_no = a.get("ayat_nomor", "pokok")
                a_text = a.get("teks", "").strip()
                if not a_text:
                    continue

                clean_a = re.sub(r"[^a-zA-Z0-9]", "", a_no) or f"{a_idx}"
                child_text = f"[{regulator} {jenis} No {nomor}/{tahun} Pasal {p_no} Ayat {a_no}]\n{a_text}"
                c_id = f"{doc_id}_hi_p{p_no}_a{clean_a}_{a_idx:02d}"

                chunks.append(
                    Chunk(
                        chunk_id=c_id,
                        doc_id=doc_id,
                        regulator=regulator,
                        jenis=jenis,
                        nomor=nomor,
                        tahun=tahun,
                        status=status,
                        tanggal_berlaku=tgl_berlaku,
                        bab_nomor=b_no,
                        bab_judul=b_jud,
                        pasal_nomor=p_no,
                        ayat_nomor=a_no,
                        strategy="hierarchical",
                        text=child_text,
                        parent_text=parent_context,
                        char_count=len(child_text),
                        token_count=len(child_text.split()),
                    )
                )

        return chunks
