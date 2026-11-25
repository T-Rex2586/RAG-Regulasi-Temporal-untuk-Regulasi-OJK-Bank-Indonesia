from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional


MONTH_MAP = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4,
    "mei": 5, "juni": 6, "juli": 7, "agustus": 8,
    "september": 9, "oktober": 10, "november": 11, "desember": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "agt": 8, "sep": 9, "oct": 10, "okt": 10,
    "nov": 11, "dec": 12, "des": 12
}


def parse_indonesian_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parses Indonesian date strings like '7 September 2023', '2023-09-07', '1 Januari 2021'."""
    if not date_str:
        return None
    s = str(date_str).strip().lower()
    
    # Try ISO YYYY-MM-DD
    iso_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if iso_match:
        try:
            return datetime(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
        except ValueError:
            pass

    # Try DD Month YYYY
    parts = re.split(r"[\s,]+", s)
    day, month, year = None, None, None
    for p in parts:
        if p.isdigit():
            val = int(p)
            if 1900 <= val <= 2100 and year is None:
                year = val
            elif 1 <= val <= 31 and day is None:
                day = val
        elif p in MONTH_MAP and month is None:
            month = MONTH_MAP[p]

    if day and month and year:
        try:
            return datetime(year, month, day)
        except ValueError:
            pass
    elif year and month:
        return datetime(year, month, 1)
    elif year:
        return datetime(year, 1, 1)

    return None


def load_parsed_doc(doc_id: str, parsed_dir: str = "data/parsed") -> Optional[Dict[str, Any]]:
    """Loads parsed json document from data/parsed/<doc_id>.json."""
    clean_id = Path(doc_id).stem
    p = Path(parsed_dir) / f"{clean_id}.json"
    if not p.exists():
        # Case-insensitive search fallback
        for candidate in Path(parsed_dir).glob("*.json"):
            if candidate.stem.lower() == clean_id.lower():
                p = candidate
                break
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def cek_masa_berlaku(doc_id: str, as_of: str, parsed_dir: str = "data/parsed") -> Dict[str, Any]:
    """Validates whether a regulation was in force on the specified as_of date.

    Tool 1 from PRD Section 7.
    """
    doc = load_parsed_doc(doc_id, parsed_dir)
    if not doc:
        return {
            "status": "error",
            "message": f"Dokumen '{doc_id}' tidak ditemukan dalam korpus.",
            "is_valid": False,
        }

    as_of_dt = parse_indonesian_date(as_of)
    eff_dt = parse_indonesian_date(doc.get("tanggal_berlaku")) or parse_indonesian_date(doc.get("tanggal_ditetapkan"))
    doc_status = (doc.get("status") or "Berlaku").strip()

    result = {
        "doc_id": doc.get("doc_id", doc_id),
        "judul": doc.get("judul"),
        "status_tercatat": doc_status,
        "as_of": as_of,
        "tanggal_berlaku": doc.get("tanggal_berlaku") or doc.get("tanggal_ditetapkan"),
        "mencabut": doc.get("mencabut", []),
        "mengubah": doc.get("mengubah", []),
    }

    if not as_of_dt or not eff_dt:
        # Qualitative date match
        result["is_valid"] = "berlaku" in doc_status.lower()
        result["verdict"] = "Berlaku (berdasarkan status master korpus)" if result["is_valid"] else "Tidak Berlaku / Dicabut"
        return result

    if as_of_dt < eff_dt:
        result["is_valid"] = False
        result["verdict"] = f"Belum berlaku pada tanggal {as_of} (Mulai berlaku: {eff_dt.strftime('%d-%m-%Y')})"
    else:
        if "dicabut" in doc_status.lower():
            result["is_valid"] = False
            result["verdict"] = f"Telah dicabut / tidak berlaku lagi pada status terkini."
        else:
            result["is_valid"] = True
            result["verdict"] = f"Berlaku pada tanggal {as_of} (Mulai berlaku: {eff_dt.strftime('%d-%m-%Y')})"

    return result


def cari_pasal(
    doc_id: str,
    pasal: str | int,
    ayat: Optional[str | int] = None,
    parsed_dir: str = "data/parsed",
) -> Dict[str, Any]:
    """Retrieves exact article (pasal) and clause (ayat) text deterministically.

    Tool 2 from PRD Section 7.
    """
    doc = load_parsed_doc(doc_id, parsed_dir)
    if not doc:
        return {
            "status": "error",
            "message": f"Dokumen '{doc_id}' tidak ditemukan dalam korpus.",
            "found": False,
        }

    pasal_target = str(pasal).strip().lower().replace("pasal", "").strip()
    ayat_target = str(ayat).strip().lower().replace("ayat", "").replace("(", "").replace(")", "").strip() if ayat else None

    matched_pasal = None
    for p in doc.get("pasal_list", []):
        p_num = str(p.get("pasal_nomor", "")).strip().lower()
        if p_num == pasal_target:
            matched_pasal = p
            break

    if not matched_pasal:
        return {
            "doc_id": doc.get("doc_id", doc_id),
            "pasal": str(pasal),
            "ayat": str(ayat) if ayat else None,
            "found": False,
            "message": f"Pasal {pasal} tidak ditemukan dalam dokumen {doc_id}.",
        }

    # If specific ayat is requested
    if ayat_target:
        matched_ayat = None
        for a in matched_pasal.get("ayat_list", []):
            a_num = str(a.get("ayat_nomor", "")).strip().lower().replace("(", "").replace(")", "")
            if a_num == ayat_target:
                matched_ayat = a
                break

        if matched_ayat:
            return {
                "doc_id": doc.get("doc_id", doc_id),
                "pasal": matched_pasal.get("pasal_nomor"),
                "ayat": matched_ayat.get("ayat_nomor"),
                "found": True,
                "text": matched_ayat.get("teks", "").strip(),
            }

    return {
        "doc_id": doc.get("doc_id", doc_id),
        "pasal": matched_pasal.get("pasal_nomor"),
        "ayat": None,
        "found": True,
        "text": matched_pasal.get("teks_lengkap", "").strip(),
    }


def bandingkan_versi(doc_id_a: str, doc_id_b: str, parsed_dir: str = "data/parsed") -> Dict[str, Any]:
    """Compares two regulations to determine amendment, revocation, or precedence relationship.

    Tool 3 from PRD Section 7.
    """
    doc_a = load_parsed_doc(doc_id_a, parsed_dir)
    doc_b = load_parsed_doc(doc_id_b, parsed_dir)

    if not doc_a or not doc_b:
        missing = []
        if not doc_a: missing.append(doc_id_a)
        if not doc_b: missing.append(doc_id_b)
        return {
            "status": "error",
            "message": f"Dokumen {', '.join(missing)} tidak ditemukan dalam korpus.",
            "relationship": "unknown",
        }

    dt_a = parse_indonesian_date(doc_a.get("tanggal_berlaku")) or parse_indonesian_date(doc_a.get("tanggal_ditetapkan"))
    dt_b = parse_indonesian_date(doc_b.get("tanggal_berlaku")) or parse_indonesian_date(doc_b.get("tanggal_ditetapkan"))

    # Check revocation & amendment strings
    mencabut_b = " ".join(doc_b.get("mencabut", [])).lower()
    mengubah_b = " ".join(doc_b.get("mengubah", [])).lower()
    mencabut_a = " ".join(doc_a.get("mencabut", [])).lower()
    mengubah_a = " ".join(doc_a.get("mengubah", [])).lower()

    nomor_a = str(doc_a.get("nomor", "")).lower()
    nomor_b = str(doc_b.get("nomor", "")).lower()

    relationship = "independen"
    details = ""

    if nomor_a and (nomor_a in mencabut_b or doc_id_a.lower() in mencabut_b):
        relationship = f"{doc_id_b}_mencabut_{doc_id_a}"
        details = f"{doc_id_b} mencabut berlakunya {doc_id_a}."
    elif nomor_a and (nomor_a in mengubah_b or doc_id_a.lower() in mengubah_b):
        relationship = f"{doc_id_b}_mengubah_{doc_id_a}"
        details = f"{doc_id_b} merupakan perubahan atas {doc_id_a}."
    elif nomor_b and (nomor_b in mencabut_a or doc_id_b.lower() in mencabut_a):
        relationship = f"{doc_id_a}_mencabut_{doc_id_b}"
        details = f"{doc_id_a} mencabut berlakunya {doc_id_b}."
    elif nomor_b and (nomor_b in mengubah_a or doc_id_b.lower() in mengubah_a):
        relationship = f"{doc_id_a}_mengubah_{doc_id_b}"
        details = f"{doc_id_a} merupakan perubahan atas {doc_id_b}."
    else:
        newer = doc_id_b if (dt_b and dt_a and dt_b > dt_a) else doc_id_a
        details = f"Kedua regulasi independen. Regulasi lebih mutakhir: {newer}."

    return {
        "doc_a": doc_id_a,
        "doc_b": doc_id_b,
        "relationship": relationship,
        "doc_a_tanggal": doc_a.get("tanggal_berlaku") or doc_a.get("tanggal_ditetapkan"),
        "doc_b_tanggal": doc_b.get("tanggal_berlaku") or doc_b.get("tanggal_ditetapkan"),
        "details": details,
    }
