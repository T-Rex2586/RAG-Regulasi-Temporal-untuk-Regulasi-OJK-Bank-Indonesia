from datetime import date, datetime
import re
from typing import Any, Dict, Optional, Tuple

MONTH_MAP_ID = {
    "januari": 1,
    "februari": 2,
    "maret": 3,
    "april": 4,
    "mei": 5,
    "juni": 6,
    "juli": 7,
    "agustus": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "desember": 12,
}


def parse_indonesian_date(date_str: Optional[str]) -> Optional[date]:
    """Parses various Indonesian legal date strings into datetime.date

    object. Examples:
    - '1 Januari 2024'
    - '15 Agustus 2023'
    - '2024-01-01'
    - 'tanggal 1 Oktober 2024'
    """
    if not date_str or not isinstance(date_str, str):
        return None

    cleaned = date_str.strip().lower()

    # Try ISO format YYYY-MM-DD
    m_iso = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", cleaned)
    if m_iso:
        try:
            return date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))
        except ValueError:
            pass

    # Try standard Indonesian format 'D Month YYYY'
    m_id = re.search(
        r"(\d{1,2})\s+([a-z]+)\s+(\d{4})",
        cleaned,
    )
    if m_id:
        day = int(m_id.group(1))
        month_name = m_id.group(2)
        year = int(m_id.group(3))
        month = MONTH_MAP_ID.get(month_name)
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                pass

    # Try year only
    m_yr = re.search(r"\b(20\d{2}|19\d{2})\b", cleaned)
    if m_yr:
        return date(int(m_yr.group(1)), 1, 1)

    return None


class TemporalChecker:
    """Verifies temporal point-in-time validity for Indonesian regulations."""

    @classmethod
    def check_temporal_validity(
        cls,
        as_of_str: Optional[str],
        doc_metadata: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Checks if a document or passage was legally in force at as_of.

        Returns (is_valid, reason).
        """
        if not as_of_str:
            # No temporal constraint specified -> active status applies
            status = doc_metadata.get("status", "Berlaku")
            if status and "cabut" in str(status).lower():
                return False, f"Regulasi berstatus '{status}' (telah dicabut)."
            return True, "Berlaku aktif (tanpa batasan as_of)."

        as_of_date = parse_indonesian_date(as_of_str)
        if not as_of_date:
            return True, f"Format tanggal as_of '{as_of_str}' tidak dapat diparse secara presisi."

        # Check effective date (tanggal_berlaku)
        tgl_berlaku_str = doc_metadata.get("tanggal_berlaku") or doc_metadata.get("diundangkan_pada")
        tgl_berlaku = parse_indonesian_date(tgl_berlaku_str)

        if tgl_berlaku and as_of_date < tgl_berlaku:
            return (
                False,
                f"Regulasi belum berlaku pada {as_of_date} (mulai berlaku: {tgl_berlaku}).",
            )

        # Check status & revocation
        status = doc_metadata.get("status", "Berlaku")
        if status and "cabut" in str(status).lower():
            # If revoked, check revocation date if available
            tgl_cabut_str = doc_metadata.get("tanggal_dicabut")
            tgl_cabut = parse_indonesian_date(tgl_cabut_str)
            if tgl_cabut and as_of_date >= tgl_cabut:
                return (
                    False,
                    f"Regulasi telah dicabut per {tgl_cabut} sebelum {as_of_date}.",
                )

        return True, f"Regulasi sah dan berlaku pada {as_of_date}."
