import re
from typing import Any, Dict, List, Optional, Set, Tuple
from src.verify.temporal_checker import TemporalChecker


def extract_keywords(text: str) -> Set[str]:
    """Extracts significant keywords (removing Indonesian stopwords)."""
    stopwords = {
        "dan", "atau", "yang", "untuk", "pada", "dalam", "dengan", "adalah",
        "sebagaimana", "dimaksud", "ayat", "pasal", "nomor", "tahun", "tentang",
        "oleh", "dari", "ke", "di", "ini", "itu", "tersebut", "secara", "dapat",
        "wajib", "harus", "atas", "bagi", "setiap", "pihak", "paling", "antara"
    }
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", text.lower())
    return {t for t in tokens if t not in stopwords}


class ClaimVerifier:
    """Independent Claim Verifier that validates generated atomic claims and citations

    against the retrieved evidence passages.
    """

    def __init__(self, support_threshold: float = 0.35):
        self.support_threshold = support_threshold

    def verify_claim(
        self,
        claim: str,
        passages: List[Dict[str, Any]],
    ) -> Tuple[bool, float, Optional[str]]:
        """Verifies if a single atomic claim is grounded in any evidence passage.

        Returns (is_supported, max_overlap_score, supporting_chunk_id).
        """
        claim_kw = extract_keywords(claim)
        if not claim_kw:
            return True, 1.0, None

        best_score = 0.0
        best_chunk_id = None

        for p in passages:
            text = p.get("text", "")
            p_kw = extract_keywords(text)
            if not p_kw:
                continue

            intersection = claim_kw.intersection(p_kw)
            score = len(intersection) / len(claim_kw)
            if score > best_score:
                best_score = score
                best_chunk_id = p.get("chunk_id")

        is_supported = best_score >= self.support_threshold
        return is_supported, round(best_score, 3), best_chunk_id

    def verify(
        self,
        claims: List[str],
        citations: List[Dict[str, Any]],
        passages: List[Dict[str, Any]],
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs comprehensive verification across all claims and citations.

        Returns structured verification report.
        """
        verified_claims = []
        unsupported_claims = []
        claim_details = []

        for c in claims:
            supported, score, chunk_id = self.verify_claim(c, passages)
            detail = {
                "claim": c,
                "supported": supported,
                "overlap_score": score,
                "supporting_chunk_id": chunk_id,
            }
            claim_details.append(detail)
            if supported:
                verified_claims.append(c)
            else:
                unsupported_claims.append(c)

        # Verify Citations against passages
        available_docs = {p.get("doc_id") for p in passages if p.get("doc_id")}
        verified_citations = []
        unverified_citations = []

        for cit in citations:
            doc_id = cit.get("doc_id", "") if isinstance(cit, dict) else str(cit)
            matched = any(
                doc_id.strip() == ad or doc_id.strip() in ad or ad in doc_id
                for ad in available_docs
            )
            if matched:
                verified_citations.append(cit)
            else:
                unverified_citations.append(cit)

        # Verify Temporal Validity for each passage
        temporal_failures = []
        if as_of:
            for p in passages:
                is_valid, reason = TemporalChecker.check_temporal_validity(as_of, p)
                if not is_valid:
                    temporal_failures.append({
                        "doc_id": p.get("doc_id"),
                        "reason": reason,
                    })

        total_claims = len(claims)
        unsupported_rate = (
            len(unsupported_claims) / total_claims if total_claims > 0 else 0.0
        )
        is_grounded = unsupported_rate <= 0.30 and len(verified_claims) > 0

        return {
            "is_grounded": is_grounded,
            "total_claims": total_claims,
            "verified_claims_count": len(verified_claims),
            "unsupported_claims_count": len(unsupported_claims),
            "unsupported_rate": round(unsupported_rate, 3),
            "verified_claims": verified_claims,
            "unsupported_claims": unsupported_claims,
            "claim_details": claim_details,
            "verified_citations": verified_citations,
            "unverified_citations": unverified_citations,
            "temporal_failures": temporal_failures,
        }
