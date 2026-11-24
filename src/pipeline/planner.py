import re
from typing import Any, Dict, List, Optional, Set, Tuple


OUT_OF_DOMAIN_KEYWORDS = [
    "paspor", "imigrasi", "retribusi", "parkir", "karantina", "pertanian",
    "tilang", "sim keliling", "bpjs kesehatan", "surat nikah", "akta kelahiran",
    "pbb-p2", "sampah", "pdam", "pemberantasan hama", "kemenkumham"
]


class QueryPlanner:
    """Plans retrieval strategy and decomposes complex regulatory queries."""

    def __init__(self):
        self.re_pasal = re.compile(r"pasal\s+(\d+[A-Z]?)", re.IGNORECASE)
        self.re_reg_no = re.compile(
            r"(?:pojk|pbi|seojk|padg)\s+(?:nomor|no\.?)\s*([\d/a-z\.]+)",
            re.IGNORECASE,
        )

    def plan(self, query: str, as_of: Optional[str] = None) -> Dict[str, Any]:
        """Analyzes query and creates structured retrieval plan."""
        q_lower = query.lower()

        # 1. Out-of-domain check
        for kw in OUT_OF_DOMAIN_KEYWORDS:
            if kw in q_lower:
                return {
                    "intent": "out_of_domain",
                    "is_out_of_domain": True,
                    "reason": f"Topik '{kw}' berada di luar yurisdiksi regulasi perbankan OJK dan Bank Indonesia.",
                    "sub_queries": [query],
                    "target_regulator": None,
                    "target_pasal": [],
                }

        # 2. Extract entities and hints
        pasal_matches = self.re_pasal.findall(query)
        reg_matches = self.re_reg_no.findall(query)

        target_regulator = None
        if "ojk" in q_lower or "otoritas jasa keuangan" in q_lower or "pojk" in q_lower or "seojk" in q_lower:
            target_regulator = "OJK"
        elif "bank indonesia" in q_lower or "bi" in q_lower or "pbi" in q_lower or "padg" in q_lower:
            target_regulator = "BI"

        # 3. Intent classification
        intent = "single_hop"
        if as_of or "per tanggal" in q_lower or "berlaku per" in q_lower or "pada tanggal" in q_lower:
            intent = "temporal_point_in_time"
        elif any(w in q_lower for w in ["perubahan", "mencabut", "mengubah", "perbedaan", "sebelumnya", "lama"]):
            intent = "konflik_versi"
        elif any(w in q_lower for w in ["koordinasi", "hubungan antara", "saling", "dan juga", "antarlembaga"]):
            intent = "multi_hop"

        # 4. Formulate targeted sub-queries
        sub_queries = [query]

        if intent == "konflik_versi":
            # Add sub-query specifically seeking amendment / revocation provisions
            sub_queries.append(f"{query} ketentuan penutup mencabut mengubah")
        elif intent == "multi_hop":
            # Add focused keyword variant
            keywords = [w for w in re.findall(r"\w+", query) if len(w) > 4]
            if len(keywords) >= 3:
                sub_queries.append(" ".join(keywords[:5]))
        elif intent == "temporal_point_in_time" and as_of:
            sub_queries.append(f"{query} berlaku {as_of}")

        return {
            "intent": intent,
            "is_out_of_domain": False,
            "reason": None,
            "sub_queries": sub_queries,
            "target_regulator": target_regulator,
            "target_pasal": pasal_matches,
            "reg_matches": reg_matches,
            "as_of": as_of,
        }


class SufficiencyScorer:
    """Evaluates whether retrieved passages provide sufficient evidence

    to answer the query accurately without hallucination.
    """

    def __init__(self, threshold: float = 0.40):
        self.threshold = threshold

    def score(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        plan: Dict[str, Any],
    ) -> Tuple[float, bool, List[str]]:
        """Scores evidence sufficiency.

        Returns (sufficiency_score, is_sufficient, missing_aspects).
        """
        if not passages:
            return 0.0, False, ["no_passages_retrieved"]

        missing_aspects = []
        score_components = []

        # 1. RRF Score strength of top passage
        top_score = float(passages[0].get("rrf_score", 0.0))
        # Normal RRF top score is around 0.02 - 0.035
        normalized_rrf = min(1.0, top_score / 0.030)
        score_components.append(0.35 * normalized_rrf)

        # 2. Term coverage: Do top passages contain key query terms?
        q_terms = {w.lower() for w in re.findall(r"\b[a-zA-Z0-9]{4,}\b", query)}
        stopwords = {"yang", "untuk", "dalam", "dengan", "adalah", "bagaimana", "apakah", "berapa", "pihak", "mana"}
        core_terms = q_terms - stopwords

        if core_terms:
            combined_text = " ".join([p.get("text", "").lower() for p in passages[:3]])
            matched_terms = {t for t in core_terms if t in combined_text}
            term_coverage = len(matched_terms) / len(core_terms)
            score_components.append(0.35 * term_coverage)
            if term_coverage < 0.35:
                missing_aspects.append("low_query_term_coverage")
        else:
            score_components.append(0.35)

        # 3. Pasal Match: If query explicitly asks for a pasal, is that pasal present?
        target_pasal = plan.get("target_pasal", [])
        if target_pasal:
            pasal_found = False
            for p in passages:
                p_pasal = str(p.get("pasal", ""))
                p_text = p.get("text", "")
                for tp in target_pasal:
                    if tp.lower() in p_pasal.lower() or f"pasal {tp}".lower() in p_text.lower():
                        pasal_found = True
                        break
                if pasal_found:
                    break
            if pasal_found:
                score_components.append(0.30)
            else:
                score_components.append(0.0)
                missing_aspects.append("target_pasal_not_found")
        else:
            score_components.append(0.30)

        final_score = round(sum(score_components), 3)
        is_sufficient = final_score >= self.threshold

        return final_score, is_sufficient, missing_aspects
