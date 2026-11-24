from collections import Counter
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple


VALID_SLICES = {
    "easy",
    "multi_hop",
    "konflik_versi",
    "temporal_point_in_time",
    "unanswerable",
}

VALID_REGULATORS = {"OJK", "BI", "BOTH", "NONE"}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def get_char_ngrams(text: str, n: int = 3) -> set[str]:
    norm = normalize_text(text)
    if len(norm) < n:
        return {norm}
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


class BenchmarkCurator:
    def __init__(
        self,
        near_dup_threshold: float = 0.85,
        ngram_size: int = 3,
        parsed_dir: str = "data/parsed",
    ):
        self.near_dup_threshold = near_dup_threshold
        self.ngram_size = ngram_size
        self.parsed_dir = Path(parsed_dir)
        self.existing_doc_ids = {
            f.stem for f in self.parsed_dir.glob("*.json")
        }

    def validate_schema(self, query_item: Dict[str, Any]) -> List[str]:
        errors = []
        qid = query_item.get("query_id", "")
        if not qid:
            errors.append("query_id is required")

        qtext = query_item.get("query", "")
        if not qtext or len(qtext.strip()) < 15:
            errors.append(f"{qid}: query text too short (<15 chars)")

        slice_name = query_item.get("slice")
        if slice_name not in VALID_SLICES:
            errors.append(f"{qid}: invalid slice '{slice_name}'")

        regulator = query_item.get("regulator")
        if regulator not in VALID_REGULATORS:
            errors.append(f"{qid}: invalid regulator '{regulator}'")

        is_ans = query_item.get("is_answerable")
        if not isinstance(is_ans, bool):
            errors.append(f"{qid}: is_answerable must be boolean")

        doc_ids = query_item.get("target_doc_ids", [])
        if is_ans and not doc_ids:
            errors.append(f"{qid}: answerable query must have target_doc_ids")
        if not is_ans and doc_ids:
            errors.append(f"{qid}: unanswerable query should have empty target_doc_ids")

        for did in doc_ids:
            if did not in self.existing_doc_ids:
                errors.append(f"{qid}: target doc_id '{did}' not found in {self.parsed_dir}")

        if slice_name == "temporal_point_in_time":
            as_of = query_item.get("as_of")
            if not as_of or not re.match(r"^\d{4}-\d{2}-\d{2}$", str(as_of)):
                errors.append(f"{qid}: temporal_point_in_time query requires valid 'as_of' date (YYYY-MM-DD)")

        return errors

    def check_duplicates(
        self, queries: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[Tuple[str, str, float]]]:
        exact_dups = []
        seen_texts: Dict[str, str] = {}

        for item in queries:
            qid = item["query_id"]
            norm = normalize_text(item["query"])
            if norm in seen_texts:
                exact_dups.append(f"Duplicate found: {qid} is exact duplicate of {seen_texts[norm]}")
            else:
                seen_texts[norm] = qid

        # Near-duplicate check via character n-grams
        near_dups: List[Tuple[str, str, float]] = []
        ngram_cache = [
            (q["query_id"], get_char_ngrams(q["query"], self.ngram_size))
            for q in queries
        ]

        for i in range(len(ngram_cache)):
            qid_a, ngrams_a = ngram_cache[i]
            for j in range(i + 1, len(ngram_cache)):
                qid_b, ngrams_b = ngram_cache[j]
                sim = jaccard_similarity(ngrams_a, ngrams_b)
                if sim >= self.near_dup_threshold:
                    near_dups.append((qid_a, qid_b, round(sim, 4)))

        return exact_dups, near_dups

    def audit_benchmark(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_errors = []
        for q in queries:
            all_errors.extend(self.validate_schema(q))

        exact_dups, near_dups = self.check_duplicates(queries)

        slice_counts = Counter(q["slice"] for q in queries)
        regulator_counts = Counter(q["regulator"] for q in queries)

        is_passed = (
            len(all_errors) == 0
            and len(exact_dups) == 0
            and len(near_dups) == 0
            and len(queries) >= 100
        )

        return {
            "total_queries": len(queries),
            "slice_distribution": dict(slice_counts),
            "regulator_distribution": dict(regulator_counts),
            "schema_errors": all_errors,
            "exact_duplicates": exact_dups,
            "near_duplicates": [
                {"query_a": a, "query_b": b, "similarity": s}
                for a, b, s in near_dups
            ],
            "passed": is_passed,
        }
