import re
from typing import Any, Dict, List, Optional, Set, Tuple


def normalize_pasal_str(pasal_val: Any) -> str:
    """Normalizes pasal identifiers e.g. 'Pasal 2' -> '2', '2A' -> '2A'."""
    if not pasal_val:
        return ""
    s = str(pasal_val).strip()
    m = re.search(r"(\d+[A-Z]?)", s, re.IGNORECASE)
    return m.group(1).upper() if m else s.lower()


def compute_f1_overlap(prediction: str, reference: str) -> float:
    """Computes token-level F1 overlap between prediction and reference."""
    pred_tokens = re.findall(r"\w+", prediction.lower())
    ref_tokens = re.findall(r"\w+", reference.lower())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens).intersection(set(ref_tokens))
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * (precision * recall) / (precision + recall)


class GenerationEvaluator:
    """Evaluates generation quality metrics:

    Citation Precision, Evidence Recall, Temporal Accuracy, Unsupported Claim
    Rate, False Refusal, False Confident, and Correctness.
    """

    def __init__(self, queries_metadata: Dict[str, Dict[str, Any]]):
        """queries_metadata maps query_id -> query dict (from dev/test.jsonl)

        with 'is_answerable', 'target_doc_ids', 'target_pasal', 'as_of',
        'reference_notes', etc.
        """
        self.metadata = queries_metadata

    def evaluate_output(
        self, query_id: str, output: Dict[str, Any], corpus_doc_ids: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """Evaluates model output dictionary for one query."""
        meta = self.metadata.get(query_id, {})
        is_answerable = meta.get("is_answerable", True)
        gold_docs = set(meta.get("target_doc_ids", []))
        gold_pasal = {normalize_pasal_str(p) for p in meta.get("target_pasal", [])}
        as_of = meta.get("as_of")
        reference_notes = meta.get("reference_notes", "")

        answer = output.get("answer", "") or ""
        claims = output.get("claims", []) or []
        citations = output.get("citations", []) or []
        verdict = output.get("verdict", "")
        abstain_reason = output.get("abstain_reason")

        # Determine if model abstained
        is_abstained = (
            verdict == "unanswerable"
            or abstain_reason is not None
            or "tidak ditemukan" in answer.lower()
            or "tidak diatur" in answer.lower()
            or "abstain" in answer.lower()
        )

        # 1. False-Refusal and False-Confident
        false_refusal = 1.0 if (is_answerable and is_abstained) else 0.0
        false_confident = 1.0 if (not is_answerable and not is_abstained) else 0.0

        # 2. Citation Precision
        # Valid citation must match a gold document (or valid corpus doc if unanswerable)
        valid_citations_count = 0
        total_citations = len(citations)

        if not is_answerable:
            # If query is unanswerable, generating citations is hallucination
            citation_precision = 1.0 if total_citations == 0 else 0.0
        else:
            if total_citations == 0:
                citation_precision = 0.0
            else:
                for cit in citations:
                    c_doc = cit.get("doc_id", "") if isinstance(cit, dict) else str(cit)
                    c_pasal = normalize_pasal_str(cit.get("pasal", "")) if isinstance(cit, dict) else ""
                    # Match if doc matches gold docs
                    doc_matched = any(c_doc.strip() == gd.strip() or c_doc.strip() in gd or gd in c_doc for gd in gold_docs)
                    if doc_matched:
                        valid_citations_count += 1
                citation_precision = valid_citations_count / total_citations

        # 3. Evidence Recall
        # Did the model cite the gold documents?
        if not is_answerable:
            evidence_recall = 1.0 if is_abstained else 0.0
        else:
            if not gold_docs:
                evidence_recall = 1.0
            else:
                cited_docs = set()
                for cit in citations:
                    c_doc = cit.get("doc_id", "") if isinstance(cit, dict) else str(cit)
                    for gd in gold_docs:
                        if c_doc.strip() == gd.strip() or c_doc.strip() in gd or gd in c_doc:
                            cited_docs.add(gd)
                evidence_recall = len(cited_docs) / len(gold_docs)

        # 4. Unsupported Claim Rate
        # A claim is unsupported if:
        # - Model answers without any citations, or citations don't match gold
        # - Or if model claims something on an unanswerable query
        if not is_answerable:
            unsupported_claim_rate = 0.0 if is_abstained else 1.0
        else:
            if not claims:
                unsupported_claim_rate = 1.0 if not is_abstained and total_citations == 0 else 0.0
            else:
                unsupported_cnt = 0
                for _ in claims:
                    # If model has 0 valid citations, claim is unsupported
                    if valid_citations_count == 0:
                        unsupported_cnt += 1
                unsupported_claim_rate = unsupported_cnt / len(claims)

        # 5. Temporal Accuracy
        # Evaluated for queries with as_of point-in-time constraints
        temporal_accuracy = None
        if as_of:
            if not is_answerable:
                temporal_accuracy = 1.0 if is_abstained else 0.0
            else:
                # Did the model cite the correct active gold document for that point in time?
                temporal_accuracy = 1.0 if evidence_recall > 0 else 0.0

        # 6. Correctness (ground truth alignment)
        if not is_answerable:
            correctness = 1.0 if is_abstained else 0.0
        else:
            f1 = compute_f1_overlap(answer, reference_notes)
            # Combine F1 with citation grounding
            correctness = round(0.5 * f1 + 0.5 * evidence_recall, 4)

        return {
            "is_answerable": is_answerable,
            "is_abstained": is_abstained,
            "false_refusal": false_refusal,
            "false_confident": false_confident,
            "citation_precision": round(citation_precision, 4),
            "evidence_recall": round(evidence_recall, 4),
            "unsupported_claim_rate": round(unsupported_claim_rate, 4),
            "temporal_accuracy": round(temporal_accuracy, 4) if temporal_accuracy is not None else None,
            "correctness": round(correctness, 4),
        }
