import math
from typing import Any, Dict, List, Optional, Set, Tuple


def compute_recall_at_k(retrieved: List[str], gold: Set[str], k: int) -> Optional[float]:
    """Computes Recall@k.

    Returns None if gold set is empty (unanswerable / no gold passages).
    """
    if not gold:
        return None
    top_k = set(retrieved[:k])
    hits = len(top_k.intersection(gold))
    return hits / len(gold)


def compute_mrr_at_k(retrieved: List[str], gold: Set[str], k: int) -> Optional[float]:
    """Computes Mean Reciprocal Rank (MRR@k).

    Returns None if gold set is empty.
    """
    if not gold:
        return None
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    retrieved: List[str], gold_graded: Dict[str, int], k: int
) -> Optional[float]:
    """Computes Normalized Discounted Cumulative Gain (nDCG@k).

    gold_graded maps item_id -> integer relevance score (e.g. 0, 1, 2).
    Returns None if all gold scores are 0 (or no gold items).
    """
    if not gold_graded or all(v == 0 for v in gold_graded.values()):
        return None

    # Compute DCG@k
    dcg = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        rel = gold_graded.get(item, 0)
        dcg += (2**rel - 1) / math.log2(rank + 1)

    # Compute Ideal DCG@k (IDCG)
    ideal_scores = sorted(gold_graded.values(), reverse=True)[:k]
    idcg = 0.0
    for rank, rel in enumerate(ideal_scores, start=1):
        idcg += (2**rel - 1) / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


class RetrievalEvaluator:
    """Evaluates retrieved passage lists against gold qrels at both

    passage-level and document-level.
    """

    def __init__(self, qrels: Dict[str, Dict[str, Any]]):
        """qrels format:

        {
            query_id: {
                "gold_passages": {passage_id: relevance_score},
                "gold_docs": {doc_id: max_relevance_score},
                "is_answerable": bool
            }
        }
        """
        self.qrels = qrels

    def evaluate_query(
        self, query_id: str, retrieved_passages: List[Dict[str, Any]], k_values: List[int] = [5, 10]
    ) -> Dict[str, Any]:
        """Evaluates retrieval for a single query.

        retrieved_passages is a list of dicts with 'chunk_id', 'doc_id',
        'score', etc.
        """
        qrel_data = self.qrels.get(query_id)
        if not qrel_data:
            return {"has_qrels": False}

        is_answerable = qrel_data.get("is_answerable", True)
        gold_passages_graded = qrel_data.get("gold_passages", {})
        gold_docs_graded = qrel_data.get("gold_docs", {})

        gold_passages_binary = {p for p, score in gold_passages_graded.items() if score >= 1}
        gold_docs_binary = {d for d, score in gold_docs_graded.items() if score >= 1}

        retrieved_chunks = [p.get("chunk_id", "") for p in retrieved_passages if p.get("chunk_id")]
        retrieved_docs = []
        for p in retrieved_passages:
            d = p.get("doc_id", "")
            if d and d not in retrieved_docs:
                retrieved_docs.append(d)

        metrics = {
            "has_qrels": True,
            "is_answerable": is_answerable,
            "total_retrieved": len(retrieved_passages),
        }

        # Passage-level metrics
        for k in k_values:
            metrics[f"passage_recall@{k}"] = compute_recall_at_k(retrieved_chunks, gold_passages_binary, k)
        metrics["passage_mrr@10"] = compute_mrr_at_k(retrieved_chunks, gold_passages_binary, 10)
        metrics["passage_ndcg@10"] = compute_ndcg_at_k(retrieved_chunks, gold_passages_graded, 10)

        # Document-level metrics (robust to chunking differences like fixed vs heading-aware)
        for k in k_values:
            metrics[f"doc_recall@{k}"] = compute_recall_at_k(retrieved_docs, gold_docs_binary, k)
        metrics["doc_mrr@10"] = compute_mrr_at_k(retrieved_docs, gold_docs_binary, 10)
        metrics["doc_ndcg@10"] = compute_ndcg_at_k(retrieved_docs, gold_docs_graded, 10)

        return metrics
