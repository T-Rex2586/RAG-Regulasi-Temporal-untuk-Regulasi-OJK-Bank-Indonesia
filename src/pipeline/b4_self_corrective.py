import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from src.retrieval.bm25_indexer import BM25Index
from src.retrieval.dense_indexer import DenseIndexer
from src.pipeline.generator import QwenGenerator, extract_json_from_text
from src.pipeline.planner import QueryPlanner, SufficiencyScorer
from src.verify.claim_verifier import ClaimVerifier
from src.verify.temporal_checker import TemporalChecker


def compute_rrf(
    bm25_results: List[Dict[str, Any]],
    dense_results: List[Dict[str, Any]],
    k: int = 60,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Merges BM25 and Dense search results using Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    passages: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(bm25_results, start=1):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
        if cid not in passages:
            passages[cid] = item

    for rank, item in enumerate(dense_results, start=1):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
        if cid not in passages:
            passages[cid] = item

    sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_n]
    results = []
    for rank, cid in enumerate(sorted_cids, start=1):
        p = dict(passages[cid])
        meta = p.get("metadata", {})
        p["rank"] = rank
        p["rrf_score"] = round(scores[cid], 6)
        p["doc_id"] = p.get("doc_id") or meta.get("doc_id", "UNKNOWN")
        p["pasal"] = p.get("pasal") or meta.get("pasal_nomor", "")
        p["status"] = p.get("status") or meta.get("status", "Berlaku")
        p["tanggal_berlaku"] = (
            p.get("tanggal_berlaku")
            or meta.get("tanggal_berlaku")
            or meta.get("diundangkan_pada")
            or ""
        )
        p["bab_judul"] = p.get("bab_judul") or meta.get("bab_judul", "")
        p["bab_nomor"] = p.get("bab_nomor") or meta.get("bab_nomor", "")
        p["text"] = p.get("text") or meta.get("text", "")
        results.append(p)

    return results


def format_context_passages(passages: List[Dict[str, Any]]) -> str:
    """Formats top passages into structured context blocks."""
    blocks = []
    for p in passages:
        doc_id = p.get("doc_id", "UNKNOWN")
        pasal = p.get("pasal", "-")
        status = p.get("status", "Berlaku")
        tgl_berlaku = p.get("tanggal_berlaku") or "N/A"
        text = p.get("text", "").strip()

        header = f"[{p.get('rank', '-')}] DOKUMEN: {doc_id} | STATUS: {status} | BERLAKU: {tgl_berlaku} | PASAL: {pasal}"
        blocks.append(f"{header}\nIsi: {text}")

    return "\n\n---\n\n".join(blocks)


class B4SelfCorrectiveBaseline:
    """Baseline B4: Self-Corrective RAG.

    Features:
    - Query Planner: Intent classification & sub-query generation
    - Out-of-Domain Guardrail: Immediate abstention for non-OJK/BI queries
    - Hybrid Retrieval: BM25 + Dense + RRF
    - Evidence Sufficiency Scorer
    - Bounded Retry Loop: Up to 2 retries for query reformulation & expansion
    - Independent Claim Verifier: Entailment and grounding verification
    - Self-Repair: Filters unsupported claims before final emission
    """

    def __init__(
        self,
        generator: QwenGenerator,
        bm25_index: BM25Index,
        dense_indexer: DenseIndexer,
        dense_collection: str = "reg_heading_aware_v1",
        top_k_bm25: int = 10,
        top_k_dense: int = 10,
        rrf_k: int = 60,
        final_top_k: int = 5,
        max_retries: int = 2,
        sufficiency_threshold: float = 0.38,
        claim_support_threshold: float = 0.35,
        prompt_file: str = "prompts/b4_generator_v1.json",
    ):
        self.generator = generator
        self.bm25_index = bm25_index
        self.dense_indexer = dense_indexer
        self.dense_collection = dense_collection
        self.top_k_bm25 = top_k_bm25
        self.top_k_dense = top_k_dense
        self.rrf_k = rrf_k
        self.final_top_k = final_top_k
        self.max_retries = max_retries

        self.planner = QueryPlanner()
        self.sufficiency_scorer = SufficiencyScorer(threshold=sufficiency_threshold)
        self.verifier = ClaimVerifier(support_threshold=claim_support_threshold)

        with open(prompt_file, "r", encoding="utf-8") as f:
            self.prompt_cfg = json.load(f)

    def _execute_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Executes one round of hybrid retrieval."""
        bm25_hits = self.bm25_index.search(query, top_k=self.top_k_bm25)
        dense_hits = self.dense_indexer.search(
            collection_name=self.dense_collection,
            query=query,
            top_k=self.top_k_dense,
        )
        return compute_rrf(
            bm25_results=bm25_hits,
            dense_results=dense_hits,
            k=self.rrf_k,
            top_n=self.final_top_k,
        )

    def run(self, query_item: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the self-corrective RAG pipeline on a single query."""
        t_start = time.time()
        query_id = query_item.get("query_id", "Q_UNKNOWN")
        raw_query = query_item.get("query", "")
        as_of = query_item.get("as_of")

        # 1. Query Planning
        plan = self.planner.plan(raw_query, as_of=as_of)

        # 2. Early Guardrail: Immediate clean abstain if query is out-of-domain
        if plan["is_out_of_domain"]:
            latency = round(time.time() - t_start, 2)
            return {
                "query_id": query_id,
                "baseline": "B4",
                "answer": (
                    "Pertanyaan ini berada di luar cakupan dan kewenangan regulasi "
                    "perbankan serta jasa keuangan Otoritas Jasa Keuangan (OJK) dan Bank Indonesia."
                ),
                "claims": [],
                "citations": [],
                "as_of": as_of,
                "verdict": "unanswerable",
                "abstain_reason": plan["reason"],
                "retrieved_passages": [],
                "latency_sec": latency,
                "tokens_generated": 0,
                "raw_output": "Out-of-domain guardrail triggered.",
                "json_parsed": True,
                "retry_count": 0,
                "retrieval_calls": 0,
                "sufficiency_score": 0.0,
                "is_repaired": False,
                "verification_report": {"is_grounded": True, "unsupported_rate": 0.0},
            }

        # 3. Initial Retrieval
        retrieval_calls = 1
        active_query = plan["sub_queries"][0]
        passages = self._execute_retrieval(active_query)

        # 4. Evidence Sufficiency Evaluation & Bounded Retry Loop
        suff_score, is_sufficient, missing = self.sufficiency_scorer.score(
            query=raw_query, passages=passages, plan=plan
        )

        retry_count = 0
        pool_passages = {p["chunk_id"]: p for p in passages}

        while not is_sufficient and retry_count < self.max_retries:
            retry_count += 1
            retrieval_calls += 1

            # Reformulate query based on missing aspects or plan sub-queries
            if retry_count < len(plan["sub_queries"]):
                reformulated = plan["sub_queries"][retry_count]
            elif "target_pasal_not_found" in missing and plan.get("target_pasal"):
                # Target pasal specifically
                tp = plan["target_pasal"][0]
                reg_name = plan["reg_matches"][0] if plan.get("reg_matches") else ""
                reformulated = f"{reg_name} Pasal {tp} {raw_query}"
            else:
                # Broaden query with general terms
                reformulated = f"{raw_query} ketentuan umum definisi"

            new_passages = self._execute_retrieval(reformulated)
            for p in new_passages:
                if p["chunk_id"] not in pool_passages:
                    pool_passages[p["chunk_id"]] = p

            # Re-rank pool passages
            passages = sorted(
                pool_passages.values(),
                key=lambda x: x.get("rrf_score", 0.0),
                reverse=True,
            )[: self.final_top_k]
            for idx, p in enumerate(passages, start=1):
                p["rank"] = idx

            suff_score, is_sufficient, missing = self.sufficiency_scorer.score(
                query=raw_query, passages=passages, plan=plan
            )

        # If still insufficient after max retries -> Abstain protocol
        if not is_sufficient and suff_score < 0.20:
            latency = round(time.time() - t_start, 2)
            return {
                "query_id": query_id,
                "baseline": "B4",
                "answer": (
                    "Bukti regulasi yang ditemukan dalam korpus tidak memadai untuk menjawab "
                    "pertanyaan ini secara akurat dan terverifikasi."
                ),
                "claims": [],
                "citations": [],
                "as_of": as_of,
                "verdict": "unanswerable",
                "abstain_reason": f"Insufficient evidence after {retry_count} retries (sufficiency_score: {suff_score}). Missing: {missing}",
                "retrieved_passages": passages,
                "latency_sec": latency,
                "tokens_generated": 0,
                "raw_output": "Abstain due to insufficient evidence.",
                "json_parsed": True,
                "retry_count": retry_count,
                "retrieval_calls": retrieval_calls,
                "sufficiency_score": suff_score,
                "is_repaired": False,
                "verification_report": {"is_grounded": True, "unsupported_rate": 0.0},
            }

        # 5. Draft Generation
        context_text = format_context_passages(passages)
        user_prompt = self.prompt_cfg["user_template"].format(
            context_text=context_text,
            query=raw_query,
            as_of=as_of or "null",
        )

        messages = [
            {"role": "system", "content": self.prompt_cfg["system_prompt"]},
            {"role": "user", "content": user_prompt},
        ]

        raw_output, gen_lat, tokens_gen = self.generator.generate(
            messages=messages,
            max_new_tokens=256,
            temperature=0.0,
        )

        # 6. Parse Draft JSON
        parsed_json = extract_json_from_text(raw_output)
        json_parsed = parsed_json is not None

        if parsed_json:
            answer = parsed_json.get("answer", "")
            claims = parsed_json.get("claims", [])
            citations = parsed_json.get("citations", [])
            verdict = parsed_json.get("verdict", "supported")
            abstain_reason = parsed_json.get("abstain_reason")
        else:
            answer = raw_output.strip()
            claims = [answer] if answer else []
            citations = []
            verdict = "supported"
            abstain_reason = None

        # 7. Independent Claim Verification & Repair
        v_report = self.verifier.verify(
            claims=claims,
            citations=citations,
            passages=passages,
            as_of=as_of,
        )

        is_repaired = False
        # Self-repair: If some claims are unsupported, filter them out
        if v_report["unsupported_claims_count"] > 0:
            if v_report["verified_claims_count"] > 0:
                # Keep verified claims and citations only
                claims = v_report["verified_claims"]
                citations = v_report["verified_citations"]
                is_repaired = True
            else:
                # Zero claims verified -> Abstain cleanly
                answer = "Klaim jawaban tidak dapat diverifikasi oleh bukti pasal yang tersedia."
                claims = []
                citations = []
                verdict = "unanswerable"
                abstain_reason = "All generated claims failed independent verification against evidence."
                is_repaired = True

        total_latency = round(time.time() - t_start, 2)

        return {
            "query_id": query_id,
            "baseline": "B4",
            "answer": answer,
            "claims": claims,
            "citations": citations,
            "as_of": as_of,
            "verdict": verdict,
            "abstain_reason": abstain_reason,
            "retrieved_passages": passages,
            "latency_sec": total_latency,
            "tokens_generated": tokens_gen,
            "raw_output": raw_output,
            "json_parsed": json_parsed,
            "retry_count": retry_count,
            "retrieval_calls": retrieval_calls,
            "sufficiency_score": suff_score,
            "is_repaired": is_repaired,
            "verification_report": v_report,
        }
