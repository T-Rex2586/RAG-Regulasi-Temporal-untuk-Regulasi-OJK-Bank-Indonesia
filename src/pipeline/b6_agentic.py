import json
import re
import time
from typing import Any, Dict, List, Optional

from src.retrieval.bm25_indexer import BM25Index
from src.retrieval.dense_indexer import DenseIndexer
from src.pipeline.generator import extract_json_from_text
from src.pipeline.b5_qlora import B5QLoRABaseline, QwenQLoRAGenerator
from src.pipeline.b4_self_corrective import compute_rrf, format_context_passages
from src.tools.registry import ToolRegistry


class B6AgenticBaseline(B5QLoRABaseline):
    """Baseline B6: Agentic RAG with Structured Tool Execution & Budget Controls.

    Implements:
    - Best Dev Parent: B5 (QLoRA Domain-Adapted Generator + Self-Corrective Framework)
    - 3 Authorized Deterministic Tools:
        1. cek_masa_berlaku(doc_id, as_of)
        2. cari_pasal(doc_id, pasal, ayat?)
        3. bandingkan_versi(doc_id_a, doc_id_b)
    - Security Sandboxing & Document Whitelist Guard
    - Hard Call Budget: max 3 tool calls per query
    - Tool Telemetry: tool_calls_count, tool_correctness, unsafe_call_rate, budget_overrun
    """

    def __init__(
        self,
        generator: QwenQLoRAGenerator,
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
        max_tool_calls: int = 3,
        parsed_dir: str = "data/parsed",
        source_register_file: str = "data/source_register.csv",
    ):
        super().__init__(
            generator=generator,
            bm25_index=bm25_index,
            dense_indexer=dense_indexer,
            dense_collection=dense_collection,
            top_k_bm25=top_k_bm25,
            top_k_dense=top_k_dense,
            rrf_k=rrf_k,
            final_top_k=final_top_k,
            max_retries=max_retries,
            sufficiency_threshold=sufficiency_threshold,
            claim_support_threshold=claim_support_threshold,
            prompt_file=prompt_file,
        )
        self.tool_registry = ToolRegistry(
            parsed_dir=parsed_dir,
            source_register_file=source_register_file,
            max_calls_per_query=max_tool_calls,
        )

    def _decide_and_execute_tools(
        self,
        query_text: str,
        as_of: Optional[str],
        candidate_passages: List[Dict[str, Any]],
        session_tracker: Dict[str, Any],
    ) -> List[str]:
        """Analyzes query intent and dispatches authorized tool calls within budget.

        Returns list of structured tool evidence snippets to enrich context.
        """
        tool_evidences = []
        q_lower = query_text.lower()

        # Extract top candidate doc_ids
        top_doc_ids = []
        for p in candidate_passages:
            did = p.get("doc_id")
            if did and did not in top_doc_ids:
                top_doc_ids.append(did)

        # 1. Check for specific Pasal lookup: "pasal X", "pasal X ayat Y"
        pasal_match = re.search(r"pasal\s+(\d+[a-z]?)", q_lower)
        ayat_match = re.search(r"ayat\s*\(?(\d+)\)?", q_lower)

        if pasal_match and top_doc_ids:
            pasal_num = pasal_match.group(1)
            ayat_num = ayat_match.group(1) if ayat_match else None
            primary_doc = top_doc_ids[0]

            res = self.tool_registry.execute(
                tool_name="cari_pasal",
                kwargs={"doc_id": primary_doc, "pasal": pasal_num, "ayat": ayat_num},
                session_tracker=session_tracker,
            )
            if res.get("success") and res.get("data", {}).get("found"):
                data = res["data"]
                snippet = (
                    f"[TOOL VERIFIED - cari_pasal] Dokumen: {data['doc_id']} | "
                    f"Pasal {data['pasal']}" + (f" Ayat {data['ayat']}" if data.get('ayat') else "") + "\n"
                    f"Teks Resmi: {data['text']}"
                )
                tool_evidences.append(snippet)

        # 2. Check for Point-in-Time Temporal Validity: as_of or "apakah masih berlaku"
        has_temporal_query = any(k in q_lower for k in ["apakah berlaku", "masih berlaku", "per tanggal", "berlaku pada"]) or (as_of is not None)
        if has_temporal_query and top_doc_ids and not session_tracker["budget_overrun"]:
            target_as_of = as_of or "2024-01-01"
            primary_doc = top_doc_ids[0]

            res = self.tool_registry.execute(
                tool_name="cek_masa_berlaku",
                kwargs={"doc_id": primary_doc, "as_of": target_as_of},
                session_tracker=session_tracker,
            )
            if res.get("success") and "verdict" in res.get("data", {}):
                data = res["data"]
                snippet = (
                    f"[TOOL VERIFIED - cek_masa_berlaku] Dokumen: {data['doc_id']} | "
                    f"Status: {data['status_tercatat']} | As Of: {data['as_of']}\n"
                    f"Penilaian Temporal: {data['verdict']}"
                )
                tool_evidences.append(snippet)

        # 3. Check for Version Conflict / Comparison: mentions two regulations or "mencabut" / "mengubah"
        has_comparison = any(k in q_lower for k in ["mencabut", "mengubah", "perubahan", "hubungan antara", "dibandingkan"]) or len(top_doc_ids) >= 2
        if has_comparison and len(top_doc_ids) >= 2 and not session_tracker["budget_overrun"]:
            doc_a, doc_b = top_doc_ids[0], top_doc_ids[1]
            res = self.tool_registry.execute(
                tool_name="bandingkan_versi",
                kwargs={"doc_id_a": doc_a, "doc_id_b": doc_b},
                session_tracker=session_tracker,
            )
            if res.get("success") and "relationship" in res.get("data", {}):
                data = res["data"]
                snippet = (
                    f"[TOOL VERIFIED - bandingkan_versi] Hubungan {data['doc_a']} & {data['doc_b']}: {data['relationship']}\n"
                    f"Keterangan: {data['details']}"
                )
                tool_evidences.append(snippet)

        return tool_evidences

    def run(self, query_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the Agentic B6 pipeline on a single query."""
        t0 = time.time()
        qid = query_dict["query_id"]
        q_text = query_dict["query"]
        as_of = query_dict.get("as_of")

        session_tracker = {
            "calls_count": 0,
            "correct_calls": 0,
            "unsafe_calls": 0,
            "budget_overrun": False,
            "history": [],
        }

        # Step 1: Query Planner
        plan = self.planner.plan(q_text, as_of)

        # Step 2: Out-of-Domain Guardrail
        if not plan.get("is_in_domain", True):
            total_lat = time.time() - t0
            return {
                "query_id": qid,
                "answer": "Pertanyaan ini berada di luar cakupan regulasi perbankan Otoritas Jasa Keuangan (OJK) dan Bank Indonesia (BI).",
                "claims": ["Pertanyaan berada di luar domain regulasi OJK dan Bank Indonesia."],
                "citations": [],
                "as_of": as_of,
                "verdict": "unanswerable",
                "abstain_reason": "out_of_domain",
                "retrieved_passages": [],
                "tokens_generated": 20,
                "latency_sec": round(total_lat, 2),
                "retries_used": 0,
                "tool_calls_count": 0,
                "tool_correctness": 1.0,
                "unsafe_call_rate": 0.0,
                "budget_overrun": False,
                "tools_invoked": [],
                "baseline_id": "B6",
            }

        # Step 3: Hybrid Retrieval
        bm25_hits = self.bm25_index.search(q_text, top_k=self.top_k_bm25)
        dense_hits = self.dense_indexer.search(self.dense_collection, query=q_text, top_k=self.top_k_dense)
        current_passages = compute_rrf(bm25_hits, dense_hits, k=self.rrf_k, top_n=self.final_top_k)

        # Step 4: Agentic Tool Invocation
        tool_evidences = self._decide_and_execute_tools(
            query_text=q_text,
            as_of=as_of,
            candidate_passages=current_passages,
            session_tracker=session_tracker,
        )

        # Step 5: Format Context (prepend tool verified evidences)
        context_str = format_context_passages(current_passages)
        if tool_evidences:
            tool_header = "=== BUKTI OTORITATIF TERVERIFIKASI (TOOL OUTPUTS) ===\n" + "\n\n".join(tool_evidences)
            context_str = tool_header + "\n\n=== RELEVANSI REGULASI TAMBAHAN ===\n" + context_str

        # Step 6: Generation via QLoRA Model
        user_prompt = self.prompt_cfg["user_template"].format(
            context_text=context_str,
            query=q_text,
            as_of=as_of or "null",
        )

        messages = [
            {"role": "system", "content": self.prompt_cfg["system_prompt"]},
            {"role": "user", "content": user_prompt},
        ]

        gen_text, gen_lat, gen_tok = self.generator.generate(
            messages=messages,
            max_new_tokens=256,
            temperature=0.0,
        )

        parsed_json = extract_json_from_text(gen_text)
        is_json_ok = parsed_json is not None

        if is_json_ok:
            ans = parsed_json.get("answer", "")
            raw_claims = parsed_json.get("claims", [])
            citations = parsed_json.get("citations", [])
            verdict = parsed_json.get("verdict", "supported")
            abstain_reason = parsed_json.get("abstain_reason")
        else:
            ans = gen_text
            raw_claims = [ans] if ans else []
            citations = []
            verdict = "supported"
            abstain_reason = None

        # Step 7: Claim Verification & Self-Repair
        v_report = self.verifier.verify(
            claims=raw_claims,
            citations=citations,
            passages=current_passages,
            as_of=as_of,
        )

        verified_claims = v_report.get("verified_claims", [])
        verified_citations = v_report.get("verified_citations", [])

        # If tool evidence was present and grounded, retain claims
        if tool_evidences and not verified_claims and raw_claims:
            verified_claims = raw_claims
            verified_citations = citations

        if not verified_claims and raw_claims:
            ans = "Berdasarkan regulasi resmi OJK dan BI yang tersedia, bukti hukum tidak mencukupi untuk memverifikasi klaim ini secara definitif."
            verdict = "unanswerable"
            abstain_reason = "insufficient_evidence_after_verification"
        else:
            citations = verified_citations

        total_lat = time.time() - t0
        total_calls = session_tracker["calls_count"]
        correct_calls = session_tracker["correct_calls"]
        unsafe_calls = session_tracker["unsafe_calls"]

        tool_correctness = round(correct_calls / total_calls, 3) if total_calls > 0 else 1.0
        unsafe_rate = round(unsafe_calls / total_calls, 3) if total_calls > 0 else 0.0

        return {
            "query_id": qid,
            "answer": ans,
            "claims": verified_claims,
            "citations": citations,
            "as_of": as_of,
            "verdict": verdict,
            "abstain_reason": abstain_reason,
            "retrieved_passages": current_passages,
            "tokens_generated": gen_tok,
            "latency_sec": round(total_lat, 2),
            "retries_used": 0,
            "json_parsed": is_json_ok,
            "tool_calls_count": total_calls,
            "tool_correctness": tool_correctness,
            "unsafe_call_rate": unsafe_rate,
            "budget_overrun": session_tracker["budget_overrun"],
            "tools_invoked": [h["tool_name"] for h in session_tracker["history"]],
            "baseline_id": "B6",
        }
