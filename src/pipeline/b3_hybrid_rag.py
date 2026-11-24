from collections import defaultdict
import json
from typing import Any, Dict, List, Optional
from src.retrieval.bm25_indexer import BM25Index
from src.retrieval.dense_indexer import DenseIndexer
from src.pipeline.generator import QwenGenerator, extract_json_from_text


def reciprocal_rank_fusion(
    bm25_results: List[Dict[str, Any]],
    dense_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Merges and ranks BM25 and Dense search results using Reciprocal Rank Fusion."""
    rrf_scores = defaultdict(float)
    chunk_meta_map = {}

    for rank, item in enumerate(bm25_results, start=1):
        cid = item["chunk_id"]
        rrf_scores[cid] += 1.0 / (rrf_k + rank)
        if cid not in chunk_meta_map:
            chunk_meta_map[cid] = item.get("metadata", {})

    for rank, item in enumerate(dense_results, start=1):
        cid = item["chunk_id"]
        rrf_scores[cid] += 1.0 / (rrf_k + rank)
        if cid not in chunk_meta_map:
            chunk_meta_map[cid] = item.get("metadata", {})

    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    fused_results = []
    for rank, (cid, score) in enumerate(sorted_chunks[:top_k], start=1):
        fused_results.append({
            "rank": rank,
            "chunk_id": cid,
            "rrf_score": round(score, 6),
            "metadata": chunk_meta_map.get(cid, {}),
        })

    return fused_results


class B3HybridRAGBaseline:
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
        prompt_file: str = "prompts/b3_hybrid_rag_v1.json",
    ):
        self.generator = generator
        self.bm25_index = bm25_index
        self.dense_indexer = dense_indexer
        self.dense_collection = dense_collection
        self.top_k_bm25 = top_k_bm25
        self.top_k_dense = top_k_dense
        self.rrf_k = rrf_k
        self.final_top_k = final_top_k

        with open(prompt_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.system_prompt = cfg["system_prompt"]
        self.user_template = cfg["user_template"]

    def retrieve_hybrid(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        # 1. BM25 Search
        res_bm25 = self.bm25_index.search(query, top_k=self.top_k_bm25)

        # 2. Dense Search
        res_dense = self.dense_indexer.search(
            collection_name=self.dense_collection,
            query=query,
            top_k=self.top_k_dense,
            filters=filters,
        )

        # 3. Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            bm25_results=res_bm25,
            dense_results=res_dense,
            rrf_k=self.rrf_k,
            top_k=self.final_top_k,
        )

        return fused

    def run(self, query_item: Dict[str, Any]) -> Dict[str, Any]:
        qid = query_item["query_id"]
        qtext = query_item["query"]
        as_of = query_item.get("as_of") or "N/A"
        regulator = query_item.get("regulator")

        # Set metadata filter if regulator is strictly OJK or BI
        filters = None
        if regulator in ("OJK", "BI"):
            filters = {"regulator": regulator}

        # 1. Hybrid Retrieval
        fused_passages = self.retrieve_hybrid(qtext, filters=filters)

        # 2. Format context with enriched metadata headers
        context_blocks = []
        retrieved_passages = []
        for p in fused_passages:
            meta = p.get("metadata", {})
            did = meta.get("doc_id", "UNKNOWN")
            pasal = meta.get("pasal_nomor", "-")
            status = meta.get("status", "Berlaku")
            tgl = meta.get("tanggal_berlaku", "N/A")
            text = meta.get("text", "")

            header = f"[{p['rank']}] DOKUMEN: {did} | STATUS: {status} | BERLAKU: {tgl} | PASAL: {pasal}"
            context_blocks.append(f"{header}\nIsi: {text}")

            retrieved_passages.append({
                "rank": p["rank"],
                "chunk_id": p.get("chunk_id"),
                "doc_id": did,
                "pasal": pasal,
                "rrf_score": p.get("rrf_score"),
                "status": status,
                "tanggal_berlaku": tgl,
                "text": text[:150],
            })

        context_str = "\n\n".join(context_blocks) if context_blocks else "Tidak ada dokumen yang ditemukan."

        user_content = self.user_template.format(
            context_text=context_str,
            query=qtext,
            as_of=as_of,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw_output, latency, tokens_gen = self.generator.generate(
            messages=messages,
            max_new_tokens=350,
            temperature=0.0,
        )

        parsed_json = extract_json_from_text(raw_output)

        if parsed_json:
            answer = parsed_json.get("answer", raw_output)
            claims = parsed_json.get("claims", [])
            citations = parsed_json.get("citations", [])
            verdict = parsed_json.get("verdict", "supported")
            abstain_reason = parsed_json.get("abstain_reason")
            out_as_of = parsed_json.get("as_of") or (None if as_of == "N/A" else as_of)
        else:
            answer = raw_output
            claims = [raw_output[:200]] if raw_output else []
            citations = [
                {"doc_id": p["doc_id"], "pasal": p.get("pasal")}
                for p in retrieved_passages[:2]
            ]
            verdict = "supported" if len(raw_output) > 20 else "unknown"
            abstain_reason = None
            out_as_of = None if as_of == "N/A" else as_of

        output_contract = {
            "query_id": qid,
            "baseline": "B3",
            "answer": answer,
            "claims": claims,
            "citations": citations,
            "as_of": out_as_of,
            "verdict": verdict,
            "abstain_reason": abstain_reason,
            "retrieved_passages": retrieved_passages,
            "latency_sec": round(latency, 2),
            "tokens_generated": tokens_gen,
            "raw_output": raw_output,
            "json_parsed": parsed_json is not None,
        }

        return output_contract
