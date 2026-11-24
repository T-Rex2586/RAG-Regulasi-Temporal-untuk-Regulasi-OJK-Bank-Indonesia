import json
from typing import Any, Dict
from src.retrieval.dense_indexer import DenseIndexer
from src.pipeline.generator import QwenGenerator, extract_json_from_text


class B2NaiveRAGBaseline:
    def __init__(
        self,
        generator: QwenGenerator,
        dense_indexer: DenseIndexer,
        collection_name: str = "reg_fixed_v1",
        top_k: int = 5,
        prompt_file: str = "prompts/b2_naive_rag_v1.json",
    ):
        self.generator = generator
        self.dense_indexer = dense_indexer
        self.collection_name = collection_name
        self.top_k = top_k

        with open(prompt_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.system_prompt = cfg["system_prompt"]
        self.user_template = cfg["user_template"]

    def retrieve(self, query: str) -> list[Dict[str, Any]]:
        return self.dense_indexer.search(
            collection_name=self.collection_name,
            query=query,
            top_k=self.top_k,
        )

    def run(self, query_item: Dict[str, Any]) -> Dict[str, Any]:
        qid = query_item["query_id"]
        qtext = query_item["query"]
        as_of = query_item.get("as_of") or "N/A"

        # 1. Dense retrieval
        retrieved = self.retrieve(qtext)

        # 2. Format context
        context_blocks = []
        retrieved_passages = []
        for i, r in enumerate(retrieved, start=1):
            meta = r.get("metadata", {})
            did = meta.get("doc_id", "UNKNOWN")
            text = meta.get("text", "")
            context_blocks.append(f"[{i}] Dokumen: {did}\nTeks: {text}")
            retrieved_passages.append({
                "rank": i,
                "chunk_id": r.get("chunk_id"),
                "doc_id": did,
                "score": r.get("score"),
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
            citations = [{"doc_id": p["doc_id"]} for p in retrieved_passages[:2]]
            verdict = "supported" if len(raw_output) > 20 else "unknown"
            abstain_reason = None
            out_as_of = None if as_of == "N/A" else as_of

        output_contract = {
            "query_id": qid,
            "baseline": "B2",
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
