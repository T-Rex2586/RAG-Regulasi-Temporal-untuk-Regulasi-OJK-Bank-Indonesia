from typing import Any, Dict
from src.pipeline.generator import QwenGenerator


class B0DirectBaseline:
    def __init__(self, generator: QwenGenerator, prompt_file: str = "prompts/b0_direct_v1.txt"):
        self.generator = generator
        with open(prompt_file, "r", encoding="utf-8") as f:
            self.template = f.read()

    def run(self, query_item: Dict[str, Any]) -> Dict[str, Any]:
        qid = query_item["query_id"]
        qtext = query_item["query"]
        as_of = query_item.get("as_of")

        prompt_str = self.template.format(query=qtext)
        messages = [
            {"role": "user", "content": prompt_str}
        ]

        raw_output, latency, tokens_gen = self.generator.generate(
            messages=messages,
            max_new_tokens=256,
            temperature=0.0,
        )

        output_contract = {
            "query_id": qid,
            "baseline": "B0",
            "answer": raw_output,
            "claims": [raw_output[:200]] if raw_output else [],
            "citations": [],
            "as_of": as_of,
            "verdict": "supported" if len(raw_output) > 20 else "unknown",
            "abstain_reason": None,
            "retrieved_passages": [],
            "latency_sec": round(latency, 2),
            "tokens_generated": tokens_gen,
            "raw_output": raw_output,
        }

        return output_contract
