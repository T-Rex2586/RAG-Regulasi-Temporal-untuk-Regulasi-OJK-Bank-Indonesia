import json
from typing import Any, Dict
from src.pipeline.generator import QwenGenerator, extract_json_from_text


class B1PromptedBaseline:
    def __init__(self, generator: QwenGenerator, prompt_file: str = "prompts/b1_prompted_v1.json"):
        self.generator = generator
        with open(prompt_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.system_prompt = cfg["system_prompt"]
        self.few_shot_examples = cfg.get("few_shot_examples", [])

    def run(self, query_item: Dict[str, Any]) -> Dict[str, Any]:
        qid = query_item["query_id"]
        qtext = query_item["query"]
        as_of = query_item.get("as_of")

        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Add few-shot examples
        for ex in self.few_shot_examples:
            messages.append({"role": "user", "content": ex["query"]})
            messages.append({"role": "assistant", "content": json.dumps(ex["response"], ensure_ascii=False)})

        # Add target user query
        user_content = qtext
        if as_of:
            user_content += f"\n(Tanggal Acuan / As-Of: {as_of})"

        messages.append({"role": "user", "content": user_content})

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
            out_as_of = parsed_json.get("as_of") or as_of
        else:
            # Fallback if model did not produce valid JSON
            answer = raw_output
            claims = [raw_output[:200]] if raw_output else []
            citations = []
            verdict = "supported" if len(raw_output) > 20 else "unknown"
            abstain_reason = None
            out_as_of = as_of

        output_contract = {
            "query_id": qid,
            "baseline": "B1",
            "answer": answer,
            "claims": claims,
            "citations": citations,
            "as_of": out_as_of,
            "verdict": verdict,
            "abstain_reason": abstain_reason,
            "retrieved_passages": [],
            "latency_sec": round(latency, 2),
            "tokens_generated": tokens_gen,
            "raw_output": raw_output,
            "json_parsed": parsed_json is not None,
        }

        return output_contract
