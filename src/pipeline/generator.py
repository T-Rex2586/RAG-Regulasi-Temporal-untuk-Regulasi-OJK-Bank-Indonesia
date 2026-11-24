import json
from pathlib import Path
import re
import time
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extracts and parses JSON object from model output text,

    handling markdown codeblocks and trailing text.
    """
    text = text.strip()

    # Try 1: Markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Try 2: Outer-most braces
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except Exception:
            # Clean common trailing commas before }
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(cleaned)
            except Exception:
                pass

    return None


class QwenGenerator:
    _instance: Optional["QwenGenerator"] = None

    def __init__(
        self,
        model_path: str = "models/Qwen2.5-1.5B-Instruct",
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.device = device

        # Set optimal intra-op CPU threads (measured peak on hybrid x86 architecture)
        torch.set_num_threads(8)

        print(f"Loading Qwen tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )

        print(f"Loading Qwen model onto {self.device}...")
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            dtype=torch.float32,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"Model loaded successfully in {time.time() - t0:.2f} seconds.")

    @classmethod
    def get_instance(
        cls,
        model_path: str = "models/Qwen2.5-1.5B-Instruct",
        device: str = "cpu",
    ) -> "QwenGenerator":
        if cls._instance is None:
            cls._instance = cls(model_path=model_path, device=device)
        return cls._instance

    def generate(
        self,
        messages: list[Dict[str, str]],
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Tuple[str, float, int]:
        """Generates text from chat messages.

        Returns (generated_text, latency_sec, tokens_generated).
        """
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)
        input_token_len = inputs.input_ids.shape[1]

        t0 = time.time()
        with torch.inference_mode():
            if temperature == 0.0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        latency = time.time() - t0

        new_tokens = outputs[0][input_token_len:]
        tokens_generated = len(new_tokens)
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return generated_text, latency, tokens_generated
