from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

from src.retrieval.bm25_indexer import BM25Index
from src.retrieval.dense_indexer import DenseIndexer
from src.pipeline.generator import QwenGenerator, extract_json_from_text
from src.pipeline.b4_self_corrective import B4SelfCorrectiveBaseline


class QwenQLoRAGenerator:
    """Generator that loads Qwen2.5-1.5B with trained QLoRA adapter for B5."""

    _instance: Optional["QwenQLoRAGenerator"] = None

    def __init__(
        self,
        base_model_path: str = "models/Qwen2.5-1.5B-Instruct",
        adapter_path: str = "models/qlora_adapter_b5",
        device: str = "cpu",
    ):
        self.device = device
        torch.set_num_threads(8)

        print(f"Loading Qwen tokenizer from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading base Qwen model onto {self.device}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.float32,
            device_map=self.device,
            trust_remote_code=True,
        )

        p_adapter = Path(adapter_path)
        if p_adapter.exists() and (p_adapter / "adapter_config.json").exists():
            if PeftModel is None:
                raise ImportError("Package 'peft' diperlukan untuk memuat adapter QLoRA B5. Silakan jalankan: pip install peft")
            print(f"Applying QLoRA adapter weights from {p_adapter}...")
            self.model = PeftModel.from_pretrained(base_model, str(p_adapter))
            print("QLoRA adapter loaded successfully.")
        else:
            print(f"Notice: Adapter not found at {p_adapter}. Running base model in zero-shot mode.")
            self.model = base_model

        self.model.eval()

    @classmethod
    def get_instance(
        cls,
        base_model_path: str = "models/Qwen2.5-1.5B-Instruct",
        adapter_path: str = "models/qlora_adapter_b5",
        device: str = "cpu",
    ) -> "QwenQLoRAGenerator":
        if cls._instance is None:
            cls._instance = cls(
                base_model_path=base_model_path,
                adapter_path=adapter_path,
                device=device,
            )
        return cls._instance

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> tuple[str, float, int]:
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)
        input_token_len = inputs.input_ids.shape[1]

        t0 = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.time() - t0
        new_tokens = outputs[0][input_token_len:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return generated_text, latency, len(new_tokens)


class B5QLoRABaseline(B4SelfCorrectiveBaseline):
    """Baseline B5: QLoRA Domain-Adapted Self-Corrective RAG.

    Combines:
    - B4: Query Planner + Evidence Sufficiency Scorer + Bounded Retry + Claim Verifier
    - B5: Parameter-Efficient Fine-Tuned (QLoRA) Qwen2.5-1.5B generator adapted on Indonesian regulatory syntax
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

    def run(self, query_item: Dict[str, Any]) -> Dict[str, Any]:
        result = super().run(query_item)
        result["baseline"] = "B5"
        return result
