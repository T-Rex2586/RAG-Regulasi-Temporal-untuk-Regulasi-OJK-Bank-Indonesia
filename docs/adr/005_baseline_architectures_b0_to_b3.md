# ADR 005: Baseline Architectures B0–B3 for Indonesian Regulatory QA

## Status
Accepted

## Context
A central requirement of the research framework is establishing an empirical capability progression (ladder of baselines) before introducing fine-tuned adapters (B4) and agentic reranking/reasoning (B5). We must evaluate four distinct baseline regimes:
1. **B0 (Direct Zero-Shot LLM)**: Model generates response directly from its internal pre-trained parametric weights without retrieval or domain-specific prompting.
2. **B1 (Structured Few-Shot Prompted LLM)**: In-context learning using structured Indonesian regulatory instructions, strict JSON schema output contract, and two few-shot exemplars.
3. **B2 (Naive RAG)**: Standard dense semantic retrieval over fixed-size chunks (`reg_fixed_v1`), passing top-5 passages directly to the generator.
4. **B3 (Hybrid RAG with RRF)**: Multi-channel retrieval combining sparse keyword search (BM25 over `heading_aware_v1`) and dense semantic embeddings (FastEmbed multilingual MiniLM over Qdrant), fused via Reciprocal Rank Fusion (RRF with smoothing constant $k=60$), injecting parsed temporal metadata (`status`, `tanggal_berlaku`, `pasal`).

All baselines must strictly adhere to the project's anti-leakage guarantee: evaluations during development are conducted exclusively on the 25-query development split (`benchmark/splits/dev.jsonl`), with dynamic SHA-256 validation of the locked test set.

## Decision

### 1. Base Model & Inference Constraints
- **Model**: `Qwen/Qwen2.5-1.5B-Instruct` locked across all baselines B0–B5.
- **Decoding Configuration**: Deterministic greedy decoding (`temperature=0.0`, `do_sample=False`) to ensure strictly reproducible outputs across benchmark runs.
- **CPU Optimization**: Thread allocation pinned to 8 intra-op threads with `torch.inference_mode()` (empirically benchmarked on x86 hybrid architecture to deliver peak throughput of 8.26 tokens/sec). Token budget bounded to 256 tokens.

### 2. Output Schema Contract
Baselines B1 through B3 enforce a standardized structured JSON output schema:
```json
{
  "answer": "string",
  "claims": ["string"],
  "citations": [{"doc_id": "string", "pasal": "string", "ayat": "string"}],
  "as_of": "string or null",
  "verdict": "supported | contradicted | unanswerable",
  "abstain_reason": "string or null"
}
```
B0 uses free-form completion which is post-processed into the unified schema for downstream evaluation compatibility.

### 3. Pipeline Implementations
- **B0 (`src/pipeline/b0_direct.py`)**: Direct chat template generation formatted with `prompts/b0_direct_v1.txt`.
- **B1 (`src/pipeline/b1_prompted.py`)**: Uses `prompts/b1_prompted_v1.json` with few-shot exemplars strictly drawn from `train.jsonl` (Q001 and Q103) demonstrating proper citation format and temporal abstention.
- **B2 (`src/pipeline/b2_naive_rag.py`)**: Dense retrieval via Qdrant collection `reg_fixed_v1` using cosine similarity over multilingual MiniLM embeddings.
- **B3 (`src/pipeline/b3_hybrid_rag.py`)**:
  - Sparse retrieval: Okapi BM25 (`k1=1.5, b=0.75`) over `index/bm25/heading_aware_v1.pkl` retrieving top-10 chunks.
  - Dense retrieval: Qdrant collection `reg_heading_aware_v1` retrieving top-10 vectors.
  - Fusion: Reciprocal Rank Fusion score:
    $$RRF(d) = \sum_{m \in \{BM25, Dense\}} \frac{1}{60 + rank_m(d)}$$
  - Passage Context: Injected with parsed regulatory metadata headers (`[OJK/BI Doc ID] [BAB ...] [Pasal ...] [Status: Berlaku]`).

## Consequences
- Allows clear attribution of performance deltas: parametric hallucination (B0/B1) vs. retrieval grounding (B2) vs. hybrid fusion + structural hierarchy (B3).
- Ensures seamless integration with Fase 7 Automated Evaluation Harness via uniform JSON line logging in `runs/b0_b3_<run_id>/`.
- Guarantees zero contamination of the benchmark test set.
