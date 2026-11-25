# ADR 008: QLoRA Domain Adaptation Architecture for Baseline B5

## Status
Accepted

## Context
Across baselines B0 through B4, the base generator `Qwen2.5-1.5B-Instruct` demonstrated consistent grounding improvements when paired with hybrid retrieval (B3) and self-corrective verification (B4). However, empirical analysis of dev set outputs revealed key generator-side bottlenecks:
1. **Instruction Formatting Fragility under Long Regulatory Contexts**: In B3 and B4, when multiple multi-hop regulatory articles are provided in the prompt, the 1.5B base model occasionally emits trailing conversational markdown before terminating the JSON closing tag.
2. **Indonesian Regulatory Syntax & Citation Precision**: Indonesian legal texts employ specific stylistic patterns (e.g., *sebagaimana dimaksud pada ayat*, *ketentuan peralihan*, *ditetapkan di Jakarta*). The zero-shot base model sometimes paraphrases legal articles into generic commercial statements, reducing citation precision.
3. **Hardware Constraints**: The local execution environment operates on CPU without dedicated CUDA GPUs, precluding full fine-tuning.

Per the PRD and course syllabus, we must perform Parameter-Efficient Fine-Tuning (PEFT) on the parent B4 architecture using QLoRA 4-bit quantization, with training offloaded to a free cloud Nvidia T4 GPU (Google Colab) and adapter weights exported for local inference.

## Decision

### 1. Anti-Leakage Isolation Guarantee
- Training and validation datasets (`data/train/qlora_train.jsonl` and `data/train/qlora_val.jsonl`) are generated exclusively from the 75 training split queries (`benchmark/splits/train.jsonl`) and corresponding parsed articles in `data/parsed/`.
- `benchmark/splits/test.jsonl` is cryptographically validated using SHA-256 (`3898015c21241292352a8a02b42aa22ab134bf2ffef65e067eb7972c8698c19b`) prior to dataset compilation. Zero test or dev instances are exposed to the training pipeline.

### 2. Quantization & LoRA Configuration
- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct` (1.54 billion parameters).
- **Quantization**: 4-bit NormalFloat4 (`nf4`) with double quantization enabled (`bnb_4bit_use_double_quant=True`) and compute dtype `bfloat16`.
- **LoRA Adapter Architecture**:
  - Rank: $r = 16$
  - Scaling factor: $\alpha = 32$
  - Dropout: $0.05$
  - Target Modules: All linear projections in self-attention and MLP blocks:
    `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
  - Trainable parameters: ~18.4 million parameters (~1.20% of total model parameters), preserving core general reasoning while adapting domain-specific task heads.

### 3. Training & Optimization Schedule
- **Optimizer**: AdamW with weight decay $0.01$.
- **Effective Batch Size**: 8 (per-device batch size 2, gradient accumulation steps 4).
- **Learning Rate**: $2 \times 10^{-4}$ with a 10% warmup ratio followed by a cosine annealing decay schedule over 3 epochs.
- **Sequence Length**: Maximum 1024 tokens with dynamic padding.
- **Hardware Profile**: Nvidia T4 GPU (16 GB VRAM) on Google Colab, estimated training duration ~15 minutes.

### 4. B5 Pipeline Integration (`src/pipeline/b5_qlora.py`)
- Baseline B5 directly inherits all algorithmic modules of B4:
  - `QueryPlanner` (intent classification & out-of-domain guardrail)
  - `SufficiencyScorer` (evidence sufficiency threshold $\tau = 0.38$)
  - `BoundedRetryLoop` ($N_{\text{retry}} \le 2$)
  - `ClaimVerifier` & `TemporalChecker`
- Swaps the parametric generator with `QwenQLoRAGenerator`, dynamically loading `PeftModel.from_pretrained(base_model, "models/qlora_adapter_b5")`.
- When deployed locally on CPU, the PEFT adapter seamlessly integrates with the quantized/float32 base model using 8-thread intra-op parallelism.

## Consequences
- Elevates JSON formatting compliance and domain citation precision on Indonesian banking regulations.
- Enables reproducible cloud training via the standalone turnkey notebook [`notebooks/Fase9_QLoRA_Training_Colab.ipynb`](../../notebooks/Fase9_QLoRA_Training_Colab.ipynb).
- Satisfies the course syllabus requirement for open-weight model parameter adaptation, providing the benchmark foundation for Ablation A4 (with vs without LoRA on B4).
