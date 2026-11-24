# ADR 006: Automated Evaluation Harness and Metric Formulations

## Status
Accepted

## Context
Evaluating Temporal Regulatory QA requires rigorous, multi-faceted metrics that distinguish retrieval quality from generation fidelity, temporal compliance, and abstention behavior. Previous baseline evaluations lacked a unified, reproducible automated harness capable of computing standard information retrieval metrics (`Recall@k`, `MRR@k`, `nDCG@k`) alongside domain-specific regulatory verification metrics (`Citation Precision`, `Evidence Recall`, `Unsupported Claim Rate`, `Temporal Accuracy`, `False Refusal`, `False Confident`).

Furthermore, per the PRD anti-leakage guarantee, the evaluation harness must strictly operate on development splits (`dev.jsonl`) during system design, enforce dynamic SHA-256 verification when evaluating the locked test set, and output standardized artifacts and manifest logs in `runs/<run_id>/`.

## Decision

### 1. Unified Metric Formulations
We implement two specialized evaluator components within `src/eval/`:

#### A. Retrieval Metrics (`RetrievalEvaluator` in `src/eval/retrieval_metrics.py`)
Evaluated against `benchmark/qrels.tsv` using gold graded relevance ($rel \in \{0, 1, 2\}$):
- **Recall@k**: Fraction of gold relevant passages/documents ($rel \ge 1$) present within the top-$k$ retrieved list.
- **MRR@10 (Mean Reciprocal Rank)**:
  $$\text{MRR} = \frac{1}{|Q_{\text{ans}}|} \sum_{q \in Q_{\text{ans}}} \frac{1}{\text{rank}_q^*}$$
  where $\text{rank}_q^*$ is the rank position of the first relevant document.
- **nDCG@10 (Normalized Discounted Cumulative Gain)**:
  $$\text{DCG}@10 = \sum_{i=1}^{10} \frac{2^{rel_i} - 1}{\log_2(i + 1)}, \quad \text{nDCG}@10 = \frac{\text{DCG}@10}{\text{IDCG}@10}$$
- **Multi-Level Evaluation**: Metrics are computed at both document-level and passage-level to provide fair comparison across varying chunking regimes (e.g. fixed-size vs heading-aware).

#### B. Generation & Regulatory QA Metrics (`GenerationEvaluator` in `src/eval/generation_metrics.py`)
- **Citation Precision**: Proportion of citations generated in `citations[]` that match valid ground truth regulations. If query is unanswerable, generating citations is penalized ($0.0$).
- **Evidence Recall**: Proportion of gold legal documents and articles cited or recovered.
- **Unsupported Claim Rate (UCR)**: Proportion of atomic claims in `claims[]` that lack grounded, valid legal citations.
- **Temporal Accuracy**: For point-in-time queries constrained by `as_of`, evaluates whether the cited regulation was legally valid as of that date ($versi\_status = \text{"BERLAKU"}$, not revoked or future-effective).
- **False Refusal Rate**: Proportion of answerable queries where the model erroneously abstained.
- **False Confident Rate**: Proportion of unanswerable queries where the model generated confident claims rather than properly abstaining.
- **Correctness**: Combined score aggregating lexical overlap / F1 with gold reference notes and citation grounding.

### 2. Operational & Latency Profiling
The harness calculates latency distributions:
- Mean Latency (seconds)
- Median ($p50$) Latency
- 95th Percentile ($p95$) Latency
- Average tokens generated and JSON schema compliance rate.

### 3. Execution & Reproducibility
The harness is invocable via a single terminal command:
```powershell
python -m src.eval.harness --run_id <run_id> --split dev
```
Upon execution, it updates `runs/<run_id>/manifest.json` and generates:
- `runs/<run_id>/eval_results_<split>.json` (full machine-readable metrics)
- `runs/<run_id>/eval_report_<split>.md` (formatted Markdown tables with slice breakdown)

## Consequences
- Eliminates manual ad-hoc scoring and guarantees 100% deterministic, audit-ready evaluation across all configurations (B0–B6, ablations A1–A7).
- Enables instant benchmarking of future architectures (B4 self-corrective planner, B5 QLoRA adapter, B6 agentic tools) against these locked B0–B3 baselines.
