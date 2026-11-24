# ADR 007: Baseline B4 — Self-Corrective RAG Architecture

## Status
Accepted

## Context
While Baseline B3 (Hybrid RAG with Reciprocal Rank Fusion) significantly improved document recall (57.5% vs 30.0% in B2) and evidence recall (50.0% vs 8.0%), empirical analysis from Fase 6 and Fase 7 revealed crucial systemic failure modes:
1. **High False-Confident Rate on Out-of-Domain / Unanswerable Queries (60.0%)**: B3 attempts to answer queries even when the subject matter (e.g. immigration passports, vehicle parking tariffs) is outside the regulatory jurisdiction of OJK and Bank Indonesia.
2. **Brittle Single-Shot Retrieval on Multi-Hop and Version-Conflict Slices**: When the initial search fails to retrieve an amending regulation or a cross-referenced pasal, B3 generates an answer based on incomplete or superseded evidence.
3. **Unsupported Claims in Generation**: 48.0% of generated atomic claims lacked verifiable grounding in the retrieved passages.

To address these limitations prior to parameter-efficient fine-tuning (B5) and external agentic tool integration (B6), we design and implement **Baseline B4: Self-Corrective RAG**.

## Decision

### 1. Architectural Components
Baseline B4 introduces a closed-loop corrective reasoning flow comprising five dedicated modules:

```
[User Query + as_of]
        │
        ▼
[1. Query Planner] ──(Out-of-Domain?)──► [Immediate Clean Abstain]
        │ (In-Domain Plan)
        ▼
[2. Hybrid Retrieval (BM25 + Qdrant RRF)]
        │
        ▼
[3. Evidence Sufficiency Scorer]
        ├── Sufficient (Score ≥ 0.38) ──► [5. Draft Generator]
        └── Insufficient (Score < 0.38)
                │
        [4. Bounded Retry Loop (Max 2x)]
                ├── Reformulate Query (Pasal / Amendment / General Terms)
                ├── Merge & Re-rank Candidate Passages
                └── Still Insufficient after 2 Retries ──► [Abstain Protocol]
        │
        ▼
[5. Draft Generator (Qwen2.5-1.5B-Instruct)]
        │ (Draft JSON Output)
        ▼
[6. Independent Claim Verifier & Temporal Checker]
        ├── Verified Claims ──► Retain & Ground
        └── Unsupported Claims ──► Prune & Self-Repair (or Abstain if all fail)
        │
        ▼
[7. Final Verified JSON Output + Audit Telemetry]
```

### 2. Module Specifications

#### A. Query Planner (`src/pipeline/planner.py`)
- **Domain Guardrail**: Early lexical and semantic screening against out-of-jurisdiction terms (`OUT_OF_DOMAIN_KEYWORDS`). Immediately triggers clean abstention without wasting inference budget.
- **Intent Classification**: Classifies queries into `single_hop`, `multi_hop`, `konflik_versi`, or `temporal_point_in_time`.
- **Sub-Query Decomposition**: Formulates focused search targets (e.g. appending amendment terms like `"ketentuan penutup mencabut mengubah"` for version conflict queries).

#### B. Evidence Sufficiency Scorer (`src/pipeline/planner.py`)
- Evaluates composite evidence strength:
  $$\text{Score} = 0.35 \cdot \text{NormalizedRRF} + 0.35 \cdot \text{TermCoverage} + 0.30 \cdot \text{PasalMatch}$$
- Threshold: $\tau_{\text{suff}} = 0.38$. If score falls below threshold, signals specific missing aspects (e.g. `low_query_term_coverage`, `target_pasal_not_found`).

#### C. Bounded Retry Mechanism
- Enforces an upper bound of $N_{\text{max\_retries}} = 2$ to prevent infinite loops and budget overruns.
- Each retry reformulates the query dynamically based on the identified missing aspect, re-queries BM25 and Qdrant, and merges candidates into an expanding pool.
- If evidence remains below critical threshold ($< 0.20$) after 2 retries, triggers formal abstention.

#### D. Independent Claim Verifier (`src/verify/claim_verifier.py`)
- Tokenizes each atomic claim and calculates keyword/entity overlap against the evidence passage pool.
- Validates citations against actual retrieved document IDs and articles.
- Integrates `TemporalChecker` (`src/verify/temporal_checker.py`) to verify that cited passages were legally active on the query's `as_of` date.
- Self-Repair: Automatically prunes unsupported claims; if all claims fail verification, converts verdict to `"unanswerable"`.

### 3. Telemetry & Cost Accounting
B4 logs detailed operational telemetry per query:
- `retry_count`: Number of retries attempted (0, 1, or 2).
- `retrieval_calls`: Total retrieval operations executed.
- `sufficiency_score`: Final evidence sufficiency score.
- `is_repaired`: Boolean flag indicating whether the verifier pruned claims.
- `verification_report`: Breakdown of supported vs unsupported claims.

## Consequences
- Directly addresses the 60.0% False-Confident rate and reduces Unsupported Claim Rate.
- Introduces moderate latency overhead for queries requiring retries, while providing bounded computation ($N_{\text{retry}} \le 2$) and zero token cost on early-abstaining out-of-domain queries.
- Establishes the exact baseline against which QLoRA adaptation (B5) and external tool calling (B6) will be benchmarked.
