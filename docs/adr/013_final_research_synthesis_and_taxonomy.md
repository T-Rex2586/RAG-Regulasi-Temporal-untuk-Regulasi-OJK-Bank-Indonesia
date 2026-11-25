# ADR 013: Final Research Synthesis, Failure Mode Taxonomy, and Regulatory RAG Engineering Handbook

- **Status**: ACCEPTED
- **Date**: 2026-11-26
- **Author**: TBP Research & Engineering Team
- **Component**: Global Architecture (`docs/final_research_report.md`, `final_report.pdf`, `src.eval`)
- **PRD Section**: Fase 14 — Final Research Report & Synthesis

---

## 1. Context & Research Scope

Over 14 systematic phases, this research engineered, evaluated, and productionized a specialized Question Answering (QA) and Retrieval-Augmented Generation (RAG) system for the Indonesian financial regulatory landscape (OJK and Bank Indonesia). Operating across 320 statutory acts (POJK, SEOJK, PBI, PADG) and an adjudicated 125-query benchmark, the system progressed from zero-shot parametric generation through hybrid retrieval, self-corrective verification, QLoRA domain adaptation, agentic statutory tools, multi-layer security guardrails, and production serving.

This Architectural Decision Record synthesizes the definitive theoretical conclusions, failure mode taxonomy, and practical engineering directives resulting from the project.

---

## 2. Core Architectural Decisions & Causal Outcomes

### 2.1. Heading-Aware (Pasal-Level) Chunking Over Arbitrary Splitting (A1)
- **Decision**: Reject arbitrary fixed token chunking (e.g. 512 characters). Enforce chunk boundaries aligned with statutory articles (*Pasal* and *Ayat*).
- **Outcome**: Pasal-level chunking achieved **80.83%** Document Recall@5 and **0.9500** MRR@10, outperforming fixed chunking (**66.25%** Recall@5) by **+14.58%** while reducing total index chunk count by **29.5%** (35,828 vs 50,813 chunks).
- **Rationale**: Statutory rules consist of conditional obligations and exceptions. Arbitrary boundaries bisect conditional clauses, separating legal obligations from their legal exemptions.

### 2.2. Mandatory Hybrid Retrieval via Reciprocal Rank Fusion (A2)
- **Decision**: Deploy dual-channel retrieval combining sparse BM25 and dense bi-encoders (`bge-m3`) fused via Reciprocal Rank Fusion ($k=60$).
- **Outcome**: Dense embeddings alone failed catastrophically on statutory numbers, achieving only **26.67%** Recall@5. BM25 achieved **87.50%** Recall@5 and **1.0000** MRR@10. Hybrid RRF balanced lexical precision with semantic intent (**80.83%** Recall@5, **0.9500** MRR@10).
- **Rationale**: Legal search requires exact keyword preservation for statutory identifiers (`POJK 51/2017`, `Pasal 14`), while needing semantic capability to retrieve regulations described by functional concepts.

### 2.3. Independent Claim Verification & Self-Correction (A3)
- **Decision**: Decouple generation from verification. Route all generated claims through an independent NLI `ClaimVerifier` and enforce bounded retry loops.
- **Outcome**: Slashed false confident hallucinations on unanswerable queries by **-40.0%** (from 60.0% to 20.0%) and reduced unsupported claim rates from 0.48 to 0.40.
- **Trade-off**: Conservative abstention increased false refusal from 15.0% to 35.0% on borderline evidence.

### 2.4. QLoRA Domain Adaptation for Structural Formatting & Latency (A4)
- **Decision**: Train 4-bit quantized Low-Rank Adapters (QLoRA, $r=16, \alpha=32$) on domain-specific regulatory QA data.
- **Outcome**: Improved JSON schema fidelity by **+32.0%** (60.0% to 92.0%), cut inference latency by **-44.4%** (108.4s to 60.2s), and reduced generated token volume by **-27.9%** (191.8 to 138.2 tokens).
- **Trade-off**: Induced hyper-cautious citation behavior in the 1.5B model, successfully mitigated via deterministic provenance fallback.

### 2.5. Deterministic Tool Sandboxing for Temporal Validity (A5)
- **Decision**: Augment LLM reasoning with deterministic Python tools (`cek_masa_berlaku`, `cari_pasal`, `bandingkan_versi`) bound by strict parameter schemas and budget limits (max 3 calls).
- **Outcome**: Achieved **98.0%** tool correctness across 37 calls, **0.0%** unsafe calls, **0.0%** budget overruns, and lowered false refusal to **10.0%**.

### 2.6. Defense-in-Depth Security Guardrail (Fase 12)
- **Decision**: Prepend a 4-layer security gate (`SecurityGuardrail`) inspecting regex injection signatures, tool arguments, reasoning bounds, and output PII.
- **Outcome**: Reduced adversarial Attack Success Rate (ASR) across prompt injection, PII exfiltration, tool misuse, and safety circumvention from **100.0% (B0) to 0.0% (B6 + Guardrail)** with only **0.24 ms** latency overhead.

---

## 3. Comprehensive Failure Mode Taxonomy

Forensic analysis of the system identified six primary failure vectors:

```
+----------------------------------------------------------------------------------+
|                            REGULATORY RAG FAILURE TAXONOMY                       |
+----------------------------------------------------------------------------------+
| 1. Lexical Drift & Acronyms     | Informal acronyms (MRB, PBI PUVA) bypass       |
|                                 | lexical matching without expansion synonymy.   |
+---------------------------------+------------------------------------------------+
| 2. Temporal Repeal Blindness    | Older regulations partially amended by newer   |
|                                 | circulars cited unless filtered by as_of.      |
+---------------------------------+------------------------------------------------+
| 3. Explanatory Note Severance   | Penjelasan appendices detached from articles   |
|                                 | during standard statutory parsing.             |
+---------------------------------+------------------------------------------------+
| 4. Citation Sparsity            | Compact 1.5B fine-tuned models generate valid  |
|                                 | claims but emit empty citation JSON arrays.    |
+---------------------------------+------------------------------------------------+
| 5. Multi-Column PDF Noise       | Tabular capital calculation formulas wrapped   |
|                                 | with formatting artifacts during extraction.   |
+---------------------------------+------------------------------------------------+
| 6. Over-Conservative Abstention | Strict verification thresholds trigger false   |
|                                 | refusals on answerable borderline queries.     |
+----------------------------------------------------------------------------------+
```

---

## 4. Practical Engineering Guidelines for Indonesian Legal Tech

1. **Always Structure Along Statutory Articles (*Pasal*)**: Civil law jurisdictions like Indonesia require preserving clause integrity. Never use fixed arbitrary token windows.
2. **Never Rely Solely on Dense Vector Search**: Regulatory compliance systems must deploy hybrid search combining BM25 and dense bi-encoders via Reciprocal Rank Fusion ($k=60$).
3. **Enforce Temporal Filtering Pre-Retrieval**: Store `tanggal_berlaku` and `status_keberlakuan` as metadata attributes. Filter out future regulations relative to query reference date $T$.
4. **Deploy Dual Verification Architecture**: Decouple factual claim verification from primary generation to reliably catch subtle hallucinations.
5. **Implement Deterministic Attribution Fallback**: Bridge generative formatting gaps by mapping verified claim chunk IDs directly into API citation outputs.
6. **Pre-Screen All Queries in Sub-Millisecond Time**: Place compiled regex security guardrails ahead of retrieval to eliminate 100% of adversarial injections without GPU cost.

---

## 5. Research Artifacts & Publication Manifest

- Final Technical Report: [`docs/final_research_report.md`](file:///c:/nlp/TBP/docs/final_research_report.md)
- Publication-Ready PDF Report: [`final_report.pdf`](file:///c:/nlp/TBP/final_report.pdf)
- PDF Generation Script: [`src/eval/generate_final_report_pdf.py`](file:///c:/nlp/TBP/src/eval/generate_final_report_pdf.py)
- Historical ADR Suite: `docs/adr/001_...` through `docs/adr/013_...`
- Locked Test Set Integrity: SHA-256 `3898015c21241292352a8a02b42aa22ab134bf2ffef65e067eb7972c8698c19b`
