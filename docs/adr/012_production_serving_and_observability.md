# ADR 012: Production Serving Architecture, Observability, and Fallback Protocol

- **Status**: ACCEPTED
- **Date**: 2026-11-25
- **Author**: TBP Research & Engineering Team
- **Component**: `src.serve` (`api.py`, `ui.py`, `test_serving.py`)
- **PRD Section**: Fase 13 — Serving (FastAPI Backend, Gradio Interactive UI, Trace Logging, and Fallback Protocol)

---

## 1. Context & Problem Statement

Following the completion of Baselines B0–B6, systematic ablation studies (A1–A7), and rigorous red-teaming (Fase 12), the Temporal Regulatory QA system required a robust, enterprise-grade serving layer for deployment to compliance officers, legal analysts, and downstream audit pipelines.

Deploying regulatory LLM systems in the Indonesian financial sector (OJK & Bank Indonesia) presents strict operational and governance challenges:
1. **Deterministic Latency & High Availability**: Financial compliance workflows require robust uptime monitoring (`/health`) and graceful handling of heavy generative LLM workloads.
2. **Auditability & Traceability**: Every legal response must be traceable back to exact retrieved chunks, prompt inputs, tool executions, and security guardrail verdicts via a persistent unique identifier (`trace_id`).
3. **Adversarial Interception**: Incoming queries must be pre-screened in sub-millisecond time by the multi-layer security guardrail before reaching retrieval or generator components.
4. **Attribution Reliability**: The compact base model (Qwen2.5-1.5B-Instruct + QLoRA) can occasionally be over-conservative in JSON formatting, omitting citations while citing clauses in prose. The serving engine must provide a deterministic provenance attribution fallback to guarantee source verification.
5. **Interactive UI**: A clean, accessible browser interface is necessary for non-technical domain experts to query regulations with temporal constraints (`as_of`), view citations, inspect full audit traces, and check system health.

---

## 2. Decision Drivers

- **Zero-Downtime Lifespan Loading**: Index resources (BM25 with 35,828 chunks, Qdrant client) and the QLoRA generator must be initialized once during application startup and shared across requests without redundant memory overhead.
- **Unified Security & Tool Sandboxing**: Requests must pass through `SecurityGuardrail` input screening prior to pipeline execution.
- **Immutable Audit Trails**: Trace records must be written asynchronously to append-only JSONL files (`logs/traces.jsonl`) and queryable via `/trace/{trace_id}`.
- **Deterministic Attribution Guarantee**: If the LLM produces a supported verdict but outputs an empty citations list, the serving layer must extract supporting chunk IDs from the verified claims and populate citations deterministically.
- **Multi-Modal Interface**: Provide both headless REST API endpoints (`/ask`, `/health`, `/trace/{id}`) for programmatic integration and an intuitive Gradio GUI for compliance officers.

---

## 3. Architecture & Design

### 3.1. System Architecture

```
                 +-----------------------------------------+
                 |       Client (Browser / Gradio UI)      |
                 +-------------------+---------------------+
                                     |
                                     | HTTP POST /ask
                                     v
                 +-------------------+---------------------+
                 |           FastAPI Gateway               |
                 |      (CORS, TraceID, Error Handling)    |
                 +-------------------+---------------------+
                                     |
                                     v
                 +-------------------+---------------------+
                 |       SecurityGuardrail (Input)         |
                 |  (ASR: 0.0%, Latency: < 0.3 ms)         |
                 +---------+-----------------------+-------+
                           | Passed                | Blocked
                           v                       v
                 +---------+-------------+   +-----+-------------------+
                 |  B6AgenticBaseline    |   | 400 Sanitized Refusal   |
                 |  - BM25 + Qdrant      |   | (Blocked by Guardrail)  |
                 |  - ToolRegistry Gate  |   +-------------------------+
                 |  - QLoRA Adapter      |
                 |  - Claim Verifier     |
                 +---------+-------------+
                           |
                           v
                 +---------+-------------------------------+
                 | Deterministic Provenance Fallback       |
                 | (Ensures valid citations from chunks)   |
                 +---------+-------------------------------+
                           |
                           v
                 +---------+-------------------------------+
                 | Asynchronous Trace Logger               |
                 | -> logs/traces.jsonl                    |
                 +---------+-------------------------------+
                           |
                           v
                 +---------+-------------------------------+
                 | JSON Response (trace_id, answer, claims,|
                 | citations, verdict, tools, latency)     |
                 +-----------------------------------------+
```

### 3.2. Endpoints Specification

1. **`GET /health`**:
   - Status check returning system uptime, Qdrant connectivity status, BM25 chunk count, model loading state, and guardrail status.
   - Example payload:
     ```json
     {
       "status": "HEALTHY",
       "uptime_sec": 14.2,
       "qdrant_connected": true,
       "bm25_chunks_loaded": 35828,
       "model_loaded": true,
       "guardrail_active": true
     }
     ```

2. **`POST /ask`**:
   - Request schema:
     ```json
     {
       "query": "Pihak mana sajakah yang diwajibkan menerapkan Keuangan Berkelanjutan?",
       "as_of": "2024-01-01"
     }
     ```
   - Workflow:
     - Assigns a UUID4 `trace_id`.
     - Passes through `SecurityGuardrail.screen_input()`. If blocked, immediately logs the refusal and returns a 200/400 safe response with `guardrail_status: "BLOCKED_BY_GUARDRAIL"` in `< 1ms`.
     - Executes `B6AgenticBaseline.run()`, invoking BM25, Qdrant, deterministic temporal tools (`cek_masa_berlaku`, `cari_pasal`, `bandingkan_versi`), QLoRA generation, and `ClaimVerifier`.
     - Runs the Provenance Attribution Fallback protocol.
     - Appends the full audit record to `logs/traces.jsonl`.
     - Returns typed response.

3. **`GET /trace/{trace_id}`**:
   - Retrieves full historical execution telemetry: query, timestamp, latency, prompt tokens, retrieved passage metadata, tool executions, guardrail verdicts, and output claims.

---

## 4. Deterministic Provenance Attribution Fallback Protocol

### 4.1. The Issue
Empirical analysis of B5/B6 revealed that fine-tuning Qwen2.5-1.5B with high temperature/formatting penalties caused the model to become hyper-conservative: while the legal claims generated were factually accurate (98.0% tool correctness, supported verdict), the generative JSON parser sometimes returned `citations: []` to avoid citation penalties.

### 4.2. Algorithmic Fallback
In `src/serve/api.py`, when `citations` is empty but `verdict == "supported"` and verified claims exist:
1. The server inspects the `ClaimVerifier` output and retrieved evidence chunks.
2. For each verified claim, the server identifies the top-ranked retrieved passage containing overlapping statutory identifiers (e.g. `POJK_51_2017`).
3. It extracts the document ID and Pasal number, constructing verified citation objects:
   ```json
   {
     "document_id": passage.doc_id,
     "pasal": passage.pasal or "General",
     "temporal_status": passage.status or "berlaku",
     "source_chunk_id": passage.chunk_id
   }
   ```
4. This guarantees that compliance officers receive actionable statutory references even if the generative module omitted JSON citation keys.

---

## 5. Gradio User Interface (`src/serve/ui.py`)

A tabbed interface was designed for diverse user personas:
1. **Interactive Regulatory QA Tab**:
   - Natural language query input with optional temporal reference date picker (`as_of`).
   - Quick preloaded regulatory examples (e.g. Sustainable Finance POJK 51/2017, PBI Rupiah transactions, Bank Perkreditan Rakyat capital requirements).
   - Display cards for Answer, Formal Claims, Citations table, Active Temporal Tools invoked, Verification Verdict, and Response Latency.
2. **Trace Inspector Tab**:
   - Allows compliance auditors to input any `trace_id` and view the raw execution trace, including retrieved chunks with similarity scores and prompt injection screening logs.
3. **System Health & Telemetry Tab**:
   - Live status monitor polling `/health` with visual indicators for vector store connection, loaded index size, and security guardrail posture.

---

## 6. Verification & Test Results

The serving engine was comprehensively validated via `src/serve/test_serving.py` using Starlette `TestClient`:

| Test Case | Method & Endpoint | Payload / Scenario | Result | Latency / SLA |
| :--- | :--- | :--- | :--- | :--- |
| **System Health** | `GET /health` | Ping server status | **PASSED** (`HEALTHY`, 35,828 chunks, Qdrant true) | 0.05s (< 1s SLA) |
| **Standard Regulatory QA** | `POST /ask` | POJK 51/2017 Sustainable Finance query | **PASSED** (Verdict: `supported`, 2 claims, tool `bandingkan_versi`) | 115.2s (CPU baseline) |
| **Trace Retrieval** | `GET /trace/{trace_id}` | Look up trace `d97f2bdc...` | **PASSED** (Complete JSONL audit record with 9 keys) | 0.04s (< 0.5s SLA) |
| **Adversarial Injection** | `POST /ask` | Prompt injection attack ("Abaikan semua instruksi...") | **PASSED** (`BLOCKED_BY_GUARDRAIL`, sanitized refusal) | **0.001s** (< 5ms SLA) |

---

## 7. Operational Recommendations for Production

1. **Hardware Acceleration**: While CPU serving with 8 threads runs stably (115s per multi-turn agentic query), production environments should deploy on GPU (e.g. NVIDIA T4 or A10G) with TensorRT-LLM or vLLM to achieve `< 3.0s` end-to-end latency.
2. **Background Trace Workers**: In high-throughput settings, trace writes to `logs/traces.jsonl` should be offloaded to a Redis queue or Kafka topic consumed by an asynchronous audit indexing worker.
3. **CORS & Rate Limiting**: The API includes permissive CORS for local UI consumption (`allow_origins=["*"]`). In production, restrict origins to internal corporate domains and attach Redis-backed token bucket rate limiters.
