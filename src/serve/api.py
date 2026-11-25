import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import yaml

from src.retrieval.bm25_indexer import BM25Index
from src.retrieval.dense_indexer import DenseIndexer
from src.pipeline.b5_qlora import QwenQLoRAGenerator
from src.pipeline.b6_agentic import B6AgenticBaseline
from src.security.guardrail import SecurityGuardrail
from src.tools.registry import ToolRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TemporalRegQA-API")

app = FastAPI(
    title="Temporal Regulatory Question Answering API (OJK & Bank Indonesia)",
    description="Production-grade AI API with Agentic Tools, Self-Correction, and Multi-Layer Security Guardrails.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container
class ServiceContainer:
    pipeline: Optional[B6AgenticBaseline] = None
    guardrail: Optional[SecurityGuardrail] = None
    traces: Dict[str, Dict[str, Any]] = {}
    init_time: datetime = datetime.now(timezone.utc)
    is_ready: bool = False

services = ServiceContainer()


# Request / Response Schemas
class AskRequest(BaseModel):
    query: str = Field(..., description="Pertanyaan hukum atau kepatuhan regulasi OJK/BI")
    as_of: Optional[str] = Field(None, description="Tanggal acuan historis (format YYYY-MM-DD)")
    top_k: Optional[int] = Field(5, description="Jumlah passage regulasi yang diambil")


class CitationItem(BaseModel):
    doc_id: str
    pasal: Optional[str] = None
    ayat: Optional[str] = None


class AskResponse(BaseModel):
    trace_id: str
    answer: str
    claims: List[str]
    citations: List[Dict[str, Any]]
    as_of: Optional[str]
    verdict: str
    abstain_reason: Optional[str]
    latency_sec: float
    tools_invoked: List[str]
    guardrail_status: str


class HealthResponse(BaseModel):
    status: str
    uptime_sec: float
    qdrant_connected: bool
    bm25_chunks_loaded: int
    model_loaded: bool
    guardrail_active: bool


@app.on_event("startup")
def startup_event():
    """Initializes the B6 Agentic Pipeline and Security Guardrail."""
    logger.info("Initializing Temporal RegQA Serving Components...")
    services.guardrail = SecurityGuardrail()

    # Load configuration
    index_cfg_path = Path("configs/index.yaml")
    if index_cfg_path.exists():
        with open(index_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["index"]
    else:
        cfg = {
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_dim": 384,
            "strategies": {
                "heading_aware": {
                    "collection_name": "reg_heading_aware_v1",
                    "bm25_index_file": "index/bm25/heading_aware_v1.pkl",
                }
            },
        }

    # Initialize Indexers
    bm25_path = Path(cfg["strategies"]["heading_aware"]["bm25_index_file"])
    if bm25_path.exists():
        bm25_index = BM25Index.load(bm25_path)
        logger.info(f"BM25 Index loaded: {len(bm25_index.chunk_ids)} chunks.")
    else:
        logger.warning(f"BM25 file not found at {bm25_path}. Creating empty fallback.")
        bm25_index = BM25Index()

    dense_indexer = DenseIndexer(
        host="127.0.0.1",
        port=6334,
        prefer_grpc=True,
        model_name=cfg["embedding_model"],
        vector_dim=cfg["embedding_dim"],
        threads=8,
    )

    # Initialize Generator (QLoRA Adapter on Base Qwen)
    generator = QwenQLoRAGenerator(
        base_model_path="models/Qwen2.5-1.5B-Instruct",
        adapter_path="models/qlora_adapter_b5",
        device="cpu",
    )

    # Initialize B6 Pipeline
    services.pipeline = B6AgenticBaseline(
        generator=generator,
        bm25_index=bm25_index,
        dense_indexer=dense_indexer,
        dense_collection=cfg["strategies"]["heading_aware"]["collection_name"],
        top_k_bm25=10,
        top_k_dense=10,
        final_top_k=5,
        max_tool_calls=3,
    )

    services.is_ready = True
    logger.info("Temporal RegQA Serving Components Successfully Initialized.")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health status check of all system sub-components."""
    uptime = (datetime.now(timezone.utc) - services.init_time).total_seconds()
    bm25_count = len(services.pipeline.bm25_index.chunk_ids) if services.pipeline else 0

    # Probe Qdrant
    qdrant_ok = False
    if services.pipeline and services.pipeline.dense_indexer:
        try:
            collections = services.pipeline.dense_indexer.client.get_collections()
            qdrant_ok = collections is not None
        except Exception:
            qdrant_ok = False

    return HealthResponse(
        status="HEALTHY" if services.is_ready else "INITIALIZING",
        uptime_sec=round(uptime, 1),
        qdrant_connected=qdrant_ok,
        bm25_chunks_loaded=bm25_count,
        model_loaded=services.pipeline is not None,
        guardrail_active=services.guardrail is not None,
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """Answers regulatory question with agentic tools, self-correction, and safety guardrails."""
    if not services.is_ready or not services.pipeline:
        raise HTTPException(status_code=503, detail="System is still initializing.")

    t_start = time.time()
    trace_id = str(uuid.uuid4())
    logger.info(f"[{trace_id}] Received query: '{request.query}' (as_of: {request.as_of})")

    # 1. Layer 1: Security Guardrail Input Validation
    input_security = services.guardrail.validate_input(request.query)
    if not input_security["is_safe"]:
        latency = round(time.time() - t_start, 3)
        safe_resp = input_security["safe_response"]
        trace_record = {
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": request.query,
            "as_of": request.as_of,
            "guardrail_status": "BLOCKED",
            "guardrail_reason": input_security["reason"],
            "response": safe_resp,
            "latency_sec": latency,
            "tools_invoked": [],
        }
        services.traces[trace_id] = trace_record
        _persist_trace(trace_record)
        return AskResponse(
            trace_id=trace_id,
            answer=safe_resp["answer"],
            claims=safe_resp["claims"],
            citations=safe_resp["citations"],
            as_of=request.as_of,
            verdict=safe_resp["verdict"],
            abstain_reason=safe_resp["abstain_reason"],
            latency_sec=latency,
            tools_invoked=[],
            guardrail_status="BLOCKED_BY_GUARDRAIL",
        )

    # 2. Execute B6 Agentic Pipeline with Fallback Protocol
    try:
        pipeline_output = services.pipeline.run({
            "query_id": trace_id,
            "query": request.query,
            "as_of": request.as_of,
        })
    except Exception as e:
        logger.error(f"[{trace_id}] Error in pipeline execution: {str(e)}", exc_info=True)
        # Graceful Fallback Protocol
        latency = round(time.time() - t_start, 3)
        return AskResponse(
            trace_id=trace_id,
            answer="Terjadi kendala teknis saat memproses penelusuran regulasi. Silakan ulangi dengan kata kunci yang lebih spesifik.",
            claims=[],
            citations=[],
            as_of=request.as_of,
            verdict="unanswerable",
            abstain_reason=f"Pipeline exception: {str(e)}",
            latency_sec=latency,
            tools_invoked=[],
            guardrail_status="FALLBACK_ERROR",
        )

    # 3. Deterministic Provenance Attribution (Citation Fallback)
    # If the model produced valid claims supported by evidence but omitted the citation array,
    # map supporting chunk metadata into citations for complete auditability.
    citations = pipeline_output.get("citations", [])
    if not citations and pipeline_output.get("verdict") == "supported":
        v_rep = pipeline_output.get("verification_report", {})
        chunk_map = {p["chunk_id"]: p for p in pipeline_output.get("retrieved_passages", [])}
        for detail in v_rep.get("claim_details", []):
            cid = detail.get("supporting_chunk_id")
            if cid and cid in chunk_map:
                p_meta = chunk_map[cid]
                cit_entry = {
                    "doc_id": p_meta.get("doc_id", "UNKNOWN"),
                    "pasal": p_meta.get("pasal", ""),
                    "ayat": p_meta.get("metadata", {}).get("ayat_nomor", ""),
                }
                if cit_entry not in citations:
                    citations.append(cit_entry)

    latency = round(time.time() - t_start, 2)
    tools_used = pipeline_output.get("tools_invoked", [])

    # 4. Layer 4: Output Sanitization
    sanitized_output = services.guardrail.sanitize_output({
        "answer": pipeline_output.get("answer", ""),
        "claims": pipeline_output.get("claims", []),
        "citations": citations,
        "as_of": request.as_of,
        "verdict": pipeline_output.get("verdict", "supported"),
        "abstain_reason": pipeline_output.get("abstain_reason"),
    })

    # Record trace
    trace_record = {
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": request.query,
        "as_of": request.as_of,
        "guardrail_status": "PASSED",
        "response": sanitized_output,
        "latency_sec": latency,
        "tools_invoked": tools_used,
        "retrieved_passages": [
            {"chunk_id": p.get("chunk_id"), "doc_id": p.get("doc_id"), "score": p.get("rrf_score")}
            for p in pipeline_output.get("retrieved_passages", [])
        ],
    }
    services.traces[trace_id] = trace_record
    _persist_trace(trace_record)

    return AskResponse(
        trace_id=trace_id,
        answer=sanitized_output["answer"],
        claims=sanitized_output["claims"],
        citations=sanitized_output["citations"],
        as_of=request.as_of,
        verdict=sanitized_output["verdict"],
        abstain_reason=sanitized_output["abstain_reason"],
        latency_sec=latency,
        tools_invoked=tools_used,
        guardrail_status="PASSED",
    )


@app.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    """Retrieves full execution trace and telemetry for a specific request ID."""
    trace = services.traces.get(trace_id)
    if not trace:
        # Check logs/traces.jsonl fallback
        trace_file = Path("logs/traces.jsonl")
        if trace_file.exists():
            with open(trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if obj.get("trace_id") == trace_id:
                            return obj
                    except Exception:
                        continue
        raise HTTPException(status_code=404, detail=f"Trace ID '{trace_id}' not found.")
    return trace


def _persist_trace(trace_data: Dict[str, Any]):
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "traces.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_data, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to persist trace log: {e}")


if __name__ == "__main__":
    uvicorn.run("src.serve.api:app", host="0.0.0.0", port=8000, reload=False)
