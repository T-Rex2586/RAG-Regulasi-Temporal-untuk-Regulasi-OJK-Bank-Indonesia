import csv
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional

from src.tools.temporal_tools import cek_masa_berlaku, cari_pasal, bandingkan_versi


class ToolRegistry:
    """Authorized Tool Execution Engine with security guardrails and budget enforcement.

    Implements PRD Section 7 and Section 4 (Fase 10: Baseline B6).
    """

    def __init__(
        self,
        parsed_dir: str = "data/parsed",
        source_register_file: str = "data/source_register.csv",
        max_calls_per_query: int = 3,
    ):
        self.parsed_dir = Path(parsed_dir)
        self.max_calls_per_query = max_calls_per_query
        self.allowed_doc_ids = self._load_allowed_doc_ids(source_register_file)

        # Registered tool functions
        self.tools: Dict[str, Callable] = {
            "cek_masa_berlaku": cek_masa_berlaku,
            "cari_pasal": cari_pasal,
            "bandingkan_versi": bandingkan_versi,
        }

    def _load_allowed_doc_ids(self, register_file: str) -> set:
        allowed = set()
        p = Path(register_file)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    doc_id = row.get("doc_id") or row.get("id")
                    if doc_id:
                        allowed.add(doc_id.strip().lower())
        # Also populate from parsed directory as fallback
        if self.parsed_dir.exists():
            for f in self.parsed_dir.glob("*.json"):
                allowed.add(f.stem.lower())
        return allowed

    def is_doc_authorized(self, doc_id: str) -> bool:
        """Validates that doc_id exists strictly within corpus whitelist."""
        if not doc_id or not isinstance(doc_id, str):
            return False
        # Reject path traversal patterns
        if ".." in doc_id or "/" in doc_id or "\\" in doc_id:
            return False
        clean = Path(doc_id).stem.lower()
        return clean in self.allowed_doc_ids

    def execute(
        self,
        tool_name: str,
        kwargs: Dict[str, Any],
        session_tracker: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Executes a tool call under strict authorization, schema checking, and budget.

        Returns output dict and updates session telemetry in session_tracker.
        """
        if session_tracker is None:
            session_tracker = {
                "calls_count": 0,
                "correct_calls": 0,
                "unsafe_calls": 0,
                "budget_overrun": False,
                "history": [],
            }

        session_tracker["calls_count"] += 1

        # 1. Budget Enforcement
        if session_tracker["calls_count"] > self.max_calls_per_query:
            session_tracker["budget_overrun"] = True
            session_tracker["unsafe_calls"] += 1
            return {
                "success": False,
                "error": "Budget overrun: maximum tool calls limit reached.",
                "tool_name": tool_name,
            }

        # 2. Tool Existence Check
        if tool_name not in self.tools:
            session_tracker["unsafe_calls"] += 1
            return {
                "success": False,
                "error": f"Tool '{tool_name}' tidak terdaftar.",
                "tool_name": tool_name,
            }

        # 3. Security & Document Whitelist Verification
        # Check all doc_id-like arguments
        doc_keys = ["doc_id", "doc_id_a", "doc_id_b"]
        for k in doc_keys:
            if k in kwargs:
                target_doc = str(kwargs[k])
                if not self.is_doc_authorized(target_doc):
                    session_tracker["unsafe_calls"] += 1
                    return {
                        "success": False,
                        "error": f"Otorisasi ditolak: Dokumen '{target_doc}' di luar cakupan korpus resmi.",
                        "tool_name": tool_name,
                    }

        # 4. Safe Tool Execution
        t0 = time.time()
        try:
            fn = self.tools[tool_name]
            output = fn(**kwargs)
            lat = round(time.time() - t0, 3)

            is_correct = output.get("status") != "error" and output.get("found", True) is not False
            if is_correct:
                session_tracker["correct_calls"] += 1

            record = {
                "tool_name": tool_name,
                "kwargs": kwargs,
                "success": True,
                "latency_sec": lat,
                "output": output,
            }
            session_tracker["history"].append(record)

            return {
                "success": True,
                "data": output,
                "latency_sec": lat,
            }

        except Exception as e:
            session_tracker["unsafe_calls"] += 1
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "tool_name": tool_name,
            }
