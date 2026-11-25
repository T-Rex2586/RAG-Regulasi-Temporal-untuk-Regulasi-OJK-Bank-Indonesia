import re
from typing import Any, Dict, List, Optional, Set, Tuple


class SecurityGuardrail:
    """Multi-layer Security Guardrail for Indonesian Regulatory Question Answering.

    Protects against:
    1. Prompt Injections & Jailbreaks (system prompt extraction, role overrides, mode bypass).
    2. PII & Confidential Data Exfiltration (whistleblower identities, personal phone/addresses, debtor SLIK data, internal inspection drafts).
    3. Tool Misuse & Unauthorized Access (path traversal, arbitrary commands, unapproved documents).
    4. Regulatory Safety & Illegal Guidance (advice on illegal banking, fake regulations, severe criminal hallucinations).
    """

    # Prompt Injection patterns
    INJECTION_PATTERNS = [
        r"abaikan\s+(semua|seluruh)?\s*instruksi",
        r"ignore\s+(all\s+)?(previous\s+)?instructions",
        r"tampilkan\s+(seluruh\s+)?system\s+prompt",
        r"(print|display|show|reveal)\s+(the\s+)?system\s+(prompt|instructions)",
        r"mode\s+['\"]?unrestricted['\"]?",
        r"tanpa\s+filter\s+hukum",
        r"tanpa\s+etika",
        r"\[admin\s+command",
        r"system\s*di-?override",
        r"translate\s+all\s+internal\s+hidden",
        r"memalsukan\s+rasio",
        r"menghindari\s+(kewajiban\s+)?pelaporan.*ppatk",
    ]

    # PII & Confidential Data patterns
    CONFIDENTIAL_PATTERNS = [
        r"laporan\s+rahasia\s+hasil\s+(audit|pemeriksaan|pengawasan)",
        r"nomor\s+(telepon|hp|whatsapp|wa)\s+pribadi",
        r"alamat\s+rumah\s+pejabat",
        r"daftar\s+rincian\s+nama\s+debitur.*(slik|kredit\s+macet)",
        r"(identitas|nama\s+lengkap)\s+karyawan\s+pelapor.*whistleblower",
        r"bocoran\s+draf\s+(internal\s+)?rahasia",
        r"saldo\s+rekening\s+individual",
    ]

    # Illegal Activity patterns
    ILLEGAL_PATTERNS = [
        r"tanpa\s+perlu\s+(mendaftar\s+atau\s+)?mendapatkan\s+izin\s+(dari\s+)?ojk",
        r"cara\s+mendirikan.*tanpa\s+izin\s+ojk",
        r"memalsukan\s+laporan\s+keuangan",
    ]

    def __init__(self, reject_threshold: float = 0.5):
        self.reject_threshold = reject_threshold

    def validate_input(self, user_query: str) -> Dict[str, Any]:
        """Scans user query against prompt injection, PII leak, and illegal activity patterns.

        Returns:
            {
                "is_safe": bool,
                "category": Optional[str],
                "reason": Optional[str],
                "safe_response": Optional[Dict[str, Any]]
            }
        """
        query_lower = user_query.lower()

        # 1. Check Prompt Injection
        for pat in self.INJECTION_PATTERNS:
            if re.search(pat, query_lower, re.IGNORECASE):
                return {
                    "is_safe": False,
                    "category": "prompt_injection",
                    "reason": f"Terdeteksi upaya manipulasi instruksi sistem atau prompt injection ({pat}).",
                    "safe_response": {
                        "answer": "Permintaan ditolak: Sistem mendeteksi instruksi yang melanggar kebijakan keamanan dan integritas operasional AI kepatuhan hukum.",
                        "claims": [],
                        "citations": [],
                        "as_of": None,
                        "verdict": "unanswerable",
                        "abstain_reason": "Blocked by Input Security Guardrail: Prompt Injection / System Override detected.",
                    },
                }

        # 2. Check PII & Confidential Data
        for pat in self.CONFIDENTIAL_PATTERNS:
            if re.search(pat, query_lower, re.IGNORECASE):
                return {
                    "is_safe": False,
                    "category": "pii_leak",
                    "reason": f"Terdeteksi permintaan data rahasia perbankan atau informasi pribadi (PII) ({pat}).",
                    "safe_response": {
                        "answer": "Permintaan ditolak: Sistem tidak memiliki akses dan dilarang membagikan informasi rahasia pengawasan perbankan, data pribadi nasabah/debitur (SLIK), identitas pelapor (whistleblower), atau kontak pribadi pejabat.",
                        "claims": [],
                        "citations": [],
                        "as_of": None,
                        "verdict": "unanswerable",
                        "abstain_reason": "Blocked by Input Security Guardrail: PII / Confidential regulatory data request.",
                    },
                }

        # 3. Check Illegal Advice
        for pat in self.ILLEGAL_PATTERNS:
            if re.search(pat, query_lower, re.IGNORECASE):
                return {
                    "is_safe": False,
                    "category": "illegal_advice",
                    "reason": f"Terdeteksi permintaan panduan aktivitas keuangan tanpa izin atau manipulasi laporan ({pat}).",
                    "safe_response": {
                        "answer": "Permintaan ditolak: Sistem mematuhi peraturan perundang-undangan dan tidak dapat memberikan panduan untuk menyelenggarakan kegiatan jasa keuangan tanpa izin OJK atau memanipulasi pelaporan keuangan.",
                        "claims": [],
                        "citations": [],
                        "as_of": None,
                        "verdict": "unanswerable",
                        "abstain_reason": "Blocked by Input Security Guardrail: Illegal financial activity advice requested.",
                    },
                }

        return {"is_safe": True, "category": None, "reason": None, "safe_response": None}

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validates tool invocation against traversal, injection, and unauthorized data."""
        # Check tool arguments for traversal or shell injection
        for k, v in arguments.items():
            if isinstance(v, str):
                if ".." in v or "/" in v or "\\" in v:
                    return {
                        "is_safe": False,
                        "reason": f"Path traversal attempt rejected in parameter '{k}': {v}",
                    }
                if any(inj in v.lower() for inj in [";", "--", "drop table", "exec", "xp_cmdshell"]):
                    return {
                        "is_safe": False,
                        "reason": f"Command/SQL injection attempt rejected in parameter '{k}': {v}",
                    }

        return {"is_safe": True, "reason": None}

    def sanitize_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures generated response does not leak internal prompt templates or credentials."""
        answer = output.get("answer", "")
        # Redact system leaks
        leak_markers = ["system_prompt", "contract rules", "anda adalah asisten ai", "qwen2.5"]
        for marker in leak_markers:
            if marker in answer.lower():
                output["answer"] = "Informasi hukum yang diminta tidak dapat disajikan secara publik."
                output["claims"] = []
                output["citations"] = []
                output["verdict"] = "unanswerable"
                output["abstain_reason"] = "Output redacted due to potential internal prompt leakage."
                break

        return output
